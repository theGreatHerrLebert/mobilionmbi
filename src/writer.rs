//! Writing `.mbi` files.
//!
//! The encoder is the exact inverse of the reader: a CSR plane becomes
//! `data-counts` (intensities verbatim) plus `data-positions` (`[start, end)`
//! runs of consecutive TOF indices, never spanning a drift scan), with
//! `index-counts` and `index-positions` giving the per-scan cumulative offsets
//! into those two arrays respectively.

use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

use hdf5_metno as hdf5;
use hdf5::types::VarLenUnicode;

use crate::{Error, Frame, Result};

/// gzip level the vendor's own writer uses.
const DEFLATE_LEVEL: u8 = 4;

/// Everything that varies per frame besides the sparse plane itself.
#[derive(Debug, Clone, Default)]
pub struct FrameExtras {
    /// Per-drift-scan trigger timestamps. Length must match the scan count.
    pub trigger_timestamps: Vec<f64>,
    /// Frame metadata attributes, written verbatim.
    pub metadata: BTreeMap<String, String>,
}

/// Builds a `.mbi` file.
pub struct MbiWriter {
    file: hdf5::File,
    frame_tics: Vec<i64>,
    n_frames: usize,
}

impl MbiWriter {
    /// Create a new file, truncating any existing one.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = hdf5::File::create(path)?;
        file.create_group("data-cubes")?;
        file.create_group("data-description")?;
        Ok(Self { file, frame_tics: Vec::new(), n_frames: 0 })
    }

    /// Write the file-level attributes under `data-description/global-description`.
    ///
    /// `acq-num-frames` is rewritten by [`Self::finish`] to the true frame count,
    /// so callers need not keep it consistent by hand.
    pub fn write_global_metadata(&mut self, metadata: &BTreeMap<String, String>) -> Result<()> {
        let g = match self.file.group("data-description/global-description") {
            Ok(g) => g,
            Err(_) => self.file.create_group("data-description/global-description")?,
        };
        write_attrs(&g, metadata)
    }

    /// Append a frame. Frames must be written in order, starting at 1.
    pub fn write_frame(&mut self, frame: &Frame, extras: &FrameExtras) -> Result<()> {
        let index = frame.index;
        if index != self.n_frames + 1 {
            return Err(Error::Inconsistent {
                index,
                detail: format!("frames must be written in order; expected {}", self.n_frames + 1),
            });
        }
        if frame.indptr.len() != frame.n_rows + 1 {
            return Err(Error::Inconsistent {
                index,
                detail: format!(
                    "indptr has {} entries but n_rows is {}",
                    frame.indptr.len(),
                    frame.n_rows
                ),
            });
        }
        if frame.indices.len() != frame.data.len() {
            return Err(Error::Inconsistent {
                index,
                detail: format!(
                    "{} indices for {} intensities",
                    frame.indices.len(),
                    frame.data.len()
                ),
            });
        }
        if frame.indptr.first() != Some(&0) {
            return Err(Error::Inconsistent {
                index,
                detail: "indptr must start at 0".to_string(),
            });
        }
        // A short final offset would write intensities into data-counts (and into
        // the frame's TIC) that no run references — a file that reads back wrong.
        if frame.indptr.last() != Some(&(frame.data.len() as u64)) {
            return Err(Error::Inconsistent {
                index,
                detail: format!(
                    "indptr ends at {:?} but there are {} intensities",
                    frame.indptr.last(),
                    frame.data.len()
                ),
            });
        }
        if !extras.trigger_timestamps.is_empty()
            && extras.trigger_timestamps.len() != frame.n_rows
        {
            return Err(Error::Inconsistent {
                index,
                detail: format!(
                    "{} trigger timestamps for {} scans",
                    extras.trigger_timestamps.len(),
                    frame.n_rows
                ),
            });
        }

        let (positions, run_offsets, at_tic) = encode_runs(frame)?;

        let g = self.file.create_group(&format!("data-cubes/frame-{index}-data"))?;

        write_1d_i32(&g, "data-counts", &frame.data)?;
        write_2d_i32_pairs(&g, "data-positions", &positions)?;

        // index-counts holds the per-scan start offset into data-counts, i.e. indptr
        // without its closing entry; index-positions does the same for the runs.
        let index_counts: Vec<i32> = frame.indptr[..frame.n_rows]
            .iter()
            .map(|&v| i32::try_from(v))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| Error::Inconsistent {
                index,
                detail: "a row offset exceeds the i32 range the format stores".to_string(),
            })?;
        write_1d_i32(&g, "index-counts", &index_counts)?;
        write_1d_i32(&g, "index-positions", &run_offsets)?;
        write_1d_i64(&g, "at-tic", &at_tic)?;

        let ts = if extras.trigger_timestamps.is_empty() {
            vec![0.0; frame.n_rows]
        } else {
            extras.trigger_timestamps.clone()
        };
        write_1d_f64(&g, "trigger-timestamps", &ts)?;

        let md = self
            .file
            .create_group(&format!("data-description/frame-{index}-metadata"))?;
        write_attrs(&md, &extras.metadata)?;

        self.frame_tics.push(frame.total_intensity());
        self.n_frames += 1;
        Ok(())
    }

    /// Write the run-level TIC series, fix up `acq-num-frames`, and close.
    pub fn finish(self) -> Result<()> {
        write_1d_i64_root(&self.file, "rt-tic", &self.frame_tics)?;

        // The frame count is authoritative here, not whatever the caller passed in.
        if let Ok(g) = self.file.group("data-description/global-description") {
            let mut one = BTreeMap::new();
            one.insert("acq-num-frames".to_string(), self.n_frames.to_string());
            write_attrs(&g, &one)?;
        }
        self.file.flush()?;
        Ok(())
    }
}

/// CSR -> (`[start, end)` runs, per-scan run offsets, per-scan TIC).
///
/// Runs are cut at drift-scan boundaries even when the TOF indices would be
/// contiguous across them, because the per-scan offsets must land on run
/// boundaries for the reader to segment the frame.
fn encode_runs(frame: &Frame) -> Result<(Vec<[i32; 2]>, Vec<i32>, Vec<i64>)> {
    let mut positions: Vec<[i32; 2]> = Vec::new();
    let mut run_offsets: Vec<i32> = Vec::with_capacity(frame.n_rows);
    let mut at_tic: Vec<i64> = Vec::with_capacity(frame.n_rows);

    for row in 0..frame.n_rows {
        run_offsets.push(positions.len() as i32);

        let (a, b) = (frame.indptr[row] as usize, frame.indptr[row + 1] as usize);
        if b < a || b > frame.data.len() {
            return Err(Error::Inconsistent {
                index: frame.index,
                detail: format!("indptr segment [{a}, {b}) is out of bounds"),
            });
        }

        at_tic.push(frame.data[a..b].iter().map(|&v| v as i64).sum());

        // TOF indices are stored as i32 on disk, so anything above i32::MAX cannot
        // be represented; casting would silently corrupt it. Rows must also be
        // strictly increasing, or the runs below would misrepresent the plane.
        for k in a..b {
            if frame.indices[k] > i32::MAX as u64 {
                return Err(Error::Inconsistent {
                    index: frame.index,
                    detail: format!(
                        "TOF index {} exceeds the i32 range the format stores",
                        frame.indices[k]
                    ),
                });
            }
            if k > a && frame.indices[k] <= frame.indices[k - 1] {
                return Err(Error::Inconsistent {
                    index: frame.index,
                    detail: format!(
                        "row {row} TOF indices are not strictly increasing at {k}"
                    ),
                });
            }
        }

        let mut i = a;
        while i < b {
            let start = frame.indices[i];
            let mut end = start + 1;
            let mut j = i + 1;
            while j < b && frame.indices[j] == end {
                end += 1;
                j += 1;
            }
            positions.push([start as i32, end as i32]);
            i = j;
        }
        if positions.len() > i32::MAX as usize {
            return Err(Error::Inconsistent {
                index: frame.index,
                detail: "run count exceeds the i32 range the format stores".to_string(),
            });
        }
    }
    Ok((positions, run_offsets, at_tic))
}

fn chunk_for(len: usize) -> usize {
    // The vendor writer chunks at roughly half the dataset; mirror that, with a
    // floor so tiny frames do not produce degenerate chunks.
    len.div_ceil(2).max(1)
}

fn write_1d_i32(g: &hdf5::Group, name: &str, data: &[i32]) -> Result<()> {
    let ds = g
        .new_dataset::<i32>()
        .shape([data.len()])
        .chunk([chunk_for(data.len())])
        .deflate(DEFLATE_LEVEL)
        .create(name)?;
    ds.write(data)?;
    Ok(())
}

fn write_1d_i64(g: &hdf5::Group, name: &str, data: &[i64]) -> Result<()> {
    let ds = g
        .new_dataset::<i64>()
        .shape([data.len()])
        .chunk([chunk_for(data.len())])
        .deflate(DEFLATE_LEVEL)
        .create(name)?;
    ds.write(data)?;
    Ok(())
}

fn write_1d_f64(g: &hdf5::Group, name: &str, data: &[f64]) -> Result<()> {
    let ds = g
        .new_dataset::<f64>()
        .shape([data.len()])
        .chunk([chunk_for(data.len())])
        .deflate(DEFLATE_LEVEL)
        .create(name)?;
    ds.write(data)?;
    Ok(())
}

fn write_1d_i64_root(f: &hdf5::File, name: &str, data: &[i64]) -> Result<()> {
    let ds = f
        .new_dataset::<i64>()
        .shape([data.len()])
        .chunk([chunk_for(data.len())])
        .deflate(DEFLATE_LEVEL)
        .create(name)?;
    ds.write(data)?;
    Ok(())
}

fn write_2d_i32_pairs(g: &hdf5::Group, name: &str, pairs: &[[i32; 2]]) -> Result<()> {
    let flat: Vec<i32> = pairs.iter().flat_map(|p| [p[0], p[1]]).collect();
    let ds = g
        .new_dataset::<i32>()
        .shape([pairs.len(), 2])
        .chunk([chunk_for(pairs.len()), 2])
        .deflate(DEFLATE_LEVEL)
        .create(name)?;
    ds.write_raw(&flat)?;
    Ok(())
}

fn write_attrs(g: &hdf5::Group, metadata: &BTreeMap<String, String>) -> Result<()> {
    for (k, v) in metadata {
        // Rewrite rather than duplicate when the attribute already exists.
        if g.attr(k).is_ok() {
            g.delete_attr(k)?;
        }
        let attr = g.new_attr::<VarLenUnicode>().create(k.as_str())?;
        let value = VarLenUnicode::from_str(v).map_err(|_| Error::Inconsistent {
            index: 0,
            detail: format!("attribute {k} is not valid unicode"),
        })?;
        attr.write_scalar(&value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MbiFile;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("mobilionmbi-test-{name}-{}.mbi", std::process::id()));
        p
    }

    /// A plane with runs of varying length, gaps, and empty scans — the shapes
    /// the run encoder has to get right.
    fn sample_frame() -> Frame {
        // scan 0: 1005..1008 and 2000..2002   scan 1: empty   scan 2: single point
        let data = vec![10, 11, 12, 20, 21, 99];
        let indices = vec![1005, 1006, 1007, 2000, 2001, 4242];
        let indptr = vec![0, 5, 5, 6];
        Frame { index: 1, data, indices, indptr, n_rows: 3, n_cols: 8192 }
    }

    #[test]
    fn run_encoder_splits_on_gaps_and_scans() {
        let f = sample_frame();
        let (positions, run_offsets, at_tic) = encode_runs(&f).unwrap();
        assert_eq!(positions, vec![[1005, 1008], [2000, 2002], [4242, 4243]]);
        // scan 0 starts at run 0, scan 1 (empty) and scan 2 both start at run 2.
        assert_eq!(run_offsets, vec![0, 2, 2]);
        assert_eq!(at_tic, vec![74, 0, 99]);
    }

    #[test]
    fn write_then_read_is_identical() {
        let path = temp_path("roundtrip");
        let frame = sample_frame();

        let mut global = BTreeMap::new();
        global.insert("adc-record-size".to_string(), "8192".to_string());
        global.insert("adc-sample-rate".to_string(), "2.0e+09".to_string());
        let mut fmeta = BTreeMap::new();
        fmeta.insert(
            "cal-ms-traditional".to_string(),
            r#"{"slope": 0.34, "intercept": 0.1, "mz_residual_terms": []}"#.to_string(),
        );

        let mut w = MbiWriter::create(&path).unwrap();
        w.write_global_metadata(&global).unwrap();
        w.write_frame(
            &frame,
            &FrameExtras {
                trigger_timestamps: vec![0.0, 1.0, 2.0],
                metadata: fmeta.clone(),
            },
        )
        .unwrap();
        w.finish().unwrap();

        let f = MbiFile::open(&path).unwrap();
        assert_eq!(f.n_frames(), 1, "finish() must fix up acq-num-frames");
        let got = f.frame(1).unwrap();
        assert_eq!(got.data, frame.data);
        assert_eq!(got.indices, frame.indices);
        assert_eq!(got.indptr, frame.indptr);
        assert_eq!(got.n_rows, frame.n_rows);
        assert_eq!(f.frame_at_tic(1).unwrap(), vec![74, 0, 99]);
        assert_eq!(f.frame_trigger_timestamps(1).unwrap(), vec![0.0, 1.0, 2.0]);
        assert_eq!(f.rt_tic().unwrap(), vec![173]);
        assert_eq!(f.frame_metadata(1).unwrap(), fmeta);
        assert!(f.calibration(1).is_ok());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn frames_must_be_written_in_order() {
        let path = temp_path("order");
        let mut w = MbiWriter::create(&path).unwrap();
        let mut f = sample_frame();
        f.index = 2;
        let err = w.write_frame(&f, &FrameExtras::default()).unwrap_err();
        assert!(matches!(err, Error::Inconsistent { .. }), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_indices_shorter_than_data() {
        let path = temp_path("shortidx");
        let mut w = MbiWriter::create(&path).unwrap();
        let mut f = sample_frame();
        f.indices.pop(); // would have panicked indexing indices[i] in encode_runs
        let err = w.write_frame(&f, &FrameExtras::default()).unwrap_err();
        assert!(matches!(err, Error::Inconsistent { .. }), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_indptr_that_does_not_close_on_the_data() {
        let path = temp_path("shortptr");
        let mut w = MbiWriter::create(&path).unwrap();
        let mut f = sample_frame();
        // Trailing intensities no run would reference: they would still land in
        // data-counts and in rt-tic, producing a file that reads back wrong.
        *f.indptr.last_mut().unwrap() -= 1;
        let err = w.write_frame(&f, &FrameExtras::default()).unwrap_err();
        assert!(matches!(err, Error::Inconsistent { .. }), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_tof_index_beyond_the_i32_the_format_stores() {
        let path = temp_path("bigtof");
        let mut w = MbiWriter::create(&path).unwrap();
        let mut f = sample_frame();
        f.indices[5] = i32::MAX as u64 + 1; // would have silently truncated
        let err = w.write_frame(&f, &FrameExtras::default()).unwrap_err();
        assert!(matches!(err, Error::Inconsistent { .. }), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_non_increasing_tof_within_a_row() {
        let path = temp_path("unsorted");
        let mut w = MbiWriter::create(&path).unwrap();
        let mut f = sample_frame();
        f.indices.swap(0, 1); // descending pair inside row 0
        let err = w.write_frame(&f, &FrameExtras::default()).unwrap_err();
        assert!(matches!(err, Error::Inconsistent { .. }), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_mismatched_trigger_timestamps() {
        let path = temp_path("triggers");
        let mut w = MbiWriter::create(&path).unwrap();
        let extras = FrameExtras { trigger_timestamps: vec![0.0], metadata: BTreeMap::new() };
        let err = w.write_frame(&sample_frame(), &extras).unwrap_err();
        assert!(matches!(err, Error::Inconsistent { .. }), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }
}
