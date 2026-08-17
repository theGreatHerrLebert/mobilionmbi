//! A pure-Rust reader for **MOBILion `.mbi`** ion mobility - mass spectrometry
//! files, with no dependency on the vendor MBI SDK.
//!
//! `.mbi` is HDF5 underneath. Each frame stores its sparse IM/MS plane in three
//! datasets that map onto CSR with one expansion step:
//!
//! | dataset | meaning |
//! |---|---|
//! | `data-counts` | intensities, one per non-zero point (CSR `data`) |
//! | `data-positions` | `[start, end)` runs of TOF indices; expanding gives CSR `indices` |
//! | `index-counts` | per-drift-scan cumulative offsets (CSR `indptr`) |
//!
//! ```no_run
//! use mobilionmbi::MbiFile;
//! let f = MbiFile::open("run.mbi")?;
//! println!("{} frames", f.n_frames());
//! let frame = f.frame(600)?;
//! let cal = f.calibration(600)?;
//! println!("{} points, first m/z {:.4}", frame.nnz(), cal.index_to_mz(frame.indices[0]));
//! # Ok::<(), mobilionmbi::Error>(())
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use hdf5_metno as hdf5;

pub mod calibration;
pub mod ccs;
pub mod writer;
pub use calibration::TofCalibration;
pub use ccs::{CcsCalibration, DriftAxis};
pub use writer::{FrameExtras, MbiWriter};

/// Errors produced while reading a `.mbi` file.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
    #[error("malformed calibration JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("frame index {index} out of range (file has {n_frames} frames, 1-based)")]
    FrameOutOfRange { index: usize, n_frames: usize },
    #[error("missing metadata key: {0}")]
    MissingMetadata(String),
    #[error("inconsistent frame {index}: {detail}")]
    Inconsistent { index: usize, detail: String },
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

/// One collision-energy setpoint within a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CeSetpoint {
    /// How many intervals this setpoint covers.
    pub interval_count: u32,
    /// Collision energy in volts.
    pub energy_v: f64,
    /// Length of one interval, in milliseconds.
    pub interval_ms: f64,
}

#[derive(serde::Deserialize)]
struct CeJson {
    #[serde(default)]
    interval_ms: f64,
    #[serde(default)]
    setpoints: Vec<CeSetpointJson>,
}

#[derive(serde::Deserialize)]
struct CeSetpointJson {
    #[serde(default)]
    interval_count: u32,
    #[serde(default)]
    collision_energy_v: f64,
}

/// One frame: a sparse `drift scan x TOF index` plane, in CSR layout.
#[derive(Debug, Clone)]
pub struct Frame {
    /// 1-based frame index, as used by the vendor API.
    pub index: usize,
    /// Intensities, one per non-zero point.
    pub data: Vec<i32>,
    /// TOF bin index of each non-zero point.
    pub indices: Vec<u64>,
    /// Row offsets, `n_rows + 1` entries.
    pub indptr: Vec<u64>,
    /// Number of drift scans.
    pub n_rows: usize,
    /// Width of the TOF axis.
    pub n_cols: usize,
}

impl Frame {
    /// Number of non-zero points.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Total ion current: the summed intensity of the frame.
    pub fn total_intensity(&self) -> i64 {
        self.data.iter().map(|&v| v as i64).sum()
    }

    /// Expand to coordinate form, returning `(rows, cols, values)`.
    pub fn to_coo(&self) -> (Vec<u64>, &[u64], &[i32]) {
        let mut rows = Vec::with_capacity(self.nnz());
        for row in 0..self.n_rows {
            let (a, b) = (self.indptr[row] as usize, self.indptr[row + 1] as usize);
            rows.extend(std::iter::repeat(row as u64).take(b - a));
        }
        (rows, &self.indices, &self.data)
    }

    /// Indices of drift scans holding at least one non-zero point.
    pub fn nonzero_scan_indices(&self) -> Vec<u64> {
        (0..self.n_rows)
            .filter(|&r| self.indptr[r + 1] > self.indptr[r])
            .map(|r| r as u64)
            .collect()
    }
}

/// An open `.mbi` file.
pub struct MbiFile {
    file: hdf5::File,
    n_frames: usize,
    sample_rate: f64,
    global: BTreeMap<String, String>,
}

impl MbiFile {
    /// Open a file and read its global description.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = hdf5::File::open(path)?;
        let global = read_attr_map(&file.group("data-description/global-description")?)?;
        let n_frames = global
            .get("acq-num-frames")
            .and_then(|v| v.trim().parse::<usize>().ok())
            .ok_or_else(|| Error::MissingMetadata("acq-num-frames".into()))?;
        let sample_rate = global
            .get("adc-sample-rate")
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(2.0e9);
        Ok(Self { file, n_frames, sample_rate, global })
    }

    /// Number of frames. Frame indices are **1-based**, as in the vendor API.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Digitiser sampling rate, samples/second.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// The file-level metadata, as stored (values are raw strings, some JSON).
    pub fn global_metadata(&self) -> &BTreeMap<String, String> {
        &self.global
    }

    /// Per-frame metadata, as stored.
    pub fn frame_metadata(&self, index: usize) -> Result<BTreeMap<String, String>> {
        self.check_index(index)?;
        read_attr_map(
            &self
                .file
                .group(&format!("data-description/frame-{index}-metadata"))?,
        )
    }

    /// The mass calibration for a frame. Constant across frames in practice, but
    /// stored per frame, so read it per frame.
    pub fn calibration(&self, index: usize) -> Result<TofCalibration> {
        let md = self.frame_metadata(index)?;
        let json = md
            .get("cal-ms-traditional")
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| Error::MissingMetadata("cal-ms-traditional".into()))?;
        Ok(TofCalibration::from_json(json, self.sample_rate)?)
    }

    /// Read one frame's sparse plane.
    pub fn frame(&self, index: usize) -> Result<Frame> {
        self.check_index(index)?;
        let g = self.file.group(&format!("data-cubes/frame-{index}-data"))?;

        let data: Vec<i32> = g.dataset("data-counts")?.read_raw()?;
        let positions: Vec<i32> = g.dataset("data-positions")?.read_raw()?;
        let offsets: Vec<i32> = g.dataset("index-counts")?.read_raw()?;

        if positions.len() % 2 != 0 {
            return Err(Error::Inconsistent {
                index,
                detail: format!("data-positions has odd length {}", positions.len()),
            });
        }

        // Expand [start, end) runs into explicit TOF indices.
        let mut indices = Vec::with_capacity(data.len());
        for pair in positions.chunks_exact(2) {
            let (start, end) = (pair[0] as i64, pair[1] as i64);
            if end < start {
                return Err(Error::Inconsistent {
                    index,
                    detail: format!("run [{start}, {end}) is reversed"),
                });
            }
            indices.extend((start..end).map(|v| v as u64));
        }
        if indices.len() != data.len() {
            return Err(Error::Inconsistent {
                index,
                detail: format!(
                    "runs expand to {} points but there are {} intensities",
                    indices.len(),
                    data.len()
                ),
            });
        }

        // index-counts holds the per-scan start offset into data-counts; the CSR
        // row pointer is that plus a closing entry.
        let n_rows = offsets.len();
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.extend(offsets.iter().map(|&v| v as u64));
        indptr.push(data.len() as u64);

        let n_cols = self
            .global
            .get("adc-record-size")
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);

        Ok(Frame { index, data, indices, indptr, n_rows, n_cols })
    }

    /// The collision-energy setpoints of a frame, in order.
    ///
    /// Stored as `frm-collision-energy`, shaped for stepping CE across a frame:
    /// `{"interval_ms": …, "setpoints": [{"interval_count": …,
    /// "collision_energy_v": …}]}`. Every file seen so far carries exactly one
    /// setpoint per frame, but the structure allows more, so this returns them
    /// all rather than pretending otherwise.
    pub fn collision_energy_setpoints(&self, index: usize) -> Result<Vec<CeSetpoint>> {
        let md = self.frame_metadata(index)?;
        let Some(raw) = md
            .get("frm-collision-energy")
            .filter(|s| !s.trim().is_empty())
        else {
            return Ok(Vec::new());
        };
        let parsed: CeJson = serde_json::from_str(raw)?;
        Ok(parsed
            .setpoints
            .into_iter()
            .map(|s| CeSetpoint {
                interval_count: s.interval_count,
                energy_v: s.collision_energy_v,
                interval_ms: parsed.interval_ms,
            })
            .collect())
    }

    /// The frame's collision energy, when it has exactly one setpoint.
    ///
    /// Returns `None` for a frame that steps CE across intervals — use
    /// [`Self::collision_energy_setpoints`] there — or that stores none.
    pub fn collision_energy(&self, index: usize) -> Result<Option<f64>> {
        let sp = self.collision_energy_setpoints(index)?;
        Ok(match sp.len() {
            1 => Some(sp[0].energy_v),
            _ => None,
        })
    }

    /// The file's CCS calibration, from the global `cal-ccs` attribute.
    ///
    /// Returns `Ok(None)` when the acquisition carried no CCS calibration, which
    /// is common — the attribute is then absent or an empty string.
    pub fn ccs_calibration(&self) -> Result<Option<CcsCalibration>> {
        // The vendor's own constant for the traditional variant is spelled
        // "cal-css-traditional" (sic), so accept that too rather than miss a
        // calibration over a typo in the format.
        let raw = ["cal-ccs", "cal-css-traditional", "cal-ccs-traditional"]
            .iter()
            .filter_map(|k| self.global.get(*k))
            .find(|s| !s.trim().is_empty());
        match raw {
            None => Ok(None),
            Some(json) => Ok(Some(CcsCalibration::from_json(json)?)),
        }
    }

    /// The drift axis of a frame: scan spacing and scan count.
    pub fn drift_axis(&self, index: usize) -> Result<DriftAxis> {
        let md = self.frame_metadata(index)?;
        let period_ms = md
            .get("frm-dt-period")
            .and_then(|v| v.trim().parse::<f64>().ok())
            .ok_or_else(|| Error::MissingMetadata("frm-dt-period".into()))?;
        let n_scans = md
            .get("frm-num-bin-dt")
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        Ok(DriftAxis { period_ms, n_scans })
    }

    /// Per-drift-scan trigger timestamps for a frame.
    pub fn frame_trigger_timestamps(&self, index: usize) -> Result<Vec<f64>> {
        self.check_index(index)?;
        let g = self.file.group(&format!("data-cubes/frame-{index}-data"))?;
        Ok(g.dataset("trigger-timestamps")?.read_raw()?)
    }

    /// Per-drift-scan total intensity for a frame, as stored.
    ///
    /// Derivable from the frame data, but read it back when checking a file
    /// rather than recomputing what you are trying to verify.
    pub fn frame_at_tic(&self, index: usize) -> Result<Vec<i64>> {
        self.check_index(index)?;
        let g = self.file.group(&format!("data-cubes/frame-{index}-data"))?;
        Ok(g.dataset("at-tic")?.read_raw()?)
    }

    /// The per-frame TIC series stored at the file root.
    pub fn rt_tic(&self) -> Result<Vec<i64>> {
        Ok(self.file.dataset("rt-tic")?.read_raw()?)
    }

    /// Retention time of each frame, in the file's own units.
    pub fn retention_times(&self) -> Result<Vec<f64>> {
        (1..=self.n_frames)
            .map(|i| {
                let md = self.frame_metadata(i)?;
                Ok(md
                    .get("frm-start-time")
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .unwrap_or(f64::NAN))
            })
            .collect()
    }

    fn check_index(&self, index: usize) -> Result<()> {
        if index == 0 || index > self.n_frames {
            Err(Error::FrameOutOfRange { index, n_frames: self.n_frames })
        } else {
            Ok(())
        }
    }
}

/// Read every attribute of a group as a string, whatever its HDF5 string flavour.
fn read_attr_map(group: &hdf5::Group) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for name in group.attr_names()? {
        let attr = group.attr(&name)?;
        let value = read_attr_string(&attr).unwrap_or_default();
        out.insert(name, value);
    }
    Ok(out)
}

fn read_attr_string(attr: &hdf5::Attribute) -> Option<String> {
    use hdf5::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};
    if let Ok(v) = attr.read_scalar::<VarLenUnicode>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<VarLenAscii>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<FixedUnicode<1024>>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<FixedAscii<1024>>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<f64>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<i64>() {
        return Some(v.to_string());
    }
    None
}
