//! Write a small `.mbi` file from nothing — the path a simulator needs.
//!
//! ```text
//! cargo run --release --example synthesize -- out.mbi
//! ```

use std::collections::BTreeMap;

use mobilionmbi::{Frame, FrameExtras, MbiFile, MbiWriter, TofCalibration};

const N_FRAMES: usize = 5;
const N_SCANS: usize = 64; // drift bins
const TOF_WIDTH: usize = 232992;
const SAMPLE_RATE: f64 = 2.0e9;

fn global_metadata() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("acq-num-frames".into(), N_FRAMES.to_string());
    m.insert("adc-record-size".into(), TOF_WIDTH.to_string());
    m.insert("adc-sample-rate".into(), format!("{SAMPLE_RATE:e}"));
    m.insert("acq-mode".into(), "HRIM".into());
    m.insert("acq-collection-mode".into(), "SIFF".into());
    m.insert("acq-ms-level".into(), "1".into());
    m.insert("acq-ms-model".into(), "synthetic".into());
    m.insert("acq-timestamp".into(), "2026-01-01 00:00:00.000000+00:00".into());
    // A CCS calibration, so the drift axis maps to cross sections. Real
    // acquisitions often omit this; both our reader and the vendor SDK treat it
    // as optional.
    m.insert("cal-ccs".into(), CCS_CAL_JSON.into());
    m
}

/// Polynomial in arrival time (ms), lowest-order first, plus the drift gas.
const CCS_CAL_JSON: &str = r#"{"coefficients": [12.5, 3.75], "min": 100.0, "max": 900.0,
 "degree": 1, "at_surfing": 0.5, "ccaps": 0, "Mass Flow.gas type": "N2",
 "gas mass": 28.0134, "version": "1.0.0"}"#;
// NB: the version must be X.Y.Z — the SDK rejects "1.0" and "1" outright, and
// treats 0.0.x (or a missing version) as the legacy coefficient scaling.

fn frame_metadata(index: usize, cal_json: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("frm-metadata-id".into(), index.to_string());
    m.insert("frm-dt-period".into(), "0.11958356190006342".into());
    m.insert("frm-num-bin-dt".into(), N_SCANS.to_string());
    m.insert("frm-start-time".into(), format!("{:.6}", index as f64 * 0.5));
    m.insert("frm-polarity".into(), "positive".into());
    m.insert("cal-ms-traditional".into(), cal_json.into());
    m
}

/// A few Gaussian-ish peaks per drift scan, so the plane looks like data.
fn make_frame(index: usize, cal: &TofCalibration) -> Frame {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0u64];

    for scan in 0..N_SCANS {
        // Two species, drifting at different mobilities.
        for (species, mz) in [(0usize, 622.0_f64), (1, 922.0)] {
            let centre_scan = 16 + species * 24;
            let d = scan as i64 - centre_scan as i64;
            if d.abs() > 4 {
                continue;
            }
            let envelope = (-(d * d) as f64 / 8.0).exp();
            let base = cal.mz_to_index(mz + index as f64 * 0.001);
            for k in 0..6u64 {
                let intensity = (2000.0 * envelope * (-(k as f64) / 3.0).exp()) as i32;
                if intensity <= 0 {
                    continue;
                }
                indices.push(base + k);
                data.push(intensity);
            }
        }
        indptr.push(data.len() as u64);
    }

    Frame { index, data, indices, indptr, n_rows: N_SCANS, n_cols: TOF_WIDTH }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).expect("usage: synthesize <out.mbi>");

    // A plausible calibration: real slope/intercept, no residual correction.
    let cal_json = r#"{"slope": 0.3478131086072915, "intercept": 0.10579624799291579, "mz_residual_terms": []}"#;
    let cal = TofCalibration::from_json(cal_json, SAMPLE_RATE)?;

    let mut writer = MbiWriter::create(&out)?;
    writer.write_global_metadata(&global_metadata())?;
    for i in 1..=N_FRAMES {
        let frame = make_frame(i, &cal);
        let extras = FrameExtras {
            trigger_timestamps: (0..N_SCANS).map(|s| s as f64 * 1.2e-4).collect(),
            metadata: frame_metadata(i, cal_json),
        };
        writer.write_frame(&frame, &extras)?;
    }
    writer.finish()?;

    // Read it back with our own reader.
    let f = MbiFile::open(&out)?;
    println!("wrote {} ({} frames)", out, f.n_frames());
    for i in 1..=f.n_frames() {
        let fr = f.frame(i)?;
        let c = f.calibration(i)?;
        let (_, cols, vals) = fr.to_coo();
        println!(
            "  frame {i}: {:5} points, tic {:8}, first m/z {:.4} (intensity {})",
            fr.nnz(),
            fr.total_intensity(),
            c.index_to_mz(cols[0]),
            vals[0]
        );
    }
    // Drift axis and CCS, now that the file carries a calibration.
    let axis = f.drift_axis(1)?;
    let ccs = f.ccs_calibration()?.expect("we just wrote one");
    println!(
        "drift axis: {} scans, {:.6} ms apart; ccs degree {}, gas {} Da",
        axis.n_scans,
        axis.period_ms,
        ccs.degree(),
        ccs.gas_mass
    );
    for scan in [12u64, 40] {
        let at = axis.arrival_time_ms(scan as usize);
        println!(
            "  scan {scan:3}: arrival {at:8.4} ms -> CCS {:9.4} A^2 (m/z 622, z=1)",
            ccs.arrival_time_to_ccs(at, 622.0, 1)
        );
    }
    println!("size: {:.1} kB", std::fs::metadata(&out)?.len() as f64 / 1e3);
    Ok(())
}
