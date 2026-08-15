//! `mbidump` - inspect a MOBILion `.mbi` file from the command line.

use std::process::ExitCode;

use mobilionmbi::MbiFile;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: mbidump <file.mbi> [frame-index]");
        return ExitCode::FAILURE;
    }

    let file = match MbiFile::open(&args[1]) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let g = file.global_metadata();
    let show = |k: &str| g.get(k).map(String::as_str).unwrap_or("-");
    println!("file        : {}", args[1]);
    println!("frames      : {}", file.n_frames());
    println!("acquired    : {}", show("acq-timestamp"));
    println!("instrument  : {} / {}", show("acq-ms-model"), show("acq-lc-model"));
    println!("mode        : {} / {}", show("acq-mode"), show("acq-collection-mode"));
    println!("tof width   : {}", show("adc-record-size"));
    println!("sample rate : {} Hz", file.sample_rate());

    let Some(idx) = args.get(2) else { return ExitCode::SUCCESS };
    let idx: usize = match idx.parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("error: frame index must be a positive integer");
            return ExitCode::FAILURE;
        }
    };

    let frame = match file.frame(idx) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cal = match file.calibration(idx) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!();
    println!("frame {idx}");
    println!("  grid            : {} scans x {} tof bins", frame.n_rows, frame.n_cols);
    println!("  non-zero points : {}", frame.nnz());
    println!("  populated scans : {}", frame.nonzero_scan_indices().len());
    println!("  total intensity : {}", frame.total_intensity());
    println!(
        "  calibration     : slope={} intercept={} residual_terms={}",
        cal.slope,
        cal.intercept,
        cal.residual_terms.len()
    );

    let axis = file.drift_axis(idx).ok();
    if let Some(a) = axis {
        println!(
            "  drift axis      : {} scans, {:.6} ms apart, 0..{:.3} ms",
            a.n_scans,
            a.period_ms,
            a.arrival_time_ms(a.n_scans.saturating_sub(1))
        );
    }
    match file.ccs_calibration() {
        Ok(Some(c)) => println!(
            "  ccs calibration : degree {}, range {:.1}..{:.1} A^2, gas {:.4} Da",
            c.degree(),
            c.min_ccs,
            c.max_ccs,
            c.gas_mass
        ),
        Ok(None) => println!("  ccs calibration : none in this file"),
        Err(e) => println!("  ccs calibration : unreadable ({e})"),
    }

    let ccs = file.ccs_calibration().ok().flatten();
    let (rows, cols, vals) = frame.to_coo();
    println!("  first 5 points  : (scan, tof, arrival ms, m/z, intensity, ccs)");
    for i in 0..rows.len().min(5) {
        let mz = cal.index_to_mz(cols[i]);
        let at = axis.map(|a| a.arrival_time_ms(rows[i] as usize));
        let ccs_val = match (&ccs, at) {
            (Some(c), Some(t)) => format!("{:9.2}", c.arrival_time_to_ccs(t, mz, 1)),
            _ => "        -".to_string(),
        };
        println!(
            "    {:5} {:8} {:11.4} {:12.4} {:8} {}",
            rows[i],
            cols[i],
            at.unwrap_or(f64::NAN),
            mz,
            vals[i],
            ccs_val
        );
    }
    ExitCode::SUCCESS
}
