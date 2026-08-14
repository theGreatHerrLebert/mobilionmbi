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

    let (rows, cols, vals) = frame.to_coo();
    println!("  first 5 points  : (scan, tof, m/z, intensity)");
    for i in 0..rows.len().min(5) {
        println!(
            "    {:5} {:8} {:12.4} {:8}",
            rows[i],
            cols[i],
            cal.index_to_mz(cols[i]),
            vals[i]
        );
    }
    ExitCode::SUCCESS
}
