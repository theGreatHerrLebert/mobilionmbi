//! Walk every frame in a file, reporting totals and throughput.
//!
//! The totals are what the vendor SDK reports for the same file, so this doubles
//! as an end-to-end check:
//!
//! ```text
//! cargo run --release --example sweep -- run.mbi
//! ```

use std::time::Instant;

use mobilionmbi::MbiFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: sweep <file.mbi>");
    let file = MbiFile::open(&path)?;
    let n = file.n_frames();

    let start = Instant::now();
    let mut total_nnz: u64 = 0;
    let mut total_intensity: i64 = 0;
    let mut populated = 0u64;
    for i in 1..=n {
        let frame = file.frame(i)?;
        total_nnz += frame.nnz() as u64;
        total_intensity += frame.total_intensity();
        populated += frame.nonzero_scan_indices().len() as u64;
    }
    let elapsed = start.elapsed();

    println!("frames           : {n}");
    println!("non-zero points  : {total_nnz}");
    println!("summed intensity : {total_intensity}");
    println!("populated scans  : {populated}");
    println!(
        "elapsed          : {:.0} ms ({:.2} ms/frame)",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / n as f64
    );
    Ok(())
}
