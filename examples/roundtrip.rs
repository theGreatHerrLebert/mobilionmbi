//! Read a `.mbi` file, write it back out, and verify the copy is identical.
//!
//! ```text
//! cargo run --release --example roundtrip -- in.mbi out.mbi
//! ```

use std::time::Instant;

use mobilionmbi::{FrameExtras, MbiFile, MbiWriter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let src = args.next().expect("usage: roundtrip <in.mbi> <out.mbi>");
    let dst = args.next().expect("usage: roundtrip <in.mbi> <out.mbi>");

    let input = MbiFile::open(&src)?;
    let n = input.n_frames();
    println!("source: {n} frames");

    let start = Instant::now();
    let mut writer = MbiWriter::create(&dst)?;
    writer.write_global_metadata(input.global_metadata())?;
    for i in 1..=n {
        let frame = input.frame(i)?;
        let extras = FrameExtras {
            trigger_timestamps: input.frame_trigger_timestamps(i)?,
            metadata: input.frame_metadata(i)?,
        };
        writer.write_frame(&frame, &extras)?;
    }
    writer.finish()?;
    println!("written in {:.1} s", start.elapsed().as_secs_f64());

    // Verify by reading the copy back and comparing everything that matters.
    let output = MbiFile::open(&dst)?;
    assert_eq!(output.n_frames(), n, "frame count");
    assert_eq!(output.global_metadata(), input.global_metadata(), "global metadata");
    assert_eq!(output.rt_tic()?, input.rt_tic()?, "rt-tic");

    let mut points = 0usize;
    for i in 1..=n {
        let a = input.frame(i)?;
        let b = output.frame(i)?;
        assert_eq!(a.data, b.data, "frame {i} data");
        assert_eq!(a.indices, b.indices, "frame {i} indices");
        assert_eq!(a.indptr, b.indptr, "frame {i} indptr");
        assert_eq!(a.n_rows, b.n_rows, "frame {i} n_rows");
        assert_eq!(
            input.frame_at_tic(i)?,
            output.frame_at_tic(i)?,
            "frame {i} at-tic"
        );
        assert_eq!(
            input.frame_trigger_timestamps(i)?,
            output.frame_trigger_timestamps(i)?,
            "frame {i} trigger-timestamps"
        );
        assert_eq!(
            input.frame_metadata(i)?,
            output.frame_metadata(i)?,
            "frame {i} metadata"
        );
        assert_eq!(
            input.calibration(i)?,
            output.calibration(i)?,
            "frame {i} calibration"
        );
        points += a.nnz();
    }

    let src_size = std::fs::metadata(&src)?.len();
    let dst_size = std::fs::metadata(&dst)?.len();
    println!("verified {n} frames, {points} points — identical");
    println!(
        "size: {:.1} MB in, {:.1} MB out ({:+.1}%)",
        src_size as f64 / 1e6,
        dst_size as f64 / 1e6,
        (dst_size as f64 / src_size as f64 - 1.0) * 100.0
    );
    Ok(())
}
