# mobilionmbi

A **pure-Rust** reader for **MOBILion `.mbi`** ion mobility - mass spectrometry files —
no vendor SDK, no proprietary shared library, no C++ ABI. Cross-platform.

## Why

MOBILion ships an MBI SDK, but going through it costs you: a closed `libmbisdk.so`
(x86-64 Linux/Windows only, glibc >= 2.29), a licence that permits object-code
redistribution only, for non-commercial use, and **explicitly forbids combination with
GPL-style code** — and Python bindings pinned to CPython 3.9 whose sparse containers are
not even wrapped.

`.mbi` turns out to be plain HDF5 with a legible sparse layout, so none of that is
necessary. This crate reads it natively.

## Status

- ✅ Frame data: CSR / COO sparse planes, **bit-exact** against the vendor SDK
  (validated on 87,004,212 points across 829 frames — zero mismatches).
- ✅ Mass calibration: matches the SDK's `IndexToMz` to **4.3e-16 relative**
  (1-2 ULP; 71% of values bitwise identical across all 232,992 TOF bins).
- ✅ `MzToIndex` inverse, matching the SDK's truncation convention.
- ✅ Global + per-frame metadata.
- 🚧 PyO3 bindings (`mobilionmbi-connector`).
- 🚧 CCS calibration — the test files carry none (`cal-dt-*` empty), so it is unwritten
  and untested rather than assumed.
- 🚧 Multiplexed acquisitions (`frm-mux-gate` / `frm-mux-sequence`) — not yet exercised.

## Usage

```rust
use mobilionmbi::MbiFile;

let f = MbiFile::open("run.mbi")?;
println!("{} frames", f.n_frames());

let frame = f.frame(600)?;              // 1-based, as in the vendor API
let cal = f.calibration(600)?;
let (rows, cols, vals) = frame.to_coo();
for i in 0..5 {
    println!("scan {} m/z {:.4} intensity {}", rows[i], cal.index_to_mz(cols[i]), vals[i]);
}
# Ok::<(), mobilionmbi::Error>(())
```

CLI:

```
cargo run --release --bin mbidump -- run.mbi 600
cargo run --release --example sweep -- run.mbi
```

## Format notes (the bits that were not obvious)

- **Container.** HDF5. Frames live at `data-cubes/frame-{N}-data` (1-based),
  metadata at `data-description/frame-{N}-metadata` and
  `data-description/global-description`, all gzip-chunked.

- **Sparse layout.** Three datasets per frame map onto CSR:
  `data-counts` is the intensity array (CSR `data`, one entry per non-zero point);
  `data-positions` is an `(M, 2)` array of **`[start, end)` TOF-index runs** which expand
  to CSR `indices`; `index-counts` holds **per-drift-scan cumulative offsets** into
  `data-counts`, i.e. CSR `indptr` less its closing entry. Note `index-counts` and
  `index-positions` are *different* arrays — the former indexes intensities, the latter
  indexes runs.

- **Mass calibration** is a traditional TOF fit *plus* a ppm residual polynomial, and you
  need both. With `t_us = index * 1e6 / sample_rate`:

  ```text
  mz_raw = (slope * (t_us - intercept))^2
  ppm    = Σ mz_residual_terms[k] * t_us^k        // the SDK's TofError()
  mz     = mz_raw * (1 - ppm / 1e6)
  ```

  `slope`, `intercept` and `mz_residual_terms` come from the frame's
  `cal-ms-traditional` attribute (JSON). **Dropping the residual term costs up to 3.9 ppm**
  (median 0.65 ppm) — invisible in a smoke test, fatal in proteomics. The residual is
  evaluated in *microseconds* and expressed in *ppm*; neither is stated anywhere.

- **`MzToIndex` truncates**, returning the largest bin whose m/z does not exceed the
  input — it does not round.

- **Calibration is stored per frame** but is constant within a file in practice. It does
  differ between files, so read it per frame rather than caching one globally.

- **TOF axis width** is `adc-record-size` in the global description (232992 in the test
  files), of which the SDK exposes a valid window via `GetToFOffset()` / `GetToFLength()`
  (30001 / 202991).

## Tests

Unit tests cover the calibration against values captured from the vendor SDK. The
end-to-end checks need a `.mbi` file, which cannot be redistributed here; a public one is
**MassIVE MSV000099577 / PXD069856** (PAMAF/SLIM HeLa, Agilent 6546 QTOF, CC0).

```
cargo test
cargo run --release --example sweep -- /path/to/run.mbi
```

## Licence

MIT OR Apache-2.0, at your option. The MOBILion SDK is *not* used, linked, or
redistributed by this crate — only observed, to validate the format notes above.
