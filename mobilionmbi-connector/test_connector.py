"""Smoke-test the PyO3 layer against a real file.

    python test_connector.py <file.mbi>

The expected values are those the vendor SDK reports for
200S-100ngHeLa-14.19.00.mbi (MassIVE MSV000099577); they are asserted only when
run against that file.
"""
import sys, time
import numpy as np
import mobilionmbi_connector as mbi

if len(sys.argv) < 2:
    sys.exit("usage: python test_connector.py <file.mbi>")
PATH = sys.argv[1]

f = mbi.MbiFile(PATH)
print(f"{f!r}")
print(f"  n_frames={f.n_frames}  sample_rate={f.sample_rate:g}")

g = f.global_metadata()
print(f"  global metadata keys: {len(g)}  instrument={g['acq-ms-model']}  mode={g['acq-mode']}")

rt = f.retention_times()
print(f"  retention_times: {rt.dtype} {rt.shape} {rt[0]:.3f}..{rt[-1]:.3f}")

fr = f.frame(600)
print(f"  {fr!r}  tic={fr.total_intensity}")
rows, cols, vals, nrow, ncol = fr.coo()
expected = [(0, 76479, 74), (74, 146624, 102), (74, 146625, 87)]
got = [(int(r), int(c), int(v)) for r, c, v in zip(rows[:3], cols[:3], vals[:3])]
assert got == expected, f"COO mismatch: {got} != {expected}"
print(f"  COO first 3 match the SDK exactly: {got}")

cal = f.calibration(600)
print(f"  {cal!r}")
mz = cal.index_to_mz(cols[:3].astype(np.uint64))
assert abs(mz[0] - 175.91827611) < 1e-6, mz[0]
assert abs(mz[1] - 648.31687165) < 1e-6, mz[1]
print(f"  index_to_mz -> {np.round(mz, 6)}  (SDK: 175.918276, 648.316872)")
assert int(cal.mz_to_index(np.array([622.0]))[0]) == 143621
print("  mz_to_index(622.0) -> 143621, matching the SDK's truncation")

# csr() and coo() consume the buffers; re-read for the second view.
fr = f.frame(600)
data, indices, indptr, nrow, ncol = fr.csr()
assert data.size == 158579 and indptr.size == 3346
assert int(data.sum()) == 18806733
print(f"  CSR: data{data.dtype}{data.shape} indptr{indptr.shape} sum={data.sum()} == frame TIC")

# Throughput over the whole file.
t0 = time.perf_counter()
nnz = 0
total = 0
for i in range(1, f.n_frames + 1):
    fr = f.frame(i)
    d, idx, ptr, _, _ = fr.csr()
    nnz += d.size
    total += int(d.sum())
dt = time.perf_counter() - t0
assert nnz == 87004212, nnz
assert total == 9924016945, total
print(f"\nfull sweep {f.n_frames} frames: {dt*1000:.0f} ms ({dt/f.n_frames*1000:.2f} ms/frame)")
print(f"  points={nnz:,} intensity={total:,} — both match the SDK")
print("\nOK")
