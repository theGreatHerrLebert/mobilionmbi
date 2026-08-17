"""Regenerate every number in docs/DATA_CHARACTERISTICS.md.

    python docs/experiments/im_rt_characteristics.py <file.mbi> [frame]

Needs mobilionmbi-connector, numpy, scipy. Defaults to frame 600, which is a
mid-gradient MS1 (CE = 0) frame in 200S-100ngHeLa-14.19.00.mbi.

Everything here is measurement, not detection: the "peak-like" and "strong"
thresholds are deliberately crude cuts used to separate genuine features from the
count-limited noise floor, NOT a calibrated peak detector.
"""

import sys

import numpy as np
from scipy import ndimage

import mobilionmbi_connector as mbi

# Crude shape cuts used to separate analytes from single-count noise.
PEAK_LIKE = (8, 5)   # (min pixels, min drift scans tall)
STRONG = (20, 8)
TOF_WINDOW = 10_000  # process the TOF axis in windows to bound memory


def drift_profile(rows, cols, vals, n_scans, tof_bin):
    """Intensity vs drift scan for one TOF bin."""
    m = cols == tof_bin
    prof = np.zeros(n_scans, dtype=np.int64)
    prof[rows[m]] = vals[m]
    return prof


def fwhm(prof):
    """Full width at half maximum, walking out from the apex."""
    apex = int(np.argmax(prof))
    half = prof[apex] / 2.0
    lo = apex
    while lo > 0 and prof[lo] >= half:
        lo -= 1
    hi = apex
    while hi < len(prof) - 1 and prof[hi] >= half:
        hi += 1
    return apex, hi - lo


def main():
    path = sys.argv[1]
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 600

    f = mbi.MbiFile(path)
    ce = f.collision_energies()
    rt = f.retention_times()
    n = f.n_frames

    print(f"file: {path}")
    print(f"  frames={n}  CE unique={np.unique(ce)}")
    ms1 = [i for i in range(1, n + 1) if ce[i - 1] == 0.0]
    print(f"  MS1 (CE=0) frames: {len(ms1)}")
    print(
        f"  RT: {rt[-1]:.1f} s total -> {rt[-1]/n:.3f} s/frame, "
        f"{2*rt[-1]/n:.3f} s per MS1->MS1"
    )

    axis = f.drift_axis(frame_idx)
    period = axis.period_ms
    fr = f.frame(frame_idx)
    rows, cols, vals, n_scans, n_tof = fr.coo()
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    vals = vals.astype(np.int64)
    print(f"\nframe {frame_idx}: {vals.size} points, "
          f"{np.unique(cols).size} distinct TOF bins, "
          f"{n_scans} drift scans x {n_tof} TOF bins")
    print(f"  drift period {period:.6f} ms -> {n_scans*period:.1f} ms cycle")

    # ---- IM peak width, from the most intense TOF bins ---------------------
    order = np.argsort(cols)
    c_s, r_s, v_s = cols[order], rows[order], vals[order]
    uniq, start = np.unique(c_s, return_index=True)
    totals = np.add.reduceat(v_s, start)
    top = np.argsort(-totals)[:12]

    print("\nIM peak width (12 most intense TOF bins):")
    widths = []
    for k in top:
        prof = drift_profile(rows, cols, vals, n_scans, uniq[k])
        apex, w = fwhm(prof)
        widths.append(w)
        lo = apex
        while lo > 0 and prof[lo - 1] > 0:
            lo -= 1
        hi = apex
        while hi < n_scans - 1 and prof[hi + 1] > 0:
            hi += 1
        print(f"  tof {uniq[k]:7d}  total {totals[k]:9d}  apex {apex:5d}  "
              f"FWHM {w:3d} scans ({w*period:.3f} ms)  "
              f"contiguous non-zero run {hi-lo+1:3d}")
    med = float(np.median(widths))
    print(f"  median FWHM {med:.1f} scans = {med*period:.3f} ms")
    print(f"  IM peak capacity ~ {n_scans/med:.0f} over the "
          f"{n_scans*period:.0f} ms cycle")

    # ---- LC peak width, tracking the strongest bin across MS1 frames -------
    target = int(uniq[top[0]])
    window = [i for i in ms1 if abs(i - frame_idx) <= 50]
    prof = []
    for i in window:
        g = f.frame(i)
        d, idx, _, _, _ = g.csr()
        prof.append(int(d[idx == target].sum()))
    prof = np.asarray(prof)
    rts = np.asarray([rt[i - 1] for i in window])
    apex = int(np.argmax(prof))
    above = np.where(prof >= prof[apex] / 2)[0]
    sampling = rts[1] - rts[0]
    print(f"\nLC peak width (TOF bin {target} across MS1 frames):")
    print(f"  apex frame {window[apex]} at {rts[apex]:.1f} s, height {prof[apex]}")
    print(f"  FWHM {above[-1]-above[0]+1} MS1 points = "
          f"{rts[above[-1]]-rts[above[0]]:.2f} s")
    print(f"  MS1 sampling {sampling:.3f} s -> "
          f"{(rts[above[-1]]-rts[above[0]])/sampling:.1f} points across FWHM")

    # ---- feature density and competition -----------------------------------
    spans, npix, inten = [], [], []
    for lo in range(0, n_tof, TOF_WINDOW):
        hi = min(lo + TOF_WINDOW, n_tof)
        m = (cols >= lo) & (cols < hi)
        if not m.any():
            continue
        img = np.zeros((n_scans, hi - lo), dtype=np.int32)
        img[rows[m], cols[m] - lo] = vals[m]
        lbl, count = ndimage.label(img > 0, structure=np.ones((3, 3)))
        if count == 0:
            continue
        index = np.arange(1, count + 1)
        sizes = ndimage.sum(img > 0, lbl, index=index)
        sums = ndimage.sum(img, lbl, index=index)
        for obj, size, total in zip(ndimage.find_objects(lbl), sizes, sums):
            spans.append((obj[0].start, obj[0].stop))
            npix.append(size)
            inten.append(total)
    spans = np.asarray(spans)
    npix = np.asarray(npix)
    inten = np.asarray(inten)
    heights = spans[:, 1] - spans[:, 0]

    print(f"\nfeature density (frame {frame_idx}), competition per drift scan:")
    populations = [
        ("all blobs", np.ones(len(spans), bool)),
        (f"peak-like (>={PEAK_LIKE[0]} px, >={PEAK_LIKE[1]} scans)",
         (npix >= PEAK_LIKE[0]) & (heights >= PEAK_LIKE[1])),
        (f"strong (>={STRONG[0]} px, >={STRONG[1]} scans)",
         (npix >= STRONG[0]) & (heights >= STRONG[1])),
    ]
    for label, mask in populations:
        occupancy = np.zeros(n_scans, dtype=int)
        for a, b in spans[mask]:
            occupancy[a:b] += 1
        print(f"  {label:42s} n={mask.sum():6d}  "
              f"median {np.median(occupancy):3.0f}  "
              f"p90 {np.percentile(occupancy, 90):3.0f}  "
              f"max {occupancy.max():3d}")
        if mask.sum():
            print(f"  {'':42s} median height {np.median(heights[mask]):.0f} scans, "
                  f"median intensity {np.median(inten[mask]):.0f}")


if __name__ == "__main__":
    main()
