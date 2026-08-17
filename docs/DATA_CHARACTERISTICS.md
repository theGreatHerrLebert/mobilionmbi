# MBI data characteristics — experimental notes

> **STATUS: EXPERIMENTAL.** Measured on two public files, and several numbers come
> from a *single frame*. Detection thresholds used below are ad hoc, not a calibrated
> detector. Treat ratios and orders of magnitude as the findings; treat absolute counts
> as indicative. Reproduce with `docs/experiments/im_rt_characteristics.py` before
> relying on any of it.

## Provenance

| | |
|---|---|
| Files | `200S-100ngHeLa-14.19.00.mbi`, `CERamp-25ngHeLa-14.52.54.mbi` |
| Source | MassIVE **MSV000099577** / PXD069856, CC0 1.0 |
| Sample | HeLa tryptic digest, 100 ng (200 SPD) and 25 ng (CE ramp) |
| Instrument | MOBILion MOBIE SLIM + Agilent 6545 QTOF, 13 m SLIM path |
| Frame used for per-frame stats | 600 (mid-gradient, CE = 0, i.e. an MS1 frame) |
| Measured | 2026-08-17, via `mobilionmbi-connector` |

## Acquisition shape

Both files are `ExperimentType = SIFF_MAF` — Mobility-Aligned Fragmentation, and
**quadrupole-free**: `HasScanDefinitions()` is false and `NumScanDefinitions()` refuses
outright, because no isolation windows exist to store. Collision energy alternates
frame by frame between 0 V and 21 V, with MS levels tracking `{1, 2}`:

```
200S    : CE = [21, 0, 21, 0, ...]   415 high / 414 low   (829 frames)
CE-ramp : CE = [0, 21, 0, 21, ...]  3254 high / 3254 low (6508 frames)
```

So precursor→fragment association has **no quad dimension to exploit**; the only
linking axes are drift time and retention time. Despite its name the "CE ramp" file
does not ramp CE *within* a frame — every frame carries a single setpoint. The
`frm-collision-energy` structure (`{"interval_ms":…, "setpoints":[…]}`) clearly
anticipates within-frame stepping, but no file we have uses it.

## Axis characteristics

| axis | peak FWHM | sampling | points across peak | resolvable positions |
|---|---|---|---|---|
| **drift (IM)** | 1.85 ms (15.5 scans) | 0.1196 ms | **~15** | **~216** |
| **retention (LC)** | 2.57 s | 0.83 s (MS1→MS1) | **~3** | — |

Drift: 3345 scans per frame at `frm-dt-period` = 0.11958 ms, a 400 ms cycle. FWHM is
the median over the 12 most intense TOF bins of frame 600. Peak capacity ≈ cycle /
FWHM ≈ 216; resolving power (apex/FWHM) ≈ 130.

Retention: 829 frames over 351.5 s = 0.424 s/frame, but MS1 and MS2 alternate, so a
precursor is sampled every 0.83 s. A representative peptide (TOF bin 171072) has an LC
FWHM of 2.57 s — **about three MS1 points across the peak**.

**The asymmetry is the headline.** The drift axis offers ~15 points per peak and
216-way separation; the retention axis offers ~3 points. Any extraction strategy that
leans on chromatographic correlation is starved here, while the mobility axis is rich.
For orientation: Waters TWIMS peak capacity is ~40–60, so this is 4–5× better than the
HDMS^E case, while quad-based diaPASEF reduces competition far below either.

## Feature density and competition

The number that decides deconvolution strategy is how many precursors share a drift
cell. Connected-component analysis of frame 600 (8-connectivity, true TOF coordinates),
binned by how peak-shaped a component is:

| population | n per frame | median height | median intensity | **components per drift scan** |
|---|---|---|---|---|
| all blobs | 55,558 | 1 scan | 185 | median 10, p90 56, max 93 |
| peak-like (≥8 px, ≥5 scans tall) | 450 | 7 scans | 3,881 | **median 0, p90 4, max 16** |
| strong (≥20 px, ≥8 scans tall) | 194 | 12 scans | 8,948 | **median 0, p90 3, max 14** |

Frame 600 holds 158,579 non-zero points across 48,272 distinct TOF bins.

**The plane is count-limited.** 55,558 connected blobs from 158,579 points means the
median "component" is a single-count speck; genuine features are a few hundred per
frame. Any statistic taken over *all* connected components describes the noise floor,
not the analytes — see the correction record below, where exactly that mistake was
made.

Taking only peak-shaped components, **precursor competition is single-digit**: zero
competitors in the median drift scan, ~4 at p90, ~16 at worst.

## What this implies for extraction

1. **Mobility profile correlation should be the primary discriminator, not a
   tiebreak.** Fragments in MAF data do not merely overlap their precursor in drift —
   they inherit the same arrival-time distribution, since the packet is fragmented
   after separation. With ~15 points across a peak and typically 0–4 competitors, a
   shape correlation on the drift axis is close to a unique assignment.

2. **Do not average the RT axis into the correlation.** With ~3 points across an LC
   peak, retention correlation is noise here; blending it with the drift axis
   dilutes a good signal with a bad one. Drift-only scoring is the right default for
   this data — the inverse of the usual timsTOF choice, where IM is the coarse axis.

3. **Joint deconvolution is a tail problem, not the main event.** Modelling each
   fragment as a non-negative mixture over competing precursors is the principled
   answer to shared fragments, but with a median of zero competitors it would be
   over-engineering to lead with. Reserve it for the dense p90 regions.

4. **Sensitivity dominates deconvolution.** ~450 peak-like features against ~55k noise
   blobs means the yield lever is pulling weak precursors out of a count-limited
   plane, not resolving contention between strong ones. Detector design (intensity-aware
   density criteria, per-location width estimation) matters more than assignment logic.

5. **Fragment drift offset is unmeasured.** Fragmentation after mobility separation may
   impart a small mass-dependent delay through the transfer region, which would sharpen
   matching further. It may also sit below the 0.12 ms bin. Measure before assuming.

## Correction record

Both errors below were made and caught during this analysis; recorded so the wrong
numbers do not get re-derived.

- **"Median drift extent = 1 scan."** Wrong twice. The TOF axis had been compacted with
  `np.unique` before labelling, so bins hundreds apart became adjacent columns and
  produced false connectivity. More importantly the median was taken over *all*
  connected blobs, which at this sparsity are single-count noise. Real precursor peaks
  span ~49 consecutive non-zero scans with ~17 scans FWHM.

- **"~10 competing precursors per drift cell."** Same root cause — noise blobs counted
  as precursors. Filtering to peak-shaped components gives a median of **0**, p90 4.
  This inverted a design conclusion: joint deconvolution went from "the real answer" to
  "reserve for the tail".

## Reproducing

```bash
python docs/experiments/im_rt_characteristics.py /path/to/200S-100ngHeLa-14.19.00.mbi
```

Requires `mobilionmbi-connector`, numpy and scipy. Prints every number in this
document.
