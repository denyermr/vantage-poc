# G2 Anchor Reads — provenance and methodology

This directory holds the audit trail for refining the Set A / B (and,
pending spot-check, Set C / D) σ° anchor values in
`phase1b/physics/g2_anchor_spec.md` from the staged source PDFs.

The `~`-values in `g2_anchor_spec.md` v0.1 were explicit read-off
scaffolding to be refined before G2 execution (DEV-1b-003). This
directory records the actual refined reads with their provenance so
an auditor can reproduce the numbers row-by-row.

## Source PDFs

| Ref | File | DOI |
|---|---|---|
| T94 | [`../Toure_1994_MIMICS_agricultural.pdf`](../Toure_1994_MIMICS_agricultural.pdf) | 10.1109/36.285188 |
| M90 | [`../McDonald_1990_MIMICS_walnut_orchard.pdf`](../McDonald_1990_MIMICS_walnut_orchard.pdf) | 10.1109/TGRS.1990.572925 (see note below) |

**Note:** `DEV-1b-003.md` cites McDonald's DOI as
`10.1109/TGRS.1990.572943`; the PDF metadata reports
`10.1109/TGRS.1990.572925`. Same paper (title, authors, journal,
volume/issue match). Likely a typo in DEV-1b-003; non-blocking.

## Methodology (v0 — label-centre calibration)

1. **Render**. Each PDF page containing a target figure rendered with
   `pdftoppm -png -r 400 -f N -l N <pdf> <out>` at 400 dpi (page images
   ~3436 × 4425 px for the Toure PDF).

2. **Plot-box localisation**. For a given panel, horizontal dark-row
   scans in the x-column-range of the panel identify top/bottom
   plot-frame borders; vertical dark-column scans in the y-row-range
   identify left/right borders. Borders were characterised as rows/cols
   with > 30 % dark-pixel density over a multi-pixel-thick band.

3. **Y-axis calibration (v0 — label-centre)**. Y-axis label glyphs
   ("0", "−10", "−20", "−30") detected by scanning the strip
   `x ∈ [l−90, l−10]` for rows with any dark pixels (label ink).
   The vertical centre of each label group used as the y-axis tick
   position for that dB value. A linear fit was performed across all
   four labels per panel to derive pixels-per-dB.

   For T94 Fig. 2:
   - CHH: `y=1391.5` (0 dB), `y=1763.5` (−30 dB) → slope 12.4 px/dB.
   - CVV: `y=1396.0` (0 dB), `y=1765.5` (−30 dB) → slope 12.32 px/dB.
   Cross-panel agreement: < 1 % slope variation, indicating no
   figure-reproduction distortion.

   **Known uncertainty:** label-vs-tick-baseline alignment.
   The calibration treats the vertical centre of each label glyph as
   the y-position of the tick. If the 1994 typesetting convention
   instead places the tick at the label's baseline (bottom edge of the
   digits), the calibration is too high by ≈ 6 px ≈ 0.5 dB. This is
   the question the human spot-check at Step 2 of the continuation
   prompt is designed to resolve. If the spot-check shows a systematic
   offset, v1 reads will use baseline-alignment calibration and
   supersede v0.

4. **X-axis calibration**. Ten x-axis ticks visible on top and bottom
   borders at labels 10°, 15°, 20°, 25°, 30°, 35°, 40°, 50°, 60°, 70°.
   Tick pixel positions extracted from both borders; inter-tick
   spacings confirmed uniform within ±2 px (Fig. 2; no axis
   discontinuity). X-axis therefore evenly spaced in tick-index,
   non-uniform in degrees (ticks at +5° for θ ≤ 40°, then +10°).

5. **Marker detection (hollow-vs-filled squares — Sets A, B)**.
   Template: 22 × 22 px window with a 3-px-thick dark rim and a bright
   interior. Score at each candidate position (y, x) in the plot
   interior is
       `score = mean_brightness(interior 8×8) − mean_brightness(rim)`.
   A high positive score indicates a hollow square (MIMICS predicted,
   `□`); a near-zero score indicates a filled square (scatterometer
   measurement, `■`) or background. Search constrained to ±12 px in x
   around each canonical θ tick position, and to the full plot interior
   in y. Top-2 candidates per θ kept and inspected to verify whether
   the hollow (upper) or filled (lower, or overlapping) marker was
   picked. Visual verification: a red circle drawn on each selected
   marker over the original panel (`t94_fig2_CHH_detected_v0.png`,
   `t94_fig2_CVV_detected_v0.png`).

6. **Marker detection (Set C — five mechanism-specific marker types)**.
   Not performed in v0. To be extended to a five-template library when
   Set C is addressed (see continuation prompt Step 3).

7. **Set E (T94 Tables V, VI)**. Not extracted in v0. The
   continuation prompt directs that Set E values are extracted from
   the PDF as text (`pdftotext`), not via pixel detection, because the
   source is a numerical table not a plot.

## Files in this directory (v0)

| File | Purpose |
|---|---|
| `t94_fig2_CHH_raw_panel.png` | Tight crop of the CHH panel, unannotated — the image the human should compare against when auditing |
| `t94_fig2_CHH_calibration_check_v0.png` | Diagnostic overlay that exposed the 5 dB off-by-one in the first calibration attempt (plot-box border vs label centre). The "0" y-axis label visibly aligns with my "−5" reference line — the error that drove switching to label-centre calibration |
| `t94_fig2_CHH_gridded_v0.png` | CHH panel with reference lines overlaid at θ=20,30,40,50 and dB every 1 dB using the final v0 calibration |
| `t94_fig2_CVV_gridded_v0.png` | Same for CVV |
| `t94_fig2_LHH_gridded_v0.png`, `t94_fig2_LVV_gridded_v0.png` | L-band panels — not used for G2 but rendered for completeness; also served as the cross-check that confirmed label-centre calibration (LHH y-axis labels detected at y=650.5, 778.5, ..., giving the same 12.8 px/dB as CHH) |
| `t94_fig2_CHH_detected_v0.png` | CHH panel with red circles on the detected hollow-square markers; also shows the second-best candidates per θ |
| `t94_fig2_CVV_detected_v0.png` | CVV panel with red circles on detected markers and θ/dB labels |
| `anchor_reads_v0.json` | Machine-readable table of the refined Set A / Set B reads with pixel coordinates, calibration version, and detection confidence |

## Version history

- **v0 (2026-04-18)** — Sets A and B refined. Calibration =
  label-centre. Awaits human spot-check at A.1 (θ=20° CHH) and
  B.3 (θ=40° CVV). Sets C, D, E not yet refined.
