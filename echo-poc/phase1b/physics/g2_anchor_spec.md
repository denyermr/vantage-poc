# Phase 1b G2 Implementation Gate — Anchor Specification

**Status:** Authoritative specification for the G2 equivalence check
**Version:** v0.3 (Session E; DEV-1b-004 + DEV-1b-005 + DEV-1b-006 amendments; supersedes v0.2)
**Source authority:** SPEC.md §4, DEV-1b-003, DEV-1b-004, DEV-1b-005, DEV-1b-006
**Changes post-sign-off:** Require a new deviation log entry
**Provenance of refined values:** [`phase1b/refs/anchor_reads/anchor_reads_v1.json`](../refs/anchor_reads/anchor_reads_v1.json) and the annotated PNGs in the same directory. v0.1 contained "~" scaffolding estimates; v0.2 replaced those with values read directly from the source PDFs per DEV-1b-003. v0.3 (this version) does **not** change any v0.2 anchor value or tolerance — it adds dielectric-configuration specificity to Set E rows E.1 / E.2 (DEV-1b-004), marks Set D as EXEMPT pending Phase 1c (DEV-1b-005), and adds a new secondary anchor block Set C2 drawn from Ulaby & Long 2014 Table 11-1 (DEV-1b-006). All additions are **additive** to the pre-registration record, not subtractive.

---

## Purpose

This document specifies the exact canonical values against which the Phase 1b differentiable PyTorch MIMICS implementation is tested at the G2 Implementation Gate. All values are drawn from peer-reviewed sources (Toure et al. 1994; McDonald, Dobson & Ulaby 1990), cited to page, figure, and table. The anchor is the pre-registered external correctness reference for the MIMICS forward model.

G2 consists of three equivalence arms, all of which must pass:

1. **numpy_port arm** — numpy and PyTorch implementations of the Phase 1b MIMICS module agree on all anchor inputs within 0.5 dB.
2. **published_table arm** — PyTorch implementation reproduces published σ° values from Sets A–D below within 0.5 dB.
3. **gradient spot-check arm** — PyTorch autograd ∂σ°/∂parameter agrees with Toure's published sensitivity coefficients (Set E) within tolerance.

---

## Reference sources

| Ref | Citation | Repository location |
|---|---|---|
| T94 | Touré, A., Thomson, K.P.B., Edwards, G., Brown, R.J., Brisco, B.G. (1994) 'Adaptation of the MIMICS backscattering model to the agricultural context — wheat and canola at L and C bands', *IEEE Trans. Geosci. Remote Sens.*, 32(1), 47–61. DOI 10.1109/36.285188. | `phase1b/refs/Toure_1994_MIMICS_agricultural.pdf` |
| M90 | McDonald, K.C., Dobson, M.C., Ulaby, F.T. (1990) 'Using MIMICS to model L-band multiangle and multitemporal backscatter from a walnut orchard', *IEEE Trans. Geosci. Remote Sens.*, 28(4), 477–491. DOI 10.1109/TGRS.1990.572925. | `phase1b/refs/McDonald_1990_MIMICS_walnut_orchard.pdf` |

**DOI correction (v0.2):** DEV-1b-003 cites McDonald's DOI as `10.1109/TGRS.1990.572943`. The PDF metadata reports `10.1109/TGRS.1990.572925`. Same paper (all other bibliographic data match); DEV-1b-003 likely contains a transcription typo. The PDF metadata is authoritative for the repository.

---

## Operating-range boundaries

All anchor points are drawn from the incidence-angle range 20°–50° (VV and HH). T94 explicitly notes that:

- At θ < 20°, MIMICS underestimates backscatter because coherent scattering components are not modelled. Anchors below 20° would be probing an acknowledged model limitation, not testing implementation correctness.
- At θ ≥ 50° for VV polarisation, MIMICS produces a Brewster-effect artefact that does not appear in scatterometer data. This artefact is a property of MIMICS itself and should be reproduced faithfully by any correct implementation, but we avoid anchoring bulk agreement at the extreme VV angles.

Our primary operating regime is θ ≈ 41.5° (Moor House mean incidence angle, SPEC §2), which sits comfortably in the interior of the 20°–50° validated range.

---

## Set A — Wheat C-band HH multiangle (primary bulk anchor)

**Source:** T94 Fig. 2, bottom-left panel (labelled CHH), page 49.
**Scenario:** Wheat, site #13, measurement date 18 July 1988.

### Canonical inputs (from T94 Fig. 2 caption, page 49, and T94 Table I, page 48)

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Frequency | f | 5.3 GHz (C-band) | T94 §I |
| Polarisation | — | HH | T94 Fig. 2 caption |
| Stem gravimetric moisture | m_g (stem) | 0.72 | T94 Fig. 2 caption |
| Stem diameter | d | 0.2 cm | T94 Table I(a) |
| Stem height | h | 0.50 m | T94 Fig. 2 caption |
| Stem density | N | 320 st./m² (site 13) | T94 Table I(a) |
| Stem PDF | — | vertical | T94 Table I(a) |
| Leaf gravimetric moisture | m_g (leaf) | 0.67 | T94 Fig. 2 caption |
| Leaf length | l | 12 cm | T94 Table I(a) |
| Leaf width | w | 1 cm | T94 Table I(a) |
| Leaf density | N | 3430 lf./m³ | T94 Fig. 2 caption (derived from LAI) |
| Leaf thickness | th. | 0.02 cm | T94 Table I(a) |
| Leaf PDF | — | uniform | T94 Table I(a) |
| Soil volumetric moisture | m_v | 0.17 g/cm³ | T94 Fig. 2 caption |
| Soil surface RMS height | s | 0.55 cm (site 13) | T94 Table I(a) |
| Soil correlation length | l_s | 4.9 cm (site 13) | T94 Table I(a) |
| Soil roughness model | — | physical optics | T94 §II.A |

### Anchor values (v0.2, refined from T94 Fig. 2 CHH panel)

Read from the open-square markers (MIMICS-modelled σ°) in T94 Fig. 2 bottom-left panel. Calibration and read methodology documented in [`phase1b/refs/anchor_reads/anchor_reads_v1.json`](../refs/anchor_reads/anchor_reads_v1.json) `set_A`. Human spot-check at A.1 confirmed method (response: "closer to −9.5 dB" vs −8.5 dB scaffolding estimate).

| Row | θ (deg) | σ°_CHH,MIMICS (dB) | Tolerance | Source |
|---|---|---|---|---|
| A.1 | 20 | **−9.48** | ±0.5 dB | T94 Fig. 2 CHH panel, open square at θ=20° |
| A.2 | 30 | **−10.44** | ±0.5 dB | T94 Fig. 2 CHH panel, open square at θ=30° |
| A.3 | 40 | **−11.57** | ±0.5 dB | T94 Fig. 2 CHH panel, open square at θ=40° |
| A.4 | 50 | **−12.70** | ±0.5 dB | T94 Fig. 2 CHH panel, open square at θ=50° |

### Pass criterion

All four rows (A.1–A.4) pass individually. Overall Set A mean absolute deviation from anchor values ≤ 0.4 dB.

---

## Set B — Wheat C-band VV multiangle (VV branch anchor)

**Source:** T94 Fig. 2, **bottom-right** panel (labelled CVV), page 49.
**Scenario:** Same as Set A (wheat, site #13, 18 July 1988).

**v0.2 correction:** v0.1 §Set B described this panel as "top-right" and flagged it as "labelled LVV … verify from PDF". This was an internal inconsistency in v0.1. Fig. 2 is a 2×2 grid: LHH top-left, LVV top-right, CHH bottom-left, **CVV bottom-right**. v0.2 uses the correct panel.

### Canonical inputs

Identical to Set A inputs, with polarisation = VV.

### Anchor values (v0.2, refined from T94 Fig. 2 CVV panel)

Human spot-check at B.2, B.3, B.4 confirmed. At B.3, v0.1's estimate of −12.5 dB was read off the filled (measured, ■) marker; the correct MIMICS value is the hollow (□) marker at −11.12 dB.

| Row | θ (deg) | σ°_CVV,MIMICS (dB) | Tolerance | Source |
|---|---|---|---|---|
| B.1 | 20 | **−9.99** | ±0.5 dB | T94 Fig. 2 CVV panel, open square at θ=20° |
| B.2 | 30 | **−10.39** | ±0.5 dB | T94 Fig. 2 CVV panel, open square at θ=30° |
| B.3 | 40 | **−11.12** | ±0.5 dB | T94 Fig. 2 CVV panel, open square at θ=40° |
| B.4 | 50 | **−14.13** | ±0.5 dB | T94 Fig. 2 CVV panel, open square at θ=50° (Brewster drop) |

### Pass criterion

All four rows (B.1–B.4) pass individually. Overall Set B mean absolute deviation ≤ 0.5 dB. The slightly wider tolerance reflects T94's own residual on this panel. B.4 tests the Brewster-effect region specifically — a correct MIMICS implementation must reproduce this drop.

---

## Set C — Wheat scattering-mechanism decomposition (structural anchor)

**Source:** T94 Fig. 7, page 54, CHH panel (bottom-left). Four panels (LHH, LVV, CHH, CVV) each showing total σ° alongside its five component mechanisms (ground-cover-ground, cover-ground, ground-cover, direct cover, direct ground) as a function of incidence angle.

**Why this set matters:** Bulk-total anchors (Sets A, B) can pass even when individual scattering mechanisms are miscomputed if the errors cancel. Set C tests each mechanism independently — this is the specific check against shared-misreading bugs in the core MIMICS radiative-transfer equations. T94 Fig. 7 for CHH (wheat) is the primary target because CHH is the most Phase-1b-relevant panel.

### Marker-to-mechanism mapping (v0.2 correction)

v0.1 of this document had a mismatch between its marker-to-mechanism mapping and the T94 Fig. 7 caption. v0.2 uses the T94 caption as authoritative:

| Mechanism | T94 caption marker | v0.1 said | v0.2 says |
|---|---|---|---|
| Total σ° | o (open circle) | open circle (C.6) | open circle ✓ |
| Ground-cover-ground | ▲ (filled triangle) | filled triangle (C.5) | filled triangle ✓ |
| Cover-ground | △ (open triangle) | filled-square (WRONG) | **open triangle** |
| Ground-cover | ■ (filled square) | open-triangle (WRONG) | **filled square** |
| Direct cover | □ (open square) | open-square (C.2) | open square ✓ |
| Direct ground | ♦ (filled diamond) | filled diamond (C.1) | filled diamond ✓ |

### Canonical inputs

Identical to Set A (wheat, site #13, 18 July 1988). T94 Fig. 7 caption confirms the inputs match Fig. 2.

### Anchor values (v0.2, refined by human transcription at θ=30°)

**v0.1 → v0.2 method change:** v0.1 relied on pixel detection. The middle cluster at ~−20 dB contains multiple overlapping markers that template scoring could not reliably distinguish, and the C.3/C.4 marker swap introduced an attribution risk. Per the continuation-prompt Step 3 fallback clause, Set C values were obtained by human transcription from the PDF.

| Row | Mechanism | Marker | σ° (dB) @ θ=30° | Tolerance | Source |
|---|---|---|---|---|---|
| C.1 | Direct ground | ♦ (filled diamond) | **−11** | ±0.5 dB | T94 Fig. 7 CHH, ♦ at θ=30° |
| C.2 | Direct cover | □ (open square) | **−20** | ±0.5 dB | T94 Fig. 7 CHH, □ at θ=30° |
| C.3 | Cover-ground | △ (open triangle) | **DROPPED** | — | Not visible at CHH θ=30° in T94 Fig. 7 (coincident with −20 dB cluster or negligible for this geometry/polarisation). See DEV-1b-003 Appendix for drop rationale. |
| C.4 | Ground-cover | ■ (filled square) | **−20** | ±0.5 dB | T94 Fig. 7 CHH, ■ at θ=30° (sits on top of / just under the □ at same dB) |
| C.5 | Ground-cover-ground | ▲ (filled triangle) | **−35** | ±2.0 dB | T94 Fig. 7 CHH, ▲ at θ=30° |
| C.6 | Total σ° | o (open circle) | **−10** | ±0.5 dB | T94 Fig. 7 CHH, o at θ=30° |

**v0.2 anchor count for Set C: 5 rows** (C.3 dropped). DEV-1b-003 Appendix records the drop.

### Notes

- Row C.5 has a wider tolerance because the ground-cover-ground mechanism is two-orders-of-magnitude smaller than the dominant mechanisms at this angle; the absolute-dB tolerance is less physically meaningful for near-zero linear values. **v0.2 note:** v0.1 estimated C.5 at −40 dB with ±2.0 dB; refined value is −35 dB, which v0.1's wider band would have missed by 3 dB. Anchor corrected.
- Row C.6 (total σ°) is a self-consistency check — the linear sum of the four remaining mechanisms (C.1 ⊕ C.2 ⊕ C.4 ⊕ C.5; C.3 dropped) should match the total σ° published in the same panel, which in turn should match Set A row A.2 (30°, CHH) at −10.44 dB. Any discrepancy between (C.1 ⊕ C.2 ⊕ C.4 ⊕ C.5) and C.6, or between C.6 and A.2, is a bug-finder.

### Pass criterion

Rows C.1, C.2, C.4 and C.6 all pass individually at ±0.5 dB. Row C.5 passes its wider ±2.0 dB tolerance. Internal consistency check (linear sum of remaining mechanisms ≈ C.6 linear) holds within 0.3 dB.

---

## Set C2 — Mechanism-ratio secondary anchor (v0.3 addition, DEV-1b-006)

**Source:** Ulaby, F.T. and Long, D.G. (2014), *Microwave Radar and Radiometric Remote Sensing*, University of Michigan Press / Artech House, Chapter 11: Volume-Scattering Models and Land Observations, **Table 11-1, page 484**. PDF staged at [`phase1b/refs/Ulaby_Long_2014_Ch11.pdf`](../refs/Ulaby_Long_2014_Ch11.pdf).

**Role.** Set C (T94 Fig. 7 CHH mechanism decomposition at θ=30°) is the primary mechanism-level anchor. Set C2 is an **independent secondary primary-source cross-check** on mechanism ratios drawn from a different scenario and textbook compilation. Having two independent primary sources for the mechanism decomposition reduces the risk that a shared misreading of a single source propagates into an implementation that passes its own anchor but is wrong. Set C2 is **informational**, not gating — rows are scored and reported but do not count against the published_table arm pass/fail predicate. This keeps the Session E G2 pass verdict driven by the sources that were pre-registered in v0.2 (T94, M90) while adding U&L 2014 as a provenance-stamped cross-check.

**Scenario.** Ulaby & Long 2014 Table 11-1 tabulates σ°_gcg/σ°_c and σ°_cgt/σ°_c ratios for a Rayleigh canopy over a ground surface at θ = 30°, across {HH, VV} × {canopy one-way transmissivity Υ ∈ {0.8, 0.5, 0.1}} × {dry ground m_v = 0, wet ground m_v = 0.35 g/cm³}. This is a structurally idealised Rayleigh scenario (not wheat, not heather, not walnut) — deliberately so, to isolate the mechanism-decomposition structure from any specific canopy.

**Canonical inputs.** PyTorch MIMICS is configured in a Rayleigh-equivalent limit matching U&L 2014's Table 11-1 scenario. Specifically the harness (when implemented in Session E Phase E-2 alongside DEV-1b-006) will invoke `mimics_toure_single_crown_breakdown_torch` with canopy parameters chosen to hit each (Υ, m_v) point: Υ controls the canopy optical depth via N_b, N_l and extinction; m_v = 0 / 0.35 with Dobson mineral dielectric sets the ground contribution. Exact canonical inputs per row are specified in the harness and recorded in `anchor_reads_v1.json` when Phase E-2 lands.

**Tolerance.** ±0.5 dB per ratio (dB-space tolerance on each ratio row). Informational.

### Anchor values

Transcribed from U&L 2014 Table 11-1, p. 484. The dB value in square brackets is `10·log10(ratio)` for direct comparison against PyTorch's mechanism-decomposition output. `σ_gcg/σ_c` is the ground-cover-ground (double-bounce) to direct-cover ratio; `σ_cgt/σ_c` is the cover-ground-total to direct-cover ratio.

| Row | Ratio | Pol | Υ (one-way) | m_v (g/cm³) | Value | Value (dB) |
|---|---|---|---|---|---|---|
| C2.01 | σ_gcg/σ_c | HH | 0.8 | 0.00 | 4.1 × 10⁻³ | −23.9 |
| C2.02 | σ_gcg/σ_c | HH | 0.5 | 0.00 | 1.6 × 10⁻³ | −28.0 |
| C2.03 | σ_gcg/σ_c | HH | 0.1 | 0.00 | 6.4 × 10⁻⁵ | −41.9 |
| C2.04 | σ_gcg/σ_c | VV | 0.8 | 0.00 | 2.3 × 10⁻³ | −26.4 |
| C2.05 | σ_gcg/σ_c | VV | 0.5 | 0.00 | 9.0 × 10⁻⁴ | −30.5 |
| C2.06 | σ_gcg/σ_c | VV | 0.1 | 0.00 | 3.6 × 10⁻⁵ | −44.4 |
| C2.07 | σ_gcg/σ_c | HH | 0.8 | 0.35 | 0.135       | −8.7 |
| C2.08 | σ_gcg/σ_c | HH | 0.5 | 0.35 | 5.3 × 10⁻² | −12.8 |
| C2.09 | σ_gcg/σ_c | HH | 0.1 | 0.35 | 2.1 × 10⁻³ | −26.8 |
| C2.10 | σ_gcg/σ_c | VV | 0.8 | 0.35 | 8.3 × 10⁻² | −10.8 |
| C2.11 | σ_gcg/σ_c | VV | 0.5 | 0.35 | 3.2 × 10⁻² | −14.9 |
| C2.12 | σ_gcg/σ_c | VV | 0.1 | 0.35 | 1.3 × 10⁻³ | −28.9 |
| C2.13 | σ_cgt/σ_c | HH | 0.8 | 0.00 | 0.40        | −4.0 |
| C2.14 | σ_cgt/σ_c | HH | 0.5 | 0.00 | 6.2 × 10⁻² | −12.1 |
| C2.15 | σ_cgt/σ_c | HH | 0.1 | 0.00 | 1.5 × 10⁻² | −18.2 |
| C2.16 | σ_cgt/σ_c | VV | 0.8 | 0.00 | 0.30        | −5.2 |
| C2.17 | σ_cgt/σ_c | VV | 0.5 | 0.00 | 4.7 × 10⁻² | −13.3 |
| C2.18 | σ_cgt/σ_c | VV | 0.1 | 0.00 | 1.1 × 10⁻² | −19.6 |
| C2.19 | σ_cgt/σ_c | HH | 0.8 | 0.35 | 2.3         | +3.6 |
| C2.20 | σ_cgt/σ_c | HH | 0.5 | 0.35 | 0.36        | −4.4 |
| C2.21 | σ_cgt/σ_c | HH | 0.1 | 0.35 | 8.6 × 10⁻² | −10.7 |
| C2.22 | σ_cgt/σ_c | VV | 0.8 | 0.35 | 1.7         | +2.3 |
| C2.23 | σ_cgt/σ_c | VV | 0.5 | 0.35 | 0.28        | −5.5 |
| C2.24 | σ_cgt/σ_c | VV | 0.1 | 0.35 | 6.6 × 10⁻² | −11.8 |

### Phase E-1 scope note

In Phase E-1 (DEV-1b-004 + DEV-1b-005), the Set C2 anchor **rows and dB values are registered in this specification** but the harness evaluation is **deferred to Phase E-2** (DEV-1b-006), where the `mimics_toure_single_crown_breakdown_torch` helper is wired into the published_table arm alongside the v0.2 physics promotion. Until Phase E-2 lands, `equivalence_check.py` reports Set C2 rows as `status: DEFERRED_PHASE_E2` and does not count them against the arm. This is scope-disciplined rather than a deviation — registering the anchor now fixes it to the pre-registration record; harnessing it in Phase E-2 is the implementation step.

### Pass criterion (when active from Phase E-2)

All 24 ratio rows within ±0.5 dB (informational). Set C2 does not gate the G2 verdict.

### Notes

- U&L 2014 Chapter 11 does not contain finite-cylinder scattering amplitude derivations (no Bessel-J₁ radial or sinc axial form factor material — those sit in Chapter 8 §8-5, which is not staged). Finite-cylinder correctness in PyTorch MIMICS is therefore established via (a) the Rayleigh-limit analytic regression derived from Ch 11 eqs 11.76 / 11.77 / 11.85, (b) the Set A / B / C published_table anchors against T94 Fig. 2, and (c) internal autograd-vs-FD consistency checks — not via an external Table 11-1-style form-factor tabulation. See `phase1b/refs/README.md` for the coverage note.
- The Set C2 scenario is Rayleigh-idealised, not heather-specific. A failure here indicates a structural issue in the mechanism decomposition; it does not directly implicate Moor House operating-regime performance.

---

## Set D — Walnut orchard L-band, non-Toure structural cross-check

**v0.3 STATUS: EXEMPT pending Phase 1c.** Per DEV-1b-005, Set D is held pending the Phase 1c trunk-layer code-path build (NISAR L-band comparative validation, Green Paper v4.1.0 §8). The four anchor values below are **retained verbatim** and will reactivate as a Phase 1c entry gate when the trunk-layer structural extensions (four branch classes, cos⁶ zenith PDF, complex vegetation dielectric with Im(ε), L-band dual-dispersion UEL) exist. In Phase 1b Session E, Set D rows return `status: EXEMPT` and do not count against the G2 published_table arm pass/fail verdict. See DEV-1b-005 for the full exemption rationale and the re-instatement criteria.

**Source:** M90 Fig. 10 (page 487). M90 Table I (canopy branch classes, page 482), M90 Table II (leaf characteristics, page 482), M90 Table III (canopy dielectric characteristics, page 486).

**v0.2 correction — figure semantics:** v0.1 described the MIMICS values as "open-circle markers". This is materially wrong. M90 Fig. 10 uses:
- **Lines** (solid=VV, dotted=HH, dashed=HV) = **MIMICS-modelled σ°**
- **Markers** (open circle with dot ⊙ = VV Measured; open square □ = HH Measured; open triangle △ = HV Measured) = **scatterometer measurements**

v0.2 reads the LINES at the target θ, not the markers. MIMICS curves in Fig. 10 end at θ=55° (no data beyond).

**Why this set matters:** M90's walnut orchard retains the trunk layer, so anchoring against M90 requires running our PyTorch MIMICS with the trunk layer re-enabled — a mode not used in Phase 1b production (§4 uses Toure-style no-trunk adaptation). This tests the crown-layer equations and the trunk-scattering equations in their native forest-canopy context, which is orthogonal to Toure's trunk-less adaptation. Bug in the crown volume-scattering equations or the extinction-matrix operations will show up here even if Set A/B/C pass.

### Canonical inputs

From M90 Tables I, II, III (reproduce in full in the anchor harness):

**Canopy architecture (M90 Table I, page 482):**

| Class | Max. diam (cm) | Min. diam (cm) | Ave. diam (cm) | Ave. length (cm) | Density (/m³) | Orientation f(θ) |
|---|---|---|---|---|---|---|
| Trunk branches | — | 4.0 | 7.3 | 92.8 | 0.13 | cos⁶ θ |
| Primary branches (crown) | 4.0 | 0.9 | 1.9 | 35.8 | 1.55 | sin⁴ 2θ |
| Secondary branches (crown) | 0.9 | 0.4 | 0.6 | 10.9 | 1.41 | sin θ |
| Stems (crown) | 0.4 | — | 0.1 | 5.0 | 900 | sin θ |

**Leaf characteristics (M90 Table II, page 482):**
- Leaf density = 652 leaves/m³
- Average leaf diameter = 5.0 cm
- Average leaf thickness = 0.02 cm
- Leaf area index = 3.2
- Leaf orientation PDF = spherical

**Layer thicknesses (M90 §V.A, page 481):**
- Crown layer height d = 2.5 m
- Trunk layer height H_t = 1.7 m

**Dielectric constants (M90 Table III, page 486, multiangle experiment):**
- Ground ε = 25 − j2.5
- Trunk ε = 45 − j11.2
- Primary branch ε = 34 − j8.5
- Secondary branch ε = 30 − j7.5
- Leaf and stem ε = 36.5 − j11.3

**Radar parameters:**
- Frequency 1.25 GHz (L-band)
- Incidence angles 40°, 55°
- Polarisations VV, HH

### Anchor values (v0.2, refined from M90 Fig. 10 MIMICS lines)

Read from MIMICS LINE positions (not markers) at θ=40° and θ=55° in M90 Fig. 10.

| Row | θ (deg) | Pol | Curve type | σ°_L,MIMICS (dB) | Tolerance | Source |
|---|---|---|---|---|---|---|
| D.1 | 40 | HH | dotted | **−11.73** | ±0.5 dB | M90 Fig. 10 dotted line at θ=40° |
| D.2 | 55 | HH | dotted | **−11.35** | ±0.5 dB | M90 Fig. 10 dotted line at θ=55° |
| D.3 | 40 | VV | solid | **−12.17** | ±0.5 dB | M90 Fig. 10 solid line at θ=40° |
| D.4 | 55 | VV | solid | **−12.17** | ±0.5 dB | M90 Fig. 10 solid line at θ=55° |

### Notes

- M90's modelled σ° is nearly flat across 40°–55° for this canopy; HH−VV separation is < 1 dB at both angles with HH slightly above VV.
- Our Phase 1b production code runs the Toure no-trunk adaptation (SPEC §4). For Set D the trunk layer must be re-enabled. Implement this as a code path gated by a `use_trunk_layer: bool` flag that defaults to `False`. The Set D test path is the ONLY code path that should ever set it to `True`. Production code paths (Phase 1b training, inference) must not expose this flag as a runtime-settable option.

### Pass criterion

All four rows (D.1–D.4) pass individually. The small slope in the 40°–55° range is reproduced within ±0.5 dB.

---

## Set E — Gradient spot-check against published sensitivities (differentiability anchor)

**Source:** T94 Table V (wheat L-band, page 57) and Table VI (canola, page 58). These tables report |∂σ°/∂P_i| × ζ_i — the error in modelled σ° propagated from a unit uncertainty ζ_i in each input parameter P_i — at θ = 20°, 30°, 40°, 50° for both VV and HH polarisations.

**Why this set matters:** Forward-mode σ° correctness (Sets A–D) does not prove gradient correctness. A PyTorch MIMICS with a wrong scattering-matrix formulation could produce correct σ° at the anchor points (by coincidence) but wrong gradients everywhere, which would cause the PINN optimiser to take wrong steps during training — silently. For a differentiable PINN the gradient is as important as the value. This set is the specific check that autograd is returning the right derivative through the MIMICS forward graph.

### Canonical inputs

**Reference cover for wheat** (T94 Table IV(a), page 51; verified via pdftotext):
- Stem m_g = 0.7, diameter = 0.2 cm, height = 0.4 m, density = 200 st./m², PDF = vertical
- Leaf m_g = 0.7, length = 12 cm, width = 1 cm, thickness = 0.02 cm, density = 3000 lf./m³, PDF = uniform
- Soil m_v = 0.2 g/cm³, C-band s = 0.45 cm, l_s = 1.2 cm, L-band s = 1 cm, l_s = 5 cm

### Dielectric configuration (v0.3 addition, DEV-1b-004)

**For E.1 and E.2 only (Soil m_v sensitivity rows):** the PyTorch MIMICS forward is evaluated under a **Dobson 1985 mineral-soil** ground dielectric with T94-consistent parameters. This reflects T94's actual configuration — T94 §II.A inherits the soil dielectric from MIMICS [Ulaby 1990 ref 19] which uses Dobson 1985, and T94 does not override (verified via text extraction of T94 §II.A, 2026-04-19).

Specifically: `phase1b.physics.mimics.ground_epsilon_dobson_torch` is called with `eps_dry = DOBSON_EPS_DRY_MINERAL = 3.0`, `eps_water = 80.0`, `alpha = DOBSON_ALPHA_MINERAL = 0.65` (Dobson 1985's generic moderate-clay-fraction mineral loam). At the T94 Table IV(a) wheat reference m_v = 0.2 g/cm³ this gives ε ≈ 30.75 with ∂ε/∂m_v ≈ 90.2 — unclamped, non-zero gradient.

**For E.3, E.4, E.5 (stem height / leaf width sensitivity rows):** the default Mironov GRMDM path is used (these rows do not perturb m_v, so the dielectric parameterisation is not the binding issue).

**For the Moor House production path** (PINN-MIMICS trainer, inference, Phase 1b λ search, diagnostics): the SPEC §6 primary Mironov GRMDM with the DEV-007 ε ≥ 1.01 clamp is used — unchanged. A regression test in `tests/unit/test_mimics_torch.py` pins the production-path call signature to `mimics_toure_single_crown(params, ground_dielectric_fn=None)`.

The amendment makes previously-implicit configuration explicit per anchor; it does not re-read any anchor value, does not loosen any tolerance, and does not remove the DEV-007 clamp.

### Anchor values (v0.2, verified from T94 Table V(a) via pdftotext)

All five rows exactly match the v0.1 spec values — no refinement needed because the table is numerical and was extracted as text. Source: T94 Table V(a), page 57.

| Row | Parameter | ζ (perturbation) | θ (deg) | Pol | \|∂σ°/∂P\| × ζ (dB) | Source |
|---|---|---|---|---|---|---|
| E.1 | Soil m_v | ±0.04 g/cm³ | 30 | VV | **1.21** | T94 Table V(a), row "m_v ±0.04 g/cm³", θ=30° VV |
| E.2 | Soil m_v | ±0.04 g/cm³ | 30 | HH | **1.16** | T94 Table V(a), same row, θ=30° HH |
| E.3 | Stem height | ±0.05 m | 30 | VV | **0.10** | T94 Table V(a), row "h ±0.05 m", θ=30° VV |
| E.4 | Stem height | ±0.05 m | 30 | HH | **0.05** | T94 Table V(a), same row, θ=30° HH |
| E.5 | Leaf width | ±0.5 cm | 30 | HH | **0.18** | T94 Table V(a), row "w ±0.5 cm", θ=30° HH |

### Computation method

For each row: perturb the named parameter by ±ζ about the reference value, compute σ° via PyTorch forward pass at each perturbation, and take the finite-difference estimate of |∂σ°/∂P| × ζ. Cross-check by computing ∂σ°/∂P via `torch.autograd.grad` and multiplying by ζ. Both the finite-difference and the autograd result should agree with T94's published value.

### Tolerance

For each row: PyTorch value within **±20% of T94 value, or ±0.1 dB, whichever is larger.** This reflects that T94's own sensitivities are finite-difference-derived (so noisy at the few-percent level) and that the reference-cover inputs are defined to limited precision in T94 Table IV.

Additional internal consistency: `finite_difference` and `autograd` results should agree with each other within ±5% or ±0.02 dB, whichever is larger. This is the direct test of differentiability.

### Pass criterion

All five rows (E.1–E.5) pass individually (PyTorch vs T94 tolerance). All five rows pass the finite-difference-vs-autograd internal consistency check.

---

## Summary table (v0.3)

| Set | Purpose | # values | Source | Status | Tolerance |
|---|---|---|---|---|---|
| A | Bulk CHH wheat | 4 | T94 Fig. 2 CHH panel | active (gating) | 0.5 dB |
| B | Bulk CVV wheat | 4 | T94 Fig. 2 CVV panel | active (gating) | 0.5 dB |
| C | Mechanism decomposition (primary) | **5** | T94 Fig. 7 CHH panel | active (gating) | 0.5 dB (2.0 dB for row C.5) |
| C2 | Mechanism ratios (secondary) | **24** | U&L 2014 Table 11-1, p. 484 | informational (non-gating; harness active from Phase E-2) | 0.5 dB per ratio |
| D | Non-Toure structural cross-check | 4 | M90 Fig. 10 (LINES, not markers) | **EXEMPT** pending Phase 1c (DEV-1b-005) | 0.5 dB (when active) |
| E | Gradient / autograd | 5 | T94 Tables V, VI | active (gating) | ±20 % or ±0.1 dB; plus autograd-vs-FD consistency |
| **Total (active + informational)** | | **42** | | | |
| **Total (gating in Session E)** | | **18** (A + B + C + E active rows) | | | |

---

## Test harness requirements

`phase1b/physics/equivalence_check.py` must:

1. Load each anchor value from a `canonical_combinations.json` file (or equivalent machine-readable source) that mirrors the tables above.
2. Run the differentiable PyTorch MIMICS implementation on each input set and record the σ° output. Set D must set `use_trunk_layer=True`; all other sets must leave `use_trunk_layer=False` (its production default).
3. For Set E, additionally compute ∂σ°/∂P via autograd AND via finite differences, and report both.
4. Emit a pass/fail verdict per row and per set, with actual-vs-expected values and per-row deviations.
5. Write a machine-readable result JSON (`outputs/g2_equivalence_result.json`) containing: overall verdict, per-arm and per-set pass count, per-row actual and expected values, and any row that failed with the diagnostic context needed to localise the bug.

If any arm fails, the script exits non-zero and G2 fails. The honest-gates protocol (SPEC §9, §13) applies: fallback is not "relax the anchor," it is "halt and fix the implementation."

---

## Change control

- Changes to this document after §14 sign-off require a DEV log entry. v0.3 amendments are covered by DEV-1b-004 (dielectric-configuration specificity for E.1 / E.2), DEV-1b-005 (Set D Phase 1c exemption), and DEV-1b-006 (Set C2 secondary anchor registration + forthcoming v0.2 physics promotion).
- Changes to the numerical anchor values post-sign-off require the entry to name which rows moved, by how much, and why. Loosening a tolerance to pass a failing test is an explicit red flag in the honest-gates protocol. **v0.3 does not change any anchor value or tolerance** — all changes are additive.
- Adding anchor rows post-sign-off is permitted (strengthens the check); removing or re-reading rows post-sign-off is not. v0.3's Set C2 registration is a strengthening addition. v0.3's Set D EXEMPT marker is a deferral, not a removal: the four rows remain in the spec with their anchor values intact and reactivate as a Phase 1c entry gate per DEV-1b-005's re-instatement criteria.

---

## v0.2 → v0.3 changelog (Session E, 2026-04-19)

All changes in v0.3 are **additive** to the pre-registration record — no existing anchor value or tolerance is modified. See DEV-1b-004, DEV-1b-005, DEV-1b-006 for the full rationale per change.

- **Set E — dielectric-configuration block added (DEV-1b-004):** rows E.1 and E.2 (Soil m_v sensitivity) now evaluate against PyTorch MIMICS configured with Dobson 1985 mineral-soil parameters (`DOBSON_EPS_DRY_MINERAL = 3.0`, `DOBSON_ALPHA_MINERAL = 0.65`). This reflects T94's actual dielectric inheritance from MIMICS [Ulaby 1990] at §II.A. Rows E.3, E.4, E.5 unchanged (use the default Mironov path). The Moor House production configuration is unchanged; a regression test pins it. Anchor values (1.21, 1.16, 0.10, 0.05, 0.18 dB) and tolerances (±20 % / ±0.1 dB vs T94; ±5 % / ±0.02 dB autograd-vs-FD) unchanged.
- **Set D — status EXEMPT (DEV-1b-005):** Set D is held pending the Phase 1c trunk-layer implementation (NISAR L-band validation, Green Paper v4.1.0 §8). Anchor values (D.1 = −11.73, D.2 = −11.35, D.3 = −12.17, D.4 = −12.17 dB) retained verbatim. Rows no longer count against G2 pass/fail in Session E; reactivated as a Phase 1c entry gate when the trunk-layer code path exists.
- **Set C2 — new secondary anchor block added (DEV-1b-006):** 24 mechanism ratios (σ_gcg/σ_c and σ_cgt/σ_c) from Ulaby & Long 2014 Table 11-1, p. 484. Informational, not gating; scored when the Phase E-2 harness lands. U&L 2014 Chapter 11 PDF staged at `phase1b/refs/Ulaby_Long_2014_Ch11.pdf`.
- **No value, tolerance, or source change** to any v0.2-signed-off anchor row in Sets A, B, C, or E.

## v0.1 → v0.2 changelog

- **Sets A, B (8 rows):** values refined from open-square marker reads in T94 Fig. 2 CHH/CVV panels at 400 dpi. Human spot-check at A.1, B.2, B.3, B.4 confirmed methodology. Calibration: y-axis label-centre, linear fit across 4 labels per panel, <1% slope variation between panels.
- **Set B panel-location correction:** v0.1 referred to "top-right" and "labelled LVV"; Fig. 2's CVV panel is bottom-right. No values affected, but the panel-location description was ambiguous.
- **Set C marker-to-mechanism mapping:** v0.1's mapping disagreed with T94 Fig. 7 caption on C.3 and C.4. v0.2 uses the caption.
- **Set C row C.3 dropped:** the open-triangle Cover-ground marker is not visible in the CHH panel at θ=30° (coincident with the −20 dB cluster or negligible). DEV-1b-003 Appendix records the drop.
- **Set C row C.5 value:** v0.1 estimate −40 dB; actual −35 dB (5 dB correction).
- **Set D figure-semantics correction:** v0.1 described MIMICS values as "open-circle markers". M90 Fig. 10 actually draws MIMICS as lines (solid/dotted/dashed) and measurements as markers. v0.2 reads lines.
- **Set E:** all 5 rows unchanged. v0.1 values matched T94 Table V(a) exactly; verified via text extraction.
- **Total anchor count:** v0.1 specified 23 rows; v0.2 retains 22 (C.3 dropped).
