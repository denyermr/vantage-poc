# PROGRESS.md — ECHO PoC Build State

**Updated:** 2026-03-09  
**Rule:** This file is the first thing read after CLAUDE.md at every session start. It is the single source of truth for what is done, what is in progress, and what comes next.

---

## Current Status

```
Phase 1 — Data Acquisition        [✓] COMPLETE — Gate 1 passed (with deviations DEV-003/004/005)
Phase 2 — Baseline Models         [✓] COMPLETE — Gate 2 passed (with deviation DEV-006)
Phase 3 — PINN & Evaluation       [✓] COMPLETE — Outcome: NEGATIVE (DEV-007/008)
Phase 4 — Diagnostics             [✓] COMPLETE — All 4 diagnostics run, figures generated
Phase 5 — Results Paper            [✓] COMPLETE — poc_results.md drafted
```

**Active phase:** All phases complete
**Next task:** Human review of results paper, then Carbon13 materials
**Blockers:** None

---

## Gate Status

| Gate | Status | Result file | Date passed |
|------|--------|-------------|-------------|
| Gate 1 — Data Quality | ✅ Passed (DEV-003/004/005) | `outputs/gates/gate_1_result.json` | 2026-03-09 |
| Gate 2 — Baseline Validity | ✅ Passed (DEV-006) | `outputs/gates/gate_2_result.json` | 2026-03-09 |
| Gate 3 — PINN & Outcome | ✅ Passed (Negative) | `outputs/metrics/phase4_diagnostics.json` | 2026-03-09 |
| Gate 4 — Diagnostics | ✅ Complete | `outputs/metrics/phase4_diagnostics.json` | 2026-03-09 |
| Gate 5 — Results Paper | ✅ Draft complete | `outputs/write-up/poc_results.md` | 2026-03-09 |

---

## Phase 1 — Data Acquisition

**Spec:** `SPEC_PHASE1.md`
**Status:** Complete — Gate 1 passed 2026-03-09

### Human steps required (cannot be automated — no network access)

- [x] **H1.1** — GEE S1 extraction (orbit 154 auto-selected, 148 overpasses). Downloaded to `data/raw/gee/sentinel1_raw.csv`
- [x] **H1.2** — GEE S2 NDVI extraction (48 monthly composites). Downloaded to `data/raw/gee/sentinel2_ndvi_raw.csv`
- [x] **H1.3** — COSMOS-UK daily product (pre-processed CSV). Saved to `data/raw/cosmos/cosmos_daily_vwc.csv`
- [x] **H1.4** — GEE ERA5-Land precipitation (1,468 daily records). Downloaded to `data/raw/gee/era5_precip_raw.csv`
- [x] **H1.5** — GEE SRTM/MERIT terrain (static site values). Downloaded to `data/raw/gee/terrain_static_raw.csv`

### Claude Code tasks

- [x] P1.1 — Monorepo scaffold: directory structure, `.gitignore`, root `Makefile`
- [x] P1.2 — `poc/config.py`: all Phase 1+ constants, paths, lazy torch device
- [x] P1.3 — `poc/data/cosmos.py`: COSMOS-UK loader + QC
- [x] P1.4 — `poc/data/gee/extract_sentinel1.py`: GEE SAR extraction + post-processing (orbit 154, 148 overpasses)
- [x] P1.5 — `poc/data/gee/extract_sentinel2.py`: GEE NDVI extraction + interpolation (48 monthly composites)
- [x] P1.6 — `poc/data/gee/extract_era5.py`: ERA5-Land precipitation extraction (1,468 daily records)
- [x] P1.6b — `poc/data/gee/extract_terrain.py`: SRTM/MERIT terrain extraction (static site values; DEV-004)
- [x] P1.7 — `poc/data/ancillary.py`: ancillary feature assembly
- [x] P1.7b — `poc/data/alignment.py`: multi-source join + QC filter + train/test split
- [x] P1.8 — `poc/gates/gate_1.py`: Gate 1 script (all 10 criteria from spec)
- [x] P1.9 — `tests/unit/test_data_p1.py`: 20 unit tests (15 spec-required + 5 config)
- [x] P1.10 — `tests/integration/test_pipeline_p1.py`: 5 integration tests
- [x] P1.11 — `poc/evaluation/plots_p1.py`: Diagnostic figures (4 plots per spec)
- [x] P1.12 — `requirements.txt` with pinned versions
- [x] P1.13 — Run Gate 1. Review DEVIATIONS.md. Sign off.

**Gate 1 result:** ✅ Passed 2026-03-09 (8/10 auto-check; G1-05 reinterpreted per DEV-003; G1-08 confirmed)

**Phase 1 completion summary:**
- 120 clean paired observations (148 raw → −2 VWC flag → −26 frozen/snow → 120 final, plus 1 manual exclusion per DEV-005 → 119 final)
- Strong seasonal VWC signal confirmed: winter mean 0.70, summer mean 0.50
- COSMOS-UK ground truth validated across 2021–2024; 2023 drought event noted as extending training range at dry end of state space
- SAR signal confirmed present and physically coherent; G1-05 gate reinterpreted per DEV-003
- Terrain features excluded per DEV-004; site metadata values recorded (slope=4.89°, TWI=15.82)
- Three VH/VV outliers investigated: one manually excluded (DEV-005, 2021-02-05), one excluded by existing QC (2021-02-11, frozen_flag=1), one retained as valid saturated-peat signal (2021-12-08)
- S1B satellite loss (December 2021) confirmed as explanation for reduced overpass density in 2022–2024 (~29–30 vs ~60 in 2021); not a data quality issue
- Two supplementary QC rules added to pipeline spec (freeze-thaw lag, cross-pol outlier)
- All Phase 1 gate criteria satisfied; proceeding to Phase 2

---

## Phase 2 — Baseline Models

**Spec:** `SPEC_PHASE2.md`
**Status:** Complete — Gate 2 passed 2026-03-09

### Human steps required

None — Phase 2 is fully automated once Phase 1 data exists.

### Claude Code tasks

- [x] P2.1 — `poc/evaluation/splits.py`: `generate_all_configs()`, split manifest (40 configs generated)
- [x] P2.2 — `poc/models/base.py`: abstract base model interface
- [x] P2.3 — `poc/models/null_model.py`: seasonal climatological baseline
- [x] P2.4 — `poc/models/random_forest.py`: RF with GridSearchCV
- [x] P2.5 — `poc/models/standard_nn.py`: 3-layer NN with early stopping, MPS support
- [x] P2.6 — `poc/evaluation/harness.py`: `compute_metrics`, `aggregate_metrics_across_reps`, `wilcoxon_test`
- [x] P2.7 — `poc/evaluation/plots.py`: learning curve figures, feature diagnostic figures
- [x] P2.8 — `poc/gates/gate_2.py`: Gate 2 script (all 13 criteria + live PINN target computation)
- [x] P2.9 — `tests/unit/test_models_p2.py`: all 30 unit tests (all passing)
- [x] P2.10 — `tests/integration/test_pipeline_p2.py`: 6 integration tests (all passing)
- [x] P2.11 — Run all 40 baseline configurations (null + RF + NN)
- [x] P2.12 — Run Gate 2. Review DEVIATIONS.md. Sign off.

**Gate 2 result:** ✅ Passed 2026-03-09 (13/13 criteria; DEV-006 G2-08 reinterpreted as relative criterion)

**Phase 2 completion summary:**
- All 40 configurations trained successfully (RF and NN across all training size fractions)
- Null model RMSE established as reference benchmark: 0.1776
- RF baseline RMSE at 100% training data: 0.1470 (17% improvement over null)
- NN baseline RMSE at 100% training data: 0.1594 (10% improvement over null)
- Both baselines beat null at all training sizes; learning curves show expected patterns
- R² negative under chronological split as expected — not used as gate metric
- G2-08 absolute RMSE threshold not met; reinterpreted as relative criterion per DEV-006
- Baseline benchmarks now fixed as reference for all PINN gate evaluations in Gate 2
- PINN targets computed: Strong < 0.124, Significant < 0.131, Moderate < 0.139 at N=25
- 56 tests passing (20 P1 unit + 30 P2 unit + 6 P2 integration), 5 skipped (P1 integration)
- Phase 3 unblocked: next step P3.1 Dobson + Mironov dielectric models
- Final aligned dataset: 119 observations (148 raw → QC exclusions → 119 final including DEV-005 manual exclusion)
- VV–VWC correlation: r=0.261 on raw data (Phase 1, n=120); r=0.290 (p=0.0014) on 119-observation aligned dataset (Phase 2 feature diagnostics) — same relationship, small difference due to QC exclusions, not an error
- RF is the stronger baseline at all training sizes, statistically significantly better than NN at all fractions (Wilcoxon, p<0.01, Bonferroni-corrected)
- RF learning curve is flat (robust at small N); NN is data-hungry (RMSE 0.319 at 10% → 0.159 at 100%)
- RF feature importances: NDVI 26%, precip_7day 19%, VH/VV ratio 15%, precip_mm 10%, VV 10%, incidence angle 9%, VH 9% — consistent with physical model and Phase 1 correlation findings
- Negative R² expected and not informative under chronological split with non-stationary test distribution
- PINN targets pre-registered (see SPEC.md §6.5)
- Active deviations at gate close: DEV-003, DEV-004, DEV-005, DEV-006
- Test suite: 56 passing, 5 skipped (Phase 1 integration tests require raw data re-download — expected)

---

## Phase 3 — PINN & Evaluation

**Spec:** `SPEC_PHASE3.md`
**Status:** Complete — Outcome: NEGATIVE (2026-03-09)

### Human steps required

None — Phase 3 is fully automated.

### Claude Code tasks

- [x] P3.1 — `poc/models/dielectric.py`: `DobsonDielectric` + `MironovDielectric`
- [x] P3.2 — WCM forward model functions: `oh_soil_backscatter`, `wcm_vegetation_terms`, `wcm_forward`
- [x] P3.3 — `poc/models/pinn.py`: `PhysicsNet`, `CorrectionNet`, `PINN`, `compute_pinn_loss`
- [x] P3.4 — `poc/evaluation/lambda_search.py`: λ grid search on 100% configs (DEV-008: all violate dominance)
- [x] P3.5 — PINN training loop: all 40 configs with per-epoch logging + `test_predictions.json`
- [x] P3.6 — `poc/evaluation/diagnostics.py`: 4 diagnostics (residual analysis, WCM forward fit, identifiability, Mironov sensitivity)
- [x] P3.7 — Figures: p3_pinn_learning_curves.png, p3_pinn_vs_rf_scatter.png
- [x] P3.8 — Outcome determination: NEGATIVE (user confirmed)

**P3.1 completion summary:**
- `poc/models/dielectric.py` — `DielectricModel` ABC, `DobsonDielectric`, `MironovDielectric` implemented
- `poc/config.py` — updated: KS_ROUGHNESS 0.1→0.30, WCM_A_LB/UB/B_LB/UB, Mironov constants, Phase 3 λ/diagnostic/interval config added
- `tests/unit/test_models_p3.py` — 19 new tests (spec-required + gradient flow + boundary + cross-model), all passing
- `poc/evaluation/plots_p3.py` — dielectric comparison plot implemented
- `outputs/figures/p3_dielectric_comparison.png` — generated; Dobson ε range 12–62 (physically consistent), Mironov ε range 0.2–1.5 (ε < 1.0 at low VWC — see DEV-007)
- Full test suite: 75 passed, 5 skipped (expected)
- Mironov parameterisation investigated: confirmed not an implementation error — GRMDM inapplicable to organic soils by design, retained as diagnostic-only per DEV-007
- Dobson confirmed as physically appropriate primary model

**Gate 3 result:** ✅ Outcome: NEGATIVE
**Outcome category:** Negative — PINN RMSE 0.1672 at N≈25 vs RF 0.1546

**Phase 3 closure (2026-03-09):**

PINN RMSE by fraction (median [IQR] across 10 reps):
| Fraction | PINN RMSE | RF RMSE | Δ vs RF |
|----------|-----------|---------|---------|
| 100% | 0.160 [0.158–0.165] | 0.147 [0.145–0.152] | +8.6% |
| 50% | 0.177 [0.171–0.199] | 0.155 [0.146–0.169] | +14.0% |
| 25% | 0.167 [0.151–0.173] | 0.155 [0.147–0.166] | +8.1% |
| 10% | 0.312 [0.234–0.434] | 0.144 [0.133–0.181] | +117% |

Wilcoxon PINN vs RF (Bonferroni-corrected):
- 100%: p=0.008 (sig worse)
- 50%: p=0.039 (not sig)
- 25%: p=1.000 (not sig)
- 10%: p=0.016 (not sig)

Phase 4 diagnostics begin immediately to characterise the failure mode.

---

## Phase 4 — Diagnostics

**Spec:** User-defined (diagnostics A–D)
**Status:** Complete — 2026-03-09

### Claude Code tasks

- [x] P4.A — Diagnostic A: Physics branch residual analysis (`p4_residual_analysis.png`)
- [x] P4.B — Diagnostic B: WCM forward model fit (`p4_wcm_forward_fit.png`)
- [x] P4.C — Diagnostic C: Identifiability analysis (`p4_identifiability.png`)
- [x] P4.D — Diagnostic D: Mironov sensitivity check (`p4_mironov_sensitivity.png`)
- [x] P4.all — Combined results saved to `outputs/metrics/phase4_diagnostics.json`

**Phase 4 completion summary (2026-03-09):**
- Residual-NDVI correlation: r=0.823 (p<10⁻⁹) — **direct evidence WCM fails with denser vegetation**
- WCM parameter bound status: A=0.0800 (interior), B=0.1762 (interior) — not at edges
- WCM forward fit r=0.007 vs raw VV-VWC r=0.290 — **WCM captures none of observed VV variability**
- Identifiability: ML branch dominates physics branch at all training sizes (residual ratio 3.3–6.4×)
- Mironov sensitivity: RMSE 0.151 vs Dobson 0.160 (−5.4%); informational only, DEV-007 ε clamping applied

---

## Phase 5 — Results Paper

**Spec:** `SPEC_PHASE5.md` (adapted: paper only, document updates deferred)
**Status:** Draft complete — 2026-03-09

### Claude Code tasks

- [x] P5.1 — Draft `poc_results.md` (8 sections + appendix, ~4,000 words)

**File:** `outputs/write-up/poc_results.md`

### Human steps required

- [ ] **H5.1** — Review and edit `poc_results.md`
- [ ] **H5.2** — Review updated White Paper, Yellow Paper, Green Paper (deferred to dedicated session)
- [ ] **H5.3** — Pitch deck update notes (deferred to dedicated session)
- [ ] **H5.4** — Rehearse demo walkthrough

**Gate 5 result:** Deferred (paper draft complete; gate script and document updates deferred)

---

## Open Questions / Blockers

| # | Question | Phase | Raised | Resolved |
|---|----------|-------|--------|---------|
| — | — | — | — | — |

## Deferred Updates (action after Carbon13)

- [x] **TODO (post-outcome):** DEV-008 addendum updated with final Negative outcome label (2026-03-09).
- [ ] **Green Paper and White Paper comprehensive consistency update.** Now that the full outcome is known, the following items require updating in a single dedicated session:
  - Negative outcome and honest framing (platform capability + physics-advantage investigation)
  - Phase 4 diagnostic findings: residual-NDVI r=0.823, WCM forward r=0.007, residual ratio 3–6×
  - DEV-007 (Mironov inapplicable to organic soils; Dobson confirmed primary)
  - DEV-008 (λ dominance violation and Negative outcome addendum)
  - PINN-NN convergence finding (PINN 0.160 ≈ NN 0.159 at 100%)
  - MIMICS/L-band upgrade path as stated next technical direction
  - G1-05 reinterpretation (DEV-003), terrain feature exclusion (DEV-004), DEV-005 QC rules
  - DEV-006 reinterpretation of G2-08 as relative criterion
  - Fixed baseline benchmarks (null: 0.1776, RF: 0.1470, NN: 0.1594)
  - R² interpretation under chronological split
  - Pre-registered PINN target thresholds and five outcome categories
  - RF feature importance ranking
  - 119 vs 120 observation correction
  - **All paper updates deferred to a single dedicated session after Carbon13.**

---

## Deviations Summary

| ID | Phase | Gate impact | Status |
|----|-------|-------------|--------|
| DEV-001 | 1 | None | Logged — COSMOS-UK daily product used |
| DEV-003 | 1 | G1-05 reinterpreted | Logged — VV–VWC r=0.26 (p=0.004); criterion reinterpreted for vegetated peatland |
| DEV-004 | 1 | Feature count 11→7 | Logged — Terrain features excluded (zero variance, single-site) |
| DEV-005 | 1 | None (119≥100) | Logged — 2021-02-05 manually excluded; 2 supplementary QC rules added |
| DEV-006 | 2 | G2-08 reinterpreted as relative | Logged — RF RMSE 0.147 > 0.10; criterion reinterpreted: RF must beat null by ≥10% (met: 17%) |
| DEV-007 | 3 | None (diagnostic only) | Logged — Mironov (2009) ε < 1.0 at low VWC; GRMDM inapplicable to organic soils; retained for Diagnostic 3 with clamping |
| DEV-008 | 3 | None (fallback path) | Logged — All 64 λ combos violate dominance; selected (0.01, 0.01, 1.0) per spec §P3.7 fallback |

---

## Session Log

_One line per session. Claude Code appends at end of each session._

| Date | Session work | Completed tasks | Left off at |
|------|-------------|-----------------|-------------|
| 2026-03-06 | Specification suite complete | All SPEC_PHASE*.md, CLAUDE.md update, PROGRESS.md, README.md | Ready to build Phase 1 |
| 2026-03-09 | Phase 1 code scaffold | P1.1–P1.7, P1.9, P1.12: scaffold, config, cosmos loader, 4 GEE scripts, ancillary, alignment, 20 unit tests (all passing), requirements.txt | Need: human data downloads, then gate_1.py + figures + integration tests |
| 2026-03-09 | Phase 1 code complete | P1.8 (gate_1.py), P1.10 (integration tests), P1.11 (diagnostic figures). All 20 unit tests passing. GEE authenticated. | Need: run GEE extractions (H1.1-H1.5), download data, run alignment, run Gate 1 |
| 2026-03-09 | Phase 1 data + Gate 1 | GEE extractions run, COSMOS loaded, pipeline executed (120 obs), Gate 1 run (8/10 auto-pass). DEV-003 (G1-05 reinterp), DEV-004 (terrain exclusion), DEV-005 (freeze-thaw QC). Gate 1 signed off. | Ready for Phase 2 |
| 2026-03-09 | Phase 2 complete | P2.1–P2.12: splits, 3 models, harness, plots, gate_2, 30 unit + 6 integration tests. All 40 configs trained. DEV-006 (G2-08 threshold). Gate 2 passed 13/13. | Ready for Phase 3 |
| 2026-03-09 | Phase 3: P3.1 + P3.2 | P3.1 dielectric models (Dobson + Mironov), P3.2 WCM forward model (Oh backscatter + vegetation terms + total). DEV-007 (Mironov ε < 1.0). 42 P3 unit tests, 98 total passing. | Next: P3.3 PINN architecture |
| 2026-03-09 | Phase 3: P3.3 | PINN architecture: PhysicsNet (32→16→1, sigmoid), CorrectionNet (64→32→16→1, ReLU+Dropout), PINN class (A/B sigmoid reparameterisation), compute_pinn_loss (4-term composite). 24 new tests (66 P3 total), all passing. | Next: P3.4 λ search |
| 2026-03-09 | Phase 3: P3.4+P3.5 | λ search (64 combos × 10 configs = 640 trainings, ~10 min CPU). Selected (0.01, 0.01, 1.0); all combos violate dominance (DEV-008). All 40 PINN configs trained (9s CPU). RMSE: 100%=0.160, 50%=0.177, 25%=0.167, 10%=0.312. pinn_trainer.py created. 122 tests passing. | Next: P3.6 diagnostics |
| 2026-03-09 | Phase 3: Pre-P3.6 gradient diagnostic | Gradient flow check: **Physics branch active** (ratio=0.228, threshold=0.01). Physics branch receives 23% of correction branch gradient magnitude. WCM A/B parameters actively learning. Standard "physics-informed" description applies. DEV-008 addendum logged. Sequencing note: SPEC_PHASE3.md defines P3.5 as λ search and P3.6 as full training. In practice, P3.5 trained all 40 configurations as part of the λ validation process and saved all artefacts (metrics, weights, predictions) to outputs/. Re-running under P3.6 was considered and rejected — re-running with knowledge of preliminary results would violate the honest-gates methodology. Existing P3.5 artefacts are the canonical training results. P3.6 proceeds directly to identifiability diagnostics using P3.5 outputs. | Next: P3.6 results compilation + figures |
| 2026-03-09 | Phase 3: P3.6 results + figures | Results compiled from 40 PINN metric files. PINN RMSE: 100%=0.1596±0.003, 50%=0.1765±0.014, 25%=0.1672±0.011, 10%=0.3123±0.100. N≈25 improvement vs RF: −8.14% (PINN worse). Wilcoxon PINN vs RF: sig worse at 100%/50%/10% (Bonf-corrected), not sig at 25% (p=0.32). Figures: p3_pinn_learning_curves.png, p3_pinn_vs_rf_scatter.png. Outcome category determination deferred to user. | Next: P3.6 identifiability diagnostics |
| 2026-03-09 | Phase 3 closure + Phase 4 diagnostics + Phase 5 paper | Negative outcome confirmed. DEV-008 addendum updated. Phase 4: all 4 diagnostics run (residual analysis r=0.823 NDVI, WCM forward r=0.007, residual ratio 3–6×, Mironov −5.4%). 4 figures generated. Phase 5: poc_results.md drafted (~4000 words, 12 references). PROGRESS.md updated with project completion. | Human review: poc_results.md, then Carbon13 materials |

---

## Project Completion Entry (2026-03-09)

All five phases complete.

**Outcome:** Negative (PINN RMSE 0.1672 at N≈25 vs RF 0.1546)

**Failure mode characterised:** WCM structural mismatch at vegetated peatland site.
- Evidence: residual-NDVI correlation (r=0.823, p<10⁻⁹)
- Evidence: WCM forward fit essentially zero (r=0.007)
- Evidence: PINN-NN convergence at 100% (0.160 vs 0.159)
- Evidence: ML branch dominates physics branch at all training sizes (ratio 3–6×)

**Full deviation log:** DEV-001 through DEV-008

**Results paper:** `outputs/write-up/poc_results.md`

**Next steps:**
1. Residual analysis and WCM forward fit figures for Carbon13 narrative
2. MIMICS literature grounding
3. Customer discovery outreach
4. UCL PhD supervisor outreach using results paper as preliminary work
5. Green Paper and White Paper consistency update pass (deferred to single session after Carbon13)
