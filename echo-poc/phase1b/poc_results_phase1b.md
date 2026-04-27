# Pre-Registered Architectural Halt of a Physics-Informed Neural Network for SAR-Based Soil Moisture Retrieval at C-band: Magnitude-Balance Saturation of a Joint Dual-Polarisation Composite-Loss Landscape

**Vantage Phase 1b Technical Results Document**

*Version 1.0 — 27 April 2026*

---

## Abstract

Phase 1b of the Vantage proof-of-concept tested whether replacing the Phase 1 Water Cloud Model (WCM) with a Toure-style single-crown adaptation of the Michigan Microwave Canopy Scattering (MIMICS) forward model — embedded as the physics branch of a two-branch PINN under a joint VV+VH composite loss — would recover a physics-informed data-efficiency advantage over a Random Forest baseline for Sentinel-1 C-band SAR soil moisture retrieval at COSMOS-UK Moor House. The experiment is reported under success criteria pre-registered against `SPEC.md` (signed via §14 on 2026-04-19) and the locked companion `phase1b/SUCCESS_CRITERIA.md` (tagged `phase1b-success-criteria-pre-registered`, 2026-04-19).

The pre-registered λ-search executed under the signed §8 joint VV+VH formulation in Session F-2b, after DEV-1b-010 corrected an implementation-vs-text divergence detected at the F-3 entry check. The result was **Tier 3 HALT**: 0 of 64 (λ_physics, λ_monotonic, λ_bounds) ∈ {0.01, 0.1, 0.5, 1.0}³ combinations satisfied the SPEC §9 primary dominance criterion under either the strict per-rep AND or the binding mean-across-reps reading of DEV-1b-009. Per the pre-authored fallback at SPEC.md §9 line 324, the experiment halts on a Phase 1b architectural failure; Phase 1's "lowest median validation loss regardless of dominance" fallback is explicitly not retained; and "the honest interpretation in this case is that the MIMICS module cannot produce a physics-dominated solution at this data volume, which is itself a publishable finding."

The sealed test set was never unsealed: the pre-registered evaluation path requires Tier 1 or Tier 2 with a selected λ. The HALT outcome together with four methodological contributions accumulated during Phase 1b execution (DEV-1b-008 implementation-correctness vs cross-configuration equivalence; DEV-1b-009 dominance-constraint aggregation-rule explicitness with its F-2b ambiguity-collapse complement; the supervisor-executor entry-check workflow with four sub-observations; DEV-1b-010 post-sign-off implementation-audit gate) and the empirical magnitude-balance saturation finding form a coherent set of five publishable contributions surfaced because the pre-registration discipline was honoured. The HALT finding is configuration-specific, not architecture-killing: it does not validate the Vantage commercial thesis at the retrieval-science level, and it also does not refute it.

---

## 1. Introduction

Phase 1 of the Vantage PoC (`outputs/write-up/poc_results.md` v1.0, 2026-03-09) reported a Negative outcome for a WCM-based PINN against a Random Forest baseline at Moor House blanket bog, with Phase 4 diagnostics identifying WCM structural mismatch — a two-parameter vegetation scattering representation applied to a layered heather/sphagnum canopy — as the binding constraint (residual–NDVI r=0.82; WCM forward fit r=0.007; ML correction branch dominating physics branch by a 3–6× residual ratio at every training size). Phase 1's diagnostic decision tree (`outputs/write-up/poc_results.md` §5.4) identified two upgrade paths: (i) MIMICS as a multi-layer canopy representation, and (ii) L-band SAR.

Phase 1b investigates path (i) at fixed sensor (Sentinel-1 C-band), fixed site (Moor House), fixed dataset (the same N=119 paired observations after Phase 1's QC pipeline, of which 83 are in the training pool and 36 are in the sealed test set; same chronological 70/30 split), with two amendments to the Phase 1 PINN: a Toure-style single-crown MIMICS forward model in place of the WCM, and a joint dual-polarisation composite-loss term `L_physics = MSE(σ°_VV) + MSE(σ°_VH)` in place of Phase 1's VV-only `L_physics = MSE(σ°_VV)` (signed `SPEC.md` §8 lines 286–300, with §3's Δ-row naming the joint VV+VH form as Phase 1b's "central change").

The rationale for the joint formulation is structural. MIMICS predicts both polarisations from a single shared parameter set; jointly fitting both polarisations is the test of whether the canopy structural parameters MIMICS introduces (branch number density N_b, leaf number density N_l, branch orientation σ_orient, branch gravimetric moisture m_g, surface roughness s) are simultaneously recoverable from C-band dual-pol observations at this site. Phase 1's VV-only formulation cannot make that test: a model that predicts only co-pol does not exercise the cross-pol structural information.

This document reports Phase 1b under the same honest-gates discipline as Phase 1: pre-registered success criteria are evaluated regardless of outcome direction; deviations from specification are logged with full impact assessment; the result is reported against the signed criteria, not against post-hoc reformulations. Phase 1b's signed criteria specified an explicit pre-authored decision tree for each fallback tier of the dominance constraint, including a Tier 3 HALT branch with a verbatim honest-interpretation passage (`SPEC.md` §9 line 324). That branch fired. The five contributions reported in §6 of this document are findings about pre-registration discipline and PINN composite-loss calibration that surfaced because the pre-registration was honoured at each branch point — they are publishable independently of any sealed-test-set retrieval outcome.

A note on what this document is and is not. This is not a retrieval-performance paper. The sealed test set (`echo-poc/data/splits/test_indices.json`, SHA-256 `a4b11206630cc80fc3e2ae5853bb114c7a4154072375654c257e51e4250f8eea`) was never unsealed in Phase 1b: the pre-registered evaluation path at SPEC §9 / §10 requires a Tier 1 or Tier 2 outcome with a selected λ, and the Tier 3 HALT outcome precludes that path. The PINN-MIMICS retrieval RMSE at N≈25 — the central Phase 1b research question — is therefore unanswered and is explicitly not reported in this document. What is reported is the λ-search outcome, the magnitude-balance characterisation that explains it, and the methodological contributions accumulated during the pre-registered execution path that produced it.

---

## 2. Study Site and Data

### 2.1 Inheritance from Phase 1

Phase 1b inherits the Moor House dataset, the COSMOS-UK quality-controlled VWC ground truth, the Sentinel-1 IW GRD descending-orbit (relative orbit 154) VV+VH SAR extraction, the Sentinel-2 monthly NDVI composite, the ERA5-Land precipitation, and the chronological 70/30 train/test split unchanged from Phase 1 (`SPEC.md` §2; `phase1b/SUCCESS_CRITERIA.md` §5). The G1 baseline reproducibility check (Session A, 2026-04-18; `phase1b/implementation_gate/results/g1_baseline_result.json`) confirmed the Phase 1 baseline numbers within 0.005 cm³/cm³ tolerance across 12 baseline rows.

| Quantity | Value | Source |
|---|---|---|
| Final paired observations (post-QC) | 119 | Phase 1 `poc_results.md` Table 1 |
| Training pool (chronological 70%) | 83 | `data/splits/test_indices.json` |
| Sealed test set (chronological 30%) | 36 | `data/splits/test_indices.json` (sealed) |
| Sealed test set SHA-256 | `a4b11206630cc80fc3e2ae5853bb114c7a4154072375654c257e51e4250f8eea` | not unsealed in Phase 1b |
| Test set date span | 2023-07-25 to 2024-12-10 | Phase 1 chronological split |
| Repetitions per training size | 10 | `SPEC.md` §2; `SUCCESS_CRITERIA.md` §5 |
| Training fractions | 100%, 50%, 25%, 10% | Phase 1 inheritance; 5% dropped per Phase 1 DEV-004 |

The Phase 1b execution path did not reach the 4×10 factorial evaluation. The sealed test set is unchanged from Phase 1 and is unread by any Phase 1b code path. The training pool was used for the λ-search only, on a stratified 20%-held-out validation split carved within the training pool per `SPEC.md` §9.

### 2.2 Phase 1b dataset deviations relative to Phase 1

No Phase 1b deviations modified the dataset. Phase 1b inherits Phase 1 deviations DEV-001 through DEV-007 by reference and adds Phase 1b deviations DEV-1b-001 through DEV-1b-005 and DEV-1b-007 through DEV-1b-010 (DEV-1b-006 retired pre-sign-off; full deviation log at Appendix A and `phase1b/deviation_log.md`). Of those, only DEV-1b-001 (NDVI → LAI → N_l prior withdrawal) and DEV-1b-002 (NDWI → m_g prior withdrawal) materially affect the input feature set — both withdrew Sentinel-2-derived priors that were named in the v0.1-draft specification but failed citation audit at pre-sign-off. NDWI continues to be extracted (Gao 1996, B08/B11) for diagnostic-only post-experiment use per the DEV-1b-002 repurposing (`SPEC.md` §11 New Diagnostic D). The post-withdrawal feature set matches Phase 1's 7-feature configuration (VV dB, VH dB, VH/VV ratio dB, NDVI, daily precipitation, 7-day precipitation, incidence angle).

---

## 3. Methods

### 3.1 PINN-MIMICS architecture

The Phase 1b PINN preserves the Phase 1 two-branch backbone (`SPEC.md` §2) and replaces the WCM forward block with a Toure-style single-crown MIMICS adaptation. Layer structure:

- **Physics branch (`PhysicsNet`):** identical to Phase 1 (Input(7) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid × θ_sat). Output: physically-bounded VWC estimate `m_v_physics` ∈ [0, 0.88].
- **Correction branch (`CorrectionNet`):** identical to Phase 1 (Input(7) → 64 → 32 → 16 → 1 with ReLU and dropout 0.2). Output: unbounded residual δ.
- **Forward chain:** `m_v_physics` → Mironov GRMDM (with the DEV-007 ε ≥ 1.01 clamp inherited from Phase 1, formalised in `SPEC.md` §6) → ε_peat. Then ε_peat with surface roughness s and incidence angle θ_i → Oh 1992 surface scattering → σ°_soil_VV, σ°_soil_VH (`SPEC.md` §7). Then crown layer parameters (N_b, N_l, σ_orient, m_g, fixed crown geometry) plus canopy loss, crown-ground coupling, and ground-direct mechanisms → MIMICS forward → σ°_VV, σ°_VH.
- **Final prediction:** `m_v_final = m_v_physics + δ`.

The five learnable structural parameters (N_b, N_l, σ_orient, m_g, s) are sigmoid-bounded at literature ranges per `SPEC.md` §5. Crown geometry is fixed from Calluna field literature (`SPEC.md` §5).

### 3.2 Composite loss (the joint VV+VH amendment)

```
L = L_data + λ_physics · L_physics + λ_monotonic · L_monotonic + λ_bounds · L_bounds
```

with

- `L_data = MSE(m_v_final, m_v_observed)` — same as Phase 1.
- `L_physics = MSE(σ°_VV_pred, VV_obs) + MSE(σ°_VH_pred, VH_obs)` — **the central change**, signed via `SPEC.md` §8 lines 292–300 and §3 Δ-row. A single shared λ_physics applies to the sum; per-polarisation separate λs are explicitly out of scope for v0.1.
- `L_monotonic`, `L_bounds`: same as Phase 1.
- `L_prior`: not used. Both Sentinel-2-derived priors originally specified in v0.1-draft were withdrawn pre-sign-off (DEV-1b-001 NDVI → LAI → N_l; DEV-1b-002 NDWI → m_g). λ_prior is no longer a model hyperparameter; the search grid is 4³ = 64 over (λ_physics, λ_monotonic, λ_bounds) only (`SPEC.md` §9; `SUCCESS_CRITERIA.md` §5 note on λ-grid arity).

### 3.3 λ-search and dominance constraint

The Phase 1b λ-search procedure (`SPEC.md` §9; `SUCCESS_CRITERIA.md` §3) evaluates all 64 grid combinations on 10 reps at the 100% training fraction, with the validation loss measured on a stratified 20%-held-out split within the training pool. The selection metric is the lowest median validation loss across the 10 reps among combinations satisfying the dominance constraint.

The dominance constraint, locked at sign-off (`SPEC.md` §9 lines 314–318; `SUCCESS_CRITERIA.md` §3 lines 144–151):

> **Primary:** L_data is the largest single term in the composite loss across the final 10 epochs of training, averaged over all 10 reps at the 100% training fraction.
>
> **Secondary:** L_physics contributes > 10% of the total loss across the final 10 epochs (averaged as above).

The pre-registered fallback procedure (`SPEC.md` §9 lines 320–324; `SUCCESS_CRITERIA.md` §3 lines 159–174) defines three tiers, with the Tier 3 branch being a pre-authored architectural-failure HALT:

> "If no combination satisfies even the primary criterion: this is a Phase 1b architectural failure that halts the experiment. The Phase 1 fallback procedure (lowest median validation loss regardless of dominance) is **not** retained for Phase 1b. The honest interpretation in this case is that the MIMICS module cannot produce a physics-dominated solution at this data volume, which is itself a publishable finding."
> — `SPEC.md` §9, line 324, signed via §14 on 2026-04-19.

The reason the Phase 1 fallback is not retained is documented in `SPEC.md` §9 line 326: Phase 1's fallback selected λ=(0.01, 0.01, 1.0) under "lowest median val loss regardless of dominance", with L_physics weighted at 0.01, which is partly attributable to Phase 1's near-zero physics-branch output diagnostic. Allowing the same fallback in Phase 1b would prevent the experiment from distinguishing "MIMICS works but the optimiser routes around it" from "MIMICS works and the optimiser uses it."

### 3.4 Pre-registration discipline at execution time

The Phase 1b execution path comprised, in temporal order: SPEC §14 sign-off (2026-04-19, commit `2331cd4`); F-1 pre-registration lock (`SUCCESS_CRITERIA.md` tagged `phase1b-success-criteria-pre-registered`, 2026-04-19, commit `cea7ec9`); F-2 v1 λ-search execution (2026-04-19/20, tag `phase1b-session-f2-lambda-selected`, commit `5a2a994`); F-3 entry check (2026-04-20, halted at the implementation-vs-text audit step); F-2b prep — DEV-1b-010 deviation entry committed at `4ebea02`, joint VV+VH implementation amendment at `dde17c6`; F-2b λ-search re-run (2026-04-20 to 2026-04-22, tag `phase1b-session-f2b-lambda-selected`, commit `1f4bfdf`).

Two adjudications during this path materially shape the F-2b result and are reported here verbatim because they bind the F-2b classification:

- **DEV-1b-009 (2026-04-20):** Aggregation-rule adjudication. The F-2 v1 implementation evaluated the SPEC §9 primary dominance criterion under a strict all-reps-AND reading inherited from Phase 1's `phase1/lambda_search.py`. The verbatim pre-registered text — "averaged over all 10 reps" — instead specifies a cross-rep mean reading. Both readings were dual-recorded in the F-2 v1 result JSON at execution time (the executor's foreseen mitigation) but the tier-classification step used only the strict reading. DEV-1b-009 adjudicated the verbatim text as binding: the cross-rep mean reading is the binding interpretation, and the F-2 v1 result JSON's `primary_dominance_mean_across_reps` field is the authoritative classification source. Both readings continue to be computed and recorded in F-2b for audit-trail transparency. Full adjudication: `phase1b/DEV-1b-009.md`.

- **DEV-1b-010 (2026-04-20):** Implementation-vs-text divergence on the joint VV+VH formulation. F-3 entry check cross-referenced `SPEC.md` §8 against `phase1b/pinn_mimics.py:315` and surfaced that the implementation computed `L_physics = MSE(σ°_VV, VV_obs)` — co-pol only — versus the signed `L_physics = MSE(σ°_VV, VV_obs) + MSE(σ°_VH, VH_obs)`. The signed text was unchanged at every tracked commit since 2026-04-17/18; the implementation drift originated at Session C/D authorship and survived three subsequent audit gates (§14 sign-off, F-1 lock, F-2 kickoff) because none required a verbatim-grep of the implementation against the SPEC. Resolution: amend implementation to match SPEC §8; re-run λ-search as F-2b under the joint formulation; re-classify per DEV-1b-009 mean-across-reps reading. F-2 tag preserved as superseded-but-intact audit record. Full adjudication: `phase1b/DEV-1b-010.md`.

These two adjudications are also Phase 1b methodological contributions in their own right (§6 below).

### 3.5 Outcome action-tier mapping (pre-registered)

`SUCCESS_CRITERIA.md` §4 pre-registers a four-row outcome action-tier matrix that translates (primary RMSE category) × (secondary criteria pattern) × (dominance status) into a downstream-decision recommendation. The matrix is reported here as locked context because the Tier 3 HALT outcome at the dominance gate is itself one of the pre-authored branches: SPEC §9 specifies the dominance constraint screen as an early diagnostic gate that, when failed, halts the experiment before the §4 action-tier matrix's primary RMSE input is available. Tier 3 HALT does not enter the action-tier matrix because the experiment halts before Phase 4 evaluation is reached.

---

## 4. Results

### 4.1 F-2b grid coverage

The F-2b λ-search executed all 64 grid combinations with 10 reps each (640 training runs total) on the 100% training fraction, with the joint VV+VH `L_physics` per the signed SPEC §8 formulation (post-DEV-1b-010 implementation correction). Wall clock: 1956.4 minutes (≈32.6 hours; compute proper ≈ 6.8 h with sleep-tax overhead ≈ 25.8 h — see `phase1b/SESSION_F2B_CHECKPOINT.md` §6 for the operational characterisation). Zero non-finite-abort events across 640 reps. Result JSON: `phase1b/lambda_search/results_f2b/lambda_search_f2_result.json`, SHA-256 `c5cbdac6414a9bf6989c0f6927462a927ebeda71120b6ab424320d6d77b6d68a`.

The integrity audit (`phase1b/SESSION_F2B_CHECKPOINT.md` §1) records six checks: 64-combo grid coverage (PASS); DEV-1b-009 dual-reading invariant present per combo (PASS); tier-classification block carrying `tier=HALT, halted=true, n_full_dominance=0, n_primary_only=0, n_any_primary=0, n_neither=64` (PASS); joint-VV+VH formulation hash (NEUTRAL — current schema does not carry an explicit loss-formulation hash; flagged for the audit-trail-strengthening proposals carried into Phase 1c per §15.3 of the decisions log); no NaN/Inf in any aggregated combo field (PASS); schema parity vs F-2 v1 (PASS).

### 4.2 Dominance distribution

Per-block aggregates (16 combos per λ_physics block; 4³=64 grid):

| λ_physics | n_combos | median val loss | mean physics_fraction | primary pass (strict) | primary pass (mean) |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 16 | 0.2171 | 0.9255 | 0/16 | 0/16 |
| 0.10 | 16 | 1.9502 | 0.9898 | 0/16 | 0/16 |
| 0.50 | 16 | 9.6585 | 0.9979 | 0/16 | 0/16 |
| 1.00 | 16 | 19.2984 | 0.9990 | 0/16 | 0/16 |

The mean physics fraction is monotone non-decreasing in λ_physics across the full grid: 0.9255 → 0.9898 → 0.9979 → 0.9990. Higher λ_physics produces higher physics fraction, never lower — the structural prediction made at the F-2b kickoff pre-flight is empirically confirmed across all 48 combos that the kickoff analysis only argued for from monotonicity.

**Most favourable single combination:** combo 2 — λ=(0.01, 0.01, 0.5) — `mean_physics_fraction = 0.9252`, `primary_dominance_mean_across_reps = False`, `primary_dominance_all_reps = False`. The single combo most favourable to L_data dominance still has weighted L_physics at 92.5% of the composite loss; the mean L_data final-window value is 0.0165 against a mean weighted L_physics of 0.2044 (`phase1b/lambda_search/results_f2b/lambda_search_f2_result.json` combinations[2]).

**Least favourable single combination:** combo 60 — λ=(1.0, 1.0, 0.01) — `mean_physics_fraction = 0.9990`, `primary_dominance_mean_across_reps = False`. L_data is approximately 0.1% of the composite loss.

**Primary dominance pass rate at the most favourable block (λ_physics=0.01):** 0/16 under both the strict per-rep AND reading and the binding mean-across-reps reading. Even at the smallest pre-registered λ_physics, no combination satisfies primary dominance.

### 4.3 Tier classification — DEV-1b-009 readings collapse

Both the strict per-rep AND reading and the binding mean-across-reps reading (DEV-1b-009) classify identically as Tier 3 HALT. 0 of 64 combinations satisfy the primary dominance criterion under either reading. This is itself a substantive observation about the F-2b regime: the DEV-1b-009 aggregation-rule ambiguity — which produced contested classifications in F-2 v1 because the VV-only landscape sat on the dominance frontier with physics_fraction ≈ 0.435 — disappears in F-2b's strongly-dominated regime where physics_fraction ≥ 0.925 across the entire grid. Cross-rep stochastic variation is irrelevant to the classification when the per-rep margin between L_physics and L_data is structural rather than incidental. The F-2b result is therefore robust to the aggregation-rule choice; this robustness is itself the empirical complement to DEV-1b-009 reported in §6 below.

### 4.4 Verbatim pre-registered honest interpretation

Per the signed SPEC §9 fallback procedure at line 324:

> "this is a Phase 1b architectural failure that halts the experiment. The Phase 1 fallback procedure (lowest median validation loss regardless of dominance) is **not** retained for Phase 1b. The honest interpretation in this case is that the MIMICS module cannot produce a physics-dominated solution at this data volume, which is itself a publishable finding."

The Tier 3 HALT branch fires. The pre-authored decision tree at the F-2b kickoff (`phase1b/SESSION_F2B_CHECKPOINT.md` §7) confirms: no F-3-the-main-experiment, no sealed-test-set unsealing, Phase 1b concludes on the HALT finding plus the methodological contributions accumulated during pre-registered execution.

### 4.5 Magnitude-balance characterisation

The F-2b magnitude-balance characterisation explains the Tier 3 HALT in terms specific enough to inform Phase 1c scoping without overgeneralising.

The pre-flight single-rep observation at λ=(0.01, 0.01, 0.10) found VH/VV ≈ 0.645 in MSE magnitude (VH MSE 7.70 dB² vs VV MSE 11.93 dB²; `phase1b/SESSION_F2B_CHECKPOINT.md` §4). Across the F-2b grid this ratio is broadly stable across configurations. Joint `L_physics = MSE(σ°_VV) + MSE(σ°_VH)` is therefore approximately 1.65× the VV-only `L_physics = MSE(σ°_VV)` would be at the same configuration. At λ_physics = 0.01 the weighted L_physics ratio against L_data on the per-observation VWC-residual scale is approximately 12–16×; at λ_physics = 1.00 the ratio is approximately 1000×.

The binding framing line for this finding (per the Phase 1b decisions log §14.7, which is the precise language locked at F-2b close):

> At C-band Moor House blanket bog, with the pre-registered MIMICS forward model and joint VV+VH `L_physics` formulation, with N=83 training points, and with the pre-registered λ grid lower-bounded at 0.01, joint VV+VH `L_physics` is intrinsically ~1.65× larger in MSE magnitude than VV-only, and the resulting composite-loss landscape is physics-saturated across the entire pre-registered λ grid {0.01, 0.1, 0.5, 1.0}³. L_data cannot be made the largest single term under any pre-registered combination.

What this finding **is** and **is not** — both halves are binding.

**The finding is** a specific, narrow empirical statement about PINN composite-loss calibration under small-N sparse-training regimes with dual-pol physics constraints: at the pre-registered configuration above, the composite loss is intrinsically physics-dominated in magnitude, and L_data cannot be made the largest single term under any pre-registered λ combination.

**The finding is not** "MIMICS is inadequate as a physics module for Moor House peatland retrieval." The MIMICS forward model itself is operating per its specification: G2 Moderate Pass per DEV-1b-008 with the numpy_port arm at machine precision (36/36 rows; max Δ = 6.17 × 10⁻⁶ dB at the 0.5 dB tolerance), the gradient arm with autograd↔FD internal consistency within 0.003 dB, and the published_table arm with characterised residuals fully traced to five pre-registered v0.1 sub-module simplifications. The HALT outcome is about composite-loss calibration, not about MIMICS forward fidelity. The distinction is binding because conflating the two would mis-license a Phase 1c scoping that abandoned MIMICS rather than addressing magnitude balance.

The finding does not preclude a later investigation — outside Phase 1b's pre-registration — into whether a loss-rescaling scheme (per-channel MSE normalisation, λ_physics extension below the pre-registered minimum, or a separable-magnitude reformulation) could rebalance the composite loss. That investigation is a Phase 1c or methods-paper question, not a Phase 1b re-scoping. The pre-registered λ grid is what binds Phase 1b's evidence claim.

---

## 5. Discussion

### 5.1 Interpretation of the HALT outcome

The Tier 3 HALT outcome is the pre-authored branch of the §9 fallback procedure that activates when the dominance landscape cannot accommodate L_data primacy at any pre-registered λ. The Phase 1b architecture, the joint VV+VH `L_physics` formulation, the pre-registered λ grid, and the N=83 small-data regime jointly produce a composite-loss landscape in which the magnitude of the physics term saturates the loss function at every grid point. Given that landscape, the dominance constraint cannot be met, and SPEC §9's pre-authored response is to halt.

The honest interpretation per SPEC.md:324 is binding: at this data volume, with this physics module under this composite-loss formulation, MIMICS cannot produce a physics-dominated solution. The pre-registration's choice to halt rather than fall back to "lowest median val loss regardless of dominance" preserves the experiment's ability to distinguish "MIMICS works but the optimiser routes around it" from "MIMICS works and the optimiser uses it" — Phase 1's diagnostic ambiguity that motivated the §9 tightening (`SPEC.md` §9 line 326) is precisely the ambiguity Phase 1b's pre-registration was designed to avoid. The HALT outcome means Phase 1b honoured that design choice.

### 5.2 Why F-2b's classification is robust

Three facts together make the F-2b Tier 3 HALT classification robust beyond the threshold of the dominance constraint itself:

1. **Both DEV-1b-009 readings collapse identically.** The strict per-rep AND reading and the binding mean-across-reps reading classify F-2b as Tier 3 HALT with `n_any_primary = 0` under both. The aggregation-rule ambiguity that had to be adjudicated in F-2 v1 is irrelevant to F-2b's classification.

2. **Monotonicity confirms across all 48 unobserved-at-pre-flight combos.** Mean physics_fraction is monotone non-decreasing in λ_physics across the entire grid (0.9255 → 0.9898 → 0.9979 → 0.9990). The structural prediction at the F-2b kickoff pre-flight — that physics-dominance must increase with λ_physics — holds empirically across all 64 combos.

3. **The most favourable combination still has physics_fraction ≈ 0.925.** Even at the cheapest band (λ_physics = 0.01), even at the most-favourable bounds and monotonic weights, the weighted L_physics is approximately 92.5% of the composite loss. There is no proximal grid point at which the dominance balance flips; the landscape is uniformly physics-dominated.

The F-2b HALT is not a marginal classification at the boundary of statistical noise; it is a uniform classification across a structurally physics-saturated landscape.

### 5.3 The HALT framing — configuration-specific, not architecture-killing

The HALT finding is configuration-specific:

- **Configuration:** C-band Sentinel-1 (single sensor); Moor House blanket bog (single site); N=83 training points (small-N regime); MIMICS Toure-style single-crown v0.1 (specific physics module); joint VV+VH `L_physics = MSE(σ°_VV) + MSE(σ°_VH)` (specific dual-pol formulation); pre-registered λ grid lower-bounded at 0.01 (specific search space).
- **Not killed:** physics-informed neural networks as a class of architectures; MIMICS as a forward model for blanket bog at other configurations (different λ grids, different polarisation weighting, different N); the Vantage commercial thesis at the platform level. None of these is refuted by F-2b's outcome; all are unaddressed.

The Phase 1c scoping path follows from this distinction: the magnitude-balance finding gives Phase 1c a specific characterised problem (composite-loss calibration on dual-pol landscapes) to address rather than a fishing expedition over the architecture space.

### 5.4 What the sealed test set was protected from

The sealed test set was never unsealed in Phase 1b. The test-set SHA-256 (`a4b11206630cc80fc3e2ae5853bb114c7a4154072375654c257e51e4250f8eea`) is unchanged since Phase 1. The pre-registered evaluation path requires a Tier 1 or Tier 2 outcome with a selected λ; the Tier 3 HALT outcome precludes that path by SPEC §9 design.

The discipline value of leaving the sealed set sealed is twofold. First, future Phase 1c work that addresses magnitude balance — via per-channel normalisation, a wider λ grid, or a reformulated `L_physics` — can be evaluated against the same sealed test set without contamination. Had Phase 1b unsealed the test set under a fallback selection (the Phase 1 pattern), Phase 1c's eventual evaluation against that test set would carry residual exposure from the Phase 1b read. Second, the sealed-test-set protocol's credibility depends on the pre-registration's commitment to halt when the dominance gate fails being honoured rather than relaxed. Honouring it costs Phase 1b a retrieval-performance answer; preserving the sealed-set integrity is the longer-run scientific asset.

---

## 6. Methodological Contributions (Five Publishable Findings)

Phase 1b's pre-registered execution path produced five findings of methodological or empirical interest, all surfaced because the pre-registration discipline was honoured at each branch point. They are presented in publication order; each is publishable independently of any retrieval-performance outcome.

### 6.1 Implementation-correctness vs cross-configuration equivalence (DEV-1b-008)

**Source:** Session E G2 closure, 2026-04-19.

**Claim:** Physics-informed ML forward-equivalence gates must explicitly distinguish two distinct tests:

1. **Implementation-correctness:** does the differentiable physics module (e.g. PyTorch) faithfully reproduce a deterministic reference implementation (e.g. numpy) of the same sub-module choices? Testable at machine precision.
2. **Cross-configuration equivalence:** does the differentiable implementation produce σ° values matching a published paper's reference configuration, given that paper's sub-module choices? Testable only if the implementation uses the paper's sub-modules.

**Observation:** Most published PINN work conflates the two. A ±0.5 dB tolerance against a published reference implicitly demands the latter; passing it requires sub-module matching that may not be appropriate for the production configuration. Phase 1b's G2 gate originally specified ±0.5 dB against published Toure 1994 and McDonald 1990 anchor tables, but the Phase 1b production configuration deliberately differs from those references at five distinct sub-module layers (simplified power-law Dobson vs full Dobson 1985; Oh 1992 vs physical-optics surface; Rayleigh + sinc² vs Ulaby-Moore-Fung finite cylinder; real-only UEL vs dual-dispersion UEL; √(σ°_oh) vs literal Fresnel Γ for crown-ground coupling). The five-way mismatch is documented in DEV-1b-008 as the pre-registered v0.1 → v0.2 promotion queue.

**Prescription:** Forward-equivalence gates should specify which test is intended, and tolerance thresholds should be calibrated to the test. Implementation-correctness tolerances should be at or near machine precision (Phase 1b's numpy_port arm achieved 6.17 × 10⁻⁶ dB max Δ across 36 anchor rows). Cross-configuration tolerances depend on sub-module differences and should characterise rather than fail when v0.1 deliberately differs from the reference configuration. Phase 1b's G2 Moderate Pass classification preserves the ±0.5 dB criterion as a pre-registered tolerance and additionally records per-row characterised residuals where it is exceeded — the characterised-residual framework is additive, not retrospective.

This contribution is independent of Phase 1b's λ-search outcome and is publishable as a methodological note for the PINN literature on equivalence testing.

### 6.2 Dominance-constraint aggregation rule explicitness (DEV-1b-009 + F-2b empirical complement)

**Source:** Session F-2 v1 execution-close adjudication, 2026-04-20; F-2b empirical complement, 2026-04-22.

**Claim:** Pre-registered PINN dominance constraints of the form "L_data must be the largest single term in the composite loss" must explicitly specify the aggregation rule across training repetitions. Two aggregation rules are plausible:

- **Per-rep AND (strict):** the constraint must hold on every individual rep.
- **Cross-rep mean (relaxed):** the constraint must hold on the cross-rep aggregate of the per-rep-window-mean loss components.

**Observation:** The two readings can produce different classifications when training results show stochastic rep-level variation around the constraint boundary. Phase 1b F-2 v1 demonstrated this on the VV-only landscape: 7-9/10 reps satisfied the primary dominance criterion under the strict reading at the favoured combinations, while 16/64 combinations satisfied it under the mean reading (with cross-rep mean physics_fraction ≈ 0.435 placing L_physics and L_data within ~5% of each other). The two readings produced contested tier classifications until DEV-1b-009 adjudicated the verbatim pre-registered text — "averaged over all 10 reps" — as binding the cross-rep mean reading.

**Prescription:** Pre-registration text should use explicit quantifier language ("in every rep" vs "in the cross-rep mean") rather than ambiguous participial phrases ("averaged over N reps"). When implementation and pre-registration text diverge, verbatim text binds. Result artefacts should record both readings for audit-trail transparency — Phase 1b F-2 v1 was salvageable as an audit record only because the executor's foreseen mitigation dual-recorded both readings in the result JSON.

**Empirical complement (F-2b):** The aggregation-rule ambiguity collapses when the composite-loss landscape is structurally lopsided rather than on the constraint frontier. F-2b at physics_fraction ≥ 0.925 across the entire grid demonstrates the regime where either reading suffices: both classify F-2b as Tier 3 HALT identically, with `n_any_primary = 0` under both. F-2 v1's ambiguity mattered because the VV-only landscape sat on the frontier; F-2b's joint-VV+VH at ≥ 0.925 does not.

The empirical complement strengthens the prescription: aggregation-rule explicitness is most critical for landscapes near the constraint frontier; in saturated regimes it is redundant. The pair (DEV-1b-009 prescription + F-2b empirical complement) is jointly publishable as a methodological note paired with the magnitude-balance finding (§6.5 below).

### 6.3 Workflow architecture: supervisor-executor entry-check discipline

**Source:** Session F-2 v1 entry check + execution close; reinforced through F-2b and Session G entry checks.

**Claim:** Pre-registration discipline can be strengthened by an explicit split between supervising-guidance authoring and executing, with entry checks at commit boundaries on both sides:

- **Executor entry check on incoming guidance:** the executor halts and flags if forward-guidance from the supervisor conflicts with locked pre-registration text.
- **Executor execution-close check on implementation-vs-text consistency:** the executor halts and flags if its own implementation diverges from verbatim pre-registration text.

**Observation (bidirectional drift catching):** Phase 1b F-2 v1 caught both supervisor drift (early Q1/Q2/Q3 conflicts at entry) and implementation drift (DEV-1b-009 aggregation rule at close). Most pre-registration literature treats pre-registration as a single-author, single-execution discipline. The supervisor-executor split with bidirectional entry-check discipline is a methodological improvement worth explicit description.

**Four sub-observations** accumulated during Phase 1b execution:

- **State-snapshot freshness.** F-2b's stalled-process episode (2026-04-21 06:00 BST) saw both supervisor and executor authorise actions on stale log-file state. The verification gap was bidirectional; neither party asked "when was this state checked, and is it still current?" before authorising. Mitigation: entry-check protocols should require explicit fresh re-read of state-bearing artefacts at every commit boundary, not just at session boundaries.

- **Cost-asymmetry case study.** The F-3 entry-check halt that surfaced DEV-1b-010 cost ~2 minutes of supervisor adjudication and saved ~32.6 hours of F-2b wall clock plus a full F-3-the-main-experiment cancellation (the F-3 budget would have run a 4×10 factorial under the wrong loss formulation). The halt-and-flag value proposition is empirically measurable in compute hours: 1000:1 in this single case.

- **Artefact-persistence drift.** Three sessions inherited a decisions-log claim about a corpus state (Yellow Paper v3.0.5, Green Paper v4.1.3) that did not exist on disk because the Session E cascade outputs were never preserved past their authoring session. The verification gap was symmetric between the author-supervisor and the inheriting-supervisor; neither verified the corpus state at the start of the inheriting session. Mitigation: persistence is explicit, not implicit; documents written to ephemeral storage do not survive without a deliberate persistence step at authoring time.

- **Cross-environment prompt drift.** Session G's original kick-off prompt assumed supervisor-side filesystem mounts that did not exist in the executor-side environment. The drift was caught at the executor's entry check. Pattern: when supervisor and executor inhabit different filesystems, the supervisor must verify environment-specific path assumptions before sending prompts. Mitigation: future supervisor prompts to the executor reference repo-relative paths only; supervisor-side mounts are supervisor-side concerns to be relayed as text rather than as path references.

The four sub-observations collectively support the workflow-architecture claim: bidirectional entry-check discipline is most effective when paired with explicit state-freshness verification, explicit cost-asymmetry awareness, explicit persistence handling, and explicit cross-environment translation.

### 6.4 Post-SPEC-sign-off implementation-audit gate (DEV-1b-010)

**Source:** Session F-3 entry check, 2026-04-20.

**Claim:** Pre-registered physics-informed ML projects need an implementation-audit gate triggered whenever an implementation file referenced by a signed SPEC section first lands in the repo, or is materially modified after sign-off.

**Observation:** Three sequential audit gates (§14 sign-off at `2331cd4`; F-1 pre-registration lock at `cea7ec9`; F-2 v1 kickoff pre-`5a2a994`) had access to the implementation `phase1b/pinn_mimics.py` — either on disk or in tracked history — but none invoked a cross-reference against SPEC §8's joint VV+VH specification of `L_physics`. F-3 entry check was the audit that caught the divergence — not by chance, but because the F-3 prompt explicitly required executor cross-reference of locked text against implementation source before execution. The pattern was systematic; its activation was late, with the cost of activation-lateness being F-2 v1's full 64-combo λ-search wall clock under the wrong formulation.

**Prescription:** Before any signed-SPEC section's implementation is exercised in execution code, the executor must grep-verify the implementation source against the SPEC section character-for-character. The cost is ~2 minutes at sign-off; the F-2 v1 case study quantifies the avoidable cost of late activation as 32.6 h of F-2b wall clock plus a full F-3 cancellation. Phase 1c onward institutionalises this as a required gate, not an ad-hoc executor-discretion check. The gate triggers at: (i) initial commit of any implementation file referenced by a signed SPEC §; (ii) any subsequent modification of such a file; (iii) any session that uses a tracked implementation file in execution code.

This contribution is independent of Phase 1b's λ-search outcome and is publishable as a methodological note for pre-registered PINN projects. It is sibling to DEV-1b-008 and DEV-1b-009 in its shared principle: verbatim signed text binds.

### 6.5 Magnitude-balance saturation of joint-VV+VH composite-loss landscapes

**Source:** Session F-2b execution close, 2026-04-22; magnitude characterisation from §4.5 above.

**Claim:** At C-band Moor House blanket bog, with the Phase 1b pre-registered MIMICS forward model and the joint VV+VH `L_physics` formulation, with N=83 training points, and with the pre-registered λ grid lower-bounded at 0.01, joint VV+VH `L_physics` is intrinsically ~1.65× larger in MSE magnitude than VV-only, and the resulting composite-loss landscape is physics-saturated across the entire pre-registered λ grid {0.01, 0.1, 0.5, 1.0}³. L_data cannot be made the largest single term under any pre-registered combination.

**Observation:** Across all 64 pre-registered grid combinations, mean physics_fraction is ≥ 0.925 (most-favourable combo: λ=(0.01, 0.01, 0.5), physics_fraction = 0.9252). At the least-favourable combo (λ=(1.0, 1.0, 0.01)) physics_fraction = 0.9990. Monotonicity confirmed: mean physics_fraction is non-decreasing in λ_physics across the grid (0.9255 → 0.9898 → 0.9979 → 0.9990).

This is a finding about PINN composite-loss calibration under small-N sparse-training regimes with dual-pol physics constraints. The PINN methods literature rarely addresses dual-pol magnitude balance in the composite loss; the pair (DEV-1b-009 aggregation-rule ambiguity + F-2b magnitude-balance saturation) form a coherent set of methodological + empirical contributions on pre-registration discipline for physics-informed losses.

**Framing constraints (binding):** the finding is **not** that MIMICS is inadequate as a physics module, and the finding does **not** preclude a later investigation into loss-rescaling schemes. Both halves of the framing are normative for downstream interpretation. The MIMICS forward model is operating per its specification (G2 Moderate Pass per DEV-1b-008); the HALT is about composite-loss calibration, not forward fidelity. The pre-registered λ grid is what binds Phase 1b's evidence claim; a broader grid (e.g. extending to 10⁻³, 10⁻⁴) or a reformulated `L_physics` (per-channel normalisation, dB-vs-linear scale changes, learned dynamic loss weights) is a Phase 1c scope question, not a Phase 1b re-scoping.

This is the headline empirical contribution of Phase 1b. It is publishable as a methodological note, paired with §6.2's aggregation-rule contribution, in the PINN methods literature.

---

## 7. Phase 1c open questions (separately scoped)

Four open questions were identified during Phase 1b execution as relevant to the magnitude-balance finding but explicitly outside Phase 1b's pre-registered deliverables. They are recorded here as Phase 1c scope candidates, not Phase 1b extensions:

1. **Per-channel L_physics normalisation.** Whether dividing each polarisation's MSE by its empirical magnitude (so that VV and VH contribute equal-scale terms regardless of the underlying dB² magnitude) would rebalance the composite loss without requiring λ extensions. The F-2b empirical observation that VH/VV ≈ 0.645 in MSE magnitude and joint L_physics ≈ 1.65× VV-only motivates the question; the Phase 1b decisions log records that F-2 v1's superseded VV-only landscape (under the implementation that diverged from signed SPEC §8 per DEV-1b-010) had physics_fraction ≈ 0.435 — close enough to L_data to make per-channel normalisation a plausible Phase 1c rebalancing direction, though the Phase 1b evidence base does not test this directly.

2. **λ grid lower bound for joint dual-pol formulations.** Whether the pre-registered λ grid's 0.01 lower bound is appropriate for joint dual-pol formulations, or whether a wider grid (extending to 10⁻³, 10⁻⁴) is needed to recover an L_data-dominant regime. The pre-registered grid was inherited from Phase 1's WCM (single-pol) and was not recalibrated for the joint dual-pol formulation at sign-off; recalibrating the grid is itself a pre-registration design decision that belongs to Phase 1c, not a within-Phase-1b adjustment.

3. **Trunk-layer scattering mechanism.** Whether the trunk-layer scattering mechanism (DEV-1b-005 Set D Phase 1c exemption) should be implemented and how that affects the magnitude-balance landscape. The Toure-style single-crown adaptation for v0.1 deliberately omitted the trunk layer per the Calluna canopy structure (heather has no woody trunk in the C-band cylinder-resonance sense); whether re-introducing a trunk-layer mechanism would shift the dominance balance is unaddressed by Phase 1b.

4. **L-band SAR generalisation.** Whether L-band SAR (NISAR / ROSE-L — the strategic Vantage roadmap per the Phase 1c plan) has materially different magnitude-balance properties to C-band Sentinel-1, and whether the Phase 1b finding generalises or is specific to C-band Moor House. L-band's lower vegetation attenuation and different scattering regime could produce a fundamentally different physics-data magnitude balance; the Phase 1b finding is silent on this.

All four items are Phase-1c-scope. Phase 1b concludes on the HALT finding plus the five publishable contributions; Phase 1c is separately scoped and is not authorised by this document.

---

## 8. What Phase 1b validates, and what it does not

This section is binding for downstream framing in the Vantage corpus. The honest reading: the HALT finding is configuration-specific, not architecture-killing. Phase 1b does not validate the Vantage commercial thesis at the retrieval-science level; it also does not refute it. Phase 1c is the next attempt at validation; the magnitude-balance finding gives Phase 1c a specific characterised problem to address rather than a fishing expedition.

### 8.1 Phase 1b does not establish

- PINN-MIMICS retrieval performance on Sentinel-1 C-band at Moor House blanket bog. The sealed test set was never unsealed; no RMSE comparison against the RF baseline was made; the central Phase 1b research question — "does PINN-MIMICS beat RF baseline at N≈25?" — is unanswered.
- Whether physics-informed neural networks are a viable retrieval architecture for peatland water-table or soil-moisture monitoring from satellite SAR, in any general sense.
- Whether the Vantage commercial thesis (physics-informed satellite monitoring producing uncertainty-quantified ecosystem state estimates more accurate and cheaper than ground-based monitoring) holds at the technical level needed to underwrite carbon credits.

### 8.2 Phase 1b does establish

- A specific empirical finding about composite-loss calibration: the magnitude-balance saturation of joint-VV+VH composite-loss landscapes under Phase 1b's pre-registered configuration (per §4.5 above and §6.5).
- A working differentiable Toure-style single-crown MIMICS implementation (numpy↔PyTorch lockstep at machine precision per the G2 numpy_port arm; autograd path alive per the G2 gradient arm; `phase1b/physics/mimics.py` and `phase1b/pinn_mimics.py` at commit `1f4bfdf`). This is a reusable engineering asset for any future PINN work, regardless of architecture.
- Four methodological contributions to physics-informed ML pre-registration discipline (§6.1–§6.4), independently publishable.

The HALT outcome was pre-authored at sign-off; halting on it preserves the experiment's epistemic integrity. The five publishable contributions are surfaced because the pre-registration discipline was honoured at each branch point. Phase 1c inherits a specific characterised problem (composite-loss magnitude balance) rather than an undiagnosed null result.

---

## 9. Venue scoping (options, not decision)

Three publication routes are plausible for the Phase 1b output. Final venue choice is supervisor adjudication and is documented separately in the Vantage corpus (Green Paper §8 publication strategy section); this document proposes options only.

**Option A — Methods-focused outlet for the four pre-registration-discipline contributions.** The four methodological contributions in §6.1–§6.4 are independently coherent and target the PINN methods literature on pre-registration discipline, equivalence testing, aggregation-rule explicitness, supervisor-executor workflow architecture, and post-sign-off audit gates. A methods-focused journal — *PLOS Computational Biology* methods section, *Methods in Ecology and Evolution*, *Geoscientific Model Development* (GMD), or *International Journal of Applied Earth Observation and Geoinformation* — would frame these as standalone contributions. This option is the cleanest framing of the methodological output and is independent of any retrieval-performance question.

**Option B — Remote Sensing of Environment (or similar domain venue) paired contribution.** A *Remote Sensing of Environment* / *IEEE TGRS* / *IEEE JSTARS* paper that pairs §6.5's magnitude-balance saturation finding with §6.2's aggregation-rule contribution and a domain-framed introduction (peatland soil moisture retrieval, dual-pol PINN composite-loss calibration). The domain venue is the natural home of the empirical finding (the magnitude-balance characterisation is specific to C-band SAR over vegetated peatland) but requires a careful framing line — per §4.5, the finding is **not** that MIMICS is inadequate, and the venue's reviewer base will need that distinction made forcefully in the introduction and discussion.

**Option C — Two-paper split.** A methods paper covering §6.1–§6.4 in a methods venue, plus a domain paper covering §6.5 plus the workflow-architecture observations from §6.3 in a remote-sensing venue. This is the maximum-coverage option but requires two parallel write-ups; whether the marginal contribution justifies the duplication is a supervisor-side judgement against current venture-development timeline pressures.

The supervisor's final venue choice depends on (i) the urgency of the Carbon13 venture-builder programme submission timeline; (ii) the corpus-side framing of the Phase 1b output in the Vantage Pitch Deck and Green Paper; (iii) the strategic value of the four methods contributions to the broader PINN community vs the strategic value of the retrieval-domain finding to the Vantage thesis specifically. None of these is resolved by this document.

---

## 10. Limitations

1. **No retrieval-performance evidence.** The sealed test set was never unsealed; this document reports no PINN-MIMICS RMSE, no Wilcoxon test, no learning curve, no comparison against the Phase 1 baselines. The central Phase 1b research question is unanswered; future Phase 1c work that resolves the magnitude-balance issue is needed before retrieval performance can be reported.

2. **Single-site, single-sensor, single-dataset scope.** The magnitude-balance finding is empirically demonstrated only at Moor House, only at Sentinel-1 C-band, only at N=83. Whether it generalises to other peatland types, other sensors (especially L-band), or larger datasets is not established by Phase 1b.

3. **Pre-registered λ grid.** The grid {0.01, 0.1, 0.5, 1.0}³ was inherited from Phase 1's WCM and was not recalibrated for the joint VV+VH formulation at sign-off. Whether a wider grid would have admitted a Tier 1 or Tier 2 outcome is a Phase 1c open question, not a Phase 1b finding.

4. **MIMICS v0.1 sub-module simplifications.** Phase 1b's v0.1 physics stack uses five deliberate sub-module simplifications relative to Toure 1994 / McDonald 1990 reference configurations (DEV-1b-008). Whether the magnitude-balance finding would survive a Phase 1c v0.2 promotion of any subset of those sub-modules is not addressed by Phase 1b.

5. **Compute-cost characterisation tied to operational details.** The 32.6 h F-2b wall clock includes ≈25.8 h of sleep-tax overhead from a `caffeinate -dimsu` defeat under clamshell-on-battery operation (`SESSION_F2B_CHECKPOINT.md` §6). The science is unaffected (zero non-finite-aborts; per-block dominance results uniform within block) but the wall-clock characterisation is operationally specific. Phase 1c compute budgeting includes lid-open or AC-power as a discipline anchor (`pmset -a sleep 0 disablesleep 1` as belt-and-braces around `caffeinate`).

6. **Corpus-side cascade.** The Vantage paper corpus (Yellow Paper v3.0.4, Green Paper v4.1.2, Pitch Deck v7.1, White Paper v11.3) references Phase 1b at various depths. Cascade revisions reflecting the HALT outcome and the five publishable contributions are scoped separately in `phase1b/cascade_plan_session_g.md` and are not part of this document.

7. **Uncertainty quantification not addressed.** The Vantage commercial thesis includes an uncertainty-quantified-state-estimate component (the credibility-interval requirement for carbon-credit underwriting). Phase 1b's PINN architecture is a point-prediction architecture; uncertainty quantification methods (Bayesian PINN extensions, deep ensembles, evidential learning) are not addressed by Phase 1b and are a separate methodological track from the magnitude-balance finding. Whether the Vantage thesis ultimately requires UQ-equipped PINNs, post-hoc UQ wrappers, or alternative architectural choices is unaddressed by Phase 1b.

---

## 11. Conclusions

Phase 1b of the Vantage PoC tested whether replacing Phase 1's WCM with a Toure-style single-crown MIMICS forward model under a joint dual-polarisation composite loss would recover a physics-informed advantage for SAR-based soil moisture retrieval at Moor House. The pre-registered λ-search at Session F-2b returned **Tier 3 HALT**: 0 of 64 pre-registered λ combinations satisfied the SPEC §9 primary dominance criterion under either DEV-1b-009 reading. Per the verbatim pre-registered honest interpretation at SPEC.md:324, "the MIMICS module cannot produce a physics-dominated solution at this data volume, which is itself a publishable finding." The sealed test set was never unsealed.

Four conclusions follow:

**The HALT outcome is configuration-specific, not architecture-killing.** The magnitude-balance saturation of the joint VV+VH composite-loss landscape across Phase 1b's pre-registered configuration explains the HALT in terms specific enough to inform Phase 1c scoping without overgeneralising. MIMICS as a physics module is operating per its specification (G2 Moderate Pass per DEV-1b-008); the HALT is about composite-loss calibration, not forward fidelity. Phase 1c open questions (per-channel normalisation, wider λ grids, trunk-layer mechanisms, L-band generalisation) are separately scoped and explicitly not Phase 1b extensions.

**The pre-registration discipline produced five publishable contributions.** DEV-1b-008's implementation-correctness vs cross-configuration equivalence framework; DEV-1b-009's dominance-constraint aggregation-rule explicitness with its F-2b ambiguity-collapse complement; the supervisor-executor entry-check workflow architecture with four sub-observations; DEV-1b-010's post-sign-off implementation-audit gate; and the empirical magnitude-balance saturation finding itself. All five are publishable independently of any sealed-test-set retrieval outcome — they were surfaced because the pre-registration discipline was honoured.

**The sealed test set is preserved as a Phase 1c scientific asset.** Phase 1b's commitment to halt rather than fall back to "lowest median val loss regardless of dominance" cost a Phase 1b retrieval-performance answer; in exchange, the sealed test set remains an uncontaminated evaluation surface for any future Phase 1c work that resolves magnitude balance. The Phase 1 honest-gates discipline is preserved through the Phase 1b conclusion.

**The Vantage thesis is unaddressed at the retrieval-science level by Phase 1b.** The HALT finding does not validate the Vantage commercial thesis; it also does not refute it. Phase 1c — separately scoped, with the magnitude-balance finding as a specific characterised problem to address rather than a fishing expedition — is the next attempt at retrieval-science validation. Carbon13 venture-builder framing should reflect this distinction precisely, neither overclaiming the Phase 1b contribution as a thesis validation nor underclaiming it as a setback.

---

## References

1. Attema, E. P. W., & Ulaby, F. T. (1978). Vegetation modeled as a water cloud. *Radio Science*, 13(2), 357–364.
2. Bechtold, M., et al. (2018). Peat moisture estimation from SAR data: a multi-temporal evaluation. *Remote Sensing*, 10(2), 199.
3. Dobson, M. C., Ulaby, F. T., Hallikainen, M. T., & El-Rayes, M. A. (1985). Microwave dielectric behavior of wet soil—Part II: Dielectric mixing models. *IEEE TGRS*, 23(1), 35–46.
4. Gao, B.-C. (1996). NDWI — A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment*, 58(3), 257–266.
5. McDonald, K. C., Dobson, M. C., & Ulaby, F. T. (1990). Using MIMICS to model L-band multiangle and multitemporal backscatter from a walnut orchard. *IEEE TGRS*, 28(4), 477–491.
6. Mironov, V. L., Dobson, M. C., Kaupp, V. H., Komarov, S. A., & Kleshchenko, V. N. (2009). Generalized refractive mixing dielectric model for moist soils. *IEEE TGRS*, 47(7), 2059–2070.
7. Oh, Y., Sarabandi, K., & Ulaby, F. T. (1992). An empirical model and an inversion technique for radar scattering from bare soil surfaces. *IEEE TGRS*, 30(2), 370–381.
8. Toure, A., Thomson, K. P. B., Edwards, G., Brown, R. J., & Brisco, B. G. (1994). Adaptation of the MIMICS backscattering model to the agricultural context — wheat and canola at L and C bands. *IEEE TGRS*, 32(1), 47–61.
9. Ulaby, F. T., Moore, R. K., & Fung, A. K. (1986). *Microwave Remote Sensing: Active and Passive*, Vol. III. Artech House.
10. Ulaby, F. T., & El-Rayes, M. A. (1987). Microwave dielectric spectrum of vegetation — Part II: Dual-dispersion model. *IEEE TGRS*, GE-25(5), 550–557.
11. Vantage PoC, Phase 1 Results Document. (2026). `outputs/write-up/poc_results.md` v1.0, 2026-03-09.

---

## Appendix A — Phase 1b Deviation Log (summary)

Full entries in `phase1b/deviation_log.md` and the individual `phase1b/DEV-1b-NNN.md` files. DEV-1b-006 retired pre-sign-off (placeholder superseded by the DEV-1b-008 five-way promotion queue).

| ID | Summary | Gate | Impact | Resolution |
|----|---------|------|--------|------------|
| DEV-1b-001 | Withdrawal of NDVI → LAI → N_l prior (citation audit failure on Lees et al. 2021) | G2 (pre-sign-off) | Reduces priors from two to one; N_l constrained by literature bounds + L_physics | Resolved before §14 sign-off |
| DEV-1b-002 | Withdrawal of NDWI → m_g prior (single-site scalar averaging defect; no heather-specific transfer function) | G2 (pre-sign-off) | L_prior removed entirely; λ_prior no longer a hyperparameter; NDWI retained as diagnostic-only | Resolved before §14 sign-off |
| DEV-1b-003 | G2 Implementation Gate anchor construction (Toure 1994 + McDonald 1990 staged) | G2 (pre-sign-off) | G2 redefined as three-arm check (numpy_port, published_table, gradient) | Resolved before §14 sign-off |
| DEV-1b-004 | Gradient arm dielectric amendment for E.1 / E.2 rows | G2 | Gradient arm autograd↔FD agreement within 0.003 dB established | Resolved at Phase E-1b |
| DEV-1b-005 | Set D Phase 1c exemption (trunk-layer mechanism out of scope for v0.1) | G2 | Set D `EXEMPT` from G2 published_table arm | Resolved at Phase E-1 |
| DEV-1b-007 | Published_table arm dielectric amendment for Sets A / B / C | G2 | Set C.1 residual reduced from 53.6 dB to 6.9 dB; A.2 from 10.2 dB to 1.96 dB; etc. | Resolved at Phase E-1b |
| DEV-1b-008 | G2 Moderate Pass classification (cross-configuration equivalence reframe) | G2 closure | G2 classified as Moderate Pass; v0.1 → v0.2 promotion queue documented | Resolved at Phase E-2 closure |
| DEV-1b-009 | Dominance-constraint aggregation-rule adjudication (verbatim text binds) | F-2 closure | Mean-across-reps reading binds; both readings dual-recorded for audit transparency | Resolved at F-2 v1 close |
| DEV-1b-010 | Implementation-vs-text divergence on joint VV+VH L_physics formulation | F-3 entry check | F-2 v1 superseded by F-2b; tag preserved as audit record | Resolved at F-3 entry; F-2b re-run |

---

*Vantage · Phase 1b · Pre-Registered Architectural Halt + Five Publishable Contributions · v1.0 · 2026-04-27*

*Recommended close tag: `phase1b-concluded-halt-finding` (outcome marker; semantically informative for future tag-history lookups).*
