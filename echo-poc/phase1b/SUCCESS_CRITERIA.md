# Phase 1b — Pre-Registered Success Criteria

**Vantage · Phase 1b · Pre-Registration Document · v1.0**
**Date locked:** 2026-04-19
**Tag at lock:** `phase1b-success-criteria-pre-registered`
**Parent commit:** `63ef5fa` (Session F handoff close)
**Companion to:** [`SPEC.md`](../SPEC.md) v0.1 (signed 2026-04-19, §14) and `vantage-mimics-litreview-v1_0_2.html` §10.

---

## 0. Purpose and authority

This document is the **formal, locked pre-registration of the Phase 1b
success criteria, experimental protocol, and diagnostic thresholds**. It
exists because the MIMICS literature review §10 proposed criteria but
explicitly noted them as "not yet pre-registered"; training without
pre-registration would break the honest-gates discipline applied
throughout Phase 1 (`outputs/write-up/poc_results.md` v1.0) and
Phase 1b Block 1 (DEV-1b-001 through DEV-1b-008).

**Authority and scope:**

- Where this document and SPEC.md conflict on a numerical threshold,
  **SPEC.md is authoritative.** SPEC.md §10 was signed at Phase E
  closure (2026-04-19) and the thresholds there are the ones the final
  Phase 1b results will be reported against.
- This document **adds** explicit numerical thresholds for the lit
  review §10 secondary criteria that SPEC §10 leaves at category level
  ("forward fit improvement", "structural adequacy", etc.), and adds
  the **action-tier mapping** that translates SPEC §10's RMSE category
  + secondary-criteria pattern into a downstream-decision recommendation.
- Once tagged (`phase1b-success-criteria-pre-registered`), any change
  to this document or to SPEC §10 thresholds requires a new DEV-1b-NNN
  entry in `phase1b/deviation_log.md` with full impact assessment, the
  same way DEV-1b-001 / DEV-1b-002 handled the pre-sign-off prior
  withdrawals. Editing the criteria after a Phase 1b training artefact
  has been produced is **enforced as a test failure** by the regression
  test in `tests/unit/test_pre_registration_lock.py` (see §5 below).

**Non-negotiable boundary:** The sealed test set
(`echo-poc/data/splits/test_indices.json`) remains sealed during F-1.
No criterion in this document is derived from observations on the
sealed test set; all thresholds come from prior literature, the Phase 1
Negative-outcome baselines, or pre-analysis on the training pool only.

---

## 1. Primary criterion (single-number retrieval outcome)

**Primary criterion (verbatim, locked):**

> PINN-MIMICS median RMSE on the sealed test set (n=36) at the N≈25
> training fraction (25%) is lower than the Random Forest baseline
> RMSE at the same fraction, with statistical significance under a
> paired Wilcoxon signed-rank test (Bonferroni-corrected
> α = 0.0125 for 4 comparisons across the four training fractions).

| Quantity | Value | Source |
|---|---|---|
| RF baseline RMSE at N≈25 (Phase 1) | **0.155 cm³/cm³** | `poc_results.md` Table 2 |
| RF baseline at 100% training (N=83) | 0.147 cm³/cm³ | `poc_results.md` Table 2 |
| Null model RMSE | 0.178 cm³/cm³ | `poc_results.md` Table 2 |
| Test set size | 36 paired observations | `data/splits/test_indices.json` (sealed) |
| Test set date span | 2023-07-25 to 2024-12-10 | Phase 1 chronological 70/30 split |
| Repetitions per training size | 10 | SPEC §2 |
| Statistical test | Paired Wilcoxon signed-rank | SPEC §2, §10 |
| Significance level (uncorrected) | α = 0.05 | SPEC §10 |
| Significance level (Bonferroni for 4 comparisons) | α = 0.0125 | SPEC §10 |

**Primary outcome categories — RMSE thresholds (authoritative copy of SPEC §10):**

| Category | RMSE threshold (cm³/cm³) | % improvement vs RF | Statistical condition |
|---|---|---|---|
| **Strong** | < 0.124 | ≥ 20% better than RF | Bonferroni-corrected p < 0.0125 |
| **Significant** | < 0.131 | ≥ 15% better than RF | Bonferroni-corrected p < 0.0125 |
| **Moderate** | < 0.139 | ≥ 10% better than RF | Uncorrected p < 0.05 (may not survive Bonferroni) |
| **Inconclusive** | 0.139 ≤ RMSE < 0.155 | < 10% better than RF | Not statistically separated from RF |
| **Negative** | ≥ 0.155 | RF matched or exceeded | Hypothesis not supported |

These thresholds are the same numerical thresholds signed in
SPEC §10 / §14 on 2026-04-19. They are reproduced here verbatim so
that this document is self-contained for the purposes of the
pre-registration lock. Re-derivation against the same RF baseline is
conditional on the SPEC §2 G1 baseline reproducibility check
confirming the Phase 1 numbers within 0.005 cm³/cm³ (G1 PASS recorded
2026-04-18, see `phase1b/implementation_gate/results/g1_baseline_result.json`).

---

## 2. Secondary criteria (structural adequacy)

The four secondary criteria below are the lit review §10 proposals
translated into **explicit numerical thresholds**. Each criterion is
evaluated regardless of the primary outcome.

### Secondary 1 — MIMICS forward fit (co-pol, VV)

| Quantity | Threshold | Phase 1 (WCM) reference |
|---|---|---|
| Pearson r between MIMICS-predicted σ°_VV and observed VV across the **training set** (n=83) | **r > 0.3** | WCM forward fit r = 0.007 |
| Computed by | Running v0.1 MIMICS in pure forward mode with the optimised parameter set, predicting σ°_VV from observed m_v over the full training pool | `poc_results.md` §4.4 / `outputs/figures/p4_wcm_forward_fit.png` |
| Failure interpretation | Implementation issue or parameter-range misspecification — halts the experiment until re-implemented (per SPEC §11 diagnostic decision tree) |

The N=83 training set is used (not the sealed test set) so the
computation does not unseal anything. Reporting on the full N=119 is
permitted as a supplementary analysis only; the threshold is evaluated
on the training pool.

### Secondary 2 — Residual ratio (structural adequacy)

| Quantity | Threshold | Phase 1 (WCM) reference |
|---|---|---|
| `std(δ_ML) / std(m_v_physics)` at N=83 (full training pool) | **< 2.0** | Phase 1 values: 3.3× to 6.4× across training fractions |
| Computed by | Final-epoch correction-branch δ vs PhysicsNet output across all 10 reps at the 100% training fraction | `poc_results.md` §4.4 / `outputs/figures/p4_identifiability.png` |
| Failure interpretation | MIMICS structurally improved but PINN optimiser still routing around it — architectural finding (per SPEC §11 decision tree) |

### Secondary 3 — Cross-pol identifiability (VH forward fit)

| Quantity | Threshold | Justification |
|---|---|---|
| Pearson r between MIMICS-predicted σ°_VH and observed VH across the training set (n=83) | **r > 0.2** | Lower than VV threshold because VH is a smaller-dynamic-range channel and is the more demanding test of crown structural parameters (N_b, σ_orient) |
| Failure interpretation | Crown structural parameters misspecified; dual-pol formulation (SPEC §8) not achieving its intended purpose — joint VV+VH constraint should provide orthogonal handle on canopy structure |

### Secondary 4 — Residual–NDVI correlation (failure-mode signature)

| Quantity | Threshold | Phase 1 (WCM) reference |
|---|---|---|
| `\|r(MIMICS_residual, NDVI)\|` across the training set (n=83) | **< 0.5** | Phase 1 WCM: r = 0.823 (p < 10⁻⁹) — direct evidence of WCM structural failure with denser vegetation |
| Failure interpretation | MIMICS still systematically failing with vegetation density — multi-layer crown extension (per lit review §5 Multi-MIMICS) needed |

The 0.5 threshold is set substantially looser than zero because some
residual–NDVI correlation is expected even from a well-specified
canopy model when the canopy itself co-varies with seasonality. The
threshold is set well above zero and well below the Phase 1 WCM value
(0.82) so that "directionally improved but not eliminated" is
distinguishable from "binding constraint unchanged from Phase 1."

---

## 3. λ search dominance constraint (rapid physics-adequacy screen)

**Constraint (verbatim, locked, mirrors SPEC §9):**

> The selected λ configuration must satisfy **both**:
>
> - **Primary:** L_data is the largest single term in the composite
>   loss across the final 10 epochs of training, averaged over all 10
>   reps at the 100% training fraction.
> - **Secondary:** L_physics contributes **> 10%** of the total loss
>   across the final 10 epochs (averaged as above).

| Quantity | Threshold | Phase 1 reference |
|---|---|---|
| Number of λ combinations evaluated | 64 = `(λ_data, λ_physics, λ_monotonic, λ_bounds) ∈ {0.01, 0.1, 0.5, 1.0}⁴` | SPEC §9 |
| Reps per combination | 10 (at 100% training fraction) | SPEC §9 |
| Validation set | Stratified 20% held-out split within the training pool | SPEC §9. **Never the sealed test set.** |
| Selection metric | Lowest median validation loss across 10 reps | SPEC §9 |

### Pre-registered fallback (locked)

- **If ≥ 1 combination satisfies both criteria:** select the
  dominance-compliant combination with the lowest median validation
  loss.
- **If no combination satisfies both criteria but ≥ 1 satisfies
  the primary criterion only:** select the primary-only-compliant
  combination with the lowest median validation loss; **log as a
  Phase 1b deviation (DEV-1b-NNN)** and report the secondary-criterion
  violation as a Phase 1b finding.
- **If no combination satisfies even the primary criterion:** the
  experiment **halts**. The Phase 1 fallback (lowest median loss
  regardless of dominance) is **not** retained — see SPEC §9 for the
  rationale. The honest interpretation is that MIMICS cannot produce
  a physics-dominated solution at this data volume, which is itself a
  publishable Phase 1b finding.

The dominance constraint is the earliest diagnostic signal in the
Phase 1b experiment. A MIMICS that fails the dominance constraint is
behaving like Phase 1's WCM from an optimisation perspective — see
DEV-008 (Phase 1) and SPEC §9. Failure mode interpretation is bound
in by §3 of this document and is reportable regardless of the primary
RMSE outcome.

---

## 4. Outcome action-tier mapping

This is a **complementary** view to §1's RMSE thresholds. The §1
thresholds report the primary outcome category. The action tiers below
combine the primary RMSE category with the §2 secondary criteria and
the §3 dominance status to give the downstream recommendation. They
are pre-registered so that the action that follows from any outcome
pattern is locked before the result is observed.

| Action tier | Primary | Secondary | Dominance | Recommended next phase |
|---|---|---|---|---|
| **Strong** | RMSE category Strong, Significant, **or** Moderate (i.e. PINN-MIMICS RMSE < 0.139) with Bonferroni-corrected p < 0.0125 | All four secondary criteria met | Both primary and secondary §3 satisfied | Phase 1c L-band NISAR comparison; physics-informed advantage established for blanket bog at C-band; MIMICS validates as the canopy structural representation |
| **Moderate** | RMSE category Strong, Significant, or Moderate (RMSE < 0.139), corrected or uncorrected significance | Some but not all secondary criteria met (typically forward-fit S1 / S3 missed) | Primary satisfied, secondary §3 may be violated under fallback | Phase 1c with **honest scoping** of what MIMICS does and does not validate; document residual identifiability/forward-fit work as pre-Phase-1c Session-G scope candidate |
| **Inconclusive** | Primary not met (Inconclusive or Negative RMSE category, i.e. RMSE ≥ 0.139) **but** Secondary 1 forward fit improves vs WCM (r > 0.007) | At least Secondary 1 strictly improves over Phase 1 WCM | Either dominance status | MIMICS partially structured but insufficient at C-band; either simplify MIMICS (e.g. trunk layer merge) or **pivot to L-band NISAR**; document and consider Session G scope authorisation |
| **Refutation** | Primary not met **and** Secondary 1 forward fit ≤ 0.007 (i.e. no improvement vs WCM) **and** Secondary 2 residual ratio still ≥ 2.0 (unchanged from Phase 1's 3.3–6.4×) | Two structural-adequacy criteria fail | Either dominance status | C-band + MIMICS insufficient for dense blanket bog. **L-band NISAR becomes the primary physics-branch target**; Phase 1c priority increases |

These tiers are NOT a relaxation of the SPEC §10 RMSE thresholds.
They are an additional commitment about what action follows from any
combination of (primary RMSE category) × (secondary pattern) × (dominance
status). The SPEC §10 categories are reported verbatim in the final
Phase 1b results document; the action tier is reported alongside.

**Deliberate edge cases:**

- An outcome with RMSE in the Inconclusive category **and** all four
  secondary criteria met (Forward fit r > 0.3, residual ratio < 2.0,
  VH r > 0.2, residual–NDVI < 0.5) is an "Inconclusive primary, Moderate
  action" pattern — the recommended action is still Phase 1c with
  honest scoping, because the structural improvements survive even
  when the retrieval RMSE doesn't separate from RF.
- An outcome with RMSE in the Strong category **but** dominance
  fallback invoked is a "Strong primary, Moderate action" pattern —
  the result holds but the Phase 1b finding (fallback DEV entry) is
  reported as a constraint on the interpretation. This is the same
  pattern that pre-registered, accountable use of fallbacks always
  produces, and is preferable to silent strict-criterion failure.

---

## 5. Experimental protocol (locked)

All numbers below are locked. No post-hoc flexibility.

| Element | Value | Source |
|---|---|---|
| Training sizes | 100%, 50%, 25%, 10% of training pool | SPEC §2; same as Phase 1, 5% dropped per Phase 1 DEV-004 (remains dropped) |
| Repetitions per training size | 10, season-stratified where N permits | SPEC §2 |
| Total configurations per model | 40 (4 × 10 factorial) | SPEC §2 |
| Primary evaluation point | N ≈ 25 (25% training fraction) | SPEC §2 / §10; matches Phase 1 |
| Test set | n=36, chronological 2023-07-25 to 2024-12-10, sealed | SPEC §2; identical to Phase 1; **never modified** |
| Test set file | `echo-poc/data/splits/test_indices.json` | Read-only |
| Statistical test | Paired Wilcoxon signed-rank, per training size | SPEC §10 |
| Multiple-comparison correction | Bonferroni for 4 comparisons (α = 0.0125) | SPEC §10 |
| λ search grid | `(λ_data, λ_physics, λ_monotonic, λ_bounds) ∈ {0.01, 0.1, 0.5, 1.0}⁴` = 64 combinations | SPEC §9 |
| λ search reps per combination | 10 (at 100% training fraction) | SPEC §9 |
| λ search dominance constraint | Per §3 of this document and SPEC §9 | SPEC §9 |
| Baselines | Null (seasonal climatological, RMSE 0.178); RF (54-combo CV, 7 features); Standard NN (64→32→16, identical to PINN correction branch) | SPEC §2; re-run at Session A 2026-04-18, all within 0.005 cm³/cm³ tolerance |
| Random seeds | `SEED = 42 + config_idx` across baselines and PINN-MIMICS | SPEC §2 |
| Physics stack version | v0.1, unchanged per DEV-1b-008 (Phase E closure 2026-04-19) | DEV-1b-008 |
| Validation set within training | Stratified 20% held-out split | SPEC §9. **Never the sealed test set.** |

**Note on the 5% training fraction:** Phase 1 DEV-004 dropped the
5% fraction because at N=4 the seasonal stratification floor
(one observation per season) cannot be enforced. Phase 1b inherits
that decision; the 4×10 factorial is over the four fractions above.

**Note on the λ-grid arity:** SPEC §9 specifies
`(λ₁, λ₂, λ₃) ∈ {0.01, 0.1, 0.5, 1.0}³`, citing 64 combinations
(after L_prior was withdrawn per DEV-1b-002, λ_prior is no longer a
hyperparameter). 4³ = 64. The "4-tuple" reading in this document's
table (§5 above) names the grid by all four loss-term coefficients
(`λ_data, λ_physics, λ_monotonic, λ_bounds`); λ_data is conventionally
fixed at 1.0 (per SPEC §8 which writes
`L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds`). The
authoritative grid arity is 64 = 4³ over the three search axes
(λ_physics, λ_monotonic, λ_bounds); the 4-tuple notation is for
clarity, not an additional search axis.

---

## 6. Phase 4 diagnostic thresholds (pre-registered triggers)

The Phase 4 diagnostic suite, identical in structure to Phase 1 §4.4,
is bound to the thresholds below. Each diagnostic is computed
post-experiment regardless of primary outcome; the **threshold** is
the value at which the diagnostic flags a specific condition.

### D-1 — Residual–NDVI correlation (MIMICS structural mismatch fingerprint)

| Quantity | Threshold | Trigger interpretation |
|---|---|---|
| `\|r(MIMICS_residual, NDVI)\|` across the training set (n=83) | **> 0.5** | Flag as "possible MIMICS structural mismatch — binding constraint not fully resolved vs Phase 1 WCM (r=0.82)" |

This is the same quantity as Secondary 4 (§2 above). When evaluated
as a **success criterion** the threshold is < 0.5 (pass). When
evaluated as a **diagnostic trigger** the threshold is > 0.5 (flag).
Reported in both forms in the Phase 1b results document.

### D-2 — Mechanism decomposition at Moor House

| Quantity | Threshold | Trigger interpretation |
|---|---|---|
| Per-mechanism contribution to total σ°_VV (direct-ground, direct-crown, crown-ground, ground-crown-ground) across training set | No single mechanism dominates (> 90% of total σ°_VV) across all samples | Reported as descriptive finding |
| Cumulative canopy-mechanism contribution at Moor House | If < 10% across training set | Reported as **finding** that C-band penetrates sufficiently that canopy complexity does not matter at this site — **not a problem**, a result |

Pre-committed framing: a low-canopy-contribution finding is reportable
as a Phase 1b result, not as a diagnostic failure. The pre-registration
of this framing means a "C-band has trivially enough penetration at
heather depth" outcome is a publishable conclusion.

### D-3 — Parameter identifiability (∂σ°/∂θ correlation matrix)

| Quantity | Threshold | Trigger interpretation |
|---|---|---|
| Any pair of rows in the 5×5 correlation matrix of ∂σ°_VV/∂θ_i and ∂σ°_VH/∂θ_i (for θ ∈ {N_b, N_l, σ_orient, m_g, s}) | `\|r\| > 0.95` | Flag as non-identifiable pair |

**Pre-commit:** any non-identifiable pairs flagged **pre-training** by
this diagnostic on the v0.1 MIMICS module will be reported in the
final Phase 1b results regardless of retrieval outcome. This is the
diagnostic that triggers the SPEC §5 v0.2 fallback paths (CI-based N_l
prior; season-stratified m_g scalar prior).

### D-4 — Dobson vs Mironov forward difference (dielectric binding)

| Quantity | Threshold | Trigger interpretation |
|---|---|---|
| `\|ε_Dobson − ε_Mironov\| / ε` over the observed m_v range (0.25–0.83 cm³/cm³) | < 5% | Dielectric choice is **not** binding; recorded as a non-binding factor in the Phase 1b interpretation |
| Same quantity | ≥ 5% | Dielectric choice **is** binding; carried as an active source of variance in the Phase 1b results |

This diagnostic was already computed at G4 (Session A, 2026-04-18) on
the pre-training calibration data and returned `binding = YES` with
max relative |Δε|/ε = 97.6% at m_v = 0.83 (see
`phase1b/implementation_gate/results/g4_dielectric.json` and
SPEC §14 sign-off block). The diagnostic is re-computed
post-experiment for confirmation, with the same threshold.

---

## 7. Pre-registration regression test (technical lock)

The technical enforcement of this pre-registration is the regression
test at `tests/unit/test_pre_registration_lock.py`. The test fails if:

1. This file (`phase1b/SUCCESS_CRITERIA.md`) has an mtime later than
   any Phase 1b training artefact (any file under
   `outputs/models/pinn_mimics/`, `phase1b/lambda_search/results/`,
   `phase1b/results/`, or any glob `outputs/metrics/phase1b_*.json`); **or**
2. SPEC.md §10 RMSE thresholds (Strong < 0.124; Significant < 0.131;
   Moderate < 0.139; Inconclusive < 0.155; Negative ≥ 0.155) cannot
   be matched verbatim against the current `SPEC.md`.

The test is dormant until Phase 1b training begins (no training
artefacts exist at the pre-registration tag). Once training begins,
any edit to this file or to the SPEC §10 thresholds will fail the
test on the next test run.

The escape hatch is a new DEV-1b-NNN entry: the test does not check
the deviation log, so a properly-logged criterion change with a fresh
DEV entry will still fail the test until the test is updated (in the
same commit as the DEV entry) to acknowledge the deviation. This is
intentional — it forces the criterion change through the same
honest-gates discipline as DEV-1b-001 / DEV-1b-002.

---

## 8. Sign-off

This pre-registration document, together with SPEC §10 (signed
2026-04-19) and SPEC §15 (added 2026-04-19, see SPEC.md), is the
formal pre-registration of Phase 1b. Once tagged
`phase1b-success-criteria-pre-registered`, the criteria above are
fixed for the duration of Phase 1b execution.

| Field | Value |
|---|---|
| Specification version | v1.0 |
| Date locked | 2026-04-19 |
| Parent commit | `63ef5fa` (Session F handoff close) |
| Tag at lock | `phase1b-success-criteria-pre-registered` |
| Lead investigator | Matthew Denyer / 2026-04-19 |
| Independent reviewer | [science agent] / 2026-04-19 |
| Authoritative companion (numerical thresholds) | SPEC.md §10, signed 2026-04-19 |
| Authoritative companion (sign-off block) | SPEC.md §14, signed 2026-04-19 |

**Block 2 (λ search + PINN-MIMICS trainer) is cleared to begin only
after this document is committed and tagged.** No training, no λ search,
no model instantiation in the same session that produced this document
(Session F-1). Session F-2 picks up Block 2 with the criteria above
locked.

---

*Vantage · Phase 1b · Pre-Registered Success Criteria · v1.0 · 2026-04-19*
