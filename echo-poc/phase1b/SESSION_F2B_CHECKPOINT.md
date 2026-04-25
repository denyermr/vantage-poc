# Session F-2b Checkpoint — λ search re-run under SPEC §8 joint VV+VH (DEV-1b-010)

**Date range:** 2026-04-20 19:48:38 BST (kickoff) → 2026-04-22 04:25:01 BST (close).
**Wall clock:** 1956.4 min ≈ **32.6 h** (compute proper ≈ 6.8 h; sleep-tax overhead ≈ 25.8 h — see §6).
**Tag (proposed):** `phase1b-session-f2b-lambda-selected`.
**Parent commit:** `dde17c6` (Phase 1b Session F-2b prep — joint VV+VH amendment).
**Predecessors:** F-2 v1 close at `5a2a994`, tag `phase1b-session-f2-lambda-selected` (preserved as superseded-but-intact audit record per DEV-1b-010 §"Not retroactive"). DEV-1b-010 (commit `4ebea02`) authorised the re-run.
**Outcome:** **Tier 3 HALT** under both DEV-1b-009 readings. No combination satisfies the SPEC §9 primary dominance criterion. Pre-authored decision-tree HALT branch fires.
**Sealed test set:** unchanged. SHA-256 `a4b11206630cc80fc3e2ae5853bb114c7a4154072375654c257e51e4250f8eea`. Not unsealed in F-2b. Pre-registered evaluation path requires Tier 1 or Tier 2 outcome with selected λ; HALT precludes unsealing.

---

## 1. Result-JSON integrity audit summary

Result JSON at `phase1b/lambda_search/results_f2b/lambda_search_f2_result.json` — SHA-256 `c5cbdac6414a9bf6989c0f6927462a927ebeda71120b6ab424320d6d77b6d68a`, 78,556 bytes.

| Check | Description | Result |
|---|---|---|
| 1 | 64 combo records, indexed 0-63 sequentially, covering `(λ_phys, λ_mono, λ_bounds) ∈ {0.01, 0.1, 0.5, 1.0}³` exactly once each | **PASS** |
| 2 | DEV-1b-009 dual-reading invariant: every combo carries `primary_dominance_all_reps`, `primary_dominance_mean_across_reps`, `secondary_dominance_all_reps`, `secondary_dominance_mean_across_reps` | **PASS** |
| 3 | Tier classification block: `tier=HALT`, `halted=true`, `n_full_dominance=0`, `n_primary_only=0`, `n_any_primary=0`, `n_neither=64`; `dominance_constraint.aggregation_rule_binding="mean_across_reps"`. Both readings classify identically (no combo satisfies primary under either reading) | **PASS** |
| 4 | Joint-VV+VH formulation hash / loss-formulation string in JSON metadata | **NEUTRAL** — current schema does not carry an explicit loss-formulation hash. Flagged for §7 audit-trail-strengthening proposals (non-blocking; the joint formulation is regression-tested by `tests/unit/test_pinn_mimics_loss_joint.py` from commit `dde17c6` and observably enforced via the `vh_db_observed` required parameter at the `compute_pinn_mimics_loss` signature). |
| 5 | No NaN/Inf in any aggregated combo field; no per-rep `non_finite_abort` events | **PASS** — 0 / 640 reps aborted on non-finite |
| 6 | Schema parity vs F-2 v1 result JSON | **PASS** (with one noted addition: `resumed_from_disk` field present on F-2 v1 combo records is absent on F-2b combos; F-2b ran without resume — see §6) |

All hard checks pass. F-2b classification is authoritative.

---

## 2. Grid coverage and dominance distribution

Per-block aggregates (16 combos per λ_physics block; 4³=64 grid):

| λ_physics | n_combos | median val loss | median wall (s) | mean physics_fraction | primary pass (strict) | primary pass (mean) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 16 | 0.2171 | 104.2 | 0.9255 | 0/16 | 0/16 |
| 0.10 | 16 | 1.9502 | 408.7 | 0.9898 | 0/16 | 0/16 |
| 0.50 | 16 | 9.6585 | 408.7 | 0.9979 | 0/16 | 0/16 |
| 1.00 | 16 | 19.2984 | 416.1 | 0.9990 | 0/16 | 0/16 |

**Monotonicity confirmed.** Mean physics_fraction is monotone non-decreasing in λ_physics across the full grid: 0.9255 → 0.9898 → 0.9979 → 0.9990. Higher λ_physics produces higher physics_fraction, never lower. This is the structural prediction made at the F-2b kickoff pre-flight; the empirical grid confirms it across all 48 unobserved combos that the kickoff analysis only argued for from monotonicity.

**Headline numbers.**

- **Most favourable single combination:** combo 2 — λ=(0.01, 0.01, 0.5) — `mean_physics_fraction = 0.9252`, `primary_dominance_mean_across_reps = False`. The single combo most favourable to L_data dominance still has L_physics at 92.5% of the composite loss.
- **Least favourable single combination:** combo 60 — λ=(1.0, 1.0, 0.01) — `mean_physics_fraction = 0.9990`, `primary_dominance_mean_across_reps = False`. L_data is approximately 0.1% of the composite loss.
- **Primary dominance pass rate at the most favourable block (λ_physics=0.01):** 0 / 16 under both readings. Even at the smallest pre-registered λ_physics, no combination satisfies primary dominance.

---

## 3. Tier classification

### Mean-across-reps reading (DEV-1b-009 binding interpretation)

**Tier 3 HALT.** 0 of 64 combinations satisfy the primary dominance criterion under the cross-rep mean of per-rep final-window means. Per SPEC §9 fallback procedure tier 3, the experiment halts. The Phase 1 fallback (lowest median val loss regardless of dominance) is **not** retained per SPEC §9.

### Strict all-reps-AND reading (transparency record)

**Tier 3 HALT.** 0 of 64 combinations satisfy primary dominance with `n_violations == 0` across the 10 reps. The strict reading produces the same classification as the mean reading.

### Methodological observation — DEV-1b-009 readings collapse here

Under both DEV-1b-009 readings, F-2b classifies identically as HALT. The two readings only differed in F-2 v1 because the VV-only landscape was on the dominance frontier — physics_fraction ≈ 0.435 placed L_physics and L_data within 5% of each other, where stochastic per-rep training variation flipped which was larger. Under joint VV+VH the landscape is structurally lopsided (physics_fraction ≥ 0.925 across the entire grid), so cross-rep stochasticity is irrelevant: every rep at every combination has L_physics > L_data with margin. The DEV-1b-009 aggregation-rule ambiguity is a real problem on dominance frontiers; it disappears in the strongly-dominated regime F-2b discovered. This is a methodological observation worth recording for the Phase 1b results paper §6.

### SPEC §9 honest interpretation (cited verbatim)

> "If no combination satisfies even the primary criterion: this is a Phase 1b architectural failure that halts the experiment. The Phase 1 fallback procedure (lowest median validation loss regardless of dominance) is **not** retained for Phase 1b. The honest interpretation in this case is that the MIMICS module cannot produce a physics-dominated solution at this data volume, which is itself a publishable finding."

— SPEC.md §9 ¶ "Fallback procedure", signed via §14 on 2026-04-19.

---

## 4. Magnitude-balance characterisation

### Empirical VH/VV magnitude

The pre-flight single-rep observation at λ=(0.01, 0.01, 0.10) found VH/VV ≈ 0.645 (VH MSE 7.70 dB² vs VV MSE 11.93 dB²). Across the F-2b grid, this ratio is broadly stable across configurations. Joint L_physics is therefore ~1.65× the VV-only L_physics would be at the same configuration — the structural shift that drove F-2 v1's selected λ from Tier 1 FULL_DOMINANCE under VV-only into Tier 3 HALT under joint.

### Empirical statement (for the Phase 1b results paper)

At C-band Moor House blanket bog with the Phase 1b pre-registered MIMICS forward model — Toure-style single-crown adaptation, peat-Mironov dielectric (DEV-007 clamp), Oh 1992 surface, Rayleigh+sinc² scatterers, real-only UEL vegetation dielectric — the joint VV+VH `L_physics = MSE(σ°_VV) + MSE(σ°_VH)` is intrinsically larger in MSE magnitude than the squared-error scale of L_data on the per-observation VWC residuals. The ratio (weighted L_physics : L_data) at the smallest pre-registered λ_physics = 0.01 is ~12-16×; at λ_physics=1.00 the ratio is ~1000×. The composite-loss landscape is **physics-saturated across the entire pre-registered λ grid {0.01, 0.1, 0.5, 1.0}³**.

### Framing — what the finding is and is not

**The finding is:** at N=83 training points with joint VV+VH L_physics at C-band, with pre-registered MIMICS v0.1 physics, with the pre-registered λ grid lower-bounded at 0.01, the composite loss is intrinsically physics-dominated in magnitude and L_data cannot be made the largest single term under any pre-registered combination.

**The finding is not:** "MIMICS is inadequate as a physics module for Moor House peatland retrieval." The MIMICS forward model itself is operating per its specification (G2 Moderate Pass per DEV-1b-008; numpy_port arm 36/36 at machine precision; gradient arm autograd↔FD within 0.003 dB; published_table arm characterised residuals fully traced to five pre-registered v0.1 sub-module simplifications). The HALT outcome is about **PINN composite-loss calibration at small-N under dual-pol physics constraints**, not about MIMICS forward fidelity.

The finding is directly relevant to the PINN methods literature: dual-pol L_physics magnitude balance is rarely addressed; the pair (DEV-1b-009 aggregation-rule ambiguity + DEV-1b-010 + the F-2b magnitude-balance observation) form a coherent set of methodological contributions on pre-registration discipline for physics-informed losses.

A later investigation — outside Phase 1b's pre-registration — into whether per-channel MSE normalisation, dB→linear scale changes, or learned dynamic loss weights could rebalance the composite loss is a Phase 1c or methods-paper scope question, not a Phase 1b re-scoping. The pre-registered λ grid is what binds Phase 1b.

---

## 5. Per-combo timing distribution

### Distribution

| Statistic | Value (s) |
|---|---:|
| min | 99.1 |
| p25 | 390.7 |
| median | 408.6 |
| mean | 1834.1 |
| p75 | 416.1 |
| max | 59,938.0 |
| sum | 117,380 (32.6 h) |

The median (408.6 s) is closely flanked by p25 / p75 (390.7 / 416.1) — the bulk of the distribution is **tight**. The mean / max divergence is driven by four outlier combos that overlap with sleep events (see §6).

### Per-block median wall clock

| λ_physics | median wall (s) | normalised vs λ=0.01 |
|---:|---:|---:|
| 0.01 | 104.2 | 1.0× |
| 0.10 | 408.7 | 3.9× |
| 0.50 | 408.7 | 3.9× |
| 1.00 | 416.1 | 4.0× |

Per-combo wall time scales sharply with λ_physics in moving from 0.01 → 0.10, then plateaus. Mechanism: at λ_physics=0.01 the optimiser quickly minimises L_data and early-stops (median trajectory ≈ 100 s at the cheapest band). At λ_physics ≥ 0.10 the physics gradient drives the optimiser through more epochs before patience-based early-stop triggers, increasing per-rep wall clock by ~4×. This is consistent with the 7.5 min/combo observed in real time during the early run.

### Pre-flight extrapolation underestimated wall clock by ~15×

Pre-flight was a single rep at the cheapest λ band (λ_physics=0.01) extrapolated as `11.9 s × 64 × 10 = 2.1 h`. Actual compute-only (excluding sleep tax) was approximately 6.8 h. Two corrections to the extrapolation:

1. The sampled rep was at λ=(0.01, 0.01, 0.10), in the cheapest band. Per-band scaling of ~4× from the cheapest to the rest of the grid means a representative single-band extrapolation underestimates by ~3× before any sleep effects.
2. The real total of 32.6 h then carries a ~5× sleep-tax multiplier on top.

Combined: pre-flight underestimated by ~15×. **This is methodology-paragraph material for Phase 1c compute budgeting and for a candid §6 observation in the Phase 1b results paper.**

---

## 6. Sleep / wake events during the F-2b run

### Summary

109 sleep events and 110 deep-idle wake events between F-2b kickoff (2026-04-20 19:48:38) and close (2026-04-22 04:25:01).

| Period | Sleep type | Notes |
|---|---|---|
| 19:48 → 21:02 | **None** | Run progressed at compute-bound rate; combos 0-22 completed on schedule. |
| 21:02 (Clamshell Sleep) | First sleep event | Triggered by lid closure, defeated `caffeinate -dimsu` on battery. |
| 21:02 → ~10:00 next morning | DarkWake / Sleep cycles ~every 15-17 min | Each DarkWake grants ~1-1000 s of CPU before re-sleep. Combo 23 completed during this window after ~16.6 h elapsed (vs ~7 min at compute-only rate). |
| 22 Apr 04:25 | Run completion | All 64 combos written; result JSON saved; HALT classification recorded. |

### Why caffeinate did not hold

The kickoff command was `caffeinate -dimsu`. Macros for those flags:

- `-d`: prevent display sleep
- `-i`: prevent idle sleep
- `-m`: prevent disk idle sleep
- `-s`: prevent system sleep on **AC power only** (per macOS docs — does NOT prevent sleep on battery)
- `-u`: declare user activity

The machine ran on battery (Charge: 100% → 94% → 100% on AC after re-plug → various) with the lid closed. **Clamshell sleep on battery is not prevented by `caffeinate -s`** because `-s` is AC-only by design. The supervisor's belt-and-braces recommendation at F-2b authorisation — `sudo pmset -a sleep 0 displaysleep 0 disksleep 0` — would have prevented this. Logged as session-ops note, not a DEV entry; this is infrastructure, not pre-registration.

### Per-combo timing anomalies attributable to sleep

Four combos with `wall_time_s > 1200` (out of 64), all overlapping with sleep windows:

| combo | λ tuple | wall (s) | wall (h) | likely cause |
|---:|---|---:|---:|---|
| 22 | (0.10, 0.10, 0.50) | 1,261.7 | 0.35 | partial sleep overlap (kickoff of clamshell-sleep period) |
| 23 | (0.10, 0.10, 1.00) | 59,938.0 | 16.65 | full overnight DarkWake/sleep cycling |
| 24 | (0.10, 0.50, 0.01) | 21,934.5 | 6.09 | late-morning DarkWake ramp-down |
| 47 | (0.50, 1.00, 1.00) | 14,434.0 | 4.01 | likely brief lid-closure during day |

Total anomalous excess: ~96,300 s ≈ 26.7 h ≈ 82% of the run's total wall clock. Compute-only wall clock ≈ 6.8 h.

### Confirmation that sleep did not corrupt the science

Each per-rep training is internally a single `train_pinn_mimics_single_rep` call running on MPS. When the system sleeps, the Python process suspends; on wake the kernel resumes the process and MPS state is intact. Verification:

- Zero `non_finite_abort` events across 640 reps → no MPS-state corruption on wake.
- Per-block dominance results are uniform within block (e.g. all 16 λ_physics=0.01 combos report median val loss 0.2169-0.2172, primary 0/10, physics_fraction 0.925-0.926) → sleep affected wall time only, not training trajectory or final val loss.
- Resume mechanism not triggered (`resumed_from_disk` field absent on F-2b combos; `n_resumed = 0`) → the run did not actually crash and re-resume; it ran continuously, just very slowly.

**Sleep events affected wall clock; they did not affect the science.** The HALT classification is robust to the sleep events.

---

## 7. Pre-authored decision-tree resolution

Per F-2b kickoff authorisation:

> "**If F-2b lands HALT (Tier 3):** No F-3-the-main-experiment. Next session is a results-adjudication session: [...] Sealed test set remains sealed — no unsealing, since the pre-registered test-set evaluation path requires a selected λ and a Tier 1 or Tier 2 outcome. Phase 1b concludes on the HALT finding. Phase 1c plans would be separately scoped."

**HALT branch fires.** No F-3-the-main-experiment. No sealed-test-set unsealing. Phase 1b concludes on the HALT finding.

---

## 8. Proposed next session scope (adjudication-of-outcome)

This is a scope proposal for supervisor adjudication, not authoring. Three deliverables:

1. **Phase 1b results document** in the structure of `outputs/write-up/poc_results.md` (Phase 1's negative-outcome paper). Reports HALT, the four methodological contributions, the pre-registration discipline as practised, the empirical magnitude-balance finding. Frames per §4 above — narrow, specific, "PINN composite-loss calibration finding under dual-pol physics constraints", not "MIMICS inadequate."

2. **Cascade plan** into the Vantage paper corpus. F-2 v1's selected λ and Tier 1 FULL_DOMINANCE outcome must be retracted from any document that referenced them; F-2b's HALT outcome and the four methodological contributions added. Likely scoped to **Yellow Paper v3.0.5 and Green Paper v4.1.3** per the supervisor's pre-authored cascade scoping at F-2b kickoff. White Paper v11.3 and Pitch Deck v7.1 likely no cascade impact (no F-2 numerics referenced at that granularity per F-2b kickoff guidance).

3. **Scope boundary statement.** Phase 1b concludes on the HALT finding. Phase 1c plans are separately scoped — specifically not part of Phase 1b's pre-registered deliverables. Any L-band NISAR follow-on, any per-channel L_physics renormalisation investigation, any v0.2 physics promotion are explicitly out-of-scope for Phase 1b's adjudication session.

The four methodological contributions referenced above:

(a) **G2 Moderate Pass framework** (DEV-1b-008) — distinguishing implementation-correctness testing from cross-configuration equivalence testing in pre-registered equivalence checks.

(b) **DEV-1b-009 aggregation-rule ambiguity** — pre-registration text for cross-rep aggregations should specify the rule unambiguously; participial-phrase attachment matters; the executor's discipline of dual-recording both readings is the right mitigation when ambiguity surfaces under execution.

(c) **DEV-1b-010 + F-2b magnitude-balance observation** — implementation-vs-signed-text divergence on a core formulation can survive multiple audit gates if the implementation is untracked at sign-off; once corrected, the structural consequence (joint vs co-pol L_physics magnitude) can shift the dominance landscape from Tier 1 FULL_DOMINANCE to Tier 3 HALT. Motivates a **required post-SPEC-sign-off implementation-audit step** for Phase 1c onward.

(d) **Supervisor-executor entry-check discipline** — the F-3 entry check that surfaced DEV-1b-010 was triggered by the F-3 prompt's explicit instruction to cross-reference locked text against implementation source before Block 1. This pattern is institutionalisable as a required post-SPEC-sign-off audit step and is the single most leveraged pre-registration discipline observation from Phase 1b.

---

## 9. Audit-trail-strengthening proposals (housekeeping, non-blocking)

Observations from the integrity audit that could tighten the Phase 1c result-JSON schema:

1. **Embed a code-version hash** (e.g. `git rev-parse HEAD` short) in result-JSON metadata. Would make Check 4 above a strict PASS rather than NEUTRAL, providing a one-step audit hook from any result JSON back to the exact commit of the loss formulation that produced it.

2. **Embed an explicit loss-formulation string** in result-JSON metadata. E.g. `"l_physics_formulation": "MSE(sigma_vv_db, vv_db_observed) + MSE(sigma_vh_db, vh_db_observed) [DEV-1b-010 joint VV+VH per SPEC §8]"`. Pairs with the hash for double-redundant audit-trail recording.

3. **Embed pre-flight summary block** at the top of any λ-search-style result JSON. Would record VV/VH magnitude ratio, per-rep wall-clock estimate, and the structural-prediction at kickoff (e.g. "expected Tier 3 HALT per pre-flight reading"), making the actual-vs-predicted comparison traceable in the artefact itself.

4. **Embed sleep/wake event count** captured from `pmset -g log` over the run window. Would make per-combo timing distribution interpretable at-a-glance; in F-2b's case, would have surfaced "109 sleep events; ~5× sleep-tax multiplier on compute-only wall clock" at the result-JSON level.

None blocking. All are additive and would be cleanly accommodated by extending `run_lambda_search_f2`'s result-JSON write step. Worth implementing pre-Phase-1c.

---

## 10. Tag and push

Proposed tag: **`phase1b-session-f2b-lambda-selected`** (per supervisor-authorised naming convention at F-2b kickoff — tag marks the session, not the outcome; symmetric with F-2 v1's `phase1b-session-f2-lambda-selected` tag).

Tag points to the close commit (this checkpoint + any housekeeping, ahead of supervisor sign-off and tag push to origin per the external-tag-push audit-permanence rule).

F-2 v1 tag `phase1b-session-f2-lambda-selected` (5a2a994) preserved as superseded-but-intact audit record per DEV-1b-010 §"Not retroactive". Both tags coexist.

---

*Vantage · Phase 1b · Session F-2b Checkpoint · 2026-04-25 · Author: executor (CC), pending supervisor sign-off.*
