# Session F-3 — Start Here

**Purpose:** Single-page landing for the F-3 kickoff. Read this first, then the files in §"Required reading order" before touching any code.

---

## One-line state

Phase 1b F-2 closed 2026-04-20 with λ selected under DEV-1b-009's binding mean-across-reps reading. F-3 is the main experiment: 4 × 10 factorial at the selected λ, sealed-test evaluation, Phase 4 diagnostic suite, and results document.

## Resume from tag

```bash
git log --oneline --decorate -1
# expected: <commit> (HEAD -> main, tag: phase1b-session-f2-lambda-selected, origin/main) ...
```

If that is not the current tag, **stop** and diagnose before proceeding.

## Absolute non-negotiables (same as Session F)

1. **Sealed test set** at `echo-poc/data/splits/test_indices.json` is accessed for the first time in F-3 at evaluation time, **never at training time**. Confirm its SHA matches Phase 1's before using it for evaluation. The chronological boundary is 2023-07-25 to 2024-12-10 (n = 36 observations).
2. **Do not modify v0.1 physics** (`phase1b/physics/mimics.py`, `phase1b/physics/reference_mimics/reference_toure.py`, `phase1b/physics/oh1992_learnable_s.py`) without a new DEV entry. The five-way DEV-1b-008 promotion queue is a registry of candidates, **not** a commitment.
3. **Do not touch Tier 1** (`phase1/`, `data/`, frozen `shared/` modules — see CLAUDE_3.md §Frozen-tier).
4. **Moor House production path pinning:** `mimics_toure_single_crown(params, ground_dielectric_fn=None)` → Mironov with DEV-007 clamp. Regression-tested in `tests/unit/test_mimics_torch.py::TestMoorHouseProductionPinning`.
5. **DEV-1b-009 binding reading** for all cross-rep aggregations: mean-across-reps is the default aggregation rule for Phase 4 diagnostic thresholds D-1 through D-4 unless a specific diagnostic's per-run execution surfaces an aggregation-dependent edge case that the cross-rep mean does not handle (in which case: new DEV entry with its own adjudication).

## Selected λ (F-2 output → F-3 input)

| Hyperparameter | Value |
|---|---|
| λ_physics | **0.01** |
| λ_monotonic | **0.01** |
| λ_bounds | **0.10** |
| λ_data (fixed) | 1.0 |

Source: [`lambda_search/results/lambda_search_f2_result.json`](lambda_search/results/lambda_search_f2_result.json) → `selected` block. Tier 1 FULL_DOMINANCE per DEV-1b-009 adjudication. See [`SESSION_F2_CHECKPOINT.md`](SESSION_F2_CHECKPOINT.md) for dual-reading context.

## Verify environment at session start

```bash
cd /Users/matthewdenyer/Documents/Vantage/Science/PoC/vantage-poc/echo-poc

git log --oneline --decorate -1
# expected tag: phase1b-session-f2-lambda-selected

.venv/bin/python -m pytest tests/unit/ -q | tail -3
# expected: 249 passed

.venv/bin/python -c "
import json
with open('phase1b/lambda_search/results/lambda_search_f2_result.json') as f:
    r = json.load(f)
assert r['three_tier_fallback_outcome']['tier'] == 'FULL_DOMINANCE'
assert r['selected']['combo_idx'] == 1
assert r['dominance_constraint']['aggregation_rule_binding'] == 'mean_across_reps'
print('F-2 result JSON OK: tier=%s selected=%s' % (
    r['three_tier_fallback_outcome']['tier'],
    r['selected']['combo_idx'],
))
"
```

If any of the three commands fail, **stop** and diagnose before doing anything else.

## Required reading order

1. [`../CLAUDE_3.md`](../CLAUDE_3.md) — tier conventions, frozen-vs-active, gate protocol, deviation discipline.
2. [`../SPEC.md`](../SPEC.md) §8 (composite loss), §9 (λ search; now closed at F-2), §10 (success criteria + outcome categories — **this is F-3's classification source**), §11 (Phase 4 diagnostic plan — **this is F-3's diagnostic scope**), §12 (risk register), §14 (signed sign-off block).
3. [`SUCCESS_CRITERIA.md`](SUCCESS_CRITERIA.md) v1.0 — the locked pre-registration. §4 outcome-category classification rules; §6 Phase 4 diagnostic thresholds (D-1 through D-4).
4. [`DEV-1b-009.md`](DEV-1b-009.md) — **the interpretation that binds going forward**. Especially the "Binding for future Phase 1b diagnostics" section, which sets the cross-rep mean as the default aggregation for D-1 through D-4.
5. [`DEV-1b-008.md`](DEV-1b-008.md) — Phase E closure. Five-way v0.1→v0.2 promotion queue and the evidence-led-promotion rule that governs whether any physics promotion is warranted after F-3 diagnostics.
6. [`SESSION_F2_CHECKPOINT.md`](SESSION_F2_CHECKPOINT.md) — F-2 close report. §3 dual-reading structure, §4 amendments, §6 honest-gates framing.
7. Phase 1 results for comparison baseline: [`../outputs/write-up/poc_results.md`](../outputs/write-up/poc_results.md).

## F-3 scope

### Build

1. **Factorial executor** — extend/adapt the F-2 trainer to run 3 models × 4 training fractions × 10 reps = 120 configurations at the selected λ. Use the 40 pre-generated config JSONs at `data/splits/configs/config_*.json` (same split structure as Phase 1). Models: **PINN-MIMICS**, **Random Forest**, **Standard NN**.
2. **Sealed-test evaluation** — after all 120 trainings complete, evaluate each on the sealed test set. Compute RMSE per model × fraction per rep.
3. **Wilcoxon signed-rank tests** — pairwise PINN-MIMICS vs RF, PINN-MIMICS vs NN, RF vs NN at each fraction. Bonferroni-corrected α = 0.05 / 4 = 0.0125.
4. **Phase 4 diagnostic suite** on the 10 × 100% PINN-MIMICS runs (SPEC §11 / SUCCESS_CRITERIA.md §6):
   - **D-1** residual-NDVI correlation |r| < 0.5 (per DEV-1b-009: cross-rep mean of per-rep Pearson r).
   - **D-2** mechanism dominance > 90 % threshold (per DEV-1b-009: cross-rep mean of per-rep per-observation contribution fractions).
   - **D-3** parameter identifiability |r_pairwise| > 0.95 (per DEV-1b-009: cross-rep mean of |r| for each parameter pair).
   - **D-4** Dobson-vs-Mironov 5 % forward sensitivity (per DEV-1b-009: cross-rep mean of per-observation maximum absolute difference).
5. **Results document** — draft `phase1b/outputs/write-up/phase1b_results.md` in the same structure as `outputs/write-up/poc_results.md` (Phase 1). Include: executive summary, methodology (cite DEV-1b-008, DEV-1b-009), dual-reading F-2 outcome table, factorial results, Wilcoxon outcomes, D-1 through D-4 diagnostic outcomes, outcome-category classification per SUCCESS_CRITERIA.md §4, and deviation summary (DEV-1b-001 through DEV-1b-009).

### Ship

- 120-run factorial result artefact with per-run RMSE on sealed test + training/validation trajectories.
- Wilcoxon p-value table (Bonferroni-corrected).
- Phase 4 diagnostic result JSONs (D-1 through D-4).
- `phase1b_results.md` drafted.
- Unit tests for any new diagnostic code; full suite stays 249+ passing.
- Session log entry appended to `SESSION_PLAN.md` at F-3 close.

### Estimated runtime

- Compute: ~45–75 min for the 120-run factorial on MPS (PINN-MIMICS at ~6 min / config based on F-2, RF + NN substantially cheaper). Use `caffeinate -dimsu` for the PINN-MIMICS subset.
- Total session (including analysis and results drafting): 3–5 days.

### Out of scope for F-3

- Any v0.1 physics promotion — these are **evidence-led** per DEV-1b-008; Session F physics work is authorised only on the basis of F-3 diagnostic evidence, and would be a subsequent session.
- Cross-document corpus audit (White / Yellow / Green / Pitch Deck) — user handles separately.
- MVP dashboard build — deferred per memory file.

## Sealed-test integrity check (run once at the start of evaluation)

Before the first read of the sealed test set, run:

```bash
.venv/bin/python -c "
import json, hashlib
with open('data/splits/test_indices.json') as f:
    content = f.read()
print('SHA-256:', hashlib.sha256(content.encode()).hexdigest())
# expected SHA to match Phase 1's last recorded value (see Phase 1 gate_2 result JSON)
obj = json.loads(content)
print('n_test:', len(obj['test_indices']))
print('first:', obj.get('test_date_range', {}).get('start') or obj.get('date_range', {}).get('start'))
print('last:',  obj.get('test_date_range', {}).get('end')   or obj.get('date_range', {}).get('end'))
"
```

Expected: test_indices.json unchanged from Phase 1, chronological boundary 2023-07-25 to 2024-12-10, n = 36 observations.

## F-3 done-when

- 120-run factorial complete; all per-run artefacts persisted.
- Sealed-test RMSE computed per model × fraction × rep.
- Wilcoxon test table computed and recorded.
- D-1 through D-4 diagnostic results JSONs produced.
- Outcome-category classification recorded per SUCCESS_CRITERIA.md §4.
- `phase1b_results.md` drafted, structured similarly to `poc_results.md`.
- Unit suite green with zero regressions.
- Session log entry appended to `SESSION_PLAN.md`.
- DEV entries opened for any training-diagnostic-implicated v0.1→v0.2 physics promotion candidates (per DEV-1b-008), **without** executing the promotion in this session.

---

**F-3 start:** TBD.
**F-3 end:** When the checklist above is green and a Block 3-close handoff is written.
