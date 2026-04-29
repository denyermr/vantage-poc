# DEV-1c-lean-008 — SPEC v0.3.3 → v0.3.4 amendment: sweep / CV procedure split

**Phase:** 1c-Lean
**Gate impact:** Pre-registration (SPEC.md §18.4.1 / §18.6.2 / §18.10 / §18.13). Phase 1c-Lean primary-metric / secondary-metric construction.
**Status:** Authored at SPEC v0.3.4 sign-off (Block D-prime halt-1 adjudication). Signed-off pre-training. Substantive in the science sense (procedure split binds primary-metric and secondary-metric construction) and substantive in the audit-trail sense (compute envelope amended).

## Summary

Block D-prime halt-1 surfaced that SPEC v0.3.3's sweep specification (108 runs over the 6×6 grid × 3 reps) and the §18.7 secondary-metric specification (PINN training-pool 5-fold CV RMSE compared against RF baseline 0.1270) were mutually inconsistent under any single interpretation:

- The §9 dominance criterion is computed on a converged model trained on the full n=83 pool (Phase 1b inheritance); this is one full-pool training per (config_idx, rep).
- The secondary metric (PINN 5-fold CV RMSE) requires 5-fold CV PINN evaluation per cell; this is 5 fold-trainings per cell on partitioned data.
- The two computations cannot be combined into a single 108-run procedure that satisfies both.
- v0.3.3 §18.10 compute envelope (~2 hours, 108 runs × ~60s) accounted for the primary procedure but not the secondary-metric CV.

Phase 1b's inheritance does not pin the resolution because Phase 1b used 40 pre-existing configs on disk (4 fractions × 10 reps factorial), each carrying its own train/val/test split. Phase 1c-Lean's 108 config_idx values have no analogous pre-existing files.

## Caught at

Block D-prime halt-1 entry-check, 2026-04-28. CC's §3 cross-check arm of the Block D-prime kickoff prompt surfaced the inconsistency between deliverable specifications and §18.10 compute envelope. CC's four-option enumeration ((i) full-pool + separate post-sweep CV; (ii) fold-mapped reps; (iii) internal 5-fold per run; (iv) fixed 80/20) provided the adjudication frame.

## Resolution

### Substantive resolution: Option (i) refined — separate Procedures A and B

**Procedure A (λ-grid sweep, 108 runs):**
- Each `(config_idx, λ_VV, λ_VH, rep_idx)` = one PINN training on full n=83 pool.
- Within-pool 80/20 split for early-stopping ONLY, split seed `SEED + 1000`.
- σ values computed on full n=83 pool's first forward pass.
- §9 dominance verdict per (config_idx, rep) emitted at converged state.
- Diagnostic battery items per §18.9 emitted.

**Procedure B (secondary-metric 5-fold CV, 180 fold-trainings):**
- For each cell `(λ_VV, λ_VH)` (36 cells), 5 fold-trainings on n=83 partitioned into 5 random-shuffle folds with seed `42 + 10000`.
- σ values recomputed per fold (~67 samples) on fold-training-portion first forward pass.
- Per-cell mean-of-per-fold RMSE emitted.
- Compared against `baselines_locked.json` RF 0.1270 for §18.7 secondary metric.

### Why Option (i) refined, not (ii) / (iii) / (iv)

- (ii) rejected: 3 reps × 5 folds is incomplete CV; secondary-metric vs RF 5-fold CV is not directly comparable.
- (iii) rejected: rep-as-fold breaks SPEC §18.6.2 v0.3.2 seed convention semantics.
- (iv) rejected: fixed 80/20 gives less CV evidence than 5-fold; not apples-to-apples vs RF baseline.

Option (i) refined is the only scheme preserving (a) §9 dominance at converged-state on full data, (b) §18.6.2 seed convention semantics, and (c) apples-to-apples secondary metric vs RF baseline.

### Procedural resolution

SPEC v0.3.3 → v0.3.4 amendment:
- §18.4.1 σ-init implementation requirements list expanded with per-procedure σ scope.
- §18.6.2 baselines/sweep procedure bullet expanded with explicit Procedure A / Procedure B language.
- §18.10 compute envelope amended from ~2 hours (108 runs) to ~3.5–4 hours total (108 + 180 = 288 runs across two procedures).
- §18.13 reserved-entries list extended.
- DEV-1c-lean-008 (this file) records the bundled resolution.

### Compute envelope (revised)

| Procedure | Runs | ~ per run | Total |
|---|---|---|---|
| A — λ-grid sweep | 108 (n=83 each) | 60s | ~108 minutes |
| B — secondary-metric CV | 180 (n=~67 each) | 30–40s | ~80–120 minutes |
| Total | 288 | | ~3.5–4 hours |

## Methodological observation

This DEV entry corresponds to **a third empirical instance of candidate sub-observation 8** (prompt-text vs SPEC cross-check at supervisor authoring time, Rule 0.9):

- Block C-prime halt-1: loss-formulation string mismatch.
- Block C-prime halt-4: null methodology vs reference number conflation.
- Block D-prime halt-1 (this entry): sweep specification did not account for secondary-metric compute cost.

Three sub-observation 8 instances in the cycle. The mitigation pattern (Rule 0.9): supervisor cross-checks every embedded normative content element against SPEC source-of-truth at authoring time. Halt-1's lesson sharpens the rule: **Rule 0.9 cross-checks must include the relationship between SPEC sections, not just within each section.** v0.3.3's §18.6.2 baselines bullet specified RF/null procedures correctly; §18.7 specified the secondary metric correctly; §18.10 specified the compute envelope. Each section was internally consistent, but the *relationships across sections* (e.g., that §18.7's PINN secondary metric implies a Procedure B not specified in §18.6.2 or accounted for in §18.10) were not cross-checked.

The empirical reinforcement: Phase 1c-Lean's intervention-isolation discipline + small-batch SPEC amendments produces a SPEC structure where individual sections are coherent but cross-section dependencies can drift. This is not a Rule 0.10 issue (which is about phantom citations / internal consistency *within* the SPEC structure); it is a Rule 0.9 *cross-section* refinement.

## Sub-observation 8 evolution: Rule 0.9 cross-section variant

Three sub-observation 8 instances now cluster into two sub-variants:
- **Sub-variant 8a (within-section)**: prompt-text drift from a specific SPEC section's content (e.g., loss-formulation string at §18.11).
- **Sub-variant 8b (cross-section)**: prompt-text or SPEC text consistent within each referenced section but inconsistent across sections (e.g., §18.7 secondary metric implying procedures not specified in §18.6.2 or §18.10).

Mitigation pattern for sub-variant 8b: at every supervisor-authored prompt and every SPEC amendment, the relationships across SPEC sections must be cross-checked, not just the content of each section. This is a stronger discipline than within-section cross-checking.

The methods-paper §6.3 enumeration may want to record this as a refinement of sub-observation 8 rather than a separate sub-observation 10 — final adjudication at methods-paper drafting pass.

## Sign-off

Founder: Matthew Denyer · Date: 2026-04-28 · *(supervisor to fill at SPEC v0.3.4 commit)*
