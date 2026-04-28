# DEV-1c-lean-006 — SPEC v0.3.1 → v0.3.2 amendment: §18.6.2 seed convention + §18 phantom-citation cleanup

**Phase:** 1c-Lean
**Gate impact:** Pre-registration (SPEC.md §18.5 / §18.6.2 / §18.7 / §18.13). Paired-comparison validity (SPEC §14, line 82) preserved.
**Status:** Authored at SPEC v0.3.2 sign-off (Block C-prime halt-2 adjudication, expanded scope per halt-3). Signed-off pre-training. Substantive in the science sense (seed scheme binds paired-comparison validity) and substantive in the audit-trail sense (phantom-citation cleanup).

## Summary

Block C-prime halt-2 surfaced two distinct issues in SPEC v0.3.1 that required bundled resolution before G3-Lean pre-training-gate work could proceed. Halt-3 expanded the phantom-citation scope from three references (§14.6 / §15.4 / §15.5) to seven references after the Rule 0.10 §4.3 cross-check enumerated all `§N.M` citations inside §18.

### Issue A — seed convention (substantive science correction)

SPEC §18.6.2 v0.3.1 specified `SEED = 42 + config_idx + rep_idx, carrying forward §14.6`. Three problems:

1. The "carrying forward §14.6" reference is non-existent (§14 has no numbered subsections per SPEC structure).
2. The actual Phase 1 / Phase 1b convention (SPEC.md:82, `phase1b/lambda_search/run_f2.py:130`, `shared/baselines/random_forest.py:56`) is `seed = config.SEED + config_idx` — a flat config_idx scheme, no `rep_idx` term.
3. The literal v0.3.1 formula `42 + config_idx + rep_idx` produces seed collisions: under config_idx ∈ [0,35] and rep_idx ∈ [0,2], the resulting seeds occupy 38 unique values across 108 (config, rep) pairs (range 42..79). Seed=43 appears at both (config_idx=0, rep_idx=1) and (config_idx=1, rep_idx=0). The kickoff-prompt §3.2 claim of "108 unique seeds" was arithmetically inconsistent with the formula it cited.

Most importantly, the paired-comparison reasoning at SPEC.md:82 ("Identical seeds across baselines and PINN-MIMICS to ensure paired comparison validity") requires the RF baseline and the PINN to use the same seed at the same config_idx. Under v0.3.1's two-index scheme, RF uses `config.SEED + config_idx_` (per `random_forest.py:56`) while PINN uses `42 + config_idx + rep_idx`; the comparison loses pairing. SPEC §18.7's secondary-metric comparator depends on paired comparison; the v0.3.1 seed scheme breaks it.

### Issue B — phantom citations in §18

Initial scope (halt-2): three references in §18 to non-existent §14.6 / §15.4 / §15.5 anchors. §14 has no numbered subsections; §15 has none either.

Expanded scope (halt-3, Rule 0.10 §4.3 cross-check): seven distinct phantom references across ten occurrences in §18:

| Phantom | Occurrences | §18.x location | Resolution |
|---|---|---|---|
| §14.2 | line 791 | §18.3 row "Site" | `SPEC §2 (line 70, "Frozen elements" Site row)` |
| §14.3 | line 792 | §18.3 row "Target variable" | `SPEC §2 (line 71, "Frozen elements" Ground truth row)` |
| §14.4 | line 793 | §18.3 row "Ground truth" | `SPEC §2 (line 71, "Frozen elements" Ground truth row); DEV-001` |
| §14.5 | lines 794, 795 | §18.3 rows "Sensors" / "Observables" | `SPEC §2 (line 72, "Frozen elements" SAR row)` |
| §14.6 | line 863 | §18.5 (season-stratification) | `SPEC §2 (line 81, training-pool meteorological-season stratification convention)` |
| §14.6 | line 889 | §18.6.2 (seed convention) | per §18.6.2 NEW text — paired-comparison reasoning at SPEC.md:82 |
| §15.4 | line 797 | §18.3 row "Sealed test set" | `SPEC §2 (line 80, sealed-set definition); status §18.5 below` |
| §15.4 | line 859 | §18.5 prose | `SPEC §14 (Phase 1 pre-registration; sealed-set lock at data/splits/test_indices.json)` |
| §15.5 | line 859 | §18.5 prose | dropped — inline numbers (PINN RMSE 0.167, etc.) stand on their own |

These were latent through Phase 1, Phase 1b, Block A's five-cycle audit, Block A-prime, Block B-prime, and Block C-prime halt-1 because no prior cycle operationally needed to follow the references.

## Caught at

Block C-prime halt-2 entry-check, 2026-04-28. CC's §6 step 3 three Phase 1b convention cross-checks (seeds, RF hyperparameters, sealed-set definition) surfaced issue A directly; in the process of grepping for §14.6 to verify the inheritance claim, CC also identified the initial three phantom citations (issue B). Halt-3 (mid-amendment) extended scope after the Rule 0.10 §4.3 cross-check arm enumerated all seven distinct phantoms.

## Resolution

### Issue A — Option A (SPEC catches up to Phase 1/1b convention)

SPEC §18.6.2 amended:
- New seed scheme: `config_idx = 18 * λ_VV_idx + 3 * λ_VH_idx + rep_idx; SEED = 42 + config_idx`. Flat config_idx ∈ [0, 107] over the (λ_VV, λ_VH, rep_idx) cross-product.
- Total unique seeds: 108 (range 42..149).
- Byte-identical to Phase 1's flat-config_idx convention encoded in `data/splits/split_manifest.json` and Phase 1b's `phase1b/lambda_search/run_f2.py:130`.
- Paired-comparison validity preserved: RF baseline at config_idx=N uses seed 42+N; PINN at config_idx=N uses seed 42+N; CV-fold splits and random-inits matched.

**Clarification on inheritance:** Phase 1c-Lean's flat config_idx ∈ [0, 107] inherits the *topology* of Phase 1's flat config_idx ∈ [0, 39] convention (per `data/splits/split_manifest.json`) and extends it to the (λ_VV × λ_VH × rep_idx) cross-product. Phase 1b's actual scheme used config_idx as an inner-rep index (`CONFIG_INDICES_100PCT = range(0,10)` per `phase1b/lambda_search/run_f2.py:79`) reused across the 64 λ-cells, producing 10 unique seeds across 640 runs — a different specialisation of the same `SEED = config.SEED + config_idx` base formula. The Phase 1c-Lean inheritance is therefore from Phase 1's flat scheme, not Phase 1b's nested scheme; what is preserved across all three is the `SEED = 42 + config_idx` base formula and the paired-comparison validity property at any single config_idx. The halt-2 adjudication's §1.2 phrasing of "byte-identical to Phase 1b inheritance" is loose in this respect; the more precise claim is "structural inheritance from Phase 1's flat scheme; same-base-formula extension."

Three options were considered (Option A: SPEC catches up to flat scheme; Option B: introduce a stride for uniqueness under two-index scheme; Option C: accept overlap as intentional). Option A is the only choice consistent with SPEC.md:82's paired-comparison reasoning.

**Formula uniqueness verified:** `18a + 3b + c` with `a ∈ [0,5]`, `b ∈ [0,5]`, `c ∈ [0,2]` produces all 108 values in [0, 107] with no collisions. Proof: since `c < 3`, `c` uniquely identifies the residue mod 3 of `(3b + c)`; since `3b + c ∈ [0, 17]`, the integer `3b + c` uniquely identifies `(b, c)`; since `18a + (3b + c)` with `3b + c ∈ [0, 17]` uniquely identifies `(a, 3b + c)`, the full triple `(a, b, c)` is recoverable.

### Issue B — phantom-citation cleanup

Seven phantoms across ten occurrences resolved per the §3.3 supervisor table in §3.3 above. All replacements use the `SPEC §N (line M, ...)` line-anchored pattern per Rule 0.10 forward operational rule.

## Methodological observations

This DEV entry corresponds to two candidate sub-observations in the Session H §28 enumeration:

### Candidate sub-observation 9 — Internal SPEC consistency cross-check

The phantom-citation finding reveals a class of audit-gap distinct from sub-observations 5–8. Mechanism: SPEC text contains internal cross-references (§N, §N.M, line numbers) that may or may not resolve to actual content; phantom references are halt-and-flag triggers when followed but latent until followed.

Mitigation pattern (Rule 0.10): every SPEC amendment that adds or modifies cross-references within the SPEC verifies that each cross-reference resolves to an actual section, subsection, or line. References to non-existent anchors are halt-and-flag triggers at SPEC amendment time, not later.

Empirical confirmation at this amendment cycle: the Rule 0.10 §4.3 cross-check arm (full grep of §18 cross-references) surfaced four additional phantom citations (§14.2 / §14.3 / §14.4 / §14.5 in §18.3) beyond the three the supervisor anticipated (§14.6 / §15.4 / §15.5). Halt-3 expanded the v0.3.2 amendment scope to bundle the additional four. The Rule 0.10 cross-check arm is doing exactly the work it was designed for at the first cycle that exercises it.

### Reinforcement of candidate sub-observation 7 (SPEC text vs implementation)

The seed-convention divergence is a fresh instance of sub-observation 7 (SPEC text vs existing implementation cross-check at SPEC sign-off). Block A-prime's Rule 0.7 cross-check arm operated on SPEC §18.4 / §18.4.2 / §18.6.1 etc. but did not operate on §18.6.2's seed convention claim because the Block A-prime cross-check arm's scope was constrained to §18.4 issues. The seed convention claim at §18.6.2 was inherited verbatim from v0.2 without verification against `run_f2.py:130`.

This reinforces that Rule 0.7 cross-check arms must be **comprehensive** at every amendment cycle — not just covering the sections being amended, but every section that references existing implementation. Forward operational rule refinement: at any SPEC amendment, the cross-check arm scope is "every SPEC section that references existing implementation," not "the sections being amended."

## Sub-finding for supervisor adjudication post-tag

A small Rule 0.10 inconsistency in the supervisor's halt-2 §5.2 NEW text was applied verbatim and is flagged here for awareness rather than corrected unilaterally:

The §18.6.2 NEW text contains the phrase `SPEC.md §14, line 82, paired-comparison reasoning` and `SPEC §14's paired-comparison reasoning`. SPEC.md line 82 actually lives in §2 (Frozen elements, starting line 64), not in §14 (Pre-registration sign-off, starting line 470). The §14 anchor is loosely defensible as "the section where Phase 1 pre-registration was signed off" (i.e., the sign-off binds line 82's content) but the more precise section anchor for line 82 itself is §2. Symmetric §14.6 references in §18.5 line 863 (season-stratification) were resolved to `SPEC §2 (line 81, ...)` per the halt-3 adjudication's `SPEC §2 (line N, "Frozen elements" row)` pattern. If supervisor prefers consistency with the halt-3 §2 pattern, a v0.3.3 follow-up could amend `§14, line 82` → `§2, line 82` in two places within §18.6.2.

This is a minor anchor-precision issue, not a substantive science finding. Flagged per Rule 0.10 audit-trail discipline.

## Sign-off

Founder: Matthew Denyer · Date: 2026-04-28 · *(supervisor to fill at SPEC v0.3.2 commit)*
