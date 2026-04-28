# DEV-1c-lean-005 — SPEC v0.3 → v0.3.1 amendment: §18.11 loss-formulation string canonicalisation

**Phase:** 1c-Lean
**Gate impact:** Pre-registration (SPEC.md §18.11 / §18.13). Result-JSON schema audit-trail clarity.
**Status:** Authored at SPEC v0.3.1 sign-off (Block C-prime halt-1 adjudication). Signed-off pre-training. Non-substantive in the science sense; substantive in the audit-trail sense.

## Summary

SPEC §18.11 item 2 v0.3 specified the loss-formulation string as `"per_channel_normalised_joint_vv_vh_composite"`. This string was carried forward from v0.2-era authoring without update during the v0.2 → v0.3 amendment cycle (Block A-prime), which focused on §18.4 / §18.6.2 / §18.7 / §18.8 / §18.10 / §18.13 changes.

Block B-prime committed G2-Lean artefacts at tag `phase1c-lean-g2-lean-passed` use the string `"v0.3_five_term_per_channel_normalised"` — a more informative identifier (`v0.3_` SPEC-version prefix + `five_term` composite-structure qualifier).

The two strings refer to the same loss formulation (v0.3 §18.4.1 five-term per-channel normalised composite). The discrepancy is in the canonical identifier, not in the formulation itself.

## Caught at

Block C-prime entry-check Rule 0.7 cross-check arm, 2026-04-28. CC reading SPEC §18.11 verbatim to author `g3_lean/preflight_schema.json` (Block C-prime deliverable 5) surfaced the divergence between SPEC §18.11 item 2 string and Block B-prime committed `phase1c-lean/g2_lean/arm_*.py` and `g2_lean_equivalence_result.json` strings.

## Resolution

Option A confirmed at Block C-prime halt-1 adjudication:
- SPEC §18.11 item 2 amended to `"v0.3_five_term_per_channel_normalised"` (SPEC catches up to implementation).
- SPEC line 3 active version metadata bumped v0.3 → v0.3.1.
- This DEV-1c-lean-005 records the resolution.
- §18.13 reserved-entries list extended.

Three grounds for Option A vs Option B (re-issue Block B-prime artefacts under SPEC string) vs Option C (third canonical string):
1. The Block B-prime string is more audit-friendly (`v0.3_` self-tags the SPEC version; `five_term` distinguishes from v0.2's three-term composite or hypothetical future variants).
2. Tag immutability discipline (Phase 1b §17.6 / Block A-prime §27): pushed tags are immutable; Option B would require re-tagging or force-pushing.
3. Block A-prime precedent: SPEC v0.2 → v0.3 was a micro-block amendment cycle; v0.3 → v0.3.1 follows the same pattern at smaller scope.

## Methodological observation

This DEV entry corresponds to a candidate **sub-observation 8** in the Session H §28 enumeration: **prompt-text vs SPEC cross-check at supervisor prompt-authoring time**. Mechanism: the Block B-prime kickoff prompt §5 result-format JSON examples specified the canonical string `"v0.3_five_term_per_channel_normalised"` without cross-checking against SPEC §18.11 item 2. CC implemented the prompt's string. The drift was supervisor-side authoring; CC is operating correctly.

The candidate sub-observation is structurally distinct from:
- Sub-observation 5 (inherited-artefact verification at session entry).
- Sub-observation 6 (transmission integrity at supervisor → executor boundary).
- Candidate sub-observation 7 (SPEC text vs existing implementation cross-check at SPEC sign-off).

The mitigation pattern for sub-observation 8 is **Rule 0.9** (forward operational rule): every supervisor-authored prompt that embeds normative content (JSON schemas, file paths, schema strings, fixed values, threshold numbers) cross-checks that content against the upstream SPEC sections it references at prompt-authoring time. Distinct from Rule 0.7 (executor-side cross-check at session entry).

## Sign-off

Founder: Matthew Denyer · Date: 2026-04-28 · *(supervisor to fill at SPEC v0.3.1 commit)*
