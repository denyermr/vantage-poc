# U-4 — NDWI → m_g mapping

**Status:** Resolved pre-sign-off (Option A — prior withdrawn)
**Decision date:** 2026-04-18
**Authoritative deviation entry:** [`../DEV-1b-002.md`](../DEV-1b-002.md)
**Related decisions:** [`U-3-ndwi-formulation.md`](U-3-ndwi-formulation.md),
[`../DEV-1b-001.md`](../DEV-1b-001.md)

---

## Decision

**Option A — Withdraw the NDWI → m_g prior entirely.**

- No mapping from NDWI to m_g is implemented.
- `L_prior` is removed from the composite loss; `λ_prior` is no
  longer a model hyperparameter.
- m_g is constrained only by its sigmoid-bounded literature range
  [0.3, 0.6] g/g and the physics-consistency term `L_physics`.
- NDWI extraction (U-3) is retained; `ndwi_gao` is re-purposed as the
  input to SPEC §11 **Diagnostic D** (post-experiment
  correlation between training-set-mean NDWI and learned m_g).

The full reasoning, spec amendments, gate impact, and pre-registered
v0.2 fallback are in [`../DEV-1b-002.md`](../DEV-1b-002.md). This
document is an index-level decision record; it does not duplicate that
content.

---

## Options considered (brief)

Full options analysis was presented during the U-4 discussion and is
preserved in the conversation history. Summarised here for future
readers:

| Option | Description | Outcome |
|---|---|---|
| **A** | **Withdraw the m_g prior; NDWI becomes diagnostic-only** | **Chosen.** |
| B | Literature scalar prior at midpoint (0.45 g/g), NDWI not used in prior | Pre-registered as the v0.2 fallback in SPEC §5 if m_g non-identifiable post-experiment (season-stratified form per lit review §8 Decision 3). Not adopted at sign-off. |
| C | Training-set-mean Gao-NDWI linearly mapped into [0.3, 0.6] g/g | Rejected — same class of invention as the generic-shrubland exponential rejected in DEV-1b-001; would be inconsistent to adopt here. |
| D | Option C plus pre-registered withdrawal fallback | Rejected for the same reason as C; the fallback clause doesn't salvage the underlying invention. |
| E | Option A plus explicit re-purposing of NDWI as diagnostic | Chosen in substance — "NDWI as diagnostic" is exactly what SPEC §11 Diagnostic D codifies. |

---

## Why Option A rather than B

The primary reason is the one stated in DEV-1b-002 §Finding
(*"scalar-over-single-site structural issue"*): the v0.1-draft
specification averages per-observation NDWI variation out before the
prior is formed, so the prior was never using NDWI as a time-varying
signal in the first place. At a single site, a training-set-mean NDWI
scalar carries no information beyond what a literature-chosen scalar
would carry. Keeping any form of "NDWI-derived" prior at v0.1 would
therefore give the appearance of empirical grounding without providing
any.

Option B (literature scalar at midpoint) does carry *some* regularisation
signal — it pulls m_g toward the range centre with a wide envelope.
But adopting it at v0.1 sign-off would commit us to a prior that
Phase 1b cannot distinguish from "no prior" at N=83 single-site
training (the wide envelope plus sigmoid bounds give the optimiser
essentially the same operating envelope as no prior does). The
pre-registered v0.2 fallback encoded in SPEC §5 takes B's principle
— a literature-grounded sanity-bound prior — and pairs it with
Decision 3's season stratification, which does introduce a
time-varying signal (summer vs winter m_g). That is a materially
stronger prior than v0.1 B would have been, and it is reserved for
deployment only if the post-experiment diagnostics show non-identifiability.

This matches the DEV-1b-001 pattern: withdraw at v0.1, commit a
qualitatively different v0.2 fallback, do not invent a substitute
that papers over the missing calibration.

---

## What this document records (that DEV-1b-002 does not)

DEV-1b-002 is the authoritative deviation entry — it carries the spec
amendments, cross-document updates, and gate-impact analysis. This
document records the *decision-process artefact* for U-4 specifically:

- Which options were considered (A–E).
- Why Option A was preferred to Option B despite Option B's retained
  regularisation value (answer: the v0.2 fallback path does Option B's
  job better, only when needed).
- Pointer to DEV-1b-002 as the binding scientific entry.

This separation exists so the pre-sign-off decision-making is
reviewable without re-reading DEV-1b-002, and so the U-item series
(U-1 through U-4) has a consistent decision-record structure.

---

## Implementation consequences

Once Phase 1b training code is written (not yet the case as of this
decision):

- No `L_prior` term in the composite loss module.
- No `lambda_prior` / `λ_prior` in the λ-search configuration.
- No `phase1b/physics/s2_priors.py` module — planned in the original
  Phase 1b task breakdown, now not required.
- Diagnostic D implementation in `phase1b/diagnostics/` — Pearson
  correlation between training-set-mean `ndwi_gao` and learned m_g
  (one value per training fraction, across the 10 reps).
- λ-search grid: exactly `(λ₁, λ₂, λ₃) ∈ {0.01, 0.1, 0.5, 1.0}³` =
  64 combinations, no other dimensions.

---

## References

1. [`../DEV-1b-002.md`](../DEV-1b-002.md) — authoritative deviation entry.
2. [`../DEV-1b-001.md`](../DEV-1b-001.md) — sibling deviation (N_l prior),
   pattern-setter for this decision.
3. [`U-3-ndwi-formulation.md`](U-3-ndwi-formulation.md) — Gao +
   McFeeters extraction decision, preserved and re-scoped.
4. SPEC.md v0.1 §5 (prior implementation + v0.2 fallback), §8
   (composite loss), §9 (λ grid), §11 (Diagnostic D), §12 R3/R5,
   §13 (DEV-1b-002 row).
5. Lit review v1.0.2 Decision 3 (promoted to v0.2 fallback) and
   Decision 5 (NDWI now diagnostic-only).
