# DEV-1c-lean-001 — Phase 1 sealed-test-set "used-once" acknowledgement

**Phase:** 1c-Lean
**Gate impact:** Tertiary evaluation interpretation (SPEC.md §18.5)
**Status:** Reserved at SPEC.md v0.2 sign-off; signed-off pre-training.

---

## Summary

The Phase 1 sealed test set defined at SPEC.md §15.4 (n=36, 2023-07-25 to 2024-12-10) was used once for Phase 1's final evaluation per §15.5 (PINN RMSE 0.167, RF 0.155 at N≈25; RF 0.147 at 100% training). Phase 1b preserved it unsealed (Tier 3 HALT before unsealing) per §17.4.

Phase 1c-Lean re-uses this set for tertiary RMSE comparison under the §18.5 unsealing policy. Because the set has been seen once in Phase 1 evaluation, Phase 1c-Lean tertiary evaluation against it is not strictly held-out. The risk is information leakage from the Phase 1 evaluation having shaped subsequent SPEC decisions.

## Mitigation

Per SPEC.md §18.5:

- Gate criteria (§18.7 primary and secondary) evaluated on training-pool 5-fold cross-validation, not on the sealed set.
- Sealed set unsealed for tertiary RMSE comparison only if all gate criteria pass.
- HALT outcomes do not unseal.

A truly fresh held-out evaluation would require extending COSMOS-UK ground truth beyond 2024-12-10 (Phase 1c-Lean-2 scope; requires fresh Sentinel-1 acquisitions and ground-truth pairing). Out of scope for Phase 1c-Lean.

## Resolution

This DEV entry stands as the explicit acknowledgement; no implementation change required at G2-Lean or G3-Lean. Phase 1c-Lean results document at execution close (analogous to `phase1b/poc_results_phase1b.md`) will reproduce this acknowledgement in the limitations section.

## Sign-off

**Founder:** Matthew Denyer · **Date:** 2026-04-28 · *(supervisor to fill at SPEC.md v0.2 commit)*
