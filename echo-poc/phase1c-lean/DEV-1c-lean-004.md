# DEV-1c-lean-004 — SPEC v0.2 → v0.3 amendment: Phase 1c-Lean composite scope correction

**Phase:** 1c-Lean
**Gate impact:** Pre-registration (SPEC.md §18.4 / §18.4.2 / §18.6.2 / §18.7 / §18.8 / §18.10 / §18.13)
**Status:** Authored at SPEC v0.3 sign-off (Block A-prime). Signed-off pre-training.

## Summary

SPEC §18.4.1 / §18.4.2 as committed at v0.2 (tag `phase1c-lean-spec-v0_2`, commit `5dbe1ba`) specified a Phase 1c-Lean composite that diverged from a minimum-deviation extension of the Phase 1b composite. Three issues:

1. **Mischaracterisation of Phase 1b composite.** §18.4.1 prose stated the F-2b composite was three-term (`L_data + λ_VV·L_phys_VV + λ_VH·L_phys_VH`). The actual F-2b composite is four-term with shared λ_physics across VV+VH plus L_monotonic and L_bounds regularisers, λ_data ≡ 1.0 (per `phase1b/pinn_mimics.py:268-373` and `phase1b/lambda_search/run_f2.py:16,60`).

2. **Scope conflation.** §18.4.1 / §18.4.2 specified a Phase 1c-Lean composite that dropped L_monotonic + L_bounds, separated λ_physics into λ_VV / λ_VH, and promoted λ_data from fixed 1.0 to a tunable axis. Three interventions, not one. SPEC §18.1 frames Phase 1c-Lean as a *"leanest-possible debug"* and §18.4.3 explicitly excludes intervention conflation. The same reasoning excludes dropping regularisers and promoting λ_data.

3. **Grid size and compute envelope inconsistency.** §18.4.2 specified 6³ = 216 cells; §18.10 specified 648 runs. Both predicated on issue 2's three-axis structure.

## Caught at

Block B halt-1 entry-check, 2026-04-28. CC reading the Phase 1b implementation at `phase1b/pinn_mimics.py` and `phase1b/lambda_search/run_f2.py` as preparation for extending the loss formulation; surfaced the v0.2 §18.4 specification did not match the implementation it referenced.

## Resolution

SPEC v0.2 → v0.3 amendment cycle (Block A-prime, 2026-04-28). Adjudication: Option α with separate λ_VV / λ_VH per Block B halt-1 adjudication §2.2. The v0.3 Phase 1c-Lean composite is:

```
L_total = L_data + λ_VV · (L_phys_VV / σ_VV) + λ_VH · (L_phys_VH / σ_VH) + λ_monotonic · L_monotonic + λ_bounds · L_bounds
```

with:
- λ_data ≡ 1.0 (fixed; same as Phase 1b).
- λ_monotonic and λ_bounds fixed at Phase 1b grid lower-bound (0.01 each, verified against `run_f2.py` at Block A-prime cross-check §3).
- σ_VV and σ_VH per-channel normalisation factors per existing v0.2 §18.4.1 specification (unchanged).
- λ_VV and λ_VH the only tunable axes; grid 6² = 36; total 108 runs; compute ~2 hours.

The single intervention being tested in Phase 1c-Lean is per-channel L_physics normalisation, in clean isolation from all other variables. L_monotonic, L_bounds, and λ_data are preserved at Phase 1b values to maintain intervention isolation per §18.4.3 reasoning.

## Methodological observation

This DEV entry corresponds to Session H decisions log §28 candidate sub-observation 7 (pre-registration text vs implementation cross-check at SPEC sign-off). Block A's five-cycle audit caught text-fidelity / structural / format / preservation / line-replacement issues but did not catch a substantive science-content error in §18.4.1 / §18.4.2. The cross-check that should have been part of Block A audit was forced by Block B's implementation entry. Forward operational rule (Rule 0.7) added to Block A-prime: every SPEC pre-registration cycle that references existing implementation files must include an explicit cross-check arm.

## Sign-off

Founder: Matthew Denyer · Date: 2026-04-28 · *(supervisor to fill at SPEC.md v0.3 commit)*
