# DEV-1c-lean-002 — Per-channel normalisation implementation: no-op deviation

**Phase:** 1c-Lean
**Gate impact:** G2-Lean (SPEC.md §18.6.1)
**Status:** Resolved at G2-Lean sign-off (Block B-prime). No-op deviation. Signed-off pre-training.

## Summary

Reserved entry slot for any aspect of σ_VV / σ_VH computation in the Phase 1c-Lean per-channel normalisation that deviates from the published Toure-style derivation (SPEC §18.4.1 v0.3).

## Resolution

Implementation matches SPEC §18.4.1 v0.3 specification exactly. σ_VV and σ_VH are computed once at training initialisation as the standard deviation of the unweighted physics losses (l_physics_vv and l_physics_vh per the existing Phase 1b implementation at `phase1b/pinn_mimics.py:268-373`) over the training set's first forward pass at randomly initialised network weights. No per-batch normalisation. No σ-floor for near-zero variance. No special treatment of NaN inputs (none observed in the training set's first forward pass at SEED=42).

σ values are saved with the model checkpoint and reproduced exactly in the pre-flight summary block per SPEC §18.11 schema item 3.

L_monotonic and L_bounds reuse the Phase 1b implementation without modification. λ_data ≡ 1.0, λ_monotonic = 0.01, λ_bounds = 0.01 are fixed coefficients per SPEC §18.4.1 v0.3. Only λ_VV and λ_VH are tunable.

Implementation pattern: **Pattern A — sibling function** per Block B-prime kickoff prompt §3. The new function `compute_pinn_mimics_loss_normalised` was added alongside the unchanged Phase 1b `compute_pinn_mimics_loss` in `phase1b/pinn_mimics.py`; F-2b reproducibility is preserved byte-for-byte (`compute_pinn_mimics_loss` signature, body, and return-dict keys unchanged). The σ-init helper `compute_init_sigma_normalisers` was added in the same module.

Two implementation sub-decisions surfaced during Block B-prime that warrant explicit recording (both within the no-op envelope):

1. **σ uses population standard deviation (ddof=0; PyTorch `unbiased=False`).** SPEC §18.4.1 says "standard deviation" without specifying sample vs population. Population std was chosen so the G2-Lean Arm 3 invariant `std(ε / σ) == 1` is exact. Sample std (ddof=1) would leave a residual factor of √(n / (n−1)) ≈ 1.006 at n=83, breaking Arm 3's 1e-6 invariant. The choice is operationally invisible at training time (it is a fixed scaling constant).

2. **σ helper raises ValueError on σ ≤ 0 or non-finite σ.** A zero-variance physics-loss series at initialisation would correspond to a degenerate PinnMimics initialisation that maps every input to identical σ°_VV / σ°_VH; this is not expected at SEED=42 and would represent a hard model-init failure. Per SPEC §18.4.1, σ-floor / clamp is forbidden, so the helper hard-errors rather than silently substituting a default. Confirmed at G2-Lean Arm 3 that σ_VV > 0 and σ_VH > 0 on the actual training set.

G2-Lean three-arm gate closed successfully at tag `phase1c-lean-g2-lean-passed`:

- Arm 1 (numpy ↔ PyTorch cross-framework consistency on the v0.3 five-term composite): PASS at machine precision (max_abs_diff = 0.0e+00, tolerance 1e-12; 64 fixture rows at SEED=42).
- Arm 2 (autograd ↔ finite-difference on the v0.3 composite gradient): PASS within 1e-3 relative tolerance (param_max_rel = 6.020e-09, λ_max_rel = 1.446e-11; n=8 fixture, float64 model, h_param = 1e-5, h_lambda = 1e-6 per SPEC §18.6.1 / DEV-1b-008 convention).
- Arm 3 (scale sanity on the actual Phase 1 training set, n=83): PASS within 1e-6 numerical tolerance (std_normalised_l_phys_vv = 1.000000000, std_normalised_l_phys_vh = 1.000000000; data source `data/processed/aligned_dataset.csv` resolved via `phase1.lambda_search.prepare_pinn_data` from `config_000.json` train_indices ∪ val_indices = 66 + 17 = 83).

Numerical values from the training-set first forward pass at SEED=42:

- σ_VV = 6.303897380828857
- σ_VH = 4.636704921722412

(Both float32, derived from PinnMimics float32 forward pass — Arm 2's float64 promotion is a numerical-precision device for FD only and does not change the production σ values.)

## Sign-off

Founder: Matthew Denyer · Date: 2026-04-28 · *(supervisor to fill at Block B-prime close commit)*
