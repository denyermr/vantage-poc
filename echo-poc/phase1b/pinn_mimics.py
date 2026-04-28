"""
Phase 1b PINN-MIMICS model and composite loss.

This module wires the v0.1 MIMICS forward physics
(`phase1b/physics/mimics.py::mimics_toure_single_crown_breakdown_torch`)
into the shared two-branch PINN backbone (`shared/pinn_backbone.py`) to
produce the Phase 1b PINN-MIMICS architecture used by Session F-2
(λ hyperparameter search) and Session F-3 (main 40-config experiment).

The architecture mirrors Phase 1's PINN exactly (PhysicsNet +
CorrectionNet, summed output) — only the physics forward model changes
from WCM to MIMICS. This is the `SPEC.md` §2 backbone-frozen contract.

Learnable parameters (SPEC §5):
    Five structural parameters are jointly learned with per-observation
    m_v via sigmoid reparameterisation on their literature-range bounds:

        N_b           ∈ [10², 10⁴] per m³   (branch density)
        N_l           ∈ [10³, 10⁵] per m³   (leaf density)
        σ_orient_deg  ∈ [15°, 60°]          (branch orientation std dev)
        m_g           ∈ [0.3, 0.6] g/g      (vegetation gravimetric moisture)
        s_cm          ∈ [1, 5] cm           (surface RMS height)

    N_b and N_l are parameterised in log-space to sample the decade range
    uniformly; σ_orient, m_g and s use linear sigmoid bounds.

Ground dielectric:
    Mironov (SPEC §6 primary) is the production default used everywhere
    in this module. DEV-1b-004's `ground_dielectric_fn` G2-harness-only
    injection point is intentionally not exposed here per the v0.1
    physics freeze (DEV-1b-008).

Composite loss (SPEC §8, joint VV+VH per DEV-1b-010):
    L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds

    L_data      = MSE(m_v_final, m_v_observed)
    L_physics   = MSE(σ°_VV_MIMICS, VV_observed) + MSE(σ°_VH_MIMICS, VH_observed)
                  (joint VV+VH, single shared λ₁ across both pol terms;
                   "the central change" of Phase 1b from Phase 1 per SPEC §3 Δ-row)
    L_monotonic = mean(ReLU(−dε_ground/dm_v))         (finite-diff probe)
    L_bounds    = mean(ReLU(−m_v_final) + ReLU(m_v_final − PEAT_THETA_SAT))

    The L_data coefficient is fixed at 1.0 per SPEC §8; the three λ's
    are the Phase 1b search axes per SPEC §9.

References:
    SPEC.md §2 (PINN backbone)
    SPEC.md §5 (parameter table)
    SPEC.md §6 (dielectric — Mironov primary)
    SPEC.md §8 (composite loss)
    SPEC.md §9 (λ search and dominance)
    DEV-1b-008 (v0.1 physics freeze)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Union

import torch
import torch.nn as nn

from shared.config import FEATURE_COLUMNS, PEAT_THETA_SAT
from shared.pinn_backbone import CorrectionNet, PhysicsNet

from phase1b.physics.mimics import (
    MimicsToureParamsTorch,
    ground_epsilon_mironov_torch,
    mimics_toure_single_crown_breakdown_torch,
)

logger = logging.getLogger(__name__)


# ─── Learnable-parameter bounds (SPEC §5) ────────────────────────────────────

# Branch and leaf densities are bounded on decade ranges — reparameterise in
# log10 space so the raw parameter maps linearly across the decade span.
NB_LOG10_LB, NB_LOG10_UB = 2.0, 4.0          # 10² … 10⁴ per m³
NL_LOG10_LB, NL_LOG10_UB = 3.0, 5.0          # 10³ … 10⁵ per m³
# Initial geometric midpoint (10^3 for N_b; 10^4 for N_l per SPEC §5).
NB_LOG10_INIT = 3.0
NL_LOG10_INIT = 4.0

SIGMA_ORIENT_LB_DEG, SIGMA_ORIENT_UB_DEG = 15.0, 60.0
SIGMA_ORIENT_INIT_DEG = 37.5                  # midpoint

MG_LB, MG_UB = 0.30, 0.60
MG_INIT = 0.45                                # midpoint per SPEC §5

S_LB_CM, S_UB_CM = 1.0, 5.0
S_INIT_CM = 3.0                               # midpoint (range 1..5)


def _sigmoid_raw_init(value: float, lb: float, ub: float) -> float:
    """Inverse sigmoid: raw s.t. sigmoid(raw) gives (value - lb)/(ub - lb)."""
    frac = (value - lb) / (ub - lb)
    frac = min(max(frac, 1e-6), 1.0 - 1e-6)
    return math.log(frac / (1.0 - frac))


# ─── PINN-MIMICS architecture ───────────────────────────────────────────────


class PinnMimics(nn.Module):
    """
    PINN with MIMICS physics forward.

    Architecture:
        X → PhysicsNet → m_v_physics ∈ [0, PEAT_THETA_SAT]
        X → CorrectionNet → δ_ML
        m_v_final = m_v_physics + δ_ML

        (m_v_physics, θ_inc, {N_b, N_l, σ_orient, m_g, s}) → MIMICS
            → σ°_VV_dB (+ mechanism breakdown + σ°_VH_dB for diagnostics)

    Learnable parameters:
        - PhysicsNet weights (≈ bounded to [0, PEAT_THETA_SAT] by sigmoid)
        - CorrectionNet weights (unbounded δ_ML)
        - 5 MIMICS structural parameters (sigmoid-bounded per SPEC §5)

    The MIMICS forward uses its default orientation-quadrature grids
    (N_THETA_SAMPLES=128, N_PHI_SAMPLES=32) unchanged from the reference
    implementation. Reducing these would be a deviation from v0.1.

    Reference:
        SPEC.md §2, §5, §8 — architecture, parameter table, composite loss
    """

    def __init__(
        self,
        n_features: int = len(FEATURE_COLUMNS),
        n_theta_quadrature: int = 128,
        n_phi_quadrature: int = 32,
    ) -> None:
        super().__init__()
        self.physics_net = PhysicsNet(n_features)
        self.correction_net = CorrectionNet(n_features)

        # MIMICS structural learnables (scalar, shared across the batch).
        # Sigmoid reparameterisation: raw ∈ ℝ, bounded = LB + (UB - LB) * sigmoid(raw).
        # Parameters initialised at midpoints so the raw value is near zero.
        self.N_b_log10_raw = nn.Parameter(torch.tensor(
            _sigmoid_raw_init(NB_LOG10_INIT, NB_LOG10_LB, NB_LOG10_UB),
            dtype=torch.float32,
        ))
        self.N_l_log10_raw = nn.Parameter(torch.tensor(
            _sigmoid_raw_init(NL_LOG10_INIT, NL_LOG10_LB, NL_LOG10_UB),
            dtype=torch.float32,
        ))
        self.sigma_orient_raw = nn.Parameter(torch.tensor(
            _sigmoid_raw_init(SIGMA_ORIENT_INIT_DEG, SIGMA_ORIENT_LB_DEG, SIGMA_ORIENT_UB_DEG),
            dtype=torch.float32,
        ))
        self.m_g_raw = nn.Parameter(torch.tensor(
            _sigmoid_raw_init(MG_INIT, MG_LB, MG_UB),
            dtype=torch.float32,
        ))
        self.s_raw = nn.Parameter(torch.tensor(
            _sigmoid_raw_init(S_INIT_CM, S_LB_CM, S_UB_CM),
            dtype=torch.float32,
        ))

        self._n_theta = int(n_theta_quadrature)
        self._n_phi = int(n_phi_quadrature)

    # ── bounded properties (no cached tensors; recomputed each access for autograd) ──

    @property
    def N_b(self) -> torch.Tensor:
        log10_nb = NB_LOG10_LB + (NB_LOG10_UB - NB_LOG10_LB) * torch.sigmoid(self.N_b_log10_raw)
        return torch.pow(torch.as_tensor(10.0, dtype=log10_nb.dtype, device=log10_nb.device), log10_nb)

    @property
    def N_l(self) -> torch.Tensor:
        log10_nl = NL_LOG10_LB + (NL_LOG10_UB - NL_LOG10_LB) * torch.sigmoid(self.N_l_log10_raw)
        return torch.pow(torch.as_tensor(10.0, dtype=log10_nl.dtype, device=log10_nl.device), log10_nl)

    @property
    def sigma_orient_deg(self) -> torch.Tensor:
        return SIGMA_ORIENT_LB_DEG + (SIGMA_ORIENT_UB_DEG - SIGMA_ORIENT_LB_DEG) * torch.sigmoid(self.sigma_orient_raw)

    @property
    def m_g(self) -> torch.Tensor:
        return MG_LB + (MG_UB - MG_LB) * torch.sigmoid(self.m_g_raw)

    @property
    def s_cm(self) -> torch.Tensor:
        return S_LB_CM + (S_UB_CM - S_LB_CM) * torch.sigmoid(self.s_raw)

    # ── forward ──

    def forward(
        self,
        X: torch.Tensor,
        theta_inc_deg: torch.Tensor,
        vv_db_observed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        PINN-MIMICS forward pass.

        Args:
            X:                Feature matrix (N, n_features), normalised.
            theta_inc_deg:    Per-observation incidence angles (N,) in degrees.
            vv_db_observed:   Observed VV backscatter (N,) in dB, unnormalised.
                              Retained in the outputs for diagnostic use but
                              not consumed by the forward itself.

        Returns:
            dict with keys:
                m_v_physics:       PhysicsNet output (N,). cm³/cm³.
                delta_ml:          CorrectionNet output (N,).
                m_v_final:         m_v_physics + delta_ml (N,).
                sigma_vv_db:       MIMICS σ°_VV (N,) in dB (for L_physics).
                sigma_vh_db:       MIMICS σ°_VH (N,) in dB (diagnostic).
                epsilon_ground:    Mironov dielectric at m_v_physics (N,).
                                   Used by the L_monotonic probe.
                vv_db_observed:    Pass-through for loss/diagnostics.
                N_b:               Current N_b value (scalar).
                N_l:               Current N_l value (scalar).
                sigma_orient_deg:  Current σ_orient (scalar).
                m_g:               Current m_g (scalar).
                s_cm:              Current s (scalar).
        """
        m_v_physics = self.physics_net(X)
        delta_ml = self.correction_net(X)
        m_v_final = m_v_physics + delta_ml

        # Build the MIMICS parameter container. N_b, N_l, σ_orient, m_g, s
        # are scalar tensors — they broadcast across the (N,) batch inside
        # the forward via torch broadcasting.
        params = MimicsToureParamsTorch(
            N_b_per_m3=self.N_b,
            N_l_per_m3=self.N_l,
            sigma_orient_deg=self.sigma_orient_deg,
            m_g=self.m_g,
            m_v=m_v_physics,
            s_cm=self.s_cm,
            theta_inc_deg=theta_inc_deg,
        )

        breakdown = mimics_toure_single_crown_breakdown_torch(
            params,
            n_theta=self._n_theta,
            n_phi=self._n_phi,
        )

        return {
            "m_v_physics": m_v_physics,
            "delta_ml": delta_ml,
            "m_v_final": m_v_final,
            "sigma_vv_db": breakdown["sigma_total_vv_db"],
            "sigma_vh_db": breakdown["sigma_total_vh_db"],
            "epsilon_ground": breakdown["epsilon_ground"],
            "vv_db_observed": vv_db_observed,
            "N_b": self.N_b,
            "N_l": self.N_l,
            "sigma_orient_deg": self.sigma_orient_deg,
            "m_g": self.m_g,
            "s_cm": self.s_cm,
        }


# ─── Composite loss (SPEC §8) ────────────────────────────────────────────────


def compute_pinn_mimics_loss(
    outputs: Dict[str, torch.Tensor],
    m_v_observed: torch.Tensor,
    vv_db_observed: torch.Tensor,
    vh_db_observed: torch.Tensor,
    lambda_physics: float,
    lambda_monotonic: float,
    lambda_bounds: float,
) -> Dict[str, torch.Tensor]:
    """
    Composite loss for PINN-MIMICS.

        L = L_data + λ_physics · L_physics + λ_mono · L_monotonic + λ_bounds · L_bounds

        L_physics = MSE(σ°_VV_pred, VV_obs) + MSE(σ°_VH_pred, VH_obs)
                  — joint VV+VH per SPEC §8 (signed 2026-04-19).
                    A single shared λ_physics weights both pol terms
                    per SPEC §8 "single shared λ₁ is used across both
                    polarisation terms; per-pol separate λs are explicitly
                    out of scope for v0.1."

    The L_data coefficient is fixed at 1.0 per SPEC §8; it is NOT a search
    axis. The three λ's are the Phase 1b λ-search axes per SPEC §9.

    DEV-1b-010 (2026-04-20) established joint VV+VH as the binding
    formulation after an F-3 entry-check cross-reference caught a VV-only
    implementation divergence from SPEC §8. Regression-tested in
    `tests/unit/test_pinn_mimics_loss_joint.py`. Any revert to VV-only
    would fail those tests and require a new DEV entry.

    Args:
        outputs:             Output dict from `PinnMimics.forward()`.
        m_v_observed:        Ground-truth VWC (N,) in cm³/cm³.
        vv_db_observed:      Observed VV backscatter (N,) in dB, unnormalised.
        vh_db_observed:      Observed VH backscatter (N,) in dB, unnormalised.
                             Required per SPEC §8 joint VV+VH formulation.
        lambda_physics:      Weight for L_physics (SPEC §9 λ₁).
        lambda_monotonic:    Weight for L_monotonic (SPEC §9 λ₂).
        lambda_bounds:       Weight for L_bounds (SPEC §9 λ₃).

    Returns:
        dict with keys (all scalar tensors):
            total:                     L.
            l_data:                    MSE(m_v_final, m_v_observed).
            l_physics:                 MSE(σ°_VV, VV_obs) + MSE(σ°_VH, VH_obs).
            l_physics_vv:              MSE(σ°_VV, VV_obs) — diagnostic breakdown.
            l_physics_vh:              MSE(σ°_VH, VH_obs) — diagnostic breakdown.
            l_monotonic:               mean(ReLU(−dε/dm_v)).
            l_bounds:                  mean(ReLU(−m_v) + ReLU(m_v − θ_sat)).
            weighted_l_physics:        λ_physics · L_physics (joint VV+VH sum).
            weighted_l_monotonic:      λ_monotonic · L_monotonic.
            weighted_l_bounds:         λ_bounds · L_bounds.

    Reference:
        SPEC.md §8 (joint VV+VH composite loss — "the central change").
        DEV-1b-010 (implementation-vs-SPEC §8 divergence resolution).
    """
    m_v_final = outputs["m_v_final"]
    m_v_physics = outputs["m_v_physics"]
    sigma_vv_db = outputs["sigma_vv_db"]
    sigma_vh_db = outputs["sigma_vh_db"]
    epsilon_base = outputs["epsilon_ground"]

    # L_data
    l_data = torch.nn.functional.mse_loss(m_v_final, m_v_observed)

    # L_physics — joint VV+VH per SPEC §8 (DEV-1b-010).
    # Single shared λ_physics weights the sum of the two per-pol MSE terms.
    # VV and VH contributions retained separately in the return dict for
    # diagnostic transparency (D-1 / D-3 / Secondary 1 / Secondary 3).
    l_physics_vv = torch.nn.functional.mse_loss(sigma_vv_db, vv_db_observed)
    l_physics_vh = torch.nn.functional.mse_loss(sigma_vh_db, vh_db_observed)
    l_physics = l_physics_vv + l_physics_vh

    # L_monotonic — dε/dm_v > 0 via finite-difference probe on Mironov.
    # Detach the anchor so the probe does not create second-order gradients.
    perturbation = 1.0e-3
    m_v_probe = m_v_physics.detach() + perturbation
    eps_probe = ground_epsilon_mironov_torch(m_v_probe)
    eps_base_detached = epsilon_base.detach()
    d_eps = (eps_probe - eps_base_detached) / perturbation
    l_monotonic = torch.relu(-d_eps).mean()

    # L_bounds
    l_bounds = (
        torch.relu(-m_v_final) + torch.relu(m_v_final - PEAT_THETA_SAT)
    ).mean()

    weighted_l_physics = lambda_physics * l_physics
    weighted_l_monotonic = lambda_monotonic * l_monotonic
    weighted_l_bounds = lambda_bounds * l_bounds

    total = l_data + weighted_l_physics + weighted_l_monotonic + weighted_l_bounds

    return {
        "total": total,
        "l_data": l_data,
        "l_physics": l_physics,
        "l_physics_vv": l_physics_vv,
        "l_physics_vh": l_physics_vh,
        "l_monotonic": l_monotonic,
        "l_bounds": l_bounds,
        "weighted_l_physics": weighted_l_physics,
        "weighted_l_monotonic": weighted_l_monotonic,
        "weighted_l_bounds": weighted_l_bounds,
    }


# ─── Phase 1c-Lean per-channel normalised composite (SPEC §18.4.1 v0.3) ──────


def compute_init_sigma_normalisers(
    model: "PinnMimics",
    X_train: torch.Tensor,
    theta_inc_deg_train: torch.Tensor,
    vv_db_train: torch.Tensor,
    vh_db_train: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute σ_VV and σ_VH at training initialisation per SPEC §18.4.1 (v0.3).

    σ_VV / σ_VH are the standard deviations of the unweighted per-sample
    physics losses (squared backscatter errors in dB²) over the training set's
    first forward pass at randomly-initialised network weights:

        ε_vv[i] = (σ°_VV_pred[i] − VV_obs[i])²
        σ_VV   = std_pop(ε_vv)            (population std, ddof=0)

    Population std (`unbiased=False`) is used so the Arm 3 invariant
    `std(ε_vv / σ_VV) == 1` is exact (a sample-std σ would leave a residual
    factor of √(n / (n−1)) on the normalised series).

    Per-batch normalisation is explicitly out of scope (SPEC §18.4.1): σ values
    are computed once here and treated as constants for the duration of
    training. They are saved to the model checkpoint and reproduced verbatim
    in the pre-flight summary block (SPEC §18.11 schema item 3).

    Args:
        model:                Initialised `PinnMimics` instance (any random seed).
        X_train:              Normalised feature matrix (N, n_features).
        theta_inc_deg_train:  Per-sample incidence angles (N,) in degrees.
        vv_db_train:          Observed VV backscatter (N,) in dB, unnormalised.
        vh_db_train:          Observed VH backscatter (N,) in dB, unnormalised.

    Returns:
        Dict with keys `sigma_vv`, `sigma_vh` — Python floats. Both are
        guaranteed nonzero finite floats (the function raises ValueError
        otherwise; a zero σ would corrupt the divisor in the normalised
        composite and is treated as a hard implementation-gate failure).

    Reference:
        SPEC.md §18.4.1 (per-channel normalisation specification, v0.3)
        SPEC.md §18.6.1 Arm 3 (scale-sanity invariant)
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(X_train, theta_inc_deg_train, vv_db_train)
            sq_err_vv = (outputs["sigma_vv_db"] - vv_db_train) ** 2
            sq_err_vh = (outputs["sigma_vh_db"] - vh_db_train) ** 2
            sigma_vv = sq_err_vv.std(unbiased=False).item()
            sigma_vh = sq_err_vh.std(unbiased=False).item()
    finally:
        if was_training:
            model.train()

    if not (math.isfinite(sigma_vv) and sigma_vv > 0.0):
        raise ValueError(
            f"σ_VV must be a positive finite float; got {sigma_vv!r}. "
            "SPEC §18.4.1 forbids σ-floor / clamp; a zero-or-NaN σ_VV at init "
            "is a hard G2-Lean implementation gate failure."
        )
    if not (math.isfinite(sigma_vh) and sigma_vh > 0.0):
        raise ValueError(
            f"σ_VH must be a positive finite float; got {sigma_vh!r}. "
            "SPEC §18.4.1 forbids σ-floor / clamp; a zero-or-NaN σ_VH at init "
            "is a hard G2-Lean implementation gate failure."
        )

    return {"sigma_vv": sigma_vv, "sigma_vh": sigma_vh}


def compute_pinn_mimics_loss_normalised(
    outputs: Dict[str, torch.Tensor],
    m_v_observed: torch.Tensor,
    vv_db_observed: torch.Tensor,
    vh_db_observed: torch.Tensor,
    lambda_vv: Union[float, torch.Tensor],
    lambda_vh: Union[float, torch.Tensor],
    lambda_monotonic: Union[float, torch.Tensor],
    lambda_bounds: Union[float, torch.Tensor],
    sigma_vv: Union[float, torch.Tensor],
    sigma_vh: Union[float, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Phase 1c-Lean per-channel normalised composite loss (SPEC §18.4.1 v0.3).

        L = L_data
            + λ_VV · (L_phys_VV / σ_VV)
            + λ_VH · (L_phys_VH / σ_VH)
            + λ_monotonic · L_monotonic
            + λ_bounds   · L_bounds

    Sibling to `compute_pinn_mimics_loss` (Phase 1b F-2b joint VV+VH formulation).
    The original is left byte-for-byte unchanged so F-2b is reproducible exactly.

    The five-term composite reuses every Phase 1b per-term computation
    verbatim (L_data MSE on m_v, per-pol MSE on σ° in dB, L_monotonic
    finite-difference probe on Mironov ε, L_bounds [0, θ_sat] hinge); the
    only intervention is per-channel σ-normalisation of the two physics
    terms with separate λ_VV / λ_VH weights. λ_data ≡ 1.0 is fixed; only
    λ_VV and λ_VH are tuned in the Phase 1c-Lean 6×6=36 grid (§18.4.2).
    λ_monotonic and λ_bounds are fixed at 0.01 each per §18.4.1, but are
    accepted as arguments here so the function remains general for the
    G2-Lean Arm 2 finite-difference gradient check.

    σ_VV and σ_VH are computed once at training initialisation by
    `compute_init_sigma_normalisers` and treated as constants in the
    composite. Per-batch re-normalisation is explicitly out of scope
    (SPEC §18.4.1): a moving-target normaliser conflates the magnitude-
    balance question with optimiser dynamics.

    Args:
        outputs:           Output dict from `PinnMimics.forward()`.
        m_v_observed:      Ground-truth VWC (N,) in cm³/cm³.
        vv_db_observed:    Observed VV backscatter (N,) in dB, unnormalised.
        vh_db_observed:    Observed VH backscatter (N,) in dB, unnormalised.
        lambda_vv:         Weight on the σ-normalised VV physics term (§18.4.2).
        lambda_vh:         Weight on the σ-normalised VH physics term (§18.4.2).
        lambda_monotonic:  Weight on L_monotonic (fixed at 0.01 in production
                           Phase 1c-Lean per §18.4.1).
        lambda_bounds:     Weight on L_bounds (fixed at 0.01 in production
                           Phase 1c-Lean per §18.4.1).
        sigma_vv:          Per-channel σ for VV (positive scalar).
        sigma_vh:          Per-channel σ for VH (positive scalar).

    Returns:
        Dict of scalar tensors. The unnormalised `l_physics_vv`, `l_physics_vh`,
        `l_physics`, `l_data`, `l_monotonic`, `l_bounds` are echoed verbatim
        from the Phase 1b loss for diagnostic continuity (§18.9 amplitude-
        residual ratio depends on the unnormalised forms). Phase 1c-Lean
        additions: `l_physics_vv_normalised`, `l_physics_vh_normalised`,
        `weighted_l_physics_vv_normalised`, `weighted_l_physics_vh_normalised`,
        `weighted_l_physics_normalised` (their sum), and the σ values echoed
        for result-JSON logging (§18.11 item 6).

    Reference:
        SPEC.md §18.4.1 (per-channel normalisation, v0.3)
        SPEC.md §18.6.1 (G2-Lean three-arm equivalence gate)
        SPEC.md §18.11 (result-JSON schema items 3 and 6)
    """
    m_v_final = outputs["m_v_final"]
    m_v_physics = outputs["m_v_physics"]
    sigma_vv_db = outputs["sigma_vv_db"]
    sigma_vh_db = outputs["sigma_vh_db"]
    epsilon_base = outputs["epsilon_ground"]

    # L_data — unchanged from Phase 1b.
    l_data = torch.nn.functional.mse_loss(m_v_final, m_v_observed)

    # L_physics_VV / L_physics_VH — unchanged per-pol MSE from Phase 1b.
    # Joint sum retained as `l_physics` for diagnostic continuity with
    # Phase 1b §18.9 amplitude-residual ratio reporting.
    l_physics_vv = torch.nn.functional.mse_loss(sigma_vv_db, vv_db_observed)
    l_physics_vh = torch.nn.functional.mse_loss(sigma_vh_db, vh_db_observed)
    l_physics = l_physics_vv + l_physics_vh

    # Per-channel σ-normalised physics terms (the v0.3 single intervention).
    l_physics_vv_normalised = l_physics_vv / sigma_vv
    l_physics_vh_normalised = l_physics_vh / sigma_vh

    # L_monotonic — finite-difference Mironov probe; unchanged from Phase 1b.
    perturbation = 1.0e-3
    m_v_probe = m_v_physics.detach() + perturbation
    eps_probe = ground_epsilon_mironov_torch(m_v_probe)
    eps_base_detached = epsilon_base.detach()
    d_eps = (eps_probe - eps_base_detached) / perturbation
    l_monotonic = torch.relu(-d_eps).mean()

    # L_bounds — unchanged from Phase 1b.
    l_bounds = (
        torch.relu(-m_v_final) + torch.relu(m_v_final - PEAT_THETA_SAT)
    ).mean()

    weighted_l_physics_vv_normalised = lambda_vv * l_physics_vv_normalised
    weighted_l_physics_vh_normalised = lambda_vh * l_physics_vh_normalised
    weighted_l_physics_normalised = (
        weighted_l_physics_vv_normalised + weighted_l_physics_vh_normalised
    )
    weighted_l_monotonic = lambda_monotonic * l_monotonic
    weighted_l_bounds = lambda_bounds * l_bounds

    total = (
        l_data
        + weighted_l_physics_normalised
        + weighted_l_monotonic
        + weighted_l_bounds
    )

    return {
        "total": total,
        "l_data": l_data,
        "l_physics": l_physics,
        "l_physics_vv": l_physics_vv,
        "l_physics_vh": l_physics_vh,
        "l_physics_vv_normalised": l_physics_vv_normalised,
        "l_physics_vh_normalised": l_physics_vh_normalised,
        "l_monotonic": l_monotonic,
        "l_bounds": l_bounds,
        "weighted_l_physics_vv_normalised": weighted_l_physics_vv_normalised,
        "weighted_l_physics_vh_normalised": weighted_l_physics_vh_normalised,
        "weighted_l_physics_normalised": weighted_l_physics_normalised,
        "weighted_l_monotonic": weighted_l_monotonic,
        "weighted_l_bounds": weighted_l_bounds,
        "sigma_vv": torch.as_tensor(sigma_vv, dtype=l_data.dtype),
        "sigma_vh": torch.as_tensor(sigma_vh, dtype=l_data.dtype),
    }
