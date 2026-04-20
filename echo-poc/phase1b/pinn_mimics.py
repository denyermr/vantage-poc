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

Composite loss (SPEC §8):
    L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds

    L_data      = MSE(m_v_final, m_v_observed)
    L_physics   = MSE(σ°_VV_MIMICS, VV_observed)      (co-pol only, VH is
                                                       evaluated post-hoc
                                                       as Secondary 3)
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
from typing import Dict

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
    lambda_physics: float,
    lambda_monotonic: float,
    lambda_bounds: float,
) -> Dict[str, torch.Tensor]:
    """
    Composite loss for PINN-MIMICS.

        L = L_data + λ_physics · L_physics + λ_mono · L_monotonic + λ_bounds · L_bounds

    The L_data coefficient is fixed at 1.0 per SPEC §8; it is NOT a search
    axis. The three λ's are the Phase 1b λ-search axes per SPEC §9.

    Args:
        outputs:             Output dict from `PinnMimics.forward()`.
        m_v_observed:        Ground-truth VWC (N,) in cm³/cm³.
        vv_db_observed:      Observed VV backscatter (N,) in dB, unnormalised.
        lambda_physics:      Weight for L_physics (SPEC §9 λ₁).
        lambda_monotonic:    Weight for L_monotonic (SPEC §9 λ₂).
        lambda_bounds:       Weight for L_bounds (SPEC §9 λ₃).

    Returns:
        dict with keys (all scalar tensors):
            total:                     L.
            l_data:                    MSE(m_v_final, m_v_observed).
            l_physics:                 MSE(σ°_VV_MIMICS, VV_observed).
            l_monotonic:               mean(ReLU(−dε/dm_v)).
            l_bounds:                  mean(ReLU(−m_v) + ReLU(m_v − θ_sat)).
            weighted_l_physics:        λ_physics · L_physics.
            weighted_l_monotonic:      λ_monotonic · L_monotonic.
            weighted_l_bounds:         λ_bounds · L_bounds.

    Reference:
        SPEC.md §8.
    """
    m_v_final = outputs["m_v_final"]
    m_v_physics = outputs["m_v_physics"]
    sigma_vv_db = outputs["sigma_vv_db"]
    epsilon_base = outputs["epsilon_ground"]

    # L_data
    l_data = torch.nn.functional.mse_loss(m_v_final, m_v_observed)

    # L_physics — VV co-pol only (VH is Secondary 3 post-hoc)
    l_physics = torch.nn.functional.mse_loss(sigma_vv_db, vv_db_observed)

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
        "l_monotonic": l_monotonic,
        "l_bounds": l_bounds,
        "weighted_l_physics": weighted_l_physics,
        "weighted_l_monotonic": weighted_l_monotonic,
        "weighted_l_bounds": weighted_l_bounds,
    }
