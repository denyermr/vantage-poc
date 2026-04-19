"""
Phase 1 Water Cloud Model physics branch + WCM-PINN wrapper.

This module contains:
  - WCM forward model functions (Oh backscatter, vegetation terms, total backscatter)
  - PINN architecture combining the PhysicsNet/CorrectionNet backbone with the WCM
  - Composite WCM PINN loss function

The WCM (Water Cloud Model) forward model is a differentiable computation graph
that maps VWC → dielectric constant → soil backscatter → total backscatter.
All operations use torch tensors for autograd compatibility.

PhysicsNet and CorrectionNet were extracted into `shared/pinn_backbone.py`
during the Phase 1b repository re-layout so that the Phase 1b MIMICS branch
can reuse the identical backbone (`SPEC.md` §2).

References:
    Oh et al. (1992), IEEE TGRS 30(2) — simplified soil backscatter model
    Attema & Ulaby (1978), Radio Science 13(2) — Water Cloud Model
    Singh et al. (2023), Remote Sensing 15(4) — C-band application
"""

import logging
import math

import torch
import torch.nn as nn

from shared.config import (
    FEATURE_COLUMNS,
    KS_ROUGHNESS,
    PEAT_THETA_SAT,
    WCM_A_INIT,
    WCM_A_LB,
    WCM_A_UB,
    WCM_B_INIT,
    WCM_B_LB,
    WCM_B_UB,
)
from shared.pinn_backbone import CorrectionNet, PhysicsNet

from phase1.physics.dielectric import DielectricModel, DobsonDielectric

logger = logging.getLogger(__name__)

# Minimum dielectric constant for Oh model input — prevents NaN from
# ε − sin²θ < 0 when Mironov produces ε < 1.0 (see DEV-007)
OH_EPSILON_MIN = 1.01


# ─── Oh (1992) soil backscatter ─────────────────────────────────────────────


def oh_soil_backscatter(
    epsilon: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    ks: float = KS_ROUGHNESS,
) -> torch.Tensor:
    """
    Simplified Oh (1992) model for soil surface backscatter at C-band VV.

    Computes Fresnel power reflectance at horizontal polarisation,
    then applies Oh roughness scaling.

    Fresnel reflectance (h-pol):
        Γ_h = |( cos θ - sqrt(ε - sin²θ) ) / ( cos θ + sqrt(ε - sin²θ) )|²

    Oh roughness correction (simplified, co-pol VV approximation):
        σ°_soil_linear = (ks^0.1 / 3) · (cos θ)^2.2 · Γ_h(ε, θ)

    Then convert to dB: σ°_soil_dB = 10 · log10(σ°_soil_linear + 1e-10)

    Args:
        epsilon:       Soil dielectric constant, shape (...). Dimensionless.
                       Clamped to min OH_EPSILON_MIN (1.01) to prevent NaN
                       when Mironov produces ε < 1.0 (see DEV-007).
        theta_inc_rad: Incidence angle in radians, shape (...).
        ks:            Surface roughness parameter (fixed = KS_ROUGHNESS = 0.30).

    Returns:
        sigma_soil_db: Soil backscatter in dB, shape (...).

    Reference:
        Oh et al. (1992), IEEE TGRS 30(2), simplified form
        Singh et al. (2023), Remote Sensing 15(4) — C-band application

    Numerical stability:
        - ε clamped to >= 1.01 to ensure ε - sin²θ > 0 (DEV-007)
        - sqrt argument clamped to > 1e-6 to prevent NaN in autograd
        - log argument offset by 1e-10 to prevent log(0)
    """
    # Clamp epsilon to physical minimum (see DEV-007 for Mironov ε < 1.0 issue)
    n_clamped = (epsilon < OH_EPSILON_MIN).sum().item()
    if n_clamped > 0:
        logger.warning(
            f"Oh backscatter: {n_clamped} ε values below {OH_EPSILON_MIN} "
            f"(min={epsilon.min().item():.4f}) — clamping to {OH_EPSILON_MIN}. "
            f"This occurs when Mironov dielectric produces ε < 1.0 (see DEV-007)."
        )
    epsilon_safe = epsilon.clamp(min=OH_EPSILON_MIN)

    cos_theta = torch.cos(theta_inc_rad)
    sin_theta = torch.sin(theta_inc_rad)

    # Fresnel reflectance (h-pol)
    inner = (epsilon_safe - sin_theta.pow(2)).clamp(min=1e-6)
    sqrt_inner = inner.sqrt()
    gamma_h = ((cos_theta - sqrt_inner) / (cos_theta + sqrt_inner + 1e-8)).pow(2)

    # Oh roughness scaling
    sigma_linear = (ks ** 0.1 / 3.0) * cos_theta.pow(2.2) * gamma_h
    sigma_db = 10.0 * torch.log10(sigma_linear + 1e-10)
    return sigma_db


# ─── WCM vegetation terms ───────────────────────────────────────────────────


def wcm_vegetation_terms(
    A: torch.Tensor,
    B: torch.Tensor,
    ndvi: torch.Tensor,
    theta_inc_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute WCM vegetation direct scattering and two-way transmissivity.

    Equations (Attema & Ulaby, 1978, Radio Science 13(2)):
        σ°_veg = A · NDVI · cos(θ)
        τ²     = exp(−2 · B · NDVI / cos(θ))

    Args:
        A:             Vegetation scattering coefficient (learnable, bounded).
        B:             Vegetation attenuation coefficient (learnable, bounded).
        ndvi:          Normalised Difference Vegetation Index, shape (...).
        theta_inc_rad: Incidence angle in radians, shape (...).

    Returns:
        sigma_veg:     Vegetation direct backscatter (linear, not dB), shape (...).
        tau_squared:   Two-way transmissivity factor [0, 1], shape (...).

    Reference:
        Attema & Ulaby (1978), Radio Science 13(2), eqs. 5, 7
    """
    cos_theta = torch.cos(theta_inc_rad)
    sigma_veg = A * ndvi * cos_theta
    tau_squared = torch.exp(-2.0 * B * ndvi / (cos_theta + 1e-8))
    return sigma_veg, tau_squared


# ─── WCM total backscatter ──────────────────────────────────────────────────


def wcm_forward(
    m_v: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    ndvi: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    dielectric_model: DielectricModel,
) -> torch.Tensor:
    """
    Full WCM forward pass: m_v → σ°_total_dB.

    σ°_total = σ°_veg + τ² · σ°_soil
    (all in linear before summing, then convert to dB)

    Args:
        m_v:              VWC estimate, shape (...). cm³/cm³.
        A:                WCM vegetation scattering coefficient (already bounded via sigmoid).
        B:                WCM vegetation attenuation coefficient (already bounded via sigmoid).
        ndvi:             NDVI, shape (...).
        theta_inc_rad:    Incidence angle, shape (...). Radians.
        dielectric_model: DobsonDielectric or MironovDielectric.

    Returns:
        sigma_total_db: Predicted total backscatter, dB, shape (...).

    Note:
        σ°_veg is in linear units (Attema & Ulaby formulation).
        σ°_soil is converted from dB before adding. Both terms summed
        in linear, then result converted to dB.

    Reference:
        Attema & Ulaby (1978), Radio Science 13(2), eq. 7
    """
    epsilon = dielectric_model(m_v)
    sigma_soil_db = oh_soil_backscatter(epsilon, theta_inc_rad)
    sigma_soil_linear = 10.0 ** (sigma_soil_db / 10.0)

    sigma_veg, tau_sq = wcm_vegetation_terms(A, B, ndvi, theta_inc_rad)

    sigma_total_linear = sigma_veg + tau_sq * sigma_soil_linear
    sigma_total_db = 10.0 * torch.log10(sigma_total_linear.clamp(min=1e-10))
    return sigma_total_db


# ─── PINN architecture ──────────────────────────────────────────────────────
# PhysicsNet and CorrectionNet now live in `shared/pinn_backbone.py` and are
# imported at module top. This module wires them up with the WCM forward pass.


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for SAR-based soil moisture retrieval.

    Combines a physics sub-network (PhysicsNet) and an ML correction sub-network
    (CorrectionNet) through the WCM forward model. The physics branch produces a
    physically-constrained VWC estimate; the correction branch learns residuals.

    WCM parameters A and B are jointly learned via sigmoid reparameterisation
    to enforce bounds [WCM_A_LB, WCM_A_UB] and [WCM_B_LB, WCM_B_UB].

    Architecture:
        X → physics_net → m_v_physics → WCM_forward → σ°_wcm
        X → correction_net → δ
        m_v_final = m_v_physics + δ

    Reference:
        SPEC_PHASE3.md §P3.5
        Attema & Ulaby (1978), Radio Science 13(2) — WCM forward model
    """

    def __init__(
        self,
        n_features: int = len(FEATURE_COLUMNS),
        dielectric_model: DielectricModel | None = None,
    ):
        """
        Args:
            n_features:      Number of input features. Default: len(FEATURE_COLUMNS) = 7.
            dielectric_model: DielectricModel instance. Default: DobsonDielectric().
        """
        super().__init__()
        self.dielectric = dielectric_model or DobsonDielectric()

        self.physics_net = PhysicsNet(n_features)
        self.correction_net = CorrectionNet(n_features)

        # WCM learnable parameters — stored as unconstrained raw values.
        # Bounded at forward pass via sigmoid reparameterisation.
        # Initialise raw values as logit of (init - LB) / (UB - LB).
        A_raw_init = math.log(
            (WCM_A_INIT - WCM_A_LB) / (WCM_A_UB - WCM_A_INIT + 1e-8)
        )
        B_raw_init = math.log(
            (WCM_B_INIT - WCM_B_LB) / (WCM_B_UB - WCM_B_INIT + 1e-8)
        )
        self.A_raw = nn.Parameter(torch.tensor(A_raw_init, dtype=torch.float32))
        self.B_raw = nn.Parameter(torch.tensor(B_raw_init, dtype=torch.float32))

    @property
    def A(self) -> torch.Tensor:
        """Bounded WCM A parameter. Always in [WCM_A_LB, WCM_A_UB]."""
        return WCM_A_LB + (WCM_A_UB - WCM_A_LB) * torch.sigmoid(self.A_raw)

    @property
    def B(self) -> torch.Tensor:
        """Bounded WCM B parameter. Always in [WCM_B_LB, WCM_B_UB]."""
        return WCM_B_LB + (WCM_B_UB - WCM_B_LB) * torch.sigmoid(self.B_raw)

    def forward(
        self,
        X: torch.Tensor,
        ndvi: torch.Tensor,
        theta_inc_rad: torch.Tensor,
        vv_db_observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full PINN forward pass.

        Returns a dict rather than a single tensor — all quantities needed
        for loss computation and diagnostics.

        Args:
            X:               Full feature matrix (N, n_features), normalised.
            ndvi:            NDVI values (N,) — extracted from X for WCM clarity.
            theta_inc_rad:   Incidence angles (N,) in radians.
            vv_db_observed:  Observed VV backscatter (N,) in dB (unnormalised).

        Returns dict with keys:
            m_v_physics:     Physics sub-network output (N,). cm³/cm³.
            delta_ml:        Correction sub-network output (N,). cm³/cm³.
            m_v_final:       m_v_physics + delta_ml (N,). cm³/cm³.
            sigma_wcm_db:    WCM forward pass on m_v_physics (N,). dB.
            epsilon:         Dielectric constant at m_v_physics (N,). For monotonic loss.
            A_current:       Current WCM A value (scalar).
            B_current:       Current WCM B value (scalar).
        """
        m_v_physics = self.physics_net(X)
        delta_ml = self.correction_net(X)
        m_v_final = m_v_physics + delta_ml

        sigma_wcm_db = wcm_forward(
            m_v_physics, self.A, self.B,
            ndvi, theta_inc_rad, self.dielectric,
        )
        epsilon = self.dielectric(m_v_physics)

        return {
            "m_v_physics": m_v_physics,
            "delta_ml": delta_ml,
            "m_v_final": m_v_final,
            "sigma_wcm_db": sigma_wcm_db,
            "epsilon": epsilon,
            "A_current": self.A,
            "B_current": self.B,
        }


# ─── Composite loss function ────────────────────────────────────────────────


def compute_pinn_loss(
    outputs: dict[str, torch.Tensor],
    m_v_observed: torch.Tensor,
    vv_db_observed: torch.Tensor,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    dielectric_model: DielectricModel | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute composite PINN loss and all component terms.

    L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds

    Loss term formulas:
        L_data      = MSE(m_v_final, m_v_observed)
        L_physics   = MSE(sigma_wcm_db, vv_db_observed)
        L_monotonic = mean(ReLU(−dε/dm_v))  (finite difference probe)
        L_bounds    = mean(ReLU(−m_v_final) + ReLU(m_v_final − PEAT_THETA_SAT))

    Args:
        outputs:          Dict from PINN.forward().
        m_v_observed:     Ground truth VWC (N,). cm³/cm³.
        vv_db_observed:   Observed VV backscatter (N,). dB (unnormalised).
        lambda1:          Weight for L_physics.
        lambda2:          Weight for L_monotonic.
        lambda3:          Weight for L_bounds.
        dielectric_model: DielectricModel for monotonic probe. Default: DobsonDielectric().

    Returns dict with keys:
        total:       Scalar total loss.
        l_data:      MSE(m_v_final, m_v_observed). Scalar.
        l_physics:   MSE(sigma_wcm_db, vv_db_observed). Scalar.
        l_monotonic: Physics constraint penalty. Scalar.
        l_bounds:    Bounds constraint penalty. Scalar.

    Reference:
        SPEC_PHASE3.md §P3.6
    """
    if dielectric_model is None:
        dielectric_model = DobsonDielectric()

    m_v_final = outputs["m_v_final"]
    sigma_wcm_db = outputs["sigma_wcm_db"]
    m_v_physics = outputs["m_v_physics"]
    epsilon_base = outputs["epsilon"]

    # L_data: MSE between predicted and observed VWC
    l_data = torch.nn.functional.mse_loss(m_v_final, m_v_observed)

    # L_physics: MSE between WCM-predicted and observed VV backscatter
    l_physics = torch.nn.functional.mse_loss(sigma_wcm_db, vv_db_observed)

    # L_monotonic: enforce dε/dm_v > 0 via finite difference probe
    # Probe at m_v_physics + small perturbation, detached to avoid second-order gradients
    perturbation = 1e-3
    m_v_probe = m_v_physics.detach() + perturbation
    eps_probe = dielectric_model(m_v_probe)
    eps_base_detached = epsilon_base.detach()
    d_eps = (eps_probe - eps_base_detached) / perturbation
    l_monotonic = torch.relu(-d_eps).mean()

    # L_bounds: penalise m_v_final outside [0, PEAT_THETA_SAT]
    l_bounds = (
        torch.relu(-m_v_final) + torch.relu(m_v_final - PEAT_THETA_SAT)
    ).mean()

    # Total composite loss
    total = l_data + lambda1 * l_physics + lambda2 * l_monotonic + lambda3 * l_bounds

    # Dominance constraint check (warning only, does not raise)
    physics_terms_sum = (
        lambda1 * l_physics + lambda2 * l_monotonic + lambda3 * l_bounds
    )
    if physics_terms_sum.item() > l_data.item():
        logger.warning(
            f"LAMBDA_DOMINANCE_CONSTRAINT: physics terms ({physics_terms_sum.item():.6f}) "
            f"> L_data ({l_data.item():.6f}). λ triple may need adjustment."
        )

    return {
        "total": total,
        "l_data": l_data,
        "l_physics": l_physics,
        "l_monotonic": l_monotonic,
        "l_bounds": l_bounds,
    }
