"""
Oh (1992) soil surface backscatter with learnable RMS height `s`.

Phase 1b variant of the Oh model: `s` (surface RMS height in cm) is a
learnable parameter, not a fixed constant. This is the change mandated
by `SPEC.md` §7:

    Phase 1 used the Oh (1992) empirical surface scattering model with
    fixed roughness ks = 0.30. Phase 1b retains Oh as the primary
    surface model, but with surface roughness s as a learnable
    parameter (range 1–5 cm) rather than a fixed value, because peat
    surface roughness at Moor House (Sphagnum hummock-hollow
    microtopography) is substantially larger than the mineral-soil
    regime Oh was calibrated on.

The forward pass is kept mathematically identical to
`phase1/physics/wcm.py::oh_soil_backscatter` at the level of the Oh
empirical equation. The only difference is where `ks` comes from:

    Phase 1:  ks = KS_ROUGHNESS (fixed constant, dimensionless)
    Phase 1b: ks = s · k        (derived from learnable s in cm and the
                                 C-band wavenumber)

Cross-polarisation (VH) is supported via the simplified Oh cross-pol
ratio; see `oh_soil_backscatter_dual_pol` below.

References:
    Oh et al. (1992), IEEE TGRS 30(2), pp. 370–381 — the empirical model
    Oh (2004), IEEE TGRS 42(3), pp. 596–601 — cross-pol extension
    SPEC.md (Phase 1b) §7 — learnable-s requirement

Gradient flow:
    All clamps are implemented with `.clamp()` rather than hard masks so
    gradient flow is preserved for the learnable `s` parameter. The
    ε-clamp at 1.01 is identical to the Phase 1 clamp inherited from
    DEV-007.
"""

from __future__ import annotations

import math

import torch

# C-band Sentinel-1 centre frequency from SPEC.md §5 fixed sensor parameters:
#   f = 5.405 GHz  →  λ = c/f ≈ 5.55 cm  →  k = 2π/λ ≈ 1.132 cm⁻¹
C_LIGHT_CM_PER_S = 2.99792458e10
SENTINEL1_FREQ_HZ = 5.405e9
SENTINEL1_LAMBDA_CM = C_LIGHT_CM_PER_S / SENTINEL1_FREQ_HZ  # ≈ 5.547 cm
SENTINEL1_K_PER_CM = 2.0 * math.pi / SENTINEL1_LAMBDA_CM    # ≈ 1.132 cm⁻¹

# Sigmoid-bounded s range from SPEC.md §5 learnable-parameter table:
#   s ∈ [1, 5] cm (Sphagnum hummock-hollow microtopography)
S_MIN_CM = 1.0
S_MAX_CM = 5.0

# ε clamp inherited from DEV-007 via SPEC §6. Soft clamp via `.clamp()` —
# gradient is zero below the floor, which is the intended behaviour.
OH_EPSILON_MIN = 1.01


def s_to_ks(s_cm: torch.Tensor | float) -> torch.Tensor:
    """
    Convert RMS height `s` (cm) to the Oh ks product at Sentinel-1 C-band.

    Args:
        s_cm: RMS height in cm. Tensor or scalar.

    Returns:
        ks: Dimensionless Oh roughness × wavenumber product.
            For s ∈ [1, 5] cm at C-band, ks spans ≈ [1.13, 5.66].

    Reference:
        Lit review §7 (Decision 4) flags that this ks range is partially
        outside Oh's nominal validity envelope (originally 0.1–6.0 on
        the calibration dataset, practically reliable over a narrower
        span). G3 `ks_validity_check.py` is the pre-experiment check
        for this.
    """
    if not torch.is_tensor(s_cm):
        s_cm = torch.as_tensor(s_cm, dtype=torch.float32)
    return s_cm * SENTINEL1_K_PER_CM


def oh_soil_backscatter_vv(
    epsilon: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    s_cm: torch.Tensor,
) -> torch.Tensor:
    """
    Oh (1992) soil surface VV backscatter, expressed in dB.

    The empirical form is the same Fresnel-times-roughness-scaling used
    in `phase1/physics/wcm.py::oh_soil_backscatter`. See that function's
    docstring for the equation derivation and the ε / sqrt / log
    numerical-stability notes. This function only differs in that
    `ks` is derived from a learnable `s` via `s_to_ks`.

    Args:
        epsilon:       Soil dielectric constant, shape (...). Dimensionless.
        theta_inc_rad: Incidence angle, shape (...). Radians.
        s_cm:          Surface RMS height, scalar or broadcastable to the
                       shape of `epsilon`. Cm.

    Returns:
        sigma_vv_db:   Soil VV backscatter, shape (...). dB.

    Reference:
        Oh et al. (1992), IEEE TGRS 30(2), simplified form.
    """
    ks = s_to_ks(s_cm)

    # DEV-007 inheritance: clamp ε to avoid sqrt of a negative when
    # Mironov produces ε < 1 at low m_v.
    epsilon_safe = epsilon.clamp(min=OH_EPSILON_MIN)

    cos_theta = torch.cos(theta_inc_rad)
    sin_theta = torch.sin(theta_inc_rad)

    # Fresnel reflectance, h-pol:
    #   Γ_h = |(cos θ − √(ε − sin²θ)) / (cos θ + √(ε − sin²θ))|²
    inner = (epsilon_safe - sin_theta.pow(2)).clamp(min=1e-6)
    sqrt_inner = inner.sqrt()
    gamma_h = ((cos_theta - sqrt_inner) / (cos_theta + sqrt_inner + 1e-8)).pow(2)

    # Oh roughness correction (simplified co-pol VV approximation):
    #   σ°_soil_linear = (ks^0.1 / 3) · (cos θ)^2.2 · Γ_h(ε, θ)
    sigma_linear = (ks.pow(0.1) / 3.0) * cos_theta.pow(2.2) * gamma_h
    sigma_db = 10.0 * torch.log10(sigma_linear + 1e-10)
    return sigma_db


def oh_cross_pol_ratio_db(
    theta_inc_rad: torch.Tensor,
    s_cm: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """
    Oh (1992 / 2004) VH/VV cross-pol ratio, in dB (i.e. σ°_VH − σ°_VV).

    The simplified Oh cross-pol form, preserving the sign convention
    that VH ≤ VV for soil surfaces at these incidence angles:

        q = σ°_VH / σ°_VV = 0.23 · √(Γ₀) · (1 − exp(−ks))

    where Γ₀ is the Fresnel reflectance at nadir (zero-angle horizontal).
    In dB:
        σ°_VH − σ°_VV = 10·log10(q)

    This is a *first-order* cross-pol correction at the Oh surface
    layer only — the MIMICS canopy module provides the bulk of the VH
    signal through cross-pol volume scattering in the crown. See
    `SPEC.md` §4 "Scattering mechanisms retained" for how VH is
    ultimately composed.

    Args:
        theta_inc_rad: Incidence angle. Radians.
        s_cm:          Surface RMS height. Cm.
        epsilon:       Soil dielectric. Dimensionless.

    Returns:
        vh_minus_vv_db: The ratio σ°_VH / σ°_VV expressed in dB,
                       i.e. σ°_VH − σ°_VV. Always ≤ 0 for physical
                       inputs. Shape is the broadcast of the inputs.

    Reference:
        Oh (2004), IEEE TGRS 42(3), eqs. 5–6 (simplified form).
    """
    ks = s_to_ks(s_cm)

    epsilon_safe = epsilon.clamp(min=OH_EPSILON_MIN)

    # Γ₀ = |(1 − √ε) / (1 + √ε)|²  (Fresnel at nadir, h-pol)
    sqrt_eps = epsilon_safe.sqrt()
    gamma_0 = ((1.0 - sqrt_eps) / (1.0 + sqrt_eps)).pow(2)

    # q = 0.23 · √(Γ₀) · (1 − exp(−ks))
    #   ks always positive for s, k > 0; clamp the factor below 1 − 1e-12
    #   to keep log10(q) finite even if ks blows up upward.
    factor = (1.0 - torch.exp(-ks)).clamp(max=1.0 - 1e-12)
    q = 0.23 * gamma_0.sqrt() * factor

    # q is strictly positive for physical inputs; clamp below at 1e-10
    # for log-safety only.
    q_safe = q.clamp(min=1e-10)
    return 10.0 * torch.log10(q_safe)


def oh_soil_backscatter_dual_pol(
    epsilon: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    s_cm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute σ°_soil in both VV and VH polarisations.

    σ°_VV is produced by `oh_soil_backscatter_vv`. σ°_VH is produced
    by adding the Oh cross-pol ratio (in dB) to σ°_VV.

    This is the dual-pol interface that SPEC.md §4 and §8 expect from
    the soil-layer contribution to MIMICS.

    Args:
        epsilon:       Soil dielectric.
        theta_inc_rad: Incidence angle in radians.
        s_cm:          Surface RMS height in cm.

    Returns:
        (sigma_vv_db, sigma_vh_db).
    """
    sigma_vv_db = oh_soil_backscatter_vv(epsilon, theta_inc_rad, s_cm)
    vh_minus_vv_db = oh_cross_pol_ratio_db(theta_inc_rad, s_cm, epsilon)
    sigma_vh_db = sigma_vv_db + vh_minus_vv_db
    return sigma_vv_db, sigma_vh_db
