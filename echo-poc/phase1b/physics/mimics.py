"""
Differentiable PyTorch port of the Toure single-crown MIMICS forward model.

This module is the training-side counterpart to
`phase1b/physics/reference_mimics/reference_toure.py` (session B). The
two must produce σ° values that agree within 0.5 dB on the canonical
parameter combinations — that check is the G2 Implementation Gate
(`SPEC.md` §4) and is wired in session D via
`phase1b/implementation_gate/equivalence_check.py`. This module is
written to mirror the v0.1 reference exactly; any promotion of the
approximations documented in the reference's "Known limitations (v0.1)"
block must be made in both modules simultaneously (session E territory).

Physics summary:
    First-order MIMICS total backscatter for a single crown above a
    rough peat ground (Toure 1994 adaptation, no trunk layer):

        σ°_total = σ°_crown_direct
                 + σ°_crown_ground
                 + τ² · σ°_ground_direct

    Scatterers: Rayleigh prolate-ellipsoid branches (Gaussian zenith,
    uniform azimuth) and Rayleigh thin-disc leaves (uniform on upper
    hemisphere), with a sinc² backscatter form factor applied to both
    σ_back and σ_ext per the reference's v0.1 simplification. Vegetation
    dielectric uses the simplified real-part-only Ulaby & El-Rayes
    linear proxy; ground dielectric uses Mironov (2009) organic-soil
    GRMDM with the SPEC §6 / DEV-007 ε ≥ 1.01 clamp. The crown-ground
    coupling uses √(σ°_oh) as a roughness-aware ground-reflectance
    proxy (same as the reference).

Differentiability contract (SPEC §4):
    - Gradient flow is preserved through every SPEC §5 learnable
      parameter (`N_b`, `N_l`, `σ_orient`, `m_g`, `s`) and through the
      per-observation `m_v` that the PhysicsNet branch will feed into
      the forward.
    - No `.detach()`, no `.item()` on learnable paths.
    - Piecewise logic uses `torch.where` (both branches differentiable)
      rather than hard numpy masks; the ε_ground ≥ 1.01 clamp is
      inherited in `.clamp()` form from `oh1992_learnable_s` / SPEC §6
      (gradient is exactly zero below the floor, which is the intended
      clamp semantics).
    - The small-κ limit of the crown-direct term is evaluated via the
      numerically stable `-expm1(-x)/x` factor with a Taylor fallback
      near x = 0, so the factor never divides by zero and is smooth
      through the transition.
    - Sentinel-1 geometry, Sphagnum peat ε, and Oh surface scattering
      are reused from `phase1b/physics/oh1992_learnable_s.py`.

Device contract:
    Device is inferred from the first input tensor that carries device
    information, defaulting to `shared.config.get_torch_device()`.
    No device is hardcoded in this module (SPEC §engineering-rules).

Shape contract:
    All learnable / per-observation inputs may be scalar (0-d) tensors
    or broadcastable batched tensors. The orientation quadrature adds
    trailing (n_theta, n_phi) dims that are reduced at the end, so a
    (B,) batch of inputs returns (B,) outputs.

References:
    See reference_toure.py. This module reimplements the same equations
    in differentiable form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from phase1b.physics.oh1992_learnable_s import (
    oh_soil_backscatter_dual_pol,
    OH_EPSILON_MIN,
)

# ─── Constants (mirror reference_toure.py at the float level) ───────────────

C_LIGHT_CM_PER_S = 2.99792458e10
SENTINEL1_FREQ_HZ = 5.405e9

MIRONOV_ND = 0.312
MIRONOV_ND1 = 1.42
MIRONOV_ND2 = 0.89
MIRONOV_MV_T = 0.36

# SPEC §6 / DEV-007 clamp. Also defined in `oh1992_learnable_s.OH_EPSILON_MIN`.
EPSILON_GROUND_MIN = OH_EPSILON_MIN  # 1.01

# Simplified Ulaby & El-Rayes (1987) vegetation dielectric (real part).
UEL_A = 1.7
UEL_B = 18.0

PEAT_THETA_SAT = 0.88

# Dobson 1985 (Bechtold et al. 2018 peat parameterisation, Moor House default).
# Mirrors the frozen `phase1/physics/dielectric.DobsonDielectric` constants so
# that the Phase 1b Moor-House production path remains bit-identical when the
# Dobson form is selected through `ground_dielectric_fn`. Tier 1 is never
# modified; this is the Phase 1b kwargs-parameterisable counterpart.
DOBSON_EPS_DRY_PEAT = 3.5
DOBSON_EPS_WATER = 80.0
DOBSON_ALPHA_PEAT = 1.4

# Dobson 1985 generic mineral-soil parameterisation used by T94 (T94 inherits
# the soil dielectric from MIMICS [19] = Ulaby 1990, which uses Dobson 1985;
# T94 does not override). The α=0.65 exponent is Dobson 1985's empirical value
# for a moderate-clay-fraction mineral loam. Used ONLY by the G2 harness for
# E.1 / E.2 per DEV-1b-004; the Moor House production path uses the peat
# defaults above.
DOBSON_EPS_DRY_MINERAL = 3.0
DOBSON_ALPHA_MINERAL = 0.65

_EPS_LOG = 1e-12

# Default orientation-quadrature sample counts — match `reference_toure.py`.
N_THETA_SAMPLES = 128
N_PHI_SAMPLES = 32


TensorOrFloat = Union[torch.Tensor, float]


# ─── Parameter container ────────────────────────────────────────────────────


@dataclass
class MimicsToureParamsTorch:
    """
    Torch analogue of `reference_toure.MimicsToureParams`.

    Fixed crown geometry / sensor fields are plain Python floats (they
    carry no gradient and do not vary across training). Everything else
    is `TensorOrFloat` so callers may pass either a numeric default or
    a `torch.Tensor` that participates in gradient flow.

    Defaults mirror SPEC §5 fixed values and learnable-range midpoints
    for the Moor House heather canopy.
    """

    # SPEC §5 fixed crown geometry (m, cm — see reference_toure.py).
    h_c_m: float = 0.4
    a_b_cm: float = 0.2
    l_b_cm: float = 8.0
    a_l_cm: float = 1.0
    t_l_cm: float = 0.03
    l_corr_cm: float = 5.0
    freq_hz: float = SENTINEL1_FREQ_HZ

    # SPEC §5 learnable (may be tensors).
    N_b_per_m3: TensorOrFloat = 1.0e3
    N_l_per_m3: TensorOrFloat = 1.0e4
    sigma_orient_deg: TensorOrFloat = 30.0
    m_g: TensorOrFloat = 0.45

    # Ground — m_v is per-observation (PhysicsNet output), s_cm is learnable.
    m_v: TensorOrFloat = 0.5
    s_cm: TensorOrFloat = 2.0

    # Sensor — theta may be per-observation.
    theta_inc_deg: TensorOrFloat = 41.5


# ─── Small helpers ──────────────────────────────────────────────────────────


def _infer_device_dtype(params: MimicsToureParamsTorch) -> Tuple[torch.device, torch.dtype]:
    """
    Pick a device/dtype for intermediate tensors.

    If any input is already a tensor, its device/dtype wins (first match
    in declaration order). Otherwise fall back to
    `shared.config.get_torch_device()` and `torch.float32`.
    """
    for name in (
        "N_b_per_m3", "N_l_per_m3", "sigma_orient_deg", "m_g",
        "m_v", "s_cm", "theta_inc_deg",
    ):
        val = getattr(params, name)
        if torch.is_tensor(val):
            return val.device, val.dtype
    # No tensor inputs: use the project default device.
    from shared import config
    return config.get_torch_device(), torch.float32


def _as_tensor(x: TensorOrFloat, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Promote a scalar or tensor to a tensor on the target device/dtype."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(float(x), device=device, dtype=dtype)


def _sinc_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalised sinc: sin(x)/x, with value 1 at x = 0. Differentiable.

    `torch.sinc` is not implemented on the MPS backend, so we evaluate
    sin(x)/x manually and select 1 at the origin via `torch.where`. Both
    branches are smooth and the select mask is a boolean predicate, so
    gradient flows correctly through either branch.
    """
    small = x.abs() < 1e-8
    # `safe_x` keeps the division path finite on the unselected branch.
    safe_x = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, torch.ones_like(x), torch.sin(x) / safe_x)


def _prolate_l_par(aspect: float) -> float:
    """
    Depolarisation factor along the long axis of a prolate ellipsoid.

    Mirrors `reference_toure._cylinder_rayleigh_polarisabilities`. The
    geometry is fixed (SPEC §5), so this is a pure Python float; no
    gradient involvement.
    """
    if aspect >= 1.0:
        return 1.0 / 3.0
    e = math.sqrt(1.0 - aspect * aspect)
    return (1.0 - e * e) / (e ** 3) * (0.5 * math.log((1.0 + e) / (1.0 - e)) - e)


def _oblate_l_normal(aspect: float) -> float:
    """Depolarisation factor along the disc normal; numpy-reference form."""
    if aspect >= 1.0:
        return 1.0 / 3.0
    e = math.sqrt(1.0 - aspect * aspect)
    return (1.0 / (e * e)) * (1.0 - (math.sqrt(1.0 - e * e) / e) * math.asin(e))


# ─── Dielectrics ────────────────────────────────────────────────────────────


def ground_epsilon_mironov_torch(m_v: torch.Tensor) -> torch.Tensor:
    """
    Real part of the peat ground dielectric via Mironov (2009) GRMDM.

    Piecewise-linear in m_v about m_v_t = 0.36, squared to give ε. The
    SPEC §6 / DEV-007 floor at ε = 1.01 is applied via `.clamp()` —
    identical semantics to `phase1/physics/wcm.oh_soil_backscatter` and
    `phase1b/physics/oh1992_learnable_s`.

    Both branches of the piecewise are computed and selected by
    `torch.where`, so gradient flows on whichever branch is active. Below
    the clamp floor the gradient is zero, as intended.
    """
    n_low = MIRONOV_ND + MIRONOV_ND1 * m_v
    n_high = (
        MIRONOV_ND
        + MIRONOV_ND1 * MIRONOV_MV_T
        + MIRONOV_ND2 * (m_v - MIRONOV_MV_T)
    )
    n = torch.where(m_v <= MIRONOV_MV_T, n_low, n_high)
    eps = n * n
    return eps.clamp(min=EPSILON_GROUND_MIN)


def ground_epsilon_dobson_torch(
    m_v: torch.Tensor,
    eps_dry: float = DOBSON_EPS_DRY_PEAT,
    eps_water: float = DOBSON_EPS_WATER,
    alpha: float = DOBSON_ALPHA_PEAT,
) -> torch.Tensor:
    """
    Real part of the ground dielectric via Dobson et al. (1985).

        ε(m_v) = ε_dry + (ε_water − 1) · m_v^α

    Defaults mirror the Moor-House peat parameterisation used by the
    frozen `phase1/physics/dielectric.DobsonDielectric` class (Bechtold
    et al. 2018 values: ε_dry = 3.5, ε_water = 80.0, α = 1.4). Those
    defaults exist so that selecting Dobson for the Moor House production
    path yields a bit-identical dielectric to the Phase 1 frozen module.

    The `eps_dry` / `eps_water` / `alpha` kwargs are used exclusively by
    the G2 harness for rows E.1 / E.2 per DEV-1b-004 — T94 inherits the
    soil dielectric from MIMICS [Ulaby 1990] which uses Dobson 1985, and
    G2 now evaluates those rows under a T94-consistent mineral-soil
    Dobson configuration (`DOBSON_EPS_DRY_MINERAL`,
    `DOBSON_ALPHA_MINERAL`). The Moor House production path (PINN-MIMICS
    trainer, inference) must never set these kwargs away from their peat
    defaults; a regression test in `tests/unit/test_mimics_torch.py`
    enforces this pinning.

    Numerical stability:
        m_v is clamped to ≥ 0 before exponentiation to prevent NaN from
        negative values raised to a fractional power. The SPEC §6 /
        DEV-007 ε ≥ 1.01 floor is applied via `.clamp()` for parity with
        `ground_epsilon_mironov_torch`; Dobson at m_v = 0 gives ε_dry
        (3.5 for peat, 3.0 for mineral), both above the floor, so the
        clamp is inactive in practice but preserved for safety.

    Reference:
        Dobson et al. (1985), IEEE TGRS GE-23(1), eq. 1.
        Bechtold et al. (2018), Remote Sensing 10(2).
    """
    m_v_safe = m_v.clamp(min=0.0)
    eps = eps_dry + (eps_water - 1.0) * m_v_safe.pow(alpha)
    return eps.clamp(min=EPSILON_GROUND_MIN)


# Callable alias used by `mimics_toure_single_crown`'s optional
# `ground_dielectric_fn` argument — see DEV-1b-004 / SPEC §6.
GroundDielectricFn = Callable[[torch.Tensor], torch.Tensor]


def vegetation_epsilon_ulaby_elrayes_torch(m_g: torch.Tensor) -> torch.Tensor:
    """
    Simplified Ulaby & El-Rayes (1987) vegetation dielectric, real part.

    `ε_v = UEL_A + UEL_B · m_g`. Uses `.clamp()` to bound m_g into
    [0, 1]; outside that range the gradient is zero, but the SPEC §5
    sigmoid-bounded range for m_g is [0.3, 0.6] so the clamp is never
    active in practice.
    """
    m_g_safe = m_g.clamp(min=0.0, max=1.0)
    return UEL_A + UEL_B * m_g_safe


# ─── Rayleigh polarisabilities ──────────────────────────────────────────────


def _cylinder_rayleigh_polarisabilities_torch(
    a_cm: float, l_cm: float, eps_v: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Rayleigh polarisabilities of a prolate dielectric cylinder.

    a_cm, l_cm are fixed geometry (SPEC §5) — depolarisation factors are
    Python floats. eps_v is a tensor and carries gradient through to
    α_par, α_perp.

    Equations match `reference_toure._cylinder_rayleigh_polarisabilities`.
    """
    a_m = a_cm * 1e-2
    l_m = l_cm * 1e-2
    full_length_m = 2.0 * l_m
    V = math.pi * a_m * a_m * full_length_m
    L_par = _prolate_l_par(a_m / l_m)
    L_perp = 0.5 * (1.0 - L_par)
    alpha_par = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_par)
    alpha_perp = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_perp)
    return V, alpha_par, alpha_perp


def _disc_rayleigh_polarisabilities_torch(
    a_cm: float, t_cm: float, eps_v: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Rayleigh polarisabilities of a thin dielectric disc.

    Same structure as `_cylinder_rayleigh_polarisabilities_torch` but
    for an oblate ellipsoid; α_normal along the disc normal, α_face in
    the disc plane.
    """
    a_m = a_cm * 1e-2
    t_m = t_cm * 1e-2
    V = math.pi * a_m * a_m * t_m
    L_normal = _oblate_l_normal(t_m / a_m)
    L_face = 0.5 * (1.0 - L_normal)
    alpha_normal = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_normal)
    alpha_face = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_face)
    return V, alpha_normal, alpha_face


# ─── Orientation-averaged cross sections ────────────────────────────────────


def _branch_cross_sections_torch(
    a_cm: float,
    l_cm: float,
    eps_v: torch.Tensor,
    sigma_orient_deg: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    k_per_m: float,
    n_theta: int,
    n_phi: int,
) -> dict:
    """
    Orientation-averaged backscatter and extinction cross sections for a
    population of Rayleigh cylinders with Gaussian zenith (std
    `sigma_orient_deg`) and uniform azimuth.

    Mirrors `reference_toure._branch_cross_sections`. The Rayleigh
    backscatter amplitudes are computed in closed form from the
    polarisability tensor Π = α_par ĉĉᵀ + α_perp (I − ĉĉᵀ); the sinc²
    form factor is evaluated at the backscatter direction and applied
    uniformly to both σ_back and σ_ext per the v0.1 reference.

    Shapes:
        Tensor inputs may be 0-d (scalar) or (..., B) batched. The
        returned tensors have shape (...) — the orientation dims are
        reduced inside.
    """
    device, dtype = eps_v.device, eps_v.dtype
    _, alpha_par, alpha_perp = _cylinder_rayleigh_polarisabilities_torch(
        a_cm, l_cm, eps_v
    )
    l_m = l_cm * 1e-2

    # Orientation grids (pure geometry — no batch).
    theta_grid = torch.linspace(
        0.0, math.pi / 2.0, n_theta, device=device, dtype=dtype
    )
    # Emulate `np.linspace(..., endpoint=False)` for the azimuth grid.
    phi_grid = torch.linspace(
        0.0, 2.0 * math.pi, n_phi + 1, device=device, dtype=dtype
    )[:-1]
    sin_theta = torch.sin(theta_grid)          # (n_theta,)
    cos_theta_b = torch.cos(theta_grid)        # (n_theta,)
    cos_phi = torch.cos(phi_grid)              # (n_phi,)
    sin_phi = torch.sin(phi_grid)              # (n_phi,)

    # Zenith weight: w(θ_b) = exp(−θ²/(2σ²)) · sin(θ), normalised.
    # σ_orient promoted to (..., 1) so it broadcasts against (n_theta,).
    sigma_rad = sigma_orient_deg * (math.pi / 180.0)
    sigma_rad_for_w = sigma_rad.unsqueeze(-1)                              # (..., 1)
    gaussian_factor = torch.exp(
        -theta_grid.pow(2) / (2.0 * sigma_rad_for_w.pow(2))
    )                                                                      # (..., n_theta)
    w_unnorm = gaussian_factor * sin_theta                                 # (..., n_theta)
    w_norm = torch.trapz(w_unnorm, theta_grid, dim=-1).unsqueeze(-1)       # (..., 1)
    w_theta = w_unnorm / w_norm                                            # (..., n_theta)

    # Unit vector ĉ on the orientation grid (geometry-only, no batch).
    c_x = sin_theta.unsqueeze(-1) * cos_phi                                # (n_theta, n_phi)
    c_y = sin_theta.unsqueeze(-1) * sin_phi
    c_z = cos_theta_b.unsqueeze(-1).expand(n_theta, n_phi)
    c_dot_h = c_y                                                          # (n_theta, n_phi)

    # Incidence-plane trig promoted to (..., 1, 1) for orientation-grid broadcast.
    cos_ti = torch.cos(theta_inc_rad).unsqueeze(-1).unsqueeze(-1)
    sin_ti = torch.sin(theta_inc_rad).unsqueeze(-1).unsqueeze(-1)

    # Dot products. c_x/y/z are (n_theta, n_phi); broadcasting with (..., 1, 1)
    # gives (..., n_theta, n_phi).
    c_dot_v_inc = c_x * cos_ti + c_z * sin_ti
    c_dot_v_sca = -c_x * cos_ti + c_z * sin_ti
    c_dot_k_inc = c_x * sin_ti - c_z * cos_ti
    # ê_v(scatter) · ê_v(incident) = sin²θ_i − cos²θ_i = −cos(2θ_i).
    e_vsca_dot_vinc = -cos_ti * cos_ti + sin_ti * sin_ti                   # (..., 1, 1)

    # Promote polarisabilities with two trailing dims for orientation broadcast.
    alpha_par_b = alpha_par.unsqueeze(-1).unsqueeze(-1)                    # (..., 1, 1)
    alpha_perp_b = alpha_perp.unsqueeze(-1).unsqueeze(-1)
    diff_alpha = alpha_par_b - alpha_perp_b

    # Scattering amplitudes f_pq = ê_p · Π · ê_q:
    #   f_vv = α_perp (ê_v_sca · ê_v_inc) + (α_par − α_perp)(ĉ · ê_v_inc)(ĉ · ê_v_sca)
    #   f_hh = α_perp (ê_h · ê_h) + (α_par − α_perp)(ĉ · ê_h)²
    #   f_vh = (α_par − α_perp)(ĉ · ê_h)(ĉ · ê_v_sca)   [since ê_v_sca · ê_h = 0]
    f_vv = alpha_perp_b * e_vsca_dot_vinc + diff_alpha * c_dot_v_inc * c_dot_v_sca
    f_hh = alpha_perp_b + diff_alpha * c_dot_h * c_dot_h
    f_vh = diff_alpha * c_dot_h * c_dot_v_sca

    # Finite-cylinder sinc² backscatter form factor (v0.1 simplification
    # — see reference_toure "Known limitations"). Applied identically to
    # extinction as a first-pass RGD surrogate.
    form_sq = _sinc_torch(2.0 * k_per_m * l_m * c_dot_k_inc).pow(2)

    k4 = k_per_m ** 4
    sigma_back_vv = 4.0 * math.pi * k4 * f_vv.pow(2) * form_sq
    sigma_back_hh = 4.0 * math.pi * k4 * f_hh.pow(2) * form_sq
    sigma_back_vh = 4.0 * math.pi * k4 * f_vh.pow(2) * form_sq

    # Rayleigh total scattering, per-polarisation:
    #   σ_s^p(ĉ) = (8π/3) k⁴ (α_par² (ĉ·ê_p)² + α_perp² (1 − (ĉ·ê_p)²))
    c_dot_v_sq = c_dot_v_inc.pow(2)
    c_dot_h_sq = c_dot_h * c_dot_h
    ap2 = alpha_par_b.pow(2)
    ape2 = alpha_perp_b.pow(2)
    sigma_ext_v = (8.0 * math.pi / 3.0) * k4 * (
        ap2 * c_dot_v_sq + ape2 * (1.0 - c_dot_v_sq)
    ) * form_sq
    sigma_ext_h = (8.0 * math.pi / 3.0) * k4 * (
        ap2 * c_dot_h_sq + ape2 * (1.0 - c_dot_h_sq)
    ) * form_sq

    def _avg(grid: torch.Tensor) -> torch.Tensor:
        """Average φ uniformly, then weight-integrate θ with w_theta."""
        azi_avg = grid.mean(dim=-1)                                        # (..., n_theta)
        return torch.trapz(azi_avg * w_theta, theta_grid, dim=-1)          # (...,)

    return {
        "sigma_back_vv": _avg(sigma_back_vv),
        "sigma_back_hh": _avg(sigma_back_hh),
        "sigma_back_vh": _avg(sigma_back_vh),
        "sigma_ext_v": _avg(sigma_ext_v),
        "sigma_ext_h": _avg(sigma_ext_h),
    }


def _leaf_cross_sections_torch(
    a_cm: float,
    t_cm: float,
    eps_v: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    k_per_m: float,
    n_theta: int,
    n_phi: int,
) -> dict:
    """
    Orientation-averaged cross sections for a uniformly oriented thin
    disc (leaf) population — no form factor.

    Same structure as `_branch_cross_sections_torch` but with:
      - α_normal / α_face instead of α_par / α_perp (disc normal n̂),
      - uniform orientation on the upper hemisphere (sin(θ) weight),
      - no sinc² form factor (discs are assumed small compared to λ).
    """
    device, dtype = eps_v.device, eps_v.dtype
    _, alpha_normal, alpha_face = _disc_rayleigh_polarisabilities_torch(
        a_cm, t_cm, eps_v
    )

    theta_grid = torch.linspace(
        0.0, math.pi / 2.0, n_theta, device=device, dtype=dtype
    )
    phi_grid = torch.linspace(
        0.0, 2.0 * math.pi, n_phi + 1, device=device, dtype=dtype
    )[:-1]
    sin_theta = torch.sin(theta_grid)
    cos_theta_n = torch.cos(theta_grid)
    cos_phi = torch.cos(phi_grid)
    sin_phi = torch.sin(phi_grid)

    # Uniform-on-sphere zenith weight: w(θ) = sin(θ) / ∫ sin(θ) dθ.
    w_theta = sin_theta / torch.trapz(sin_theta, theta_grid)

    n_x = sin_theta.unsqueeze(-1) * cos_phi
    n_y = sin_theta.unsqueeze(-1) * sin_phi
    n_z = cos_theta_n.unsqueeze(-1).expand(n_theta, n_phi)
    n_dot_h = n_y

    cos_ti = torch.cos(theta_inc_rad).unsqueeze(-1).unsqueeze(-1)
    sin_ti = torch.sin(theta_inc_rad).unsqueeze(-1).unsqueeze(-1)
    n_dot_v_inc = n_x * cos_ti + n_z * sin_ti
    n_dot_v_sca = -n_x * cos_ti + n_z * sin_ti
    e_vsca_dot_vinc = -cos_ti * cos_ti + sin_ti * sin_ti

    alpha_normal_b = alpha_normal.unsqueeze(-1).unsqueeze(-1)
    alpha_face_b = alpha_face.unsqueeze(-1).unsqueeze(-1)
    diff_alpha = alpha_normal_b - alpha_face_b

    f_vv = alpha_face_b * e_vsca_dot_vinc + diff_alpha * n_dot_v_inc * n_dot_v_sca
    f_hh = alpha_face_b + diff_alpha * n_dot_h * n_dot_h
    f_vh = diff_alpha * n_dot_h * n_dot_v_sca

    k4 = k_per_m ** 4
    sigma_back_vv = 4.0 * math.pi * k4 * f_vv.pow(2)
    sigma_back_hh = 4.0 * math.pi * k4 * f_hh.pow(2)
    sigma_back_vh = 4.0 * math.pi * k4 * f_vh.pow(2)

    n_dot_v_sq = n_dot_v_inc.pow(2)
    n_dot_h_sq = n_dot_h * n_dot_h
    an2 = alpha_normal_b.pow(2)
    af2 = alpha_face_b.pow(2)
    sigma_ext_v = (8.0 * math.pi / 3.0) * k4 * (
        an2 * n_dot_v_sq + af2 * (1.0 - n_dot_v_sq)
    )
    sigma_ext_h = (8.0 * math.pi / 3.0) * k4 * (
        an2 * n_dot_h_sq + af2 * (1.0 - n_dot_h_sq)
    )

    def _avg(grid: torch.Tensor) -> torch.Tensor:
        azi_avg = grid.mean(dim=-1)
        return torch.trapz(azi_avg * w_theta, theta_grid, dim=-1)

    return {
        "sigma_back_vv": _avg(sigma_back_vv),
        "sigma_back_hh": _avg(sigma_back_hh),
        "sigma_back_vh": _avg(sigma_back_vh),
        "sigma_ext_v": _avg(sigma_ext_v),
        "sigma_ext_h": _avg(sigma_ext_h),
    }


# ─── First-order MIMICS forward ─────────────────────────────────────────────


def _one_minus_exp_over_x(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable `(1 − exp(−x))/x`, returning 1 as x → 0.

    Uses `-torch.expm1(-x)/x` for the bulk and a three-term Taylor
    expansion for |x| below a small threshold. Both branches are
    differentiable in torch.where, and the Taylor branch is exact at the
    origin so the crown-direct limit σ_vol · h is recovered smoothly.
    """
    small = x.abs() < 1e-6
    taylor = 1.0 - x / 2.0 + x * x / 6.0
    # Use clamp on the denominator for the non-selected branch — this
    # path is discarded by torch.where but still traced; the clamp keeps
    # its numerics finite without affecting the selected (exact) branch.
    exact = -torch.expm1(-x) / x.clamp(min=1e-30)
    return torch.where(small, taylor, exact)


def _forward_internal(
    params: MimicsToureParamsTorch,
    n_theta: int,
    n_phi: int,
    ground_dielectric_fn: Optional[GroundDielectricFn],
) -> dict:
    """
    Shared forward core used by both `mimics_toure_single_crown` (which
    returns (σ°_VV_dB, σ°_VH_dB)) and `mimics_toure_single_crown_breakdown_torch`
    (which returns the mechanism decomposition in the same dict shape as
    the numpy reference's `mimics_toure_single_crown_breakdown`).

    Keeping a single code path for both entry points means the
    PyTorch-vs-numpy numpy_port arm agreement on the totals implies
    agreement on every mechanism — the breakdown is read from the same
    intermediate tensors that form the total.

    The `ground_dielectric_fn` argument is the DEV-1b-004 injection
    point: when `None` (production default) the path uses the SPEC §6
    primary Mironov GRMDM with the DEV-007 clamp. The G2 harness passes
    a Dobson callable (with T94 mineral-soil kwargs) for E.1 / E.2 only;
    the Moor House production path never sets this away from `None`.
    """
    device, dtype = _infer_device_dtype(params)

    N_b = _as_tensor(params.N_b_per_m3, device, dtype)
    N_l = _as_tensor(params.N_l_per_m3, device, dtype)
    sigma_orient_deg = _as_tensor(params.sigma_orient_deg, device, dtype)
    m_g = _as_tensor(params.m_g, device, dtype)
    m_v = _as_tensor(params.m_v, device, dtype)
    s_cm = _as_tensor(params.s_cm, device, dtype)
    theta_inc_deg = _as_tensor(params.theta_inc_deg, device, dtype)

    theta_rad = theta_inc_deg * (math.pi / 180.0)
    cos_t = torch.cos(theta_rad)

    # Sentinel-1 C-band geometry (may be overridden via params.freq_hz
    # for the G2 Set E L-band anchor).
    lam_m = (C_LIGHT_CM_PER_S * 1e-2) / params.freq_hz
    k_per_m = 2.0 * math.pi / lam_m

    # Dielectrics. Mironov is the SPEC §6 primary production model; the
    # G2 harness injects a Dobson callable for E.1 / E.2 per DEV-1b-004.
    if ground_dielectric_fn is None:
        eps_g = ground_epsilon_mironov_torch(m_v)
    else:
        eps_g = ground_dielectric_fn(m_v)
    eps_v = vegetation_epsilon_ulaby_elrayes_torch(m_g)

    # Ground Oh backscatter — reuses the SPEC §7 learnable-s implementation.
    sigma_oh_vv_db, sigma_oh_vh_db = oh_soil_backscatter_dual_pol(
        eps_g, theta_rad, s_cm
    )
    sigma_oh_vv_lin = torch.pow(torch.as_tensor(10.0, device=device, dtype=dtype),
                                sigma_oh_vv_db / 10.0)
    sigma_oh_vh_lin = torch.pow(torch.as_tensor(10.0, device=device, dtype=dtype),
                                sigma_oh_vh_db / 10.0)

    # Orientation-averaged single-scatterer cross sections.
    branch = _branch_cross_sections_torch(
        params.a_b_cm, params.l_b_cm, eps_v, sigma_orient_deg,
        theta_rad, k_per_m, n_theta=n_theta, n_phi=n_phi,
    )
    leaf = _leaf_cross_sections_torch(
        params.a_l_cm, params.t_l_cm, eps_v, theta_rad, k_per_m,
        n_theta=n_theta, n_phi=n_phi,
    )

    # Volume backscatter per unit volume (m⁻¹).
    sigma_vol_vv = (
        N_b * branch["sigma_back_vv"] + N_l * leaf["sigma_back_vv"]
    )
    sigma_vol_vh = (
        N_b * branch["sigma_back_vh"] + N_l * leaf["sigma_back_vh"]
    )

    # Extinction (m⁻¹) per polarisation.
    kappa_e_v = N_b * branch["sigma_ext_v"] + N_l * leaf["sigma_ext_v"]
    kappa_e_h = N_b * branch["sigma_ext_h"] + N_l * leaf["sigma_ext_h"]

    # Two-way canopy transmissivity.
    h_c = params.h_c_m
    opt_depth_v = kappa_e_v * h_c / cos_t
    opt_depth_h = kappa_e_h * h_c / cos_t
    tau2_v = torch.exp(-2.0 * opt_depth_v)
    tau2_h = torch.exp(-2.0 * opt_depth_h)

    # Crown direct: σ°_pp = σ_vol_pp · h · (1 − exp(−2κh/cosθ)) / (2κh/cosθ).
    x_v = 2.0 * kappa_e_v * h_c / cos_t
    sigma_crown_vv = sigma_vol_vv * h_c * _one_minus_exp_over_x(x_v)
    x_sum = (kappa_e_v + kappa_e_h) * h_c / cos_t
    sigma_crown_vh = sigma_vol_vh * h_c * _one_minus_exp_over_x(x_sum)

    # Crown–ground coupling.
    sigma_crown_ground_vv = (
        4.0 * sigma_vol_vv * torch.sqrt(sigma_oh_vv_lin)
        * torch.sqrt(tau2_v * tau2_v) * h_c / cos_t
    )
    sigma_crown_ground_vh = (
        4.0 * sigma_vol_vh * torch.sqrt(sigma_oh_vh_lin)
        * torch.sqrt(tau2_v * tau2_h) * h_c / cos_t
    )

    # Ground direct (attenuated by canopy two-way loss).
    sigma_gd_vv = tau2_v * sigma_oh_vv_lin
    sigma_gd_vh = torch.sqrt(tau2_v * tau2_h) * sigma_oh_vh_lin

    sigma_total_vv = sigma_crown_vv + sigma_crown_ground_vv + sigma_gd_vv
    sigma_total_vh = sigma_crown_vh + sigma_crown_ground_vh + sigma_gd_vh

    sigma_vv_db = 10.0 * torch.log10(sigma_total_vv.clamp(min=_EPS_LOG))
    sigma_vh_db = 10.0 * torch.log10(sigma_total_vh.clamp(min=_EPS_LOG))

    return {
        "sigma_vv_db": sigma_vv_db,
        "sigma_vh_db": sigma_vh_db,
        "sigma_crown_vv": sigma_crown_vv,
        "sigma_crown_vh": sigma_crown_vh,
        "sigma_crown_ground_vv": sigma_crown_ground_vv,
        "sigma_crown_ground_vh": sigma_crown_ground_vh,
        "sigma_gd_vv": sigma_gd_vv,
        "sigma_gd_vh": sigma_gd_vh,
        "sigma_total_vv_lin": sigma_total_vv,
        "sigma_total_vh_lin": sigma_total_vh,
        "eps_g": eps_g,
        "eps_v": eps_v,
        "kappa_e_v": kappa_e_v,
        "kappa_e_h": kappa_e_h,
        "tau2_v": tau2_v,
        "tau2_h": tau2_h,
        "sigma_oh_vv_db": sigma_oh_vv_db,
        "sigma_oh_vh_db": sigma_oh_vh_db,
        "sigma_vol_vv": sigma_vol_vv,
        "sigma_vol_vh": sigma_vol_vh,
        "wavelength_m": lam_m,
        "k_per_m": k_per_m,
    }


def mimics_toure_single_crown(
    params: MimicsToureParamsTorch,
    n_theta: int = N_THETA_SAMPLES,
    n_phi: int = N_PHI_SAMPLES,
    *,
    use_trunk_layer: bool = False,
    ground_dielectric_fn: Optional[GroundDielectricFn] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Toure-style single-crown MIMICS first-order backscatter (PyTorch).

    The forward equations match `reference_toure.mimics_toure_single_crown`
    at the physics level. See that module's docstring for the per-term
    derivation; this implementation differs only in that every term is
    a differentiable torch operation so gradient flows through SPEC §5
    learnables and through per-observation m_v.

    `use_trunk_layer` is a COMPILE-TIME flag for the G2 Set D anchor
    only (SPEC g2_anchor_spec.md §Set D). It defaults to `False` — the
    production Phase 1b code path. It must NEVER be exposed as a
    runtime-settable option in training or inference code. `True` runs
    the McDonald 1990 walnut-orchard structural cross-check path with a
    trunk layer re-enabled; this code path is only exercised by
    `phase1b/physics/equivalence_check.py` for Set D.

    `ground_dielectric_fn` is a COMPILE-TIME injection point for the
    DEV-1b-004 dielectric-configuration amendment. When `None` (default,
    production) the SPEC §6 primary Mironov GRMDM with the DEV-007 clamp
    is used. The G2 harness passes `ground_epsilon_dobson_torch` with
    T94 mineral-soil kwargs for rows E.1 / E.2 only; the Moor House
    PINN-MIMICS trainer and inference paths must never set this away
    from `None`. A regression test in `tests/unit/test_mimics_torch.py`
    enforces this pinning.

    Args:
        params: MimicsToureParamsTorch with SPEC §5 learnables +
            per-observation inputs (m_v, theta_inc).
        n_theta, n_phi: orientation-quadrature sample counts.
        use_trunk_layer (keyword-only): set True ONLY for the G2 Set D
            anchor. Not implemented in this session; raises
            NotImplementedError with a clear message directing the
            caller to session E.
        ground_dielectric_fn (keyword-only): callable
            `(m_v_tensor) -> eps_tensor` overriding the default Mironov
            ground dielectric. G2-harness-only per DEV-1b-004.

    Returns:
        (sigma_vv_db, sigma_vh_db). Tensor shapes follow the broadcast
        shape of the learnable / per-observation inputs (scalar by
        default).
    """
    if use_trunk_layer:
        # Set D (M90 walnut orchard) requires:
        #  - Trunk layer with cos^6(theta) orientation PDF.
        #  - Four canopy branch classes (Trunk / Primary / Secondary / Stems)
        #    rather than a single branch-density parameter.
        #  - Complex-valued vegetation dielectric (M90 Table III uses
        #    ε = 25−j2.5 ground, 45−j11.2 trunk, etc.) — the v0.1
        #    real-only UEL proxy cannot represent this.
        #  - L-band (1.25 GHz) wavelength, not Sentinel-1 C-band.
        # None of these extensions are in scope for Session D. The harness
        # at phase1b/physics/equivalence_check.py is expected to catch
        # this NotImplementedError for Set D rows and record them as
        # UNIMPLEMENTED in the result JSON rather than propagating the
        # failure as a G2 gate failure. Escalate to session E to build
        # the trunk-layer code path.
        raise NotImplementedError(
            "use_trunk_layer=True is the G2 Set D (M90 walnut orchard) "
            "code path. Per DEV-1b-005 (Set D Phase 1c exemption), Set D "
            "anchors are held pending the trunk-layer implementation, "
            "which is a Phase 1c entry gate (NISAR L-band comparative "
            "validation). The trunk layer requires structural extensions "
            "(four branch classes, cos^6 orientation PDF, complex "
            "vegetation dielectric, L-band wavelength) and is explicitly "
            "out of Session E scope."
        )

    out = _forward_internal(
        params, n_theta=n_theta, n_phi=n_phi,
        ground_dielectric_fn=ground_dielectric_fn,
    )
    return out["sigma_vv_db"], out["sigma_vh_db"]


def mimics_toure_single_crown_breakdown_torch(
    params: MimicsToureParamsTorch,
    n_theta: int = N_THETA_SAMPLES,
    n_phi: int = N_PHI_SAMPLES,
    *,
    ground_dielectric_fn: Optional[GroundDielectricFn] = None,
) -> dict:
    """
    PyTorch analogue of `reference_toure.mimics_toure_single_crown_breakdown`.

    Returns the three-mechanism decomposition (crown-direct,
    crown-ground, ground-direct-attenuated) plus intermediate
    dielectrics / extinctions / transmissivities — in the same dict
    shape and key set as the numpy reference, with tensor values
    replacing numpy floats.

    This is the P3 deliverable for DEV-1b-006: it surfaces the PyTorch
    forward's internal mechanism values so the G2 Set C anchor can be
    evaluated directly against the PyTorch implementation (previously
    the harness fell back to the numpy reference's breakdown per Session
    D's Set C limitation — see `equivalence_check.py::_run_set_C` prior
    to Session E).

    The decomposition is extracted from the same intermediate tensors
    the total σ° is built from, so the numpy-port arm agreement on the
    totals implies agreement on every mechanism (no new physics).

    Args:
        params, n_theta, n_phi: same as `mimics_toure_single_crown`.
        ground_dielectric_fn (keyword-only): DEV-1b-004 harness-only
            injection point; see `mimics_toure_single_crown`.

    Returns:
        dict with keys matching
        `reference_toure.mimics_toure_single_crown_breakdown` at the
        mechanism / total-σ° level. Tensor values; callers that need
        Python floats should call `.item()` themselves.
    """
    out = _forward_internal(
        params, n_theta=n_theta, n_phi=n_phi,
        ground_dielectric_fn=ground_dielectric_fn,
    )

    def _db(x: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10(x.clamp(min=_EPS_LOG))

    return {
        "wavelength_m": out["wavelength_m"],
        "k_per_m": out["k_per_m"],
        "epsilon_ground": out["eps_g"],
        "epsilon_vegetation": out["eps_v"],
        "kappa_e_v_per_m": out["kappa_e_v"],
        "kappa_e_h_per_m": out["kappa_e_h"],
        "tau2_v": out["tau2_v"],
        "tau2_h": out["tau2_h"],
        "sigma_vol_vv_per_m": out["sigma_vol_vv"],
        "sigma_vol_vh_per_m": out["sigma_vol_vh"],
        "sigma_oh_vv_db": out["sigma_oh_vv_db"],
        "sigma_oh_vh_db": out["sigma_oh_vh_db"],
        "mechanisms_vv_linear": {
            "crown_direct": out["sigma_crown_vv"],
            "crown_ground": out["sigma_crown_ground_vv"],
            "ground_direct_attenuated": out["sigma_gd_vv"],
        },
        "mechanisms_vh_linear": {
            "crown_direct": out["sigma_crown_vh"],
            "crown_ground": out["sigma_crown_ground_vh"],
            "ground_direct_attenuated": out["sigma_gd_vh"],
        },
        "mechanisms_vv_db": {
            "crown_direct": _db(out["sigma_crown_vv"]),
            "crown_ground": _db(out["sigma_crown_ground_vv"]),
            "ground_direct_attenuated": _db(out["sigma_gd_vv"]),
        },
        "mechanisms_vh_db": {
            "crown_direct": _db(out["sigma_crown_vh"]),
            "crown_ground": _db(out["sigma_crown_ground_vh"]),
            "ground_direct_attenuated": _db(out["sigma_gd_vh"]),
        },
        "sigma_total_vv_db": out["sigma_vv_db"],
        "sigma_total_vh_db": out["sigma_vh_db"],
    }
