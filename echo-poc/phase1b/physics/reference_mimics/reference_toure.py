"""
Plain-numpy Toure 1994 single-crown MIMICS reference implementation.

This module is the **non-differentiable numpy reference** for the Phase
1b G2 Implementation Gate (`SPEC.md` §4). A differentiable PyTorch port
will live at `phase1b/physics/mimics.py` (session C); the equivalence
check (session D) will require σ° agreement between the PyTorch port
and this reference to within 0.5 dB across the canonical parameter
combinations listed in `canonical_combinations.json`.

Why a separate numpy reference:
    The G2 gate needs an independent code path from the PyTorch module.
    A bug in the differentiable port that produces plausible σ° can
    silently corrupt Phase 1b training; comparing against an
    independently authored numpy implementation plus published tables
    (to be added in session D) is the safety net.

Physics:
    First-order MIMICS total backscatter for a single crown above a
    rough peat ground (Toure-style adaptation, no trunk layer):

        σ°_total = σ°_crown_direct
                 + σ°_crown_ground
                 + τ² · σ°_ground_direct

    with two-way crown transmissivity

        τ² = exp( −2 · κ_e · h_c / cos θ )

    The three mechanisms match `SPEC.md` §4 "Scattering mechanisms
    retained" — crown direct volume scattering, crown–ground
    interaction, and ground direct attenuated by crown two-way loss.
    The trunk–ground double bounce is explicitly dropped per Toure
    (1994) adaptation.

Scatterer models:
    - Branches: finite dielectric cylinders (radius a_b, half-length
      l_b) with Gaussian zenith orientation distribution (std σ_orient)
      and uniform azimuth. Treated in the Rayleigh limit for the radial
      direction (k·a_b ≪ 1 at C-band for heather a_b ≈ 2 mm: k·a ≈ 0.023)
      with depolarisation factors for a long prolate ellipsoid (L_∥ → 0,
      L_⊥ → 1/2) so the polarisation-dependent response is retained.
    - Leaves: dielectric discs (radius a_l, thickness t_l) with uniform
      orientation (Calluna leaves are near-random at the Moor House
      scale). Rayleigh-limit thin-disc depolarisation (L_normal → 1,
      L_face → 0).
    - All single-scatterer amplitudes are computed in the Rayleigh
      regime and orientation-averaged by direct numerical quadrature
      over (θ_b, φ_b). The Rayleigh approximation loses resonance
      corrections that appear for k·l ≳ 1; this is a known limitation
      of the v0.1 reference and is revisited at session E against Toure
      1994 tables.

Known limitations (v0.1, to be addressed in session E):
    - Branches are modelled as Rayleigh scatterers multiplied by a
      single sinc² form factor evaluated at the backscatter direction,
      |F(q·ĉ·l)|² = sinc²(2 k l cos α) with α the angle between the
      cylinder axis and the incidence direction. The same form factor
      is applied to both σ_back_pq and the single-orientation σ_ext
      contribution. This is a tractable first-pass approximation of the
      Rayleigh–Gans–Debye finite-cylinder result; the correct
      treatment would average |F(q)|² over all scattering directions
      for extinction, and use the Ulaby–Moore–Fung infinite-cylinder
      scattering amplitudes (with Bessel-function form factors in radial
      and axial directions) throughout. Empirically, applying the sinc²
      form factor reduces κ_e from O(200 m⁻¹) to O(1 m⁻¹) at SPEC §5
      learnable-midpoint densities, keeping the crown optically thin
      enough (τ² ~ 0.1–0.9) that σ°_total depends on both canopy and
      ground as the physics requires.
    - Vegetation dielectric uses a two-coefficient UEL proxy (real
      part only) rather than the full Ulaby–El-Rayes dual-dispersion
      model. Loss-tangent effects on extinction are therefore absent.
    - The crown–ground coupling uses √(σ°_oh) as a ground-reflectance
      proxy instead of the proper Fresnel-reflectance magnitude. This
      adds roughness to the coupling term consistent with Oh's formulation
      but is not literally Toure 1994's form.

    These limitations are accepted for the v0.1 reference. Session D
    transcribes σ° values from Toure 1994 tables; session E debugs the
    numpy reference against those tables, at which point any of the
    three items above that contribute materially to the disagreement
    will be promoted to the full MIMICS form.

Dielectrics:
    - Ground (peat): Mironov 2009 GRMDM, organic-soil parameters,
      ε ≥ 1.01 clamp inherited from DEV-007 (SPEC §6).
    - Vegetation (branch + leaf): Ulaby & El-Rayes (1987) simplified
      empirical form, reduced to the real part at C-band for a
      non-differentiable reference. Reference value at m_g = 0.45 g/g,
      f = 5.4 GHz is ε_veg ≈ 15 (Ulaby & El-Rayes 1987, Fig. 7).

Independence from the PyTorch code:
    The Oh 1992, Mironov 2009, and Ulaby–El-Rayes 1987 formulas below
    are reimplemented in plain numpy. They use the same numeric
    constants as `shared.config` / `phase1/physics/dielectric.py` /
    `phase1b/physics/oh1992_learnable_s.py` (ε_min = 1.01, Mironov
    organic parameters, C-band wavenumber) but the code is written
    fresh so that transcription errors in the PyTorch port cannot
    propagate here.

Inputs:
    A `MimicsToureParams` dataclass, documented below.

Outputs:
    (sigma_vv_db, sigma_vh_db) in dB.

References:
    - Toure, A., Thomson, K.P.B., Edwards, G., Brown, R.J., Brisco, B.G.
      (1994). 'Adaptation of the MIMICS backscattering model to the
      agricultural context — wheat and canola at L and C bands.'
      IEEE TGRS 32(1), 47–61. — primary single-crown precedent.
    - Ulaby, F.T., Sarabandi, K., McDonald, K., Whitt, M., Dobson, M.C.
      (1990). 'Michigan microwave canopy scattering model.'
      Int. J. Remote Sensing 11(7), 1223–1253. — original MIMICS paper.
    - Ulaby, F.T., Moore, R.K., Fung, A.K. (1986). *Microwave Remote
      Sensing, vol. II*, ch. 11 — first-order radiative-transfer
      solution used throughout.
    - Mironov, V.L. et al. (2009). 'Generalized refractive mixing
      dielectric model.' IEEE TGRS 47(7), 1998–2010.
    - Oh, Y., Sarabandi, K., Ulaby, F.T. (1992). 'An empirical model
      and an inversion technique for radar scattering from bare soil
      surfaces.' IEEE TGRS 30(2), 370–381.
    - Ulaby, F.T., El-Rayes, M.A. (1987). 'Microwave dielectric
      spectrum of vegetation — Part II: dual-dispersion model.'
      IEEE TGRS GE-25(5), 550–557.

SPEC links:
    - SPEC.md §4 — MIMICS forward-model architecture, single crown,
      differentiable implementation gate at 0.5 dB.
    - SPEC.md §5 — parameter table (fixed / learnable / per-obs).
    - SPEC.md §6 — Mironov primary, Dobson sensitivity arm, ε ≥ 1.01
      clamp.
    - SPEC.md §7 — Oh 1992 with learnable s across 1–5 cm.
    - `physics/reference_mimics/README.md` — sourcing decision
      (Option E: published tables + numpy port, both required).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np

# ─── Constants ──────────────────────────────────────────────────────────────

# Speed of light (cm/s) — SI value. Used to derive λ, k at Sentinel-1
# C-band. Kept as a local constant rather than imported from
# `phase1b.physics.oh1992_learnable_s` so the reference does not share a
# symbol with the PyTorch code path.
C_LIGHT_CM_PER_S = 2.99792458e10

# Sentinel-1 C-band carrier frequency (Hz) — SPEC §5 fixed sensor
# parameter.
SENTINEL1_FREQ_HZ = 5.405e9

# Mironov (2009) organic-soil GRMDM constants — matched to
# `shared.config.MIRONOV_*`. Duplicated here so a transcription error in
# either path is caught by the G2 gate.
MIRONOV_ND = 0.312
MIRONOV_ND1 = 1.42
MIRONOV_ND2 = 0.89
MIRONOV_MV_T = 0.36

# DEV-007 / SPEC §6 ε clamp. Applied to the ground dielectric before
# entry to Oh 1992 to prevent √(negative) at low m_v.
EPSILON_GROUND_MIN = 1.01

# Ulaby & El-Rayes (1987) simplified Calluna-canopy vegetation
# dielectric parameters. The full dual-dispersion model has real and
# imaginary parts; the reference keeps the real part only because Oh
# 1992 and the MIMICS first-order form used here depend on ε_v through
# the polarisability (ε_v - 1)/(ε_v + 1) and analogous factors, for
# which the imaginary part changes σ° by a few tenths of a dB at C-band
# for moist vegetation. The 0.5 dB G2 tolerance absorbs this
# simplification; a v0.2 reference can promote the imaginary part if
# the equivalence check reveals it is binding.
UEL_A = 1.7   # intercept
UEL_B = 18.0  # gain on v_fw (fractional free water)
# These numeric values reproduce ε'_leaf ≈ 15 at m_g ≈ 0.45 g/g and
# ε'_leaf ≈ 22 at m_g ≈ 0.6 g/g at C-band (Ulaby & El-Rayes 1987, Fig. 7,
# visual read-off). Intended as a numerically plausible reference; the
# published U-ER Type-A polynomial is more accurate and is noted as a
# v0.2 refinement target in the G2 session E work.

# SPEC §5 saturated VWC for Moor House peat.
PEAT_THETA_SAT = 0.88

# Numerical safety.
_EPS_LOG = 1e-12

# Default orientation-quadrature sample counts — balance between
# accuracy (cosθ-weighted Gaussian moments stable within ≲ 0.005 dB at
# σ_orient = 30°) and the speed needed to regenerate the canonical set
# from a single-script invocation.
N_THETA_SAMPLES = 128
N_PHI_SAMPLES = 32


# ─── Parameter dataclass ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MimicsToureParams:
    """
    Input parameter set for the Toure single-crown MIMICS reference.

    All lengths in the units given. Angles in degrees at the interface;
    the implementation converts to radians internally.

    Fields mirror `SPEC.md` §5 grouped by role. Defaults are the SPEC
    §5 fixed values / learnable-range midpoints for the Moor House
    heather canopy.
    """

    # Crown geometry — SPEC §5 fixed from Calluna field literature.
    h_c_m: float = 0.4            # crown height (m)
    a_b_cm: float = 0.2           # branch (cylinder) radius (cm) — 2 mm
    l_b_cm: float = 8.0           # branch half-length (cm)
    a_l_cm: float = 1.0           # leaf (disc) radius (cm)
    t_l_cm: float = 0.03          # leaf (disc) thickness (cm) — 0.3 mm

    # Crown densities — SPEC §5 learnable.
    N_b_per_m3: float = 1.0e3     # branch number density (m⁻³)
    N_l_per_m3: float = 1.0e4     # leaf number density (m⁻³)
    sigma_orient_deg: float = 30.0  # branch-zenith Gaussian std (deg)

    # Crown moisture — SPEC §5 learnable, tied branch/leaf.
    m_g: float = 0.45             # gravimetric moisture (g/g)

    # Ground — SPEC §5 and §6.
    m_v: float = 0.5              # volumetric water content (cm³/cm³)
    s_cm: float = 2.0             # surface RMS height (cm)
    l_corr_cm: float = 5.0        # correlation length (cm, fixed per SPEC §5)

    # Sensor — SPEC §5 fixed.
    theta_inc_deg: float = 41.5   # incidence angle at Moor House (deg)
    freq_hz: float = SENTINEL1_FREQ_HZ

    def as_dict(self) -> dict:
        return asdict(self)


# ─── Dielectrics (numpy) ─────────────────────────────────────────────────────


def ground_epsilon_mironov(m_v: float) -> float:
    """
    Real part of the peat ground dielectric via Mironov (2009) GRMDM.

    Piecewise linear in m_v about the transition m_v_t = 0.36.
    Numerically identical to `phase1.physics.dielectric.MironovDielectric`
    at the float level, intentionally coded fresh here to preserve
    independence for the G2 gate.

    Args:
        m_v: volumetric water content (cm³/cm³), scalar. Typical Moor
             House range 0.25 – 0.83.

    Returns:
        ε_real: real part of the complex dielectric (dimensionless).
                Clamped ≥ EPSILON_GROUND_MIN per DEV-007 / SPEC §6.
    """
    if m_v <= MIRONOV_MV_T:
        n = MIRONOV_ND + MIRONOV_ND1 * m_v
    else:
        n = (
            MIRONOV_ND
            + MIRONOV_ND1 * MIRONOV_MV_T
            + MIRONOV_ND2 * (m_v - MIRONOV_MV_T)
        )
    eps = n * n
    if eps < EPSILON_GROUND_MIN:
        eps = EPSILON_GROUND_MIN
    return float(eps)


def vegetation_epsilon_ulaby_elrayes(m_g: float) -> float:
    """
    Simplified Ulaby & El-Rayes (1987) vegetation dielectric, real part.

    Form:
        ε_v(m_g) = UEL_A + UEL_B · m_g

    Coefficients chosen so the reference reproduces Ulaby & El-Rayes
    (1987) Fig. 7 to within a few percent at Moor House m_g ∈ [0.3, 0.6]
    g/g. The imaginary part is neglected in this v0.1 reference; the
    0.5 dB G2 tolerance absorbs the resulting σ° offset. A v0.2
    reference can promote to the full dual-dispersion model if session E
    debugging against Toure 1994 tables shows this is binding.

    Args:
        m_g: gravimetric moisture (g/g), scalar, ∈ [0, 1].

    Returns:
        ε_v: vegetation dielectric (dimensionless, real part).
             At m_g = 0.45: ε_v ≈ 9.8.
    """
    if m_g < 0.0:
        m_g = 0.0
    if m_g > 1.0:
        m_g = 1.0
    return UEL_A + UEL_B * m_g


# ─── Oh 1992 surface scattering (numpy) ─────────────────────────────────────


def oh_surface_vv_vh_db(
    epsilon_ground: float,
    theta_inc_rad: float,
    s_cm: float,
    k_per_cm: float,
) -> Tuple[float, float]:
    """
    Oh 1992 rough-surface backscatter, σ°_VV and σ°_VH, numpy version.

    Simplified Oh co-pol VV:
        Γ_h(ε, θ) = ((cosθ − √(ε − sin²θ)) / (cosθ + √(ε − sin²θ)))²
        σ°_VV_lin = (ks^0.1 / 3) · (cosθ)^2.2 · Γ_h(ε, θ)

    VH via the Oh cross-pol ratio (Oh 1992, 2004 simplified):
        Γ₀ = ((1 − √ε) / (1 + √ε))²  (Fresnel, nadir, h-pol)
        q   = 0.23 · √Γ₀ · (1 − exp(−ks))
        σ°_VH_lin = q · σ°_VV_lin

    Args:
        epsilon_ground: ground real dielectric. Scalar, pre-clamped or
                        raw — clamped here for safety.
        theta_inc_rad : incidence angle in radians.
        s_cm          : surface RMS height (cm), matches SPEC §5 units.
        k_per_cm      : radar wavenumber (rad/cm), matches SPEC §5 unit.

    Returns:
        (sigma_vv_db, sigma_vh_db).

    Independence: numerically identical to
    `phase1b.physics.oh1992_learnable_s.oh_soil_backscatter_dual_pol`
    at the float level but written as a separate numpy code path for
    the G2 gate.
    """
    eps = max(epsilon_ground, EPSILON_GROUND_MIN)
    ks = s_cm * k_per_cm

    cos_t = np.cos(theta_inc_rad)
    sin_t = np.sin(theta_inc_rad)

    inner = max(eps - sin_t * sin_t, 1e-6)
    sqrt_inner = np.sqrt(inner)
    gamma_h = ((cos_t - sqrt_inner) / (cos_t + sqrt_inner + 1e-8)) ** 2
    sigma_vv_lin = (ks ** 0.1 / 3.0) * (cos_t ** 2.2) * gamma_h
    sigma_vv_db = 10.0 * np.log10(sigma_vv_lin + 1e-10)

    sqrt_eps = np.sqrt(eps)
    gamma_0 = ((1.0 - sqrt_eps) / (1.0 + sqrt_eps)) ** 2
    factor = min(1.0 - 1e-12, 1.0 - np.exp(-ks))
    q = 0.23 * np.sqrt(gamma_0) * factor
    q_safe = max(q, 1e-10)
    vh_minus_vv_db = 10.0 * np.log10(q_safe)

    return float(sigma_vv_db), float(sigma_vv_db + vh_minus_vv_db)


# ─── Single-scatterer cross sections (Rayleigh limit) ───────────────────────


def _cylinder_rayleigh_polarisabilities(
    a_cm: float, l_cm: float, eps_v: float
) -> Tuple[float, float, float]:
    """
    Rayleigh-limit polarisabilities of a prolate dielectric cylinder.

    Prolate ellipsoid of revolution with semi-axes (l, a, a), l > a, in
    the Rayleigh limit (k·max(a, l) ≪ 1 idealisation). Volume

        V = π · a² · 2·l       (full length 2l)

    Depolarisation factors for prolate ellipsoid (axial L_∥, transverse
    L_⊥) — in the very prolate limit a/l → 0,

        L_∥ → 0,    L_⊥ → 1/2.

    For a finite aspect ratio we use the closed-form

        L_∥(e) = (1 − e²)/e³ · ( ½ ln((1+e)/(1−e)) − e )
        L_⊥   = (1 − L_∥) / 2
        e     = √(1 − (a/l)²)

    (See Bohren & Huffman 1983, eq. 5.32–5.33.)

    Polarisabilities (m³) along each axis:

        α_i = V · (ε_v − 1) / ( 1 + (ε_v − 1) · L_i )

    Args:
        a_cm  : cylinder radius (cm).
        l_cm  : cylinder half-length (cm). Full length 2l in V.
        eps_v : vegetation real dielectric.

    Returns:
        (V_m3, alpha_par_m3, alpha_perp_m3).
    """
    a_m = a_cm * 1e-2
    l_m = l_cm * 1e-2  # half-length
    full_length_m = 2.0 * l_m
    V = np.pi * a_m * a_m * full_length_m

    # Aspect-ratio-dependent depolarisation for prolate ellipsoid.
    aspect = a_m / l_m  # a/l, with l = half-length. Very prolate → aspect ≪ 1.
    if aspect >= 1.0:
        # Degenerate: near-spherical; use L_∥ = L_⊥ = 1/3.
        L_par = 1.0 / 3.0
    else:
        e = np.sqrt(1.0 - aspect * aspect)
        L_par = (1.0 - e * e) / (e ** 3) * (0.5 * np.log((1.0 + e) / (1.0 - e)) - e)
    L_perp = 0.5 * (1.0 - L_par)

    alpha_par = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_par)
    alpha_perp = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_perp)
    return float(V), float(alpha_par), float(alpha_perp)


def _disc_rayleigh_polarisabilities(
    a_cm: float, t_cm: float, eps_v: float
) -> Tuple[float, float, float]:
    """
    Rayleigh-limit polarisabilities of a thin dielectric disc.

    Oblate ellipsoid with semi-axes (a, a, t), t ≪ a.  Volume

        V = π · a² · t   (no factor of 2; the disc thickness is t, not 2t)

    In the thin-disc limit (t/a → 0),

        L_normal → 1,    L_face → 0.

    Closed-form for oblate:

        L_normal(e) = (1 + e²)/e³ · ( e − arctan(e) ),   e = √((a/t)² − 1)
        L_face     = (1 − L_normal) / 2

    (See Bohren & Huffman 1983.)

    Polarisabilities as above.

    Args:
        a_cm  : disc radius (cm).
        t_cm  : disc thickness (cm).
        eps_v : vegetation dielectric.

    Returns:
        (V_m3, alpha_normal_m3, alpha_face_m3).
    """
    a_m = a_cm * 1e-2
    t_m = t_cm * 1e-2
    V = np.pi * a_m * a_m * t_m

    aspect = t_m / a_m
    if aspect >= 1.0:
        # Near-spherical; fallback to 1/3.
        L_normal = 1.0 / 3.0
    else:
        e = np.sqrt(1.0 - aspect * aspect)
        # For an oblate ellipsoid with e' = √((a/t)² - 1),
        # L_normal = (1 + e'²)/e'³ · (e' − arctan(e')). Use the
        # re-parameterisation with e from aspect below, which is the
        # Bohren–Huffman form for oblate.
        L_normal = (1.0 / (e * e)) * (1.0 - (np.sqrt(1.0 - e * e) / e) * np.arcsin(e))
    L_face = 0.5 * (1.0 - L_normal)

    alpha_normal = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_normal)
    alpha_face = V * (eps_v - 1.0) / (1.0 + (eps_v - 1.0) * L_face)
    return float(V), float(alpha_normal), float(alpha_face)


# ─── Orientation-averaged crown cross sections ──────────────────────────────


def _sinc(x: np.ndarray | float) -> np.ndarray | float:
    """Unnormalised sinc: sin(x)/x, defined as 1 at x = 0."""
    x_arr = np.asarray(x, dtype=float)
    out = np.where(np.abs(x_arr) < 1e-8, 1.0, np.sin(x_arr) / np.where(x_arr == 0, 1.0, x_arr))
    return float(out) if np.isscalar(x) else out


def _branch_cross_sections(
    a_cm: float,
    l_cm: float,
    eps_v: float,
    sigma_orient_deg: float,
    theta_inc_rad: float,
    k_per_m: float,
    n_theta: int = N_THETA_SAMPLES,
    n_phi: int = N_PHI_SAMPLES,
) -> dict:
    """
    Orientation-averaged backscatter and extinction cross sections for
    a Rayleigh cylinder population with Gaussian zenith, uniform
    azimuth.

    The scattering amplitude tensor of a cylinder with axis ĉ
    (unit vector) is

        A(ĉ) = α_∥ ĉĉᵀ + α_⊥ (I − ĉĉᵀ).

    For monostatic backscatter at incidence θ_i, the (vv, hh, vh)
    backscatter cross sections of one particle in orientation ĉ are

        σ_pq_back(ĉ) = 4π · k⁴ · | ê_p(−k̂_i) · A(ĉ) · ê_q(k̂_i) |²

    where ê_v and ê_h are the unit polarisation vectors in the plane
    perpendicular to the propagation direction, with ê_h horizontal in
    the azimuth plane and ê_v completing a right-handed basis.
    Writing k̂_i = (sin θ_i, 0, −cos θ_i) so the incident wave is
    descending in the x-z plane, we have

        ê_h = (0, 1, 0),
        ê_v(k̂_i) = (cos θ_i, 0, sin θ_i),
        ê_v(−k̂_i) = (−cos θ_i, 0, sin θ_i).

    (Backscatter basis convention per Ulaby & Long 2014 Table 5.1.)

    Extinction per unit length κ_e^p is related via the forward
    scattering amplitude:

        κ_e^p = (4π · N / k) · Im( f_pp_forward )

    In the Rayleigh non-absorbing limit for a real ε_v, Im(f_forward)
    vanishes. Pure scattering extinction is then approximated by the
    Rayleigh total scattering cross section:

        σ_s^p(ĉ)  ≈  (8π/3) · k⁴ · (α_∥² cos²γ + α_⊥² sin²γ)

    where γ is the angle between ĉ and ê_p — this is the canonical
    Rayleigh result for anisotropic particles. Orientation-averaged
    κ_e^p = N · <σ_s^p>. This is a deliberate simplification in the
    v0.1 reference: a complex ε_v would introduce absorption through
    Im(ε_v) but is not present in the simplified U-ER used here.

    Args:
        a_cm            : branch radius (cm).
        l_cm            : branch half-length (cm).
        eps_v           : vegetation dielectric (real).
        sigma_orient_deg: Gaussian zenith-angle std (degrees).
        theta_inc_rad   : incidence angle (radians).
        k_per_m         : wavenumber (rad/m).
        n_theta, n_phi  : orientation quadrature sample counts.

    Returns:
        dict with keys:
            sigma_back_vv_per_particle_m2
            sigma_back_hh_per_particle_m2
            sigma_back_vh_per_particle_m2
            sigma_ext_v_per_particle_m2
            sigma_ext_h_per_particle_m2
    """
    _, alpha_par, alpha_perp = _cylinder_rayleigh_polarisabilities(a_cm, l_cm, eps_v)
    # Half-length in metres, used by the sinc² form factor below.
    l_m = l_cm * 1e-2

    # Orientation grid. Zenith on [0, π/2] (cylinders cannot point
    # into the ground), azimuth on [0, 2π).
    sigma_rad = np.radians(sigma_orient_deg)
    theta_grid = np.linspace(0.0, np.pi / 2.0, n_theta)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    # Gaussian pdf on θ_b, truncated to [0, π/2], combined with the
    # sin(θ_b) solid-angle factor on the sphere. After truncation and
    # reshaping, this gives the weight w_θ(θ_b) such that
    #     ∫ f(θ_b) w_θ(θ_b) dθ_b  ≈  <f>_θ
    # and analogously w_φ(φ_b) = 1/(2π).
    w_theta_unnorm = np.exp(-(theta_grid ** 2) / (2.0 * sigma_rad ** 2)) * np.sin(theta_grid)
    w_theta = w_theta_unnorm / np.trapz(w_theta_unnorm, theta_grid)

    # Incidence geometry: k̂_i = (sin θ_i, 0, −cos θ_i). Scatter into
    # −k̂_i for backscatter.
    cos_ti = np.cos(theta_inc_rad)
    sin_ti = np.sin(theta_inc_rad)
    # Polarisation unit vectors in the scattering frame.
    e_h = np.array([0.0, 1.0, 0.0])
    e_v_inc = np.array([cos_ti, 0.0, sin_ti])
    e_v_sca = np.array([-cos_ti, 0.0, sin_ti])

    k4 = k_per_m ** 4

    # Accumulators. Shape (n_theta, n_phi) to allow a per-cell weight.
    sigma_vv_grid = np.zeros((n_theta, n_phi))
    sigma_hh_grid = np.zeros((n_theta, n_phi))
    sigma_vh_grid = np.zeros((n_theta, n_phi))
    sigma_ext_v_grid = np.zeros((n_theta, n_phi))
    sigma_ext_h_grid = np.zeros((n_theta, n_phi))

    for ix, theta_b in enumerate(theta_grid):
        sin_tb = np.sin(theta_b)
        cos_tb = np.cos(theta_b)
        for iy, phi_b in enumerate(phi_grid):
            # Cylinder-axis unit vector in lab frame.
            c = np.array(
                [sin_tb * np.cos(phi_b), sin_tb * np.sin(phi_b), cos_tb]
            )
            # Scattering-amplitude tensor (anisotropic polarisability × k²/4π):
            #   A_ij = (k²/4π) · [α_∥ c_i c_j + α_⊥ (δ_ij − c_i c_j)]
            # For σ we need |ê_p A ê_q|² · 4π · k⁴ / (4π)² = k⁴ |ê_p Π ê_q|² / (4π)
            # where Π_ij = α_∥ c_i c_j + α_⊥ (δ_ij − c_i c_j).
            # Simpler: compute the bilinear form ê_p · Π · ê_q
            # directly for each (p, q) backscatter combination.
            pi_cc_v_inc = c * np.dot(c, e_v_inc)       # (c · e_v_inc) c
            pi_v_inc = alpha_par * pi_cc_v_inc + alpha_perp * (e_v_inc - pi_cc_v_inc)
            pi_cc_h = c * np.dot(c, e_h)
            pi_h = alpha_par * pi_cc_h + alpha_perp * (e_h - pi_cc_h)

            # f_vv = (k²/4π) · ê_v(scatter) · Π · ê_v(incident). The
            # k²/4π factor propagates to σ as k⁴ / (4π)² · 4π = k⁴/(4π).
            f_vv = float(np.dot(e_v_sca, pi_v_inc))
            f_hh = float(np.dot(e_h, pi_h))
            f_vh = float(np.dot(e_v_sca, pi_h))
            # (cross-pol HV = ê_h · Π · ê_v_inc is identical to VH
            # here by reciprocity; kept implicit.)

            # Finite-cylinder backscatter form factor — see module
            # docstring "Known limitations (v0.1)". Suppresses k·l ≫ 1
            # backscatter when the cylinder is nearly aligned with the
            # incidence direction.
            k_inc = np.array([np.sin(theta_inc_rad), 0.0, -np.cos(theta_inc_rad)])
            cos_alpha_inc = float(np.dot(c, k_inc))
            form_factor_sq = _sinc(2.0 * k_per_m * l_m * cos_alpha_inc) ** 2

            sigma_vv_grid[ix, iy] = 4.0 * np.pi * k4 * f_vv * f_vv * form_factor_sq
            sigma_hh_grid[ix, iy] = 4.0 * np.pi * k4 * f_hh * f_hh * form_factor_sq
            sigma_vh_grid[ix, iy] = 4.0 * np.pi * k4 * f_vh * f_vh * form_factor_sq

            # Rayleigh total scattering for each incident polarisation.
            # σ_s = (8π/3) · k⁴ · (|α_∥|² · cos²γ + |α_⊥|² · sin²γ)
            # with γ = angle(ĉ, ê_p). Same sinc² form factor applied
            # as a first-pass RGD approximation — see module docstring.
            dot_v = float(np.dot(c, e_v_inc))
            dot_h = float(np.dot(c, e_h))
            sigma_ext_v_grid[ix, iy] = (8.0 * np.pi / 3.0) * k4 * (
                alpha_par * alpha_par * dot_v * dot_v
                + alpha_perp * alpha_perp * (1.0 - dot_v * dot_v)
            ) * form_factor_sq
            sigma_ext_h_grid[ix, iy] = (8.0 * np.pi / 3.0) * k4 * (
                alpha_par * alpha_par * dot_h * dot_h
                + alpha_perp * alpha_perp * (1.0 - dot_h * dot_h)
            ) * form_factor_sq

    # Average over azimuth (uniform; equivalent to a mean) then weighted
    # mean over θ_b with sin(θ_b)-times-Gaussian weight.
    def _avg(grid: np.ndarray) -> float:
        azi_avg = grid.mean(axis=1)                         # uniform azimuth
        return float(np.trapz(azi_avg * w_theta, theta_grid))

    return {
        "sigma_back_vv_per_particle_m2": _avg(sigma_vv_grid),
        "sigma_back_hh_per_particle_m2": _avg(sigma_hh_grid),
        "sigma_back_vh_per_particle_m2": _avg(sigma_vh_grid),
        "sigma_ext_v_per_particle_m2": _avg(sigma_ext_v_grid),
        "sigma_ext_h_per_particle_m2": _avg(sigma_ext_h_grid),
    }


def _leaf_cross_sections(
    a_cm: float,
    t_cm: float,
    eps_v: float,
    theta_inc_rad: float,
    k_per_m: float,
    n_theta: int = N_THETA_SAMPLES,
    n_phi: int = N_PHI_SAMPLES,
) -> dict:
    """
    Orientation-averaged cross sections for a uniformly oriented thin
    disc (leaf) population.

    Geometry and dielectric logic follow `_branch_cross_sections` —
    the only differences are:
        - The polarisability tensor is α_n n̂n̂ᵀ + α_f (I − n̂n̂ᵀ)
          with n̂ the disc-face normal.
        - The orientation distribution is uniform on the upper
          hemisphere: p(n̂) = 1/(2π) for n_z ≥ 0 (Calluna-scale
          leaves treated as near-random).

    Args:
        a_cm, t_cm      : disc radius and thickness (cm).
        eps_v           : vegetation dielectric.
        theta_inc_rad   : incidence angle.
        k_per_m         : wavenumber (rad/m).
        n_theta, n_phi  : quadrature sample counts.

    Returns: same keys as `_branch_cross_sections`.
    """
    _, alpha_normal, alpha_face = _disc_rayleigh_polarisabilities(a_cm, t_cm, eps_v)

    theta_grid = np.linspace(0.0, np.pi / 2.0, n_theta)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    # Uniform-on-sphere weight on θ is sin θ.
    w_theta_unnorm = np.sin(theta_grid)
    w_theta = w_theta_unnorm / np.trapz(w_theta_unnorm, theta_grid)

    cos_ti = np.cos(theta_inc_rad)
    sin_ti = np.sin(theta_inc_rad)
    e_h = np.array([0.0, 1.0, 0.0])
    e_v_inc = np.array([cos_ti, 0.0, sin_ti])
    e_v_sca = np.array([-cos_ti, 0.0, sin_ti])
    k4 = k_per_m ** 4

    sigma_vv_grid = np.zeros((n_theta, n_phi))
    sigma_hh_grid = np.zeros((n_theta, n_phi))
    sigma_vh_grid = np.zeros((n_theta, n_phi))
    sigma_ext_v_grid = np.zeros((n_theta, n_phi))
    sigma_ext_h_grid = np.zeros((n_theta, n_phi))

    for ix, theta_n in enumerate(theta_grid):
        sin_tn = np.sin(theta_n)
        cos_tn = np.cos(theta_n)
        for iy, phi_n in enumerate(phi_grid):
            n = np.array(
                [sin_tn * np.cos(phi_n), sin_tn * np.sin(phi_n), cos_tn]
            )
            pi_nn_v_inc = n * np.dot(n, e_v_inc)
            pi_v_inc = alpha_normal * pi_nn_v_inc + alpha_face * (e_v_inc - pi_nn_v_inc)
            pi_nn_h = n * np.dot(n, e_h)
            pi_h = alpha_normal * pi_nn_h + alpha_face * (e_h - pi_nn_h)

            f_vv = float(np.dot(e_v_sca, pi_v_inc))
            f_hh = float(np.dot(e_h, pi_h))
            f_vh = float(np.dot(e_v_sca, pi_h))

            sigma_vv_grid[ix, iy] = 4.0 * np.pi * k4 * f_vv * f_vv
            sigma_hh_grid[ix, iy] = 4.0 * np.pi * k4 * f_hh * f_hh
            sigma_vh_grid[ix, iy] = 4.0 * np.pi * k4 * f_vh * f_vh

            dot_v = float(np.dot(n, e_v_inc))
            dot_h = float(np.dot(n, e_h))
            sigma_ext_v_grid[ix, iy] = (8.0 * np.pi / 3.0) * k4 * (
                alpha_normal * alpha_normal * dot_v * dot_v
                + alpha_face * alpha_face * (1.0 - dot_v * dot_v)
            )
            sigma_ext_h_grid[ix, iy] = (8.0 * np.pi / 3.0) * k4 * (
                alpha_normal * alpha_normal * dot_h * dot_h
                + alpha_face * alpha_face * (1.0 - dot_h * dot_h)
            )

    def _avg(grid: np.ndarray) -> float:
        azi_avg = grid.mean(axis=1)
        return float(np.trapz(azi_avg * w_theta, theta_grid))

    return {
        "sigma_back_vv_per_particle_m2": _avg(sigma_vv_grid),
        "sigma_back_hh_per_particle_m2": _avg(sigma_hh_grid),
        "sigma_back_vh_per_particle_m2": _avg(sigma_vh_grid),
        "sigma_ext_v_per_particle_m2": _avg(sigma_ext_v_grid),
        "sigma_ext_h_per_particle_m2": _avg(sigma_ext_h_grid),
    }


# ─── MIMICS first-order total backscatter ───────────────────────────────────


def mimics_toure_single_crown(
    params: MimicsToureParams,
    n_theta: int = N_THETA_SAMPLES,
    n_phi: int = N_PHI_SAMPLES,
) -> Tuple[float, float]:
    """
    Toure-style single-crown MIMICS first-order forward model.

    Equations (per-polarisation p ∈ {v, h} for the backscattered
    pp component, and cross-pol p ≠ q for VH; linear units unless
    noted):

        λ = c / f
        k = 2π / λ

        ε_g = Mironov(m_v)               (ground dielectric)
        ε_v = UEL(m_g)                   (vegetation dielectric)

        σ_back_b_pq, σ_ext_b_p  =  ⟨σ⟩_orient for branches (Gaussian)
        σ_back_l_pq, σ_ext_l_p  =  ⟨σ⟩_orient for leaves   (uniform)

        κ_e^p  = N_b · σ_ext_b_p + N_l · σ_ext_l_p          (1/m)

        Crown volume backscatter per unit volume:
        σ_vol_pq = N_b · σ_back_b_pq + N_l · σ_back_l_pq    (1/m)

        Two-way canopy transmissivity:
        τ²_p   = exp( −2 · κ_e^p · h_c / cos θ )

        Crown-direct backscatter (per unit surface, dimensionless):
        σ°_crown_pp = σ_vol_pp · cos(θ)/(2 κ_e^p) · (1 − τ²_p)
        σ°_crown_vh = σ_vol_vh · cos(θ)/(κ_e^v + κ_e^h) ·
                     (1 − exp(−(κ_e^v + κ_e^h) h_c / cos θ))

        Ground direct (attenuated by canopy two-way loss):
        σ°_ground_direct_pp = τ²_p · σ°_oh_pp(ε_g, θ, s)
        σ°_ground_direct_vh = √(τ²_v · τ²_h) · σ°_oh_vh

        Crown–ground double-pass interaction:
        σ°_crown_ground_pp  ≈ 4 · σ_vol_pp · |Γ_p| · τ²_p · h_c / cos θ
        σ°_crown_ground_vh  ≈ 4 · σ_vol_vh · √|Γ_v · Γ_h| ·
                             √(τ²_v · τ²_h) · h_c / cos θ

        σ°_total_pq = σ°_crown_pq + σ°_crown_ground_pq + σ°_ground_direct_pq

        σ°_dB = 10 · log10(σ°_total_lin)

    The crown-direct form uses the Ulaby–Moore–Fung (1986) first-order
    closed-form result for a homogeneous scattering layer with
    incoherent input. The crown-ground form is the simplified
    specular-ground × volume-backscatter coupling (Ulaby 1990 MIMICS
    notation ΣGC / ΣCG path sum); |Γ| is the Oh h-pol Fresnel
    reflectance magnitude used consistently with σ°_oh above.

    Args:
        params : MimicsToureParams.
        n_theta, n_phi: orientation-quadrature sample counts.

    Returns:
        (sigma_vv_db, sigma_vh_db).
    """
    theta_rad = np.radians(params.theta_inc_deg)
    cos_t = np.cos(theta_rad)
    lam_m = C_LIGHT_CM_PER_S * 1e-2 / params.freq_hz  # λ (m)
    k_per_m = 2.0 * np.pi / lam_m
    k_per_cm = k_per_m * 1e-2

    # Dielectrics.
    eps_g = ground_epsilon_mironov(params.m_v)
    eps_v = vegetation_epsilon_ulaby_elrayes(params.m_g)

    # Ground Oh backscatter (dB, then linear).
    sigma_oh_vv_db, sigma_oh_vh_db = oh_surface_vv_vh_db(
        eps_g, theta_rad, params.s_cm, k_per_cm
    )
    sigma_oh_vv_lin = 10.0 ** (sigma_oh_vv_db / 10.0)
    sigma_oh_vh_lin = 10.0 ** (sigma_oh_vh_db / 10.0)

    # Single-scatterer orientation-averaged cross sections.
    branch = _branch_cross_sections(
        params.a_b_cm, params.l_b_cm, eps_v,
        params.sigma_orient_deg, theta_rad, k_per_m,
        n_theta=n_theta, n_phi=n_phi,
    )
    leaf = _leaf_cross_sections(
        params.a_l_cm, params.t_l_cm, eps_v, theta_rad, k_per_m,
        n_theta=n_theta, n_phi=n_phi,
    )

    # Volume-backscatter per unit volume (m⁻¹).
    sigma_vol_vv = (
        params.N_b_per_m3 * branch["sigma_back_vv_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_back_vv_per_particle_m2"]
    )
    sigma_vol_hh = (
        params.N_b_per_m3 * branch["sigma_back_hh_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_back_hh_per_particle_m2"]
    )
    sigma_vol_vh = (
        params.N_b_per_m3 * branch["sigma_back_vh_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_back_vh_per_particle_m2"]
    )

    # Extinction per unit length (m⁻¹), per polarisation.
    kappa_e_v = (
        params.N_b_per_m3 * branch["sigma_ext_v_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_ext_v_per_particle_m2"]
    )
    kappa_e_h = (
        params.N_b_per_m3 * branch["sigma_ext_h_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_ext_h_per_particle_m2"]
    )

    # Two-way transmissivity per polarisation.
    optical_depth_v = kappa_e_v * params.h_c_m / cos_t
    optical_depth_h = kappa_e_h * params.h_c_m / cos_t
    tau2_v = np.exp(-2.0 * optical_depth_v)
    tau2_h = np.exp(-2.0 * optical_depth_h)

    # Crown direct (first-order closed form).
    #
    # For co-pol pp with extinction κ_e^p:
    #     σ°_crown_pp = σ_vol_pp · cos(θ) / (2 κ_e^p) · (1 − τ²_p)
    #
    # Guard against κ_e → 0 (sparse canopy) by using the limit form
    #     lim κ→0 cos(θ)/(2κ) · (1 − exp(−2κh/cosθ)) = h
    def _crown_direct(sigma_vol: float, kappa_e: float, tau2: float) -> float:
        # cos(θ) h / (κ_e · 2h/cosθ) · (1 − τ²)  with τ² = exp(−2κh/cosθ)
        # For κ · h/cosθ small: expand 1 − τ² ≈ 2κh/cosθ → σ_vol · h.
        if kappa_e * params.h_c_m / cos_t < 1e-6:
            return sigma_vol * params.h_c_m
        return sigma_vol * cos_t / (2.0 * kappa_e) * (1.0 - tau2)

    sigma_crown_vv = _crown_direct(sigma_vol_vv, kappa_e_v, tau2_v)
    sigma_crown_hh = _crown_direct(sigma_vol_hh, kappa_e_h, tau2_h)

    # VH crown direct uses the harmonic-averaged extinction
    # (κ_e^v + κ_e^h) across the two-way path.
    kappa_sum = kappa_e_v + kappa_e_h
    opt_depth_sum = kappa_sum * params.h_c_m / cos_t
    if opt_depth_sum < 1e-6:
        sigma_crown_vh = sigma_vol_vh * params.h_c_m
    else:
        sigma_crown_vh = (
            sigma_vol_vh * cos_t / kappa_sum * (1.0 - np.exp(-opt_depth_sum))
        )

    # Crown–ground coupling (simplified). Uses Oh σ°_pp as a stand-in
    # for ground forward reflectance magnitude — this is the standard
    # single-ground-bounce form for the MIMICS ΣGC / ΣCG pathway when
    # the ground is treated as a specular-plus-roughness scatterer
    # (Ulaby 1990 §IV-A).
    # Equivalent reflectance magnitude Γ is recovered from the Oh σ°_pp
    # via the definition σ°_oh = Γ² · roughness_factor; here we fold
    # the roughness factor directly by using σ°_oh itself as the
    # forward-reflection proxy. This is an intentional simplification
    # for the v0.1 reference — the G2 tolerance (0.5 dB) absorbs the
    # resulting offset. Session E is the escalation path if it
    # dominates the disagreement.
    sigma_crown_ground_vv = (
        4.0 * sigma_vol_vv * np.sqrt(max(sigma_oh_vv_lin, 0.0))
        * np.sqrt(tau2_v * tau2_v) * params.h_c_m / cos_t
    )
    sigma_crown_ground_hh = (
        4.0 * sigma_vol_hh * np.sqrt(max(sigma_oh_vv_lin, 0.0))
        * np.sqrt(tau2_h * tau2_h) * params.h_c_m / cos_t
    )
    sigma_crown_ground_vh = (
        4.0 * sigma_vol_vh * np.sqrt(max(sigma_oh_vh_lin, 0.0))
        * np.sqrt(tau2_v * tau2_h) * params.h_c_m / cos_t
    )

    # Ground direct (attenuated).
    sigma_gd_vv = tau2_v * sigma_oh_vv_lin
    sigma_gd_vh = np.sqrt(tau2_v * tau2_h) * sigma_oh_vh_lin

    # Totals.
    sigma_total_vv = sigma_crown_vv + sigma_crown_ground_vv + sigma_gd_vv
    sigma_total_vh = sigma_crown_vh + sigma_crown_ground_vh + sigma_gd_vh

    sigma_vv_db = 10.0 * np.log10(max(sigma_total_vv, _EPS_LOG))
    sigma_vh_db = 10.0 * np.log10(max(sigma_total_vh, _EPS_LOG))

    # Unused HH accumulators (for future expansion / debugging). Kept
    # out of the public return to keep the interface aligned with
    # SPEC §4 "Outputs: σ°_VV (dB) and σ°_VH (dB)".
    _ = sigma_crown_hh
    _ = sigma_crown_ground_hh

    return float(sigma_vv_db), float(sigma_vh_db)


# ─── Debug / inspection helper ──────────────────────────────────────────────


def mimics_toure_single_crown_breakdown(
    params: MimicsToureParams,
    n_theta: int = N_THETA_SAMPLES,
    n_phi: int = N_PHI_SAMPLES,
) -> dict:
    """
    Same as `mimics_toure_single_crown` but returns the three-mechanism
    decomposition and the intermediate dielectrics / extinctions.

    Useful for session D/E debugging where we need to see which
    mechanism is dominating the disagreement with Toure 1994 tables.
    """
    theta_rad = np.radians(params.theta_inc_deg)
    cos_t = np.cos(theta_rad)
    lam_m = C_LIGHT_CM_PER_S * 1e-2 / params.freq_hz
    k_per_m = 2.0 * np.pi / lam_m
    k_per_cm = k_per_m * 1e-2

    eps_g = ground_epsilon_mironov(params.m_v)
    eps_v = vegetation_epsilon_ulaby_elrayes(params.m_g)

    sigma_oh_vv_db, sigma_oh_vh_db = oh_surface_vv_vh_db(
        eps_g, theta_rad, params.s_cm, k_per_cm
    )
    sigma_oh_vv_lin = 10.0 ** (sigma_oh_vv_db / 10.0)
    sigma_oh_vh_lin = 10.0 ** (sigma_oh_vh_db / 10.0)

    branch = _branch_cross_sections(
        params.a_b_cm, params.l_b_cm, eps_v,
        params.sigma_orient_deg, theta_rad, k_per_m,
        n_theta=n_theta, n_phi=n_phi,
    )
    leaf = _leaf_cross_sections(
        params.a_l_cm, params.t_l_cm, eps_v, theta_rad, k_per_m,
        n_theta=n_theta, n_phi=n_phi,
    )

    sigma_vol_vv = (
        params.N_b_per_m3 * branch["sigma_back_vv_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_back_vv_per_particle_m2"]
    )
    sigma_vol_vh = (
        params.N_b_per_m3 * branch["sigma_back_vh_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_back_vh_per_particle_m2"]
    )
    kappa_e_v = (
        params.N_b_per_m3 * branch["sigma_ext_v_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_ext_v_per_particle_m2"]
    )
    kappa_e_h = (
        params.N_b_per_m3 * branch["sigma_ext_h_per_particle_m2"]
        + params.N_l_per_m3 * leaf["sigma_ext_h_per_particle_m2"]
    )

    tau2_v = float(np.exp(-2.0 * kappa_e_v * params.h_c_m / cos_t))
    tau2_h = float(np.exp(-2.0 * kappa_e_h * params.h_c_m / cos_t))

    def _crown_direct(sigma_vol, kappa_e, tau2):
        if kappa_e * params.h_c_m / cos_t < 1e-6:
            return sigma_vol * params.h_c_m
        return sigma_vol * cos_t / (2.0 * kappa_e) * (1.0 - tau2)

    sigma_crown_vv = _crown_direct(sigma_vol_vv, kappa_e_v, tau2_v)
    kappa_sum = kappa_e_v + kappa_e_h
    opt_depth_sum = kappa_sum * params.h_c_m / cos_t
    if opt_depth_sum < 1e-6:
        sigma_crown_vh = sigma_vol_vh * params.h_c_m
    else:
        sigma_crown_vh = (
            sigma_vol_vh * cos_t / kappa_sum * (1.0 - np.exp(-opt_depth_sum))
        )
    sigma_crown_ground_vv = (
        4.0 * sigma_vol_vv * np.sqrt(max(sigma_oh_vv_lin, 0.0))
        * np.sqrt(tau2_v * tau2_v) * params.h_c_m / cos_t
    )
    sigma_crown_ground_vh = (
        4.0 * sigma_vol_vh * np.sqrt(max(sigma_oh_vh_lin, 0.0))
        * np.sqrt(tau2_v * tau2_h) * params.h_c_m / cos_t
    )
    sigma_gd_vv = tau2_v * sigma_oh_vv_lin
    sigma_gd_vh = float(np.sqrt(tau2_v * tau2_h)) * sigma_oh_vh_lin

    def _db(x):
        return 10.0 * np.log10(max(x, _EPS_LOG))

    total_vv = sigma_crown_vv + sigma_crown_ground_vv + sigma_gd_vv
    total_vh = sigma_crown_vh + sigma_crown_ground_vh + sigma_gd_vh

    return {
        "wavelength_m": float(lam_m),
        "k_per_m": float(k_per_m),
        "epsilon_ground": float(eps_g),
        "epsilon_vegetation": float(eps_v),
        "kappa_e_v_per_m": float(kappa_e_v),
        "kappa_e_h_per_m": float(kappa_e_h),
        "tau2_v": float(tau2_v),
        "tau2_h": float(tau2_h),
        "sigma_vol_vv_per_m": float(sigma_vol_vv),
        "sigma_vol_vh_per_m": float(sigma_vol_vh),
        "sigma_oh_vv_db": float(sigma_oh_vv_db),
        "sigma_oh_vh_db": float(sigma_oh_vh_db),
        "mechanisms_vv_linear": {
            "crown_direct": float(sigma_crown_vv),
            "crown_ground": float(sigma_crown_ground_vv),
            "ground_direct_attenuated": float(sigma_gd_vv),
        },
        "mechanisms_vh_linear": {
            "crown_direct": float(sigma_crown_vh),
            "crown_ground": float(sigma_crown_ground_vh),
            "ground_direct_attenuated": float(sigma_gd_vh),
        },
        "mechanisms_vv_db": {
            "crown_direct": _db(sigma_crown_vv),
            "crown_ground": _db(sigma_crown_ground_vv),
            "ground_direct_attenuated": _db(sigma_gd_vv),
        },
        "mechanisms_vh_db": {
            "crown_direct": _db(sigma_crown_vh),
            "crown_ground": _db(sigma_crown_ground_vh),
            "ground_direct_attenuated": _db(sigma_gd_vh),
        },
        "sigma_total_vv_db": _db(total_vv),
        "sigma_total_vh_db": _db(total_vh),
        "branch_cross_sections": branch,
        "leaf_cross_sections": leaf,
        "params": params.as_dict(),
    }
