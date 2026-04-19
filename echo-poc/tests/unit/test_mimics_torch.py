"""
Unit tests for the differentiable PyTorch MIMICS port at
`phase1b/physics/mimics.py` (session C).

Scope — physics-only. Per the session C plan in
`phase1b/SESSION_PLAN.md`, the equivalence check against the numpy
reference at 0.5 dB is session D's responsibility. These tests verify:

  1. Monotonicity in N_l for σ°_VV — more leaves → higher saturated
     ceiling.
  2. Finite outputs (no NaN / Inf) across the SPEC §5 learnable ranges
     with gradient-enabled forward.
  3. Gradients exist and are finite for each SPEC §5 learnable at the
     range midpoints under a simple loss.backward() pass.
  4. VH ≤ VV at the sparse-canopy probe, mirroring
     `test_sparse_canopy_has_vv_above_vh` from session B.

The tests deliberately avoid comparing σ° values against the numpy
reference: that is the G2 gate and belongs in session D.
"""

from __future__ import annotations

import math

import pytest
import torch

from phase1b.physics.mimics import (
    DOBSON_ALPHA_MINERAL,
    DOBSON_ALPHA_PEAT,
    DOBSON_EPS_DRY_MINERAL,
    DOBSON_EPS_DRY_PEAT,
    DOBSON_EPS_WATER,
    EPSILON_GROUND_MIN,
    MimicsToureParamsTorch,
    N_PHI_SAMPLES,
    N_THETA_SAMPLES,
    _one_minus_exp_over_x,
    ground_epsilon_dobson_torch,
    ground_epsilon_mironov_torch,
    mimics_toure_single_crown,
    mimics_toure_single_crown_breakdown_torch,
    vegetation_epsilon_ulaby_elrayes_torch,
)


# ─── Low-level helpers ──────────────────────────────────────────────────────


class TestMironovTorch:
    """The torch Mironov must be differentiable and produce the reference values."""

    def test_clamped_at_low_mv(self):
        eps = ground_epsilon_mironov_torch(torch.tensor(0.0))
        assert float(eps) == pytest.approx(EPSILON_GROUND_MIN, abs=1e-5)

    def test_monotone_above_clamp(self):
        mvs = torch.linspace(0.5, 0.83, 50)
        eps = torch.stack([ground_epsilon_mironov_torch(m) for m in mvs])
        diffs = eps[1:] - eps[:-1]
        assert bool((diffs >= -1e-6).all())

    def test_gradient_flows_above_clamp(self):
        m_v = torch.tensor(0.6, requires_grad=True)
        eps = ground_epsilon_mironov_torch(m_v)
        eps.backward()
        assert m_v.grad is not None
        assert torch.isfinite(m_v.grad)
        assert float(m_v.grad) > 0.0

    def test_gradient_is_zero_below_clamp(self):
        # At m_v = 0 ε is clamped; the gradient wrt m_v is exactly zero
        # (this is the intended clamp semantics inherited from DEV-007).
        m_v = torch.tensor(0.0, requires_grad=True)
        eps = ground_epsilon_mironov_torch(m_v)
        eps.backward()
        assert m_v.grad is not None
        assert float(m_v.grad) == pytest.approx(0.0, abs=1e-9)


class TestVegetationEpsilonTorch:
    def test_midpoint_value(self):
        eps = vegetation_epsilon_ulaby_elrayes_torch(torch.tensor(0.45))
        # UEL_A + UEL_B · 0.45 = 1.7 + 18·0.45 = 9.8
        assert float(eps) == pytest.approx(9.8, abs=1e-6)

    def test_gradient_equals_uel_b(self):
        m_g = torch.tensor(0.45, requires_grad=True)
        eps = vegetation_epsilon_ulaby_elrayes_torch(m_g)
        eps.backward()
        assert float(m_g.grad) == pytest.approx(18.0, abs=1e-6)


class TestOneMinusExpOverX:
    """The crown-direct small-κ factor must be continuous across x = 0."""

    def test_value_at_zero(self):
        x = torch.tensor(0.0)
        assert float(_one_minus_exp_over_x(x)) == pytest.approx(1.0, abs=1e-9)

    def test_small_x_taylor(self):
        x = torch.tensor(1e-8)
        # Taylor branch: 1 − x/2 + x²/6 ≈ 1 − 5e-9
        val = float(_one_minus_exp_over_x(x))
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_large_x(self):
        # At x = 10, (1 − exp(−10))/10 ≈ (1 − 4.54e-5)/10 ≈ 0.09995.
        # float32 precision absorbs a few ULP, hence abs=1e-5.
        val = float(_one_minus_exp_over_x(torch.tensor(10.0)))
        assert val == pytest.approx((1.0 - math.exp(-10.0)) / 10.0, abs=1e-5)

    def test_monotone_decreasing(self):
        xs = torch.linspace(0.0, 20.0, 200)
        vals = _one_minus_exp_over_x(xs)
        diffs = vals[1:] - vals[:-1]
        assert bool((diffs <= 1e-6).all())


# ─── MIMICS forward — structural / physical invariants ─────────────────────


def _spec5_midpoint_params(**overrides) -> MimicsToureParamsTorch:
    """SPEC §5 midpoint-range parameter set, overridable per test."""
    defaults = dict(
        N_b_per_m3=1.0e3,
        N_l_per_m3=1.0e4,
        sigma_orient_deg=30.0,
        m_g=0.45,
        m_v=0.5,
        s_cm=2.0,
        theta_inc_deg=41.5,
    )
    defaults.update(overrides)
    return MimicsToureParamsTorch(**defaults)


class TestMimicsTorchForward:
    """High-level behavioural tests of the full MIMICS forward."""

    def test_default_params_produce_finite_sigma(self):
        vv, vh = mimics_toure_single_crown(_spec5_midpoint_params())
        assert torch.isfinite(vv)
        assert torch.isfinite(vh)
        # Plausibility envelope matches the numpy reference's bound.
        assert -40.0 < float(vv) < 0.0
        assert -40.0 < float(vh) < 0.0

    def test_sparse_canopy_has_vv_above_vh(self):
        # Mirror `test_sparse_canopy_has_vv_above_vh` from session B —
        # at SPEC §5 lower-bound densities the Rayleigh+sinc² reference
        # gives VH < VV.
        params = _spec5_midpoint_params(
            N_b_per_m3=1e2, N_l_per_m3=1e3,
        )
        vv, vh = mimics_toure_single_crown(params)
        assert float(vv) > float(vh)

    def test_sigma_vv_increases_with_leaf_density(self):
        # Monotonicity in N_l. Small tolerance because the crown-direct
        # ceiling saturates at high density — the reference allows a
        # −0.05 dB tolerance against numerical noise, we do the same.
        results = [
            float(mimics_toure_single_crown(
                _spec5_midpoint_params(N_l_per_m3=N_l)
            )[0])
            for N_l in (1e3, 1e4, 1e5)
        ]
        diffs = [results[i + 1] - results[i] for i in range(len(results) - 1)]
        assert all(d >= -0.05 for d in diffs), (
            f"σ°_VV non-monotone in N_l: {results}"
        )


# ─── Gradient flow across SPEC §5 learnables ────────────────────────────────


class TestMimicsTorchGradients:
    """
    Every SPEC §5 learnable plus the per-obs m_v must carry a finite,
    non-NaN gradient from a simple σ°.sum().backward() pass.
    """

    @pytest.mark.parametrize(
        "name,init",
        [
            ("N_b_per_m3", 1.0e3),
            ("N_l_per_m3", 1.0e4),
            ("sigma_orient_deg", 30.0),
            ("m_g", 0.45),
            ("m_v", 0.5),
            ("s_cm", 2.0),
        ],
    )
    def test_gradient_finite_for_each_learnable(self, name: str, init: float):
        leaf = torch.tensor(init, dtype=torch.float32, requires_grad=True)
        kwargs = {name: leaf}
        params = _spec5_midpoint_params(**kwargs)
        vv, vh = mimics_toure_single_crown(params)
        loss = vv + vh
        loss.backward()
        assert leaf.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(leaf.grad), f"{name} has non-finite gradient: {leaf.grad}"

    def test_simultaneous_gradient_for_all_learnables(self):
        # All five SPEC §5 learnables plus per-obs m_v in one forward pass.
        N_b = torch.tensor(1.0e3, requires_grad=True)
        N_l = torch.tensor(1.0e4, requires_grad=True)
        sigma = torch.tensor(30.0, requires_grad=True)
        m_g = torch.tensor(0.45, requires_grad=True)
        s_cm = torch.tensor(2.0, requires_grad=True)
        m_v = torch.tensor(0.5, requires_grad=True)

        params = MimicsToureParamsTorch(
            N_b_per_m3=N_b, N_l_per_m3=N_l, sigma_orient_deg=sigma,
            m_g=m_g, m_v=m_v, s_cm=s_cm,
        )
        vv, vh = mimics_toure_single_crown(params)
        (vv + vh).backward()

        for name, leaf in {
            "N_b": N_b, "N_l": N_l, "sigma_orient": sigma,
            "m_g": m_g, "s_cm": s_cm, "m_v": m_v,
        }.items():
            assert leaf.grad is not None, f"{name} missing gradient"
            assert torch.isfinite(leaf.grad), (
                f"{name} has non-finite gradient: {leaf.grad}"
            )


# ─── Finite outputs under SPEC §5 learnable-range sweeps ────────────────────


class TestMimicsTorchRangeSweeps:
    """Finite (no NaN / Inf) across the SPEC §5 learnable ranges."""

    @pytest.mark.parametrize(
        "field,values",
        [
            # N_b ∈ [10², 10⁴] m⁻³ (SPEC §5)
            ("N_b_per_m3", [1e2, 1e3, 1e4]),
            # N_l ∈ [10³, 10⁵] m⁻³ (SPEC §5)
            ("N_l_per_m3", [1e3, 1e4, 1e5]),
            # σ_orient ∈ [15°, 60°] (SPEC §5)
            ("sigma_orient_deg", [15.0, 30.0, 60.0]),
            # m_g ∈ [0.3, 0.6] g/g (SPEC §5)
            ("m_g", [0.3, 0.45, 0.6]),
            # s ∈ [1, 5] cm (SPEC §5 / §7)
            ("s_cm", [1.0, 3.0, 5.0]),
            # m_v ∈ [0.0, 0.88] cm³/cm³ (SPEC §5, per-obs)
            ("m_v", [0.0, 0.25, 0.5, 0.83]),
        ],
    )
    def test_finite_outputs_along_range(self, field: str, values: list):
        for v in values:
            params = _spec5_midpoint_params(**{field: v})
            vv, vh = mimics_toure_single_crown(params)
            assert torch.isfinite(vv), f"{field}={v} gave non-finite σ°_VV={vv}"
            assert torch.isfinite(vh), f"{field}={v} gave non-finite σ°_VH={vh}"


# ─── Phase E-1 additions (DEV-1b-004 + P3) ──────────────────────────────────


class TestDobsonTorch:
    """
    `ground_epsilon_dobson_torch` (added in Phase E-1 per DEV-1b-004) is
    a parameterised differentiable Dobson 1985 dielectric with default
    kwargs = Moor House peat (bit-identical to the frozen
    `phase1/physics/dielectric.DobsonDielectric`). The G2 harness calls
    it with T94-consistent mineral-soil kwargs for rows E.1 / E.2 only.
    """

    def test_peat_defaults_match_frozen_dobson(self):
        """
        Default kwargs must reproduce `phase1.physics.dielectric.DobsonDielectric`
        bit-identically. This test is the guardrail against accidental
        drift between the Phase 1b torch Dobson and the Tier 1 frozen
        one.
        """
        from phase1.physics.dielectric import DobsonDielectric
        frozen = DobsonDielectric()
        mvs = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.83])
        eps_p1 = frozen(mvs)
        eps_p1b = ground_epsilon_dobson_torch(mvs)
        assert torch.allclose(eps_p1, eps_p1b, atol=1e-6), (
            f"Phase 1b Dobson drifted from Phase 1 frozen at default kwargs: "
            f"{eps_p1} vs {eps_p1b}"
        )

    def test_mineral_kwargs_unclamped_at_t94_wheat_mv(self):
        """
        At T94 Table IV(a) wheat reference m_v = 0.2 g/cm³ under
        Dobson-mineral, ε must be well above the DEV-007 ε ≥ 1.01 floor
        (i.e. unclamped), with a non-zero gradient ∂ε/∂m_v. This is the
        core DEV-1b-004 claim — the peat-Mironov clamp is no longer
        binding at this operating point.
        """
        m_v = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        eps = ground_epsilon_dobson_torch(
            m_v,
            eps_dry=DOBSON_EPS_DRY_MINERAL,
            eps_water=DOBSON_EPS_WATER,
            alpha=DOBSON_ALPHA_MINERAL,
        )
        assert float(eps) > 5.0, f"Dobson-mineral at m_v=0.2 is unexpectedly low: ε={float(eps)}"
        grad = torch.autograd.grad(eps, m_v)[0]
        assert float(grad) > 0.0, f"∂ε/∂m_v should be positive, got {float(grad)}"

    def test_clamp_preserved_at_zero_mv(self):
        """
        At m_v = 0 the DEV-007 ε ≥ 1.01 clamp must still be applied by
        `ground_epsilon_dobson_torch` (Dobson ε_dry = 3.5 at peat
        defaults so the clamp is inactive in practice, but the machinery
        must be in place so the function's semantics match the Mironov
        version).
        """
        m_v = torch.tensor(0.0)
        eps = ground_epsilon_dobson_torch(m_v)
        # Peat defaults: ε_dry = 3.5 > 1.01, clamp not active.
        assert float(eps) == pytest.approx(DOBSON_EPS_DRY_PEAT, abs=1e-6)
        # Artificial sub-floor case.
        eps_sub = ground_epsilon_dobson_torch(
            torch.tensor(0.0), eps_dry=0.5,
        )
        assert float(eps_sub) >= EPSILON_GROUND_MIN - 1e-6

    def test_gradient_nonzero_at_t94_wheat(self):
        """
        Integration-style check: ∂σ°_VV/∂m_v at T94 wheat reference
        m_v=0.2 under Dobson-mineral is non-zero — the Phase E-1 fix for
        gradient-arm rows E.1 and E.2.
        """
        m_v = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        dobson_mineral = lambda mv: ground_epsilon_dobson_torch(
            mv, eps_dry=DOBSON_EPS_DRY_MINERAL,
            eps_water=DOBSON_EPS_WATER, alpha=DOBSON_ALPHA_MINERAL,
        )
        params = MimicsToureParamsTorch(
            h_c_m=0.4, a_b_cm=0.1, l_b_cm=0.4 * 50.0,
            a_l_cm=0.5, t_l_cm=0.02, l_corr_cm=5.0,
            freq_hz=1.25e9, N_b_per_m3=500.0, N_l_per_m3=3000.0,
            sigma_orient_deg=5.0, m_g=0.7,
            m_v=m_v, s_cm=1.0, theta_inc_deg=30.0,
        )
        vv, _ = mimics_toure_single_crown(
            params, ground_dielectric_fn=dobson_mineral,
        )
        grad = torch.autograd.grad(vv, m_v)[0]
        # Non-zero and finite. Magnitude check deliberately loose —
        # closing to T94's published 1.21 dB/ζ (per DEV-1b-006) is a
        # Phase E-2 concern, not a Phase E-1 unit test.
        assert float(abs(grad)) > 0.0
        assert torch.isfinite(grad)


class TestTorchBreakdown:
    """
    `mimics_toure_single_crown_breakdown_torch` (P3 deliverable) must
    produce mechanism values that sum (in linear space) to the total
    σ° returned by `mimics_toure_single_crown` on the same inputs.
    Extracting from the same intermediate tensors guarantees numerical
    consistency at machine precision.
    """

    def test_breakdown_sum_matches_total(self):
        params = _spec5_midpoint_params()
        with torch.no_grad():
            bd = mimics_toure_single_crown_breakdown_torch(params)
            vv, vh = mimics_toure_single_crown(params)
        mech_vv = bd["mechanisms_vv_linear"]
        sum_vv_lin = (
            mech_vv["crown_direct"]
            + mech_vv["crown_ground"]
            + mech_vv["ground_direct_attenuated"]
        )
        total_vv_db = 10.0 * torch.log10(sum_vv_lin.clamp(min=1e-12))
        assert torch.allclose(total_vv_db, vv, atol=1e-4), (
            f"breakdown linear sum ({float(total_vv_db)} dB) disagrees with "
            f"`mimics_toure_single_crown` total ({float(vv)} dB)"
        )

    def test_breakdown_keys_match_numpy_reference(self):
        """
        The torch breakdown dict must expose the same top-level keys
        and mechanism names as the numpy reference breakdown helper, so
        callers can use either interchangeably.
        """
        from phase1b.physics.reference_mimics.reference_toure import (
            mimics_toure_single_crown_breakdown as numpy_bd,
        )
        from phase1b.physics.reference_mimics.reference_toure import (
            MimicsToureParams,
        )

        params_torch = _spec5_midpoint_params()
        params_np = MimicsToureParams(
            h_c_m=0.4, a_b_cm=0.2, l_b_cm=8.0, a_l_cm=1.0, t_l_cm=0.03,
            l_corr_cm=5.0, freq_hz=5.405e9,
            N_b_per_m3=1e3, N_l_per_m3=1e4, sigma_orient_deg=30.0,
            m_g=0.45, m_v=0.5, s_cm=2.0, theta_inc_deg=41.5,
        )
        with torch.no_grad():
            bd_t = mimics_toure_single_crown_breakdown_torch(params_torch)
        bd_n = numpy_bd(params_np)

        # Top-level keys: torch breakdown is a proper subset of numpy
        # (numpy also exposes branch_cross_sections and leaf_cross_sections
        # internals for session B debugging; torch does not).
        required_top_keys = {
            "mechanisms_vv_linear", "mechanisms_vh_linear",
            "mechanisms_vv_db", "mechanisms_vh_db",
            "sigma_total_vv_db", "sigma_total_vh_db",
            "epsilon_ground", "epsilon_vegetation",
            "kappa_e_v_per_m", "kappa_e_h_per_m",
            "tau2_v", "tau2_h",
        }
        assert required_top_keys.issubset(bd_t.keys()), (
            f"torch breakdown missing keys: {required_top_keys - bd_t.keys()}"
        )
        assert required_top_keys.issubset(bd_n.keys()), (
            f"numpy breakdown missing keys: {required_top_keys - bd_n.keys()}"
        )
        # Mechanism names inside the per-pol dicts must match.
        mech_names = {"crown_direct", "crown_ground", "ground_direct_attenuated"}
        assert set(bd_t["mechanisms_vv_db"].keys()) == mech_names
        assert set(bd_n["mechanisms_vv_db"].keys()) == mech_names

    def test_breakdown_agrees_with_numpy_reference_on_totals(self):
        """
        The torch breakdown's total σ° must agree with the numpy
        reference's total σ° on the same Moor-House-midpoint inputs
        within 0.5 dB — this is the same invariant the numpy_port G2
        arm checks.
        """
        from phase1b.physics.reference_mimics.reference_toure import (
            mimics_toure_single_crown_breakdown as numpy_bd,
            MimicsToureParams,
        )
        params_torch = _spec5_midpoint_params()
        params_np = MimicsToureParams(
            h_c_m=0.4, a_b_cm=0.2, l_b_cm=8.0, a_l_cm=1.0, t_l_cm=0.03,
            l_corr_cm=5.0, freq_hz=5.405e9,
            N_b_per_m3=1e3, N_l_per_m3=1e4, sigma_orient_deg=30.0,
            m_g=0.45, m_v=0.5, s_cm=2.0, theta_inc_deg=41.5,
        )
        with torch.no_grad():
            bd_t = mimics_toure_single_crown_breakdown_torch(params_torch)
        bd_n = numpy_bd(params_np)
        # Totals within 0.5 dB (numpy_port arm tolerance).
        assert abs(float(bd_t["sigma_total_vv_db"]) - bd_n["sigma_total_vv_db"]) < 0.5
        assert abs(float(bd_t["sigma_total_vh_db"]) - bd_n["sigma_total_vh_db"]) < 0.5


class TestMoorHouseProductionPinning:
    """
    Regression: the Moor House production call signature must remain
    `mimics_toure_single_crown(params)` with `ground_dielectric_fn=None`,
    so that the DEV-1b-004 harness-only Dobson path cannot accidentally
    leak into training or inference. DEV-1b-004 Resolution § requires
    this regression test.
    """

    def test_default_ground_dielectric_is_none_implies_mironov(self):
        """
        Calling without `ground_dielectric_fn` must yield the same σ° as
        calling with `ground_dielectric_fn=None` (both → Mironov path).
        """
        params = _spec5_midpoint_params()
        with torch.no_grad():
            vv_a, vh_a = mimics_toure_single_crown(params)
            vv_b, vh_b = mimics_toure_single_crown(params, ground_dielectric_fn=None)
        assert torch.allclose(vv_a, vv_b)
        assert torch.allclose(vh_a, vh_b)

    def test_mironov_is_the_default_path(self):
        """
        Production path ε at Moor House m_v = 0.5 must come from
        Mironov, not from any Dobson variant. Direct probe via the
        breakdown helper's `epsilon_ground` field.
        """
        params = _spec5_midpoint_params(m_v=0.5)
        with torch.no_grad():
            bd = mimics_toure_single_crown_breakdown_torch(params)
        # At m_v = 0.5 (below Mironov's mv_t = 0.36 crossover, but the
        # Mironov linear branches + square + clamp produce ε ≈ 1.055
        # which is near the clamp floor for peat-Mironov). The Dobson
        # peat values at m_v = 0.5 would give ε ≈ 31, very different.
        eps_production = float(bd["epsilon_ground"])
        eps_dobson_peat = float(ground_epsilon_dobson_torch(torch.tensor(0.5)))
        assert eps_production < eps_dobson_peat, (
            f"Production ε={eps_production} is not the Mironov path "
            f"(Dobson-peat would give {eps_dobson_peat})"
        )

    def test_dobson_mineral_kwargs_can_coexist_with_production(self):
        """
        The harness can pass a custom dielectric callable in one call
        without affecting a subsequent production call. No module-level
        state leaks.

        Uses a sparse-canopy + high-m_v regime where the ground
        contribution is visible in the total σ°, so the Mironov /
        Dobson-mineral dielectric difference propagates to the output.
        At SPEC §5 midpoints the canopy is optically thick and the
        ground path is fully saturated, hiding any dielectric-choice
        difference from the total σ° — that is a feature of the forward
        model at dense-canopy operating points, not a test failure.
        """
        # Sparse canopy (N_b = 1e2, N_l = 1e3 — SPEC §5 lower bounds),
        # high m_v = 0.80 (above Mironov's clamp floor at ~0.56 so both
        # dielectrics produce distinct, finite ε values).
        params = _spec5_midpoint_params(
            N_b_per_m3=1e2, N_l_per_m3=1e3, m_v=0.80,
        )
        dobson_mineral = lambda mv: ground_epsilon_dobson_torch(
            mv, eps_dry=DOBSON_EPS_DRY_MINERAL, alpha=DOBSON_ALPHA_MINERAL,
        )
        with torch.no_grad():
            vv_harness, _ = mimics_toure_single_crown(
                params, ground_dielectric_fn=dobson_mineral,
            )
            vv_production, _ = mimics_toure_single_crown(params)
        # The two calls must produce different totals — distinct ε paths.
        # Dobson-mineral at m_v=0.80 → ε ≈ 73; Mironov at m_v=0.80 →
        # ε ≈ 1.48. Very different ground σ° → very different total σ°.
        assert not torch.allclose(vv_harness, vv_production, atol=0.5), (
            "harness Dobson call and production Mironov call produced "
            f"identical σ° (Dobson={float(vv_harness):.3f}, "
            f"Mironov={float(vv_production):.3f}); one of the paths is "
            "not being used"
        )


class TestSetDExemptMessage:
    """
    DEV-1b-005: Set D's `use_trunk_layer=True` path remains
    NotImplemented in Phase 1b; the exception message must cross-
    reference DEV-1b-005 and Phase 1c so future callers know where to
    look.
    """

    def test_use_trunk_layer_true_still_raises(self):
        params = _spec5_midpoint_params()
        with pytest.raises(NotImplementedError) as exc_info:
            mimics_toure_single_crown(params, use_trunk_layer=True)
        msg = str(exc_info.value).lower()
        assert "phase 1c" in msg or "dev-1b-005" in msg, (
            f"NotImplementedError message should reference DEV-1b-005 "
            f"/ Phase 1c per DEV-1b-005; got: {exc_info.value}"
        )
