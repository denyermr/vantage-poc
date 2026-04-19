"""
Unit tests for Phase 1b pre-training gates G3 and G4 and the
supporting `phase1b/physics/oh1992_learnable_s.py` module.

Scope:
  - `phase1b.physics.oh1992_learnable_s` — parameterisation and
    polarimetric behaviour of the learnable-s Oh module.
  - `phase1b.implementation_gate.dielectric_diagnostic` — G4
    numerical correctness and pass-either-way convention.
  - `phase1b.implementation_gate.ks_validity_check` — G3 pass
    logic, monotonicity check, cell-level reporting.

The gates are run entirely forward-mode in production, so the tests
are forward-mode too — no autograd checks here. A later session that
writes the PINN-MIMICS composite loss will add gradient-flow tests.
"""

from __future__ import annotations

import math

import pytest
import torch

from phase1b.physics.oh1992_learnable_s import (
    SENTINEL1_K_PER_CM,
    SENTINEL1_LAMBDA_CM,
    S_MAX_CM,
    S_MIN_CM,
    oh_cross_pol_ratio_db,
    oh_soil_backscatter_dual_pol,
    oh_soil_backscatter_vv,
    s_to_ks,
)
from phase1b.implementation_gate.dielectric_diagnostic import (
    BINDING_THRESHOLD,
    MV_MAX as G4_MV_MAX,
    MV_MIN as G4_MV_MIN,
    compute_dielectric_diagnostic,
)
from phase1b.implementation_gate.ks_validity_check import (
    MONOTONIC_WIGGLE_TOL_DB,
    run_ks_validity_check,
)


# ─── oh1992_learnable_s ─────────────────────────────────────────────────────


class TestSentinel1Constants:
    """SPEC.md §5 sensor-parameter table fixes f = 5.405 GHz."""

    def test_wavelength_near_5_55_cm(self):
        assert SENTINEL1_LAMBDA_CM == pytest.approx(5.547, abs=0.01)

    def test_wavenumber_near_1_13_inverse_cm(self):
        # Lit review Decision 4 quotes k ≈ 1.13 cm⁻¹.
        assert SENTINEL1_K_PER_CM == pytest.approx(1.132, abs=0.01)


class TestSToKs:
    """s → ks conversion: ks = s · k."""

    def test_s_equals_1cm(self):
        ks = float(s_to_ks(1.0))
        assert ks == pytest.approx(SENTINEL1_K_PER_CM, abs=1e-6)

    def test_s_equals_5cm(self):
        ks = float(s_to_ks(5.0))
        assert ks == pytest.approx(5 * SENTINEL1_K_PER_CM, abs=1e-5)

    def test_tensor_input(self):
        s = torch.tensor([1.0, 2.0, 3.0])
        ks = s_to_ks(s)
        assert ks.shape == s.shape
        assert torch.allclose(ks, s * SENTINEL1_K_PER_CM, atol=1e-6)


class TestOhVV:
    """Oh VV forward pass: correctness and monotonicity."""

    def test_finite_at_moor_house_operating_point(self):
        """Typical Moor House state: ε≈30 (Dobson at m_v≈0.5), θ=41.5°, s=2 cm."""
        eps = torch.tensor(30.0)
        theta = torch.tensor(math.radians(41.5))
        s = torch.tensor(2.0)
        vv = oh_soil_backscatter_vv(eps, theta, s)
        assert torch.isfinite(vv)
        # Phase 1 frozen Oh (ks=0.3) at ε=30 gives σ°≈-11 dB; learnable-s
        # at s=2 cm (ks≈2.27) should land in the same ballpark since
        # Oh's dependence on ks is gentle (ks^0.1).
        assert -20.0 < float(vv) < -5.0

    def test_monotonic_in_epsilon(self):
        """σ°_VV must increase with ε (stronger dielectric contrast → more reflection)."""
        eps_vals = torch.tensor([2.0, 5.0, 10.0, 20.0, 40.0, 60.0])
        theta = torch.tensor(math.radians(41.5))
        s = torch.tensor(3.0)
        vv = oh_soil_backscatter_vv(eps_vals, theta, s)
        diffs = vv[1:] - vv[:-1]
        assert (diffs > 0).all(), f"σ°_VV non-monotone in ε: {vv.tolist()}"

    def test_epsilon_clamp_at_1_01(self):
        """ε below 1.01 is clamped; result should equal σ° at exactly 1.01."""
        theta = torch.tensor(math.radians(41.5))
        s = torch.tensor(2.0)
        sigma_low = oh_soil_backscatter_vv(torch.tensor(0.5), theta, s)
        sigma_clamp = oh_soil_backscatter_vv(torch.tensor(1.01), theta, s)
        assert float(sigma_low) == pytest.approx(float(sigma_clamp), abs=1e-4)


class TestOhCrossPolRatio:
    """Oh VH/VV ratio: always ≤ 0 dB (VH ≤ VV) and increases with ks."""

    def test_ratio_is_nonpositive(self):
        theta = torch.tensor(math.radians(41.5))
        for s_cm in [1.0, 3.0, 5.0]:
            for eps in [5.0, 20.0, 40.0]:
                ratio = oh_cross_pol_ratio_db(
                    theta, torch.tensor(s_cm), torch.tensor(eps)
                )
                assert float(ratio) <= 0.0, (
                    f"VH − VV > 0 at s={s_cm}, ε={eps}: {float(ratio):.2f} dB"
                )

    def test_ratio_increases_with_ks(self):
        """q = 0.23 · √Γ₀ · (1 − exp(−ks)) monotone in ks → ratio (dB) monotone in ks."""
        theta = torch.tensor(math.radians(41.5))
        eps = torch.tensor(20.0)
        ratios = [
            float(oh_cross_pol_ratio_db(theta, torch.tensor(s), eps))
            for s in [1.0, 2.0, 3.0, 4.0, 5.0]
        ]
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i - 1] - 1e-6, (
                f"cross-pol ratio non-monotone in s: {ratios}"
            )


class TestOhDualPol:
    """Dual-pol interface: consistency with single-pol functions."""

    def test_vv_matches_single_pol_function(self):
        eps = torch.tensor([5.0, 15.0, 40.0])
        theta = torch.tensor([math.radians(41.5)] * 3)
        s = torch.tensor([2.0, 2.0, 2.0])
        vv_dual, _ = oh_soil_backscatter_dual_pol(eps, theta, s)
        vv_single = oh_soil_backscatter_vv(eps, theta, s)
        assert torch.allclose(vv_dual, vv_single)

    def test_vh_less_than_or_equal_vv(self):
        eps = torch.tensor([5.0, 15.0, 40.0])
        theta = torch.tensor([math.radians(41.5)] * 3)
        s = torch.tensor([2.0, 2.0, 2.0])
        vv, vh = oh_soil_backscatter_dual_pol(eps, theta, s)
        assert (vh <= vv + 1e-6).all()


# ─── G4 dielectric diagnostic ───────────────────────────────────────────────


class TestG4DielectricDiagnostic:
    """G4: Dobson vs Mironov over m_v ∈ [0.25, 0.83]."""

    @pytest.fixture(scope="class")
    def result(self):
        return compute_dielectric_diagnostic()

    def test_gate_passes_either_way(self, result):
        """SPEC §6: diagnostic records the outcome; pass either way."""
        assert result["pass"] is True

    def test_m_v_range_matches_spec(self, result):
        assert result["m_v_range"] == [G4_MV_MIN, G4_MV_MAX]
        assert result["m_v_range"] == [0.25, 0.83]

    def test_binding_threshold_five_percent(self, result):
        assert result["binding_threshold_relative"] == BINDING_THRESHOLD == 0.05

    def test_binding_flag_computed_correctly(self, result):
        """`binding` must be consistent with `max_relative_diff`."""
        expected = result["max_relative_diff"] >= result["binding_threshold_relative"]
        assert result["binding"] == expected

    def test_relative_diff_within_zero_to_one(self, result):
        """Relative diff = |Δε| / max(ε_D, ε_M) is in [0, 1] by construction."""
        assert 0.0 <= result["max_relative_diff"] <= 1.0

    def test_samples_include_expected_keys(self, result):
        s = result["samples"]
        required = {
            "m_v",
            "epsilon_dobson_clamped",
            "epsilon_mironov_clamped",
            "epsilon_dobson_raw",
            "epsilon_mironov_raw",
            "abs_diff_clamped",
            "relative_diff_clamped",
        }
        assert required.issubset(s.keys())
        assert len(s["m_v"]) >= 10


# ─── G3 ks-validity check ───────────────────────────────────────────────────


class TestG3KsValidity:
    """G3: Oh behaviour across (dielectric × s × θ)."""

    @pytest.fixture(scope="class")
    def result(self):
        return run_ks_validity_check()

    def test_all_cells_pass(self, result):
        """With the learnable-s module wired correctly, all cells should pass."""
        assert result["pass"] is True, (
            f"G3 failed on {result['n_cells_failed']} cells; "
            f"failures: {result['failed_cells']}"
        )

    def test_s_grid_covers_spec_range(self, result):
        """SPEC §5 says s ∈ [1, 5] cm; grid must cover both endpoints."""
        assert min(result["s_grid_cm"]) == S_MIN_CM == 1.0
        assert max(result["s_grid_cm"]) == S_MAX_CM == 5.0

    def test_each_cell_reports_finite_sigma(self, result):
        """No NaN/Inf contamination in any cell."""
        for cell in result["cells"]:
            assert cell["n_nan"]["vv"] == 0, cell
            assert cell["n_nan"]["vh"] == 0, cell
            assert cell["n_inf"]["vv"] == 0, cell
            assert cell["n_inf"]["vh"] == 0, cell

    def test_monotonicity_tolerance_honoured(self, result):
        """No cell's worst adjacent decrease may exceed the wiggle tolerance."""
        for cell in result["cells"]:
            assert cell["worst_adjacent_decrease_db"] >= -MONOTONIC_WIGGLE_TOL_DB, (
                f"Cell {cell['dielectric']}/s={cell['s_cm']}/θ={cell['theta_inc_deg']}: "
                f"worst decrease {cell['worst_adjacent_decrease_db']:+.3f} dB "
                f"exceeds tolerance {-MONOTONIC_WIGGLE_TOL_DB:+.3f}"
            )

    def test_observational_envelope_reported(self, result):
        """The Moor House envelope summary must be present for reporting."""
        envelope = result["moor_house_observational_envelope"]
        assert len(envelope) == result["n_cells"]
        for e in envelope:
            assert e["moor_house_mv_range"] == [0.25, 0.83]

    def test_dobson_envelope_matches_observations(self, result):
        """Dobson σ°_VV at Moor House m_v at θ=41.5° should land near Phase 1 observed range."""
        # Phase 1 observed mean VV at Moor House is around -10 to -12 dB.
        # Dobson with s=2 cm should reproduce that.
        for e in result["moor_house_observational_envelope"]:
            if (
                e["dielectric"] == "Dobson"
                and e["s_cm"] == 2.0
                and abs(e["theta_inc_deg"] - 41.5) < 0.01
            ):
                lo, hi = e["sigma_vv_db_range"]
                assert -20.0 < lo < -5.0
                assert -20.0 < hi < -5.0
                return
        pytest.fail("No Dobson/s=2/θ=41.5° envelope cell found")
