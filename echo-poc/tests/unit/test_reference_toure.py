"""
Unit tests for `phase1b/physics/reference_mimics/reference_toure.py`
and the companion canonical-combinations generator.

Scope (session B):
  - Cross-checks of the numpy-ported dielectric and surface models
    against the frozen `phase1/physics/dielectric.py` and
    `phase1b/physics/oh1992_learnable_s.py` PyTorch implementations
    (the independence of the code path is only useful for the G2
    gate if the two paths agree when inputs match).
  - Structural checks on the MIMICS first-order forward (finite
    outputs, VH ≤ VV for the canonical sparse-canopy probe, non-zero
    VV/VH contrast once the crown is not fully saturated, preservation
    of the three-mechanism decomposition).
  - `canonical_combinations.json` round-trip: regeneration is
    deterministic, numpy-port rows are over-written, other sources are
    preserved.

The G2 gate itself (numerical equivalence between PyTorch MIMICS and
these references at 0.5 dB) is tested in a separate module written in
session D.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phase1.physics.dielectric import DobsonDielectric, MironovDielectric
from phase1b.physics.oh1992_learnable_s import (
    oh_soil_backscatter_dual_pol,
    SENTINEL1_K_PER_CM,
)
from phase1b.physics.reference_mimics.reference_toure import (
    C_LIGHT_CM_PER_S,
    EPSILON_GROUND_MIN,
    MIRONOV_MV_T,
    MimicsToureParams,
    PEAT_THETA_SAT,
    SENTINEL1_FREQ_HZ,
    ground_epsilon_mironov,
    mimics_toure_single_crown,
    mimics_toure_single_crown_breakdown,
    oh_surface_vv_vh_db,
    vegetation_epsilon_ulaby_elrayes,
    _cylinder_rayleigh_polarisabilities,
    _disc_rayleigh_polarisabilities,
    _sinc,
)
from phase1b.physics.reference_mimics.generate_numpy_port_combinations import (
    CANONICAL_JSON,
    generate_combinations,
    write_canonical,
)


# ─── Dielectric equivalence with frozen PyTorch versions ────────────────────


class TestMironovGroundDielectric:
    """The numpy Mironov path must match the PyTorch Mironov at float level."""

    @pytest.mark.parametrize("m_v", [0.0, 0.1, 0.25, MIRONOV_MV_T, 0.5, 0.7, 0.83])
    def test_matches_phase1_mironov_within_clamp(self, m_v: float):
        # Phase 1 Mironov returns raw n² without the DEV-007 clamp; apply
        # the clamp on the reference side too so both paths are in the
        # same regime.
        torch_eps = float(MironovDielectric()(torch.tensor(m_v)))
        torch_eps_clamped = max(torch_eps, EPSILON_GROUND_MIN)
        numpy_eps = ground_epsilon_mironov(m_v)
        assert numpy_eps == pytest.approx(torch_eps_clamped, abs=1e-6)

    def test_clamp_enforced(self):
        # At m_v = 0 the raw Mironov n² = 0.312² ≈ 0.097 → below clamp.
        assert ground_epsilon_mironov(0.0) == pytest.approx(EPSILON_GROUND_MIN, abs=1e-9)

    def test_monotone_over_observational_range(self):
        # Above the clamp floor the numpy Mironov should be non-decreasing.
        mvs = np.linspace(0.5, 0.83, 50)
        eps = np.array([ground_epsilon_mironov(m) for m in mvs])
        diffs = np.diff(eps)
        # Allow exact ties where the clamp is active.
        assert (diffs >= -1e-9).all()


class TestUlabyElRayesVegetation:
    """Vegetation-dielectric sanity."""

    def test_monotone_in_mg(self):
        mgs = np.linspace(0.0, 1.0, 50)
        eps = np.array([vegetation_epsilon_ulaby_elrayes(m) for m in mgs])
        diffs = np.diff(eps)
        assert (diffs > 0).all()

    def test_dry_value_reasonable(self):
        # Dry vegetation should have ε close to 2 (Ulaby & El-Rayes 1987).
        assert vegetation_epsilon_ulaby_elrayes(0.0) == pytest.approx(1.7, abs=0.1)

    def test_moist_value_c_band_reference(self):
        # Reference: ε ≈ 9.8 at m_g = 0.45 g/g (see module docstring).
        assert vegetation_epsilon_ulaby_elrayes(0.45) == pytest.approx(9.8, abs=0.2)


# ─── Oh 1992 numpy port vs PyTorch ──────────────────────────────────────────


class TestOhNumpyMatchesTorch:
    """The numpy Oh and the torch Oh must give identical σ° at float level."""

    @pytest.mark.parametrize("m_v,s_cm,theta_deg", [
        (0.25, 1.0, 41.5),
        (0.5, 2.0, 41.5),
        (0.7, 3.0, 29.0),
        (0.83, 5.0, 46.0),
    ])
    def test_dual_pol_agreement_under_1e3_db(
        self, m_v: float, s_cm: float, theta_deg: float,
    ):
        eps = ground_epsilon_mironov(m_v)
        theta_rad = np.radians(theta_deg)
        vv_np, vh_np = oh_surface_vv_vh_db(
            eps, theta_rad, s_cm, SENTINEL1_K_PER_CM
        )
        vv_t, vh_t = oh_soil_backscatter_dual_pol(
            torch.tensor(eps, dtype=torch.float32),
            torch.tensor(theta_rad, dtype=torch.float32),
            torch.tensor(s_cm, dtype=torch.float32),
        )
        assert vv_np == pytest.approx(float(vv_t), abs=1e-3)
        assert vh_np == pytest.approx(float(vh_t), abs=1e-3)


# ─── Scatterer polarisabilities ──────────────────────────────────────────────


class TestCylinderPolarisabilities:
    """
    Rayleigh polarisabilities of a prolate ellipsoid at heather-branch
    scale (a/l = 0.025) approach the infinitely-prolate limit
        L_∥ → 0,   L_⊥ → 1/2.
    """

    def test_prolate_limit(self):
        _, a_par, a_perp = _cylinder_rayleigh_polarisabilities(
            a_cm=0.2, l_cm=8.0, eps_v=9.8
        )
        # Very prolate → α_∥ approaches V(ε−1).
        V = np.pi * (0.2e-2) ** 2 * 2 * 8.0e-2
        assert a_par == pytest.approx(V * (9.8 - 1.0), rel=0.02)
        # α_⊥ approaches 2V(ε−1)/(ε+1).
        assert a_perp == pytest.approx(2 * V * (9.8 - 1.0) / (9.8 + 1.0), rel=0.05)

    def test_sphere_limit(self):
        # At a = l the code falls back to L = 1/3; α = V(ε−1)/(1 + (ε−1)/3).
        _, a_par, a_perp = _cylinder_rayleigh_polarisabilities(
            a_cm=1.0, l_cm=1.0, eps_v=9.8
        )
        assert a_par == pytest.approx(a_perp, rel=1e-6)


class TestDiscPolarisabilities:
    """
    Thin oblate limit: L_normal → 1, L_face → 0.
        α_normal ≈ V(ε-1)/ε,   α_face ≈ V(ε-1).
    """

    def test_thin_disc_limit(self):
        # For t/a = 0.03 the finite-aspect-ratio correction leaves L_face
        # ≈ 0.023 (not quite zero) and L_normal ≈ 0.954 (not quite one),
        # so α_face and α_normal sit ~15-20% inside the exact
        # thin-disc asymptote. The ratio test below is tighter and is
        # the one that actually pins the limit behaviour.
        V_expected = np.pi * (1e-2) ** 2 * 3e-4  # a = 1 cm, t = 0.03 cm
        _, a_n, a_f = _disc_rayleigh_polarisabilities(
            a_cm=1.0, t_cm=0.03, eps_v=9.8
        )
        assert a_f == pytest.approx(V_expected * (9.8 - 1.0), rel=0.25)
        assert a_n == pytest.approx(V_expected * (9.8 - 1.0) / 9.8, rel=0.2)
        # Ratio α_face / α_normal → ε in the thin-disc limit. For t/a = 0.03
        # the approach to ε is imperfect (~17%); use 0.3 for the relative
        # tolerance.
        assert (a_f / a_n) == pytest.approx(9.8, rel=0.3)

    def test_thinner_disc_ratio_closer_to_epsilon(self):
        # Tightening t/a should bring α_face / α_normal closer to ε.
        _, a_n_thick, a_f_thick = _disc_rayleigh_polarisabilities(
            a_cm=1.0, t_cm=0.1, eps_v=9.8
        )
        _, a_n_thin, a_f_thin = _disc_rayleigh_polarisabilities(
            a_cm=1.0, t_cm=0.005, eps_v=9.8
        )
        ratio_thick = a_f_thick / a_n_thick
        ratio_thin = a_f_thin / a_n_thin
        assert ratio_thin > ratio_thick
        assert ratio_thin == pytest.approx(9.8, rel=0.15)


class TestSincHelper:
    def test_zero_point(self):
        assert _sinc(0.0) == 1.0

    def test_at_pi(self):
        assert _sinc(np.pi) == pytest.approx(0.0, abs=1e-9)

    def test_vector(self):
        x = np.array([0.0, np.pi, 2 * np.pi])
        out = _sinc(x)
        assert isinstance(out, np.ndarray)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.0, abs=1e-9)


# ─── MIMICS first-order forward ──────────────────────────────────────────────


class TestMimicsForward:
    def test_default_params_produce_finite_sigma(self):
        vv, vh = mimics_toure_single_crown(MimicsToureParams())
        assert np.isfinite(vv)
        assert np.isfinite(vh)
        # Plausibility envelope — C-band heather-canopy σ° should live
        # in [-40, 0] dB in both co- and cross-pol for SPEC §5 params.
        assert -40.0 < vv < 0.0
        assert -40.0 < vh < 0.0

    def test_sparse_canopy_has_vv_above_vh(self):
        # At SPEC §5 lower-bound densities the crown is not yet saturated,
        # and the v0.1 Rayleigh+sinc² reference gives VH < VV.
        params = MimicsToureParams(N_b_per_m3=1e2, N_l_per_m3=1e3)
        vv, vh = mimics_toure_single_crown(params)
        assert vv > vh

    def test_sigma_vv_increases_with_leaf_density(self):
        # N_l goes from 10³ → 10⁵; σ° VV should be non-decreasing in dB
        # (more leaves → more crown-direct backscatter at saturation).
        results = [
            mimics_toure_single_crown(MimicsToureParams(N_l_per_m3=N_l))[0]
            for N_l in (1e3, 1e4, 1e5)
        ]
        diffs = np.diff(results)
        assert (diffs >= -0.05).all(), f"non-monotone in N_l: {results}"

    def test_moor_house_m_v_sensitivity_in_sparse_probe(self):
        # In the sparse-canopy probe, σ° should measurably depend on m_v
        # above the Mironov clamp. At m_v = 0.83 (Mironov ε ≈ 1.54) the
        # ground-direct term re-emerges relative to m_v = 0.25
        # (ε clamped to 1.01).
        low = mimics_toure_single_crown(
            MimicsToureParams(N_b_per_m3=1e2, N_l_per_m3=1e3, m_v=0.25, s_cm=1.0)
        )[0]
        high = mimics_toure_single_crown(
            MimicsToureParams(N_b_per_m3=1e2, N_l_per_m3=1e3, m_v=0.83, s_cm=1.0)
        )[0]
        # High-m_v should give higher σ° — even a modest 0.3 dB
        # difference is enough to certify that the clamp boundary is
        # wired correctly.
        assert high > low

    def test_breakdown_sum_matches_total(self):
        bd = mimics_toure_single_crown_breakdown(MimicsToureParams())
        # Reconstruct total from linear mechanisms.
        total_lin = sum(bd["mechanisms_vv_linear"].values())
        recon_db = 10.0 * np.log10(total_lin + 1e-18)
        assert recon_db == pytest.approx(bd["sigma_total_vv_db"], abs=1e-6)

    def test_wavelength_matches_sentinel1(self):
        bd = mimics_toure_single_crown_breakdown(MimicsToureParams())
        expected_lambda = C_LIGHT_CM_PER_S * 1e-2 / SENTINEL1_FREQ_HZ
        assert bd["wavelength_m"] == pytest.approx(expected_lambda, rel=1e-9)


# ─── Canonical-combinations round-trip ──────────────────────────────────────


class TestCanonicalCombinations:
    def test_regeneration_is_deterministic(self):
        """Two back-to-back generate_combinations() calls produce identical σ°."""
        a = generate_combinations()
        b = generate_combinations()
        assert len(a) == len(b)
        for ea, eb in zip(a, b):
            assert ea["id"] == eb["id"]
            assert ea["reference_sigma"] == eb["reference_sigma"]

    def test_schema_keys_stable(self):
        combos = generate_combinations()
        for entry in combos:
            assert set(entry.keys()) == {
                "id", "source", "parameters", "reference_sigma", "notes",
            }
            assert set(entry["source"]).issuperset({
                "type", "module", "function", "code_sha256", "version",
            })
            assert entry["source"]["type"] == "numpy_port"
            assert set(entry["reference_sigma"]) == {"sigma_vv_db", "sigma_vh_db"}

    def test_contains_sparse_probe_entries(self):
        combos = generate_combinations()
        ids = [c["id"] for c in combos]
        # At least one sparse-canopy probe must be present so the G2
        # equivalence check covers a regime where the ground terms are
        # numerically non-negligible.
        assert any("sparse" in i for i in ids)

    def test_covers_spec_learnable_range_endpoints(self):
        combos = generate_combinations()
        s_values = {c["parameters"]["s_cm"] for c in combos}
        m_v_values = {c["parameters"]["m_v"] for c in combos}
        # SPEC §5 / §7 s range endpoints must both be present.
        assert 1.0 in s_values
        assert 5.0 in s_values
        # Moor House envelope endpoints.
        assert any(abs(v - 0.25) < 1e-9 for v in m_v_values)
        assert any(abs(v - 0.83) < 1e-9 for v in m_v_values)

    def test_json_round_trip_preserves_non_numpy_port(self, tmp_path: Path):
        """Manually injected non-numpy_port rows must survive regeneration."""
        path = tmp_path / "canonical.json"
        # Seed with a manual entry.
        seed = {
            "schema_version": 1,
            "description": "test",
            "tolerance_db": 0.5,
            "created_at": None,
            "last_regenerated_at": None,
            "combinations": [
                {
                    "id": "manual_seed",
                    "source": {"type": "manual"},
                    "parameters": {"dummy": 1.0},
                    "reference_sigma": {"sigma_vv_db": -10.0, "sigma_vh_db": -15.0},
                    "notes": "seeded for round-trip test",
                }
            ],
        }
        path.write_text(json.dumps(seed, indent=2), encoding="utf-8")

        out = write_canonical(path)
        kept = [c for c in out["combinations"] if c["source"]["type"] != "numpy_port"]
        assert any(c["id"] == "manual_seed" for c in kept)
        # And numpy-port entries must have been added.
        added = [c for c in out["combinations"] if c["source"]["type"] == "numpy_port"]
        assert len(added) > 0

    def test_canonical_json_exists_and_matches_current_reference(self):
        """
        The checked-in canonical_combinations.json must be in sync with
        what the current reference code produces. This guards against
        drift between `reference_toure.py` and the canonical file — if
        someone edits the reference without re-running the generator,
        this test fires.
        """
        assert CANONICAL_JSON.exists(), (
            f"Expected canonical_combinations.json at {CANONICAL_JSON}. "
            "Run `python phase1b/physics/reference_mimics/"
            "generate_numpy_port_combinations.py`."
        )
        stored = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        stored_numpy = [
            c for c in stored["combinations"]
            if c["source"]["type"] == "numpy_port"
        ]
        regenerated = generate_combinations()
        assert len(stored_numpy) == len(regenerated), (
            "stored numpy_port count differs from regenerated — "
            "re-run generate_numpy_port_combinations.py"
        )
        for s, r in zip(stored_numpy, regenerated):
            assert s["id"] == r["id"]
            assert s["reference_sigma"]["sigma_vv_db"] == pytest.approx(
                r["reference_sigma"]["sigma_vv_db"], abs=1e-6
            )
            assert s["reference_sigma"]["sigma_vh_db"] == pytest.approx(
                r["reference_sigma"]["sigma_vh_db"], abs=1e-6
            )


# ─── Independence check — numpy Mironov vs Dobson path ──────────────────────


class TestDielectricIndependence:
    """
    Sanity — Dobson and Mironov produce measurably different ε at
    Moor House m_v, confirming the two code paths are truly computing
    different physics. (The G4 diagnostic (SPEC §6) already quantifies
    this; this test just guards against a future refactor accidentally
    aliasing Mironov to Dobson in `reference_toure`.)
    """

    def test_mironov_differs_from_dobson_at_m_v_083(self):
        mv = 0.83
        dobson_eps = float(DobsonDielectric()(torch.tensor(mv)))
        mironov_eps = ground_epsilon_mironov(mv)
        # The G4 diagnostic quotes ~97% relative difference here.
        rel_diff = abs(dobson_eps - mironov_eps) / dobson_eps
        assert rel_diff > 0.5, (
            f"Dobson ({dobson_eps}) and Mironov ({mironov_eps}) are too "
            "close — possible aliasing"
        )
