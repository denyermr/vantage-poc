"""
Phase 3 unit tests — dielectric models, physics equations, PINN architecture.

Tests are grouped by component. Each physics equation has a dedicated test that
verifies against known-good values from the published literature.

Coverage target: ≥ 80% line coverage on poc/models/dielectric.py and poc/models/pinn.py.
"""

import logging
import math

import pytest
import torch

from shared.config import (
    DOBSON_ALPHA,
    EPSILON_DRY_PEAT,
    EPSILON_WATER,
    KS_ROUGHNESS,
    MIRONOV_MV_T,
    MIRONOV_ND,
    MIRONOV_ND1,
    MIRONOV_ND2,
    PEAT_THETA_SAT,
    WCM_A_INIT,
    WCM_A_LB,
    WCM_A_UB,
    WCM_B_INIT,
    WCM_B_LB,
    WCM_B_UB,
)
from phase1.physics.dielectric import DobsonDielectric, MironovDielectric
from phase1.physics.wcm import (
    OH_EPSILON_MIN,
    PINN,
    CorrectionNet,
    PhysicsNet,
    compute_pinn_loss,
    oh_soil_backscatter,
    wcm_forward,
    wcm_vegetation_terms,
)


# ─── Dobson dielectric model tests ──────────────────────────────────────────


class TestDobsonDielectric:
    """Tests for DobsonDielectric — Dobson et al. (1985) mixing model."""

    def setup_method(self):
        self.model = DobsonDielectric()

    def test_dobson_at_zero_moisture(self):
        """ε(0) = EPSILON_DRY_PEAT — dry soil returns dry peat constant."""
        eps = self.model(torch.tensor(0.0))
        assert eps.item() == pytest.approx(EPSILON_DRY_PEAT, abs=1e-6)

    def test_dobson_increases_with_moisture(self):
        """ε(0.3) > ε(0.1) — dielectric constant monotonically increases with VWC."""
        eps_low = self.model(torch.tensor(0.1))
        eps_high = self.model(torch.tensor(0.3))
        assert eps_high.item() > eps_low.item()

    def test_dobson_at_saturation(self):
        """ε(PEAT_THETA_SAT) < EPSILON_WATER — saturation bounded below free water."""
        eps_sat = self.model(torch.tensor(PEAT_THETA_SAT))
        assert eps_sat.item() < EPSILON_WATER

    def test_dobson_known_value_at_half(self):
        """
        Manual calculation cross-check at m_v = 0.5.

        ε(0.5) = 3.5 + (80.0 - 1.0) * 0.5^1.4
                = 3.5 + 79.0 * 0.5^1.4
                = 3.5 + 79.0 * 0.37892914...
                = 3.5 + 29.935...
                = 33.435...

        Reference: Manual computation from Dobson et al. (1985), eq. 1
        with Bechtold et al. (2018) peat parameterisation.
        """
        eps = self.model(torch.tensor(0.5))
        expected = EPSILON_DRY_PEAT + (EPSILON_WATER - 1.0) * (0.5 ** DOBSON_ALPHA)
        assert eps.item() == pytest.approx(expected, abs=1e-4)

    def test_dobson_monotonic_across_range(self):
        """Verify monotonicity across the full observed VWC range at Moor House."""
        m_v = torch.linspace(0.0, PEAT_THETA_SAT, 100)
        eps = self.model(m_v)
        diffs = eps[1:] - eps[:-1]
        assert (diffs > 0).all(), "Dobson must be strictly monotonically increasing"

    def test_dobson_gradient_flows(self):
        """Gradient d(ε)/d(m_v) is computable and positive (needed for PINN backprop)."""
        m_v = torch.tensor(0.4, requires_grad=True)
        eps = self.model(m_v)
        eps.backward()
        assert m_v.grad is not None, "Gradient must flow through Dobson model"
        assert m_v.grad.item() > 0, "d(ε)/d(m_v) must be positive"

    def test_dobson_batch_input(self):
        """Model handles batched inputs correctly."""
        m_v = torch.tensor([0.0, 0.2, 0.5, PEAT_THETA_SAT])
        eps = self.model(m_v)
        assert eps.shape == (4,)
        assert eps[0].item() == pytest.approx(EPSILON_DRY_PEAT, abs=1e-6)
        assert eps[-1].item() < EPSILON_WATER

    def test_dobson_negative_moisture_clamped(self):
        """Negative m_v is clamped to 0, returning ε_dry (no NaN/error)."""
        eps = self.model(torch.tensor(-0.1))
        assert eps.item() == pytest.approx(EPSILON_DRY_PEAT, abs=1e-6)


# ─── Mironov dielectric model tests ─────────────────────────────────────────


class TestMironovDielectric:
    """Tests for MironovDielectric — Mironov et al. (2009) refractive index model."""

    def setup_method(self):
        self.model = MironovDielectric()

    def test_mironov_piecewise_boundary(self):
        """
        Continuous at m_v = MIRONOV_MV_T — no step discontinuity.

        At the transition point, both branches must give the same value.
        """
        mv_t = MIRONOV_MV_T
        # Approach from below
        eps_below = self.model(torch.tensor(mv_t - 1e-6))
        # Approach from above
        eps_above = self.model(torch.tensor(mv_t + 1e-6))
        # At boundary exactly
        eps_at = self.model(torch.tensor(mv_t))
        assert eps_below.item() == pytest.approx(eps_at.item(), abs=1e-3)
        assert eps_above.item() == pytest.approx(eps_at.item(), abs=1e-3)

    def test_mironov_increases_with_moisture(self):
        """ε(0.4) > ε(0.2) — monotonically increasing across transition boundary."""
        eps_low = self.model(torch.tensor(0.2))
        eps_high = self.model(torch.tensor(0.4))
        assert eps_high.item() > eps_low.item()

    def test_mironov_at_zero_moisture(self):
        """ε(0) = nd² — dry soil returns squared dry refractive index."""
        eps = self.model(torch.tensor(0.0))
        expected = MIRONOV_ND ** 2
        assert eps.item() == pytest.approx(expected, abs=1e-6)

    def test_mironov_known_value_below_transition(self):
        """
        Manual calculation at m_v = 0.2 (below transition).

        n(0.2) = 0.312 + 1.42 * 0.2 = 0.312 + 0.284 = 0.596
        ε(0.2) = 0.596² = 0.355216

        Reference: Mironov et al. (2009), eq. applied manually.
        """
        eps = self.model(torch.tensor(0.2))
        n = MIRONOV_ND + MIRONOV_ND1 * 0.2
        expected = n ** 2
        assert eps.item() == pytest.approx(expected, abs=1e-4)

    def test_mironov_known_value_above_transition(self):
        """
        Manual calculation at m_v = 0.5 (above transition).

        n(0.5) = 0.312 + 1.42*0.36 + 0.89*(0.5 - 0.36)
               = 0.312 + 0.5112 + 0.89*0.14
               = 0.312 + 0.5112 + 0.1246
               = 0.9478
        ε(0.5) = 0.9478² = 0.89832...

        Reference: Mironov et al. (2009), eq. applied manually.
        """
        eps = self.model(torch.tensor(0.5))
        n = MIRONOV_ND + MIRONOV_ND1 * MIRONOV_MV_T + MIRONOV_ND2 * (0.5 - MIRONOV_MV_T)
        expected = n ** 2
        assert eps.item() == pytest.approx(expected, abs=1e-4)

    def test_mironov_monotonic_across_range(self):
        """Verify monotonicity across the full observed VWC range."""
        m_v = torch.linspace(0.0, PEAT_THETA_SAT, 100)
        eps = self.model(m_v)
        diffs = eps[1:] - eps[:-1]
        assert (diffs > 0).all(), "Mironov must be strictly monotonically increasing"

    def test_mironov_gradient_flows(self):
        """Gradient d(ε)/d(m_v) is computable (needed for PINN backprop)."""
        m_v = torch.tensor(0.4, requires_grad=True)
        eps = self.model(m_v)
        eps.backward()
        assert m_v.grad is not None, "Gradient must flow through Mironov model"
        assert m_v.grad.item() > 0, "d(ε)/d(m_v) must be positive"

    def test_mironov_batch_input(self):
        """Model handles batched inputs including values on both sides of transition."""
        m_v = torch.tensor([0.0, 0.2, MIRONOV_MV_T, 0.5, PEAT_THETA_SAT])
        eps = self.model(m_v)
        assert eps.shape == (5,)
        # All values should be positive
        assert (eps > 0).all()


# ─── Cross-model comparison tests ───────────────────────────────────────────


class TestDielectricCrossModel:
    """Tests comparing Dobson and Mironov — ensures both produce physically coherent output."""

    def test_both_models_positive_across_range(self):
        """Both models produce positive dielectric constant for all valid VWC."""
        m_v = torch.linspace(0.0, PEAT_THETA_SAT, 100)
        dobson = DobsonDielectric()
        mironov = MironovDielectric()
        assert (dobson(m_v) > 0).all()
        assert (mironov(m_v) > 0).all()

    def test_both_models_at_saturation_below_water(self):
        """At saturation, both models produce ε < ε_water."""
        m_v = torch.tensor(PEAT_THETA_SAT)
        dobson = DobsonDielectric()
        mironov = MironovDielectric()
        assert dobson(m_v).item() < EPSILON_WATER
        assert mironov(m_v).item() < EPSILON_WATER

    def test_dobson_larger_than_mironov_at_high_moisture(self):
        """
        At high VWC, Dobson gives much higher ε than Mironov.

        This is expected: Dobson uses the full water dielectric (80)
        in its mixing formula, while Mironov is based on refractive indices
        with smaller numerical range. This is a known characteristic difference
        between the models at C-band on organic soils.
        """
        m_v = torch.tensor(0.7)
        dobson = DobsonDielectric()
        mironov = MironovDielectric()
        eps_dobson = dobson(m_v).item()
        eps_mironov = mironov(m_v).item()
        assert eps_dobson > eps_mironov


# ─── Oh (1992) soil backscatter tests ────────────────────────────────────────


class TestOhSoilBackscatter:
    """Tests for Oh (1992) simplified soil backscatter model."""

    def test_oh_backscatter_range(self):
        """σ°_soil_dB in [−30, 0] for ε ∈ [3, 80], θ ∈ [25°, 50°]."""
        eps_vals = torch.tensor([3.0, 10.0, 30.0, 50.0, 80.0])
        theta_vals = torch.deg2rad(torch.tensor([25.0, 30.0, 35.0, 40.0, 45.0, 50.0]))
        for eps in eps_vals:
            for theta in theta_vals:
                sigma = oh_soil_backscatter(eps.unsqueeze(0), theta.unsqueeze(0))
                assert -30.0 <= sigma.item() <= 0.0, (
                    f"σ°_soil_dB = {sigma.item():.2f} out of range for ε={eps.item()}, "
                    f"θ={math.degrees(theta.item()):.0f}°"
                )

    def test_oh_backscatter_increases_with_epsilon(self):
        """Higher ε → higher (less negative) backscatter at fixed θ."""
        theta = torch.deg2rad(torch.tensor([38.0]))
        eps_low = torch.tensor([5.0])
        eps_high = torch.tensor([40.0])
        sigma_low = oh_soil_backscatter(eps_low, theta)
        sigma_high = oh_soil_backscatter(eps_high, theta)
        assert sigma_high.item() > sigma_low.item()

    def test_oh_backscatter_no_nan_no_inf(self):
        """No NaN/inf for any physically valid input."""
        eps = torch.linspace(3.0, 80.0, 50)
        theta = torch.deg2rad(torch.tensor(38.0)).expand(50)
        sigma = oh_soil_backscatter(eps, theta)
        assert not torch.isnan(sigma).any(), "NaN in Oh backscatter output"
        assert not torch.isinf(sigma).any(), "Inf in Oh backscatter output"

    def test_oh_backscatter_gradient_flows(self):
        """Gradient ∂σ°/∂ε is computable (needed for physics branch backprop)."""
        eps = torch.tensor(20.0, requires_grad=True)
        theta = torch.deg2rad(torch.tensor(38.0))
        sigma = oh_soil_backscatter(eps, theta)
        sigma.backward()
        assert eps.grad is not None
        assert eps.grad.item() != 0.0

    def test_oh_epsilon_clamping_warning(self, caplog):
        """When ε < OH_EPSILON_MIN, a warning is logged and output is finite."""
        eps = torch.tensor([0.5, 0.8])  # below 1.01 — Mironov at low VWC
        theta = torch.deg2rad(torch.tensor([38.0, 38.0]))
        with caplog.at_level(logging.WARNING):
            sigma = oh_soil_backscatter(eps, theta)
        assert not torch.isnan(sigma).any(), "NaN after clamping"
        assert not torch.isinf(sigma).any(), "Inf after clamping"
        assert "clamping" in caplog.text.lower()

    def test_oh_epsilon_clamping_no_warning_for_valid(self, caplog):
        """No warning when all ε values are above OH_EPSILON_MIN."""
        eps = torch.tensor([5.0, 20.0])
        theta = torch.deg2rad(torch.tensor([38.0, 38.0]))
        with caplog.at_level(logging.WARNING):
            oh_soil_backscatter(eps, theta)
        assert "clamping" not in caplog.text.lower()

    def test_oh_known_value(self):
        """
        Manual calculation cross-check at ε=20, θ=38°, ks=0.30.

        cos(38°) = 0.78801
        sin(38°) = 0.61566

        Inner = 20 - 0.61566² = 20 - 0.37903 = 19.62097
        sqrt(inner) = 4.42956

        Γ_h = ((0.78801 - 4.42956) / (0.78801 + 4.42956))²
            = ((-3.64155) / (5.21757))²
            = (-0.69797)²
            = 0.48717

        σ_linear = (0.30^0.1 / 3.0) * 0.78801^2.2 * 0.48717
                 = (0.88715 / 3.0) * 0.59771 * 0.48717
                 = 0.29572 * 0.29120
                 = 0.08611

        σ_dB = 10 * log10(0.08611) = 10 * (-1.06497) = -10.650

        Reference: Manual computation from Oh et al. (1992) simplified form.
        """
        eps = torch.tensor([20.0])
        theta = torch.deg2rad(torch.tensor([38.0]))
        sigma = oh_soil_backscatter(eps, theta, ks=0.30)
        assert sigma.item() == pytest.approx(-10.65, abs=0.15)


# ─── WCM vegetation terms tests ─────────────────────────────────────────────


class TestWCMVegetationTerms:
    """Tests for WCM vegetation scattering and transmissivity."""

    def test_wcm_vegetation_tau_bounded(self):
        """τ² ∈ [0, 1] for all NDVI ∈ [0, 1], B > 0, θ ∈ [25°, 50°]."""
        ndvi = torch.linspace(0.0, 1.0, 50)
        theta = torch.deg2rad(torch.tensor(38.0)).expand(50)
        A = torch.tensor(WCM_A_INIT)
        B = torch.tensor(WCM_B_INIT)
        _, tau_sq = wcm_vegetation_terms(A, B, ndvi, theta)
        assert (tau_sq >= 0.0).all(), "τ² must be non-negative"
        assert (tau_sq <= 1.0).all(), "τ² must be ≤ 1.0"

    def test_wcm_vegetation_tau_decreases_with_ndvi(self):
        """Higher NDVI → lower transmissivity (more vegetation attenuation)."""
        theta = torch.deg2rad(torch.tensor(38.0))
        A = torch.tensor(WCM_A_INIT)
        B = torch.tensor(WCM_B_INIT)
        _, tau_low = wcm_vegetation_terms(A, B, torch.tensor(0.2), theta)
        _, tau_high = wcm_vegetation_terms(A, B, torch.tensor(0.8), theta)
        assert tau_low.item() > tau_high.item()

    def test_wcm_vegetation_sigma_veg_nonneg(self):
        """σ°_veg ≥ 0 for NDVI ≥ 0 and A > 0."""
        ndvi = torch.linspace(0.0, 1.0, 20)
        theta = torch.deg2rad(torch.tensor(38.0)).expand(20)
        A = torch.tensor(WCM_A_INIT)
        B = torch.tensor(WCM_B_INIT)
        sigma_veg, _ = wcm_vegetation_terms(A, B, ndvi, theta)
        assert (sigma_veg >= 0.0).all()

    def test_wcm_vegetation_known_values(self):
        """
        Manual calculation at A=0.1, B=0.15, NDVI=0.4, θ=38°.

        cos(38°) = 0.78801

        σ°_veg = 0.1 * 0.4 * 0.78801 = 0.031520
        τ² = exp(-2 * 0.15 * 0.4 / 0.78801) = exp(-0.15228) = 0.85872

        Reference: Attema & Ulaby (1978), eqs. 5, 7, applied manually.
        """
        A = torch.tensor(0.1)
        B = torch.tensor(0.15)
        ndvi = torch.tensor(0.4)
        theta = torch.deg2rad(torch.tensor(38.0))
        sigma_veg, tau_sq = wcm_vegetation_terms(A, B, ndvi, theta)
        assert sigma_veg.item() == pytest.approx(0.03152, abs=1e-4)
        assert tau_sq.item() == pytest.approx(0.8587, abs=1e-3)

    def test_wcm_vegetation_gradient_flows_through_A_B(self):
        """Gradients ∂τ²/∂B and ∂σ°_veg/∂A are computable."""
        A = torch.tensor(0.1, requires_grad=True)
        B = torch.tensor(0.15, requires_grad=True)
        ndvi = torch.tensor(0.4)
        theta = torch.deg2rad(torch.tensor(38.0))
        sigma_veg, tau_sq = wcm_vegetation_terms(A, B, ndvi, theta)
        (sigma_veg + tau_sq).backward()
        assert A.grad is not None and A.grad.item() != 0.0
        assert B.grad is not None and B.grad.item() != 0.0


# ─── WCM forward model tests ────────────────────────────────────────────────


class TestWCMForward:
    """Tests for the full WCM forward pass: m_v → σ°_total_dB."""

    def setup_method(self):
        self.dobson = DobsonDielectric()

    def test_wcm_total_range(self):
        """σ°_total_dB in [−20, −5] for typical peatland inputs."""
        m_v = torch.tensor([0.3, 0.5, 0.7])
        A = torch.tensor(WCM_A_INIT)
        B = torch.tensor(WCM_B_INIT)
        ndvi = torch.tensor([0.3, 0.5, 0.7])
        theta = torch.deg2rad(torch.tensor([38.0, 38.0, 38.0]))
        sigma = wcm_forward(m_v, A, B, ndvi, theta, self.dobson)
        for i in range(3):
            assert -20.0 <= sigma[i].item() <= -5.0, (
                f"σ°_total_dB = {sigma[i].item():.2f} out of [-20, -5] "
                f"at m_v={m_v[i].item()}, NDVI={ndvi[i].item()}"
            )

    def test_wcm_known_values(self):
        """
        Manual calculation cross-check at A=0.1, B=0.15, NDVI=0.4, θ=38°, m_v=0.6.

        Step 1: Dobson dielectric
            ε(0.6) = 3.5 + 79.0 * 0.6^1.4 = 3.5 + 79.0 * 0.48159 = 3.5 + 38.046 = 41.546

        Step 2: Oh backscatter (ks=0.30, θ=38°)
            cos(38°) = 0.78801, sin(38°) = 0.61566
            inner = 41.546 - 0.37903 = 41.167
            sqrt_inner = 6.41614
            Γ_h = ((0.78801 - 6.41614) / (0.78801 + 6.41614))²
                = ((-5.62813) / (7.20415))² = 0.61058
            σ_soil_linear = (0.30^0.1 / 3) * 0.78801^2.2 * 0.61058
                          = 0.29572 * 0.59771 * 0.61058
                          = 0.10791
            σ_soil_dB = 10 * log10(0.10791) = -9.669

        Step 3: WCM vegetation (from test above)
            σ°_veg = 0.03152 (linear)
            τ² = 0.8587

        Step 4: Total
            σ_soil_linear = 10^(-9.669/10) = 0.10791
            σ_total_linear = 0.03152 + 0.8587 * 0.10791 = 0.03152 + 0.09268 = 0.12420
            σ_total_dB = 10 * log10(0.12420) = -9.060

        Reference: Manual computation from Attema & Ulaby (1978), eq. 7
        with Dobson (1985) and Oh (1992) sub-models.
        """
        m_v = torch.tensor(0.6)
        A = torch.tensor(0.1)
        B = torch.tensor(0.15)
        ndvi = torch.tensor(0.4)
        theta = torch.deg2rad(torch.tensor(38.0))
        sigma = wcm_forward(m_v, A, B, ndvi, theta, self.dobson)
        assert sigma.item() == pytest.approx(-9.06, abs=0.2)

    def test_wcm_gradient_flows_through_m_v(self):
        """∂σ°_total/∂m_v is computable through the full Dobson→Oh→WCM chain."""
        m_v = torch.tensor(0.5, requires_grad=True)
        A = torch.tensor(WCM_A_INIT)
        B = torch.tensor(WCM_B_INIT)
        ndvi = torch.tensor(0.4)
        theta = torch.deg2rad(torch.tensor(38.0))
        sigma = wcm_forward(m_v, A, B, ndvi, theta, self.dobson)
        sigma.backward()
        assert m_v.grad is not None, "Gradient must flow through full VWC→σ° chain"
        assert m_v.grad.item() != 0.0, "∂σ°/∂m_v must be non-zero"

    def test_wcm_gradient_flows_through_A_and_B(self):
        """∂σ°_total/∂A and ∂σ°_total/∂B are computable."""
        m_v = torch.tensor(0.5)
        A = torch.tensor(WCM_A_INIT, requires_grad=True)
        B = torch.tensor(WCM_B_INIT, requires_grad=True)
        ndvi = torch.tensor(0.4)
        theta = torch.deg2rad(torch.tensor(38.0))
        sigma = wcm_forward(m_v, A, B, ndvi, theta, self.dobson)
        sigma.backward()
        assert A.grad is not None and A.grad.item() != 0.0
        assert B.grad is not None and B.grad.item() != 0.0

    def test_wcm_with_mironov_finite_output(self):
        """WCM with Mironov produces finite output (ε clamping handles ε < 1)."""
        mironov = MironovDielectric()
        m_v = torch.tensor([0.3, 0.5, 0.7])
        A = torch.tensor(WCM_A_INIT)
        B = torch.tensor(WCM_B_INIT)
        ndvi = torch.tensor([0.3, 0.5, 0.7])
        theta = torch.deg2rad(torch.tensor([38.0, 38.0, 38.0]))
        sigma = wcm_forward(m_v, A, B, ndvi, theta, mironov)
        assert not torch.isnan(sigma).any(), "NaN with Mironov dielectric"
        assert not torch.isinf(sigma).any(), "Inf with Mironov dielectric"

    def test_wcm_batch_consistency(self):
        """Batch processing matches individual element processing."""
        m_v = torch.tensor([0.3, 0.6])
        A = torch.tensor(WCM_A_INIT)
        B = torch.tensor(WCM_B_INIT)
        ndvi = torch.tensor([0.3, 0.5])
        theta = torch.deg2rad(torch.tensor([38.0, 38.0]))
        sigma_batch = wcm_forward(m_v, A, B, ndvi, theta, self.dobson)
        sigma_0 = wcm_forward(m_v[0:1], A, B, ndvi[0:1], theta[0:1], self.dobson)
        sigma_1 = wcm_forward(m_v[1:2], A, B, ndvi[1:2], theta[1:2], self.dobson)
        assert sigma_batch[0].item() == pytest.approx(sigma_0.item(), abs=1e-6)
        assert sigma_batch[1].item() == pytest.approx(sigma_1.item(), abs=1e-6)


# ─── Sigmoid reparameterisation tests ────────────────────────────────────────


class TestSigmoidReparameterisation:
    """Tests for sigmoid-based parameter bounding used in WCM learnable parameters."""

    def test_sigmoid_bounds_A_across_range(self):
        """A = WCM_A_LB + (WCM_A_UB - WCM_A_LB) * sigmoid(raw) stays in bounds."""
        raw_vals = torch.linspace(-10.0, 10.0, 1000)
        A = WCM_A_LB + (WCM_A_UB - WCM_A_LB) * torch.sigmoid(raw_vals)
        assert (A >= WCM_A_LB).all(), f"A below lower bound: min={A.min().item()}"
        assert (A <= WCM_A_UB).all(), f"A above upper bound: max={A.max().item()}"

    def test_sigmoid_bounds_B_across_range(self):
        """B = WCM_B_LB + (WCM_B_UB - WCM_B_LB) * sigmoid(raw) stays in bounds."""
        raw_vals = torch.linspace(-10.0, 10.0, 1000)
        B = WCM_B_LB + (WCM_B_UB - WCM_B_LB) * torch.sigmoid(raw_vals)
        assert (B >= WCM_B_LB).all(), f"B below lower bound: min={B.min().item()}"
        assert (B <= WCM_B_UB).all(), f"B above upper bound: max={B.max().item()}"

    def test_sigmoid_init_close_to_literature_A(self):
        """Sigmoid reparameterised A at init raw value ≈ WCM_A_INIT ± 0.01."""
        A_raw_init = math.log(
            (WCM_A_INIT - WCM_A_LB) / (WCM_A_UB - WCM_A_INIT + 1e-8)
        )
        A = WCM_A_LB + (WCM_A_UB - WCM_A_LB) * torch.sigmoid(torch.tensor(A_raw_init))
        assert A.item() == pytest.approx(WCM_A_INIT, abs=0.01)

    def test_sigmoid_init_close_to_literature_B(self):
        """Sigmoid reparameterised B at init raw value ≈ WCM_B_INIT ± 0.01."""
        B_raw_init = math.log(
            (WCM_B_INIT - WCM_B_LB) / (WCM_B_UB - WCM_B_INIT + 1e-8)
        )
        B = WCM_B_LB + (WCM_B_UB - WCM_B_LB) * torch.sigmoid(torch.tensor(B_raw_init))
        assert B.item() == pytest.approx(WCM_B_INIT, abs=0.01)

    def test_sigmoid_gradient_flows(self):
        """Gradient flows through sigmoid reparameterisation."""
        raw = torch.tensor(0.0, requires_grad=True)
        A = WCM_A_LB + (WCM_A_UB - WCM_A_LB) * torch.sigmoid(raw)
        A.backward()
        assert raw.grad is not None and raw.grad.item() != 0.0


# ─── PINN architecture tests ────────────────────────────────────────────────


class TestPINNArchitecture:
    """Tests for the PINN class — architecture, forward pass, parameter bounding."""

    def setup_method(self):
        torch.manual_seed(42)
        self.n_features = len(
            ["vv_db", "vh_db", "vhvv_db", "ndvi",
             "precip_mm", "precip_7day_mm", "incidence_angle_mean"]
        )  # 7 features per DEV-004
        self.model = PINN(n_features=self.n_features)
        self.N = 16
        self.X = torch.randn(self.N, self.n_features)
        self.ndvi = torch.rand(self.N) * 0.6 + 0.2  # [0.2, 0.8]
        self.theta = torch.deg2rad(torch.full((self.N,), 38.0))
        self.vv_db = torch.randn(self.N) * 2.0 - 12.0  # ~[-16, -8] dB

    def test_pinn_forward_returns_all_keys(self):
        """PINN forward returns dict with all 7 required keys."""
        outputs = self.model(self.X, self.ndvi, self.theta, self.vv_db)
        required_keys = {
            "m_v_physics", "delta_ml", "m_v_final",
            "sigma_wcm_db", "epsilon", "A_current", "B_current",
        }
        assert set(outputs.keys()) == required_keys

    def test_pinn_m_v_final_is_sum(self):
        """m_v_final = m_v_physics + delta_ml (exactly)."""
        outputs = self.model(self.X, self.ndvi, self.theta, self.vv_db)
        expected = outputs["m_v_physics"] + outputs["delta_ml"]
        assert torch.allclose(outputs["m_v_final"], expected, atol=1e-7)

    def test_pinn_physics_net_smaller_than_correction(self):
        """PhysicsNet has fewer parameters than CorrectionNet (capacity asymmetry)."""
        physics_params = sum(p.numel() for p in self.model.physics_net.parameters())
        correction_params = sum(p.numel() for p in self.model.correction_net.parameters())
        assert physics_params < correction_params, (
            f"PhysicsNet ({physics_params}) should have fewer params "
            f"than CorrectionNet ({correction_params})"
        )

    def test_pinn_A_bounded(self):
        """PINN.A always in [WCM_A_LB, WCM_A_UB] regardless of A_raw value."""
        for raw_val in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            self.model.A_raw.data = torch.tensor(raw_val)
            A = self.model.A.item()
            # float32 sigmoid approaches but never exactly reaches 0 or 1
            assert A >= WCM_A_LB - 1e-6, f"A={A} below lower bound at A_raw={raw_val}"
            assert A <= WCM_A_UB + 1e-6, f"A={A} above upper bound at A_raw={raw_val}"

    def test_pinn_B_bounded(self):
        """PINN.B always in [WCM_B_LB, WCM_B_UB] regardless of B_raw value."""
        for raw_val in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            self.model.B_raw.data = torch.tensor(raw_val)
            B = self.model.B.item()
            assert B >= WCM_B_LB - 1e-6, f"B={B} below lower bound at B_raw={raw_val}"
            assert B <= WCM_B_UB + 1e-6, f"B={B} above upper bound at B_raw={raw_val}"

    def test_pinn_A_init_close_to_literature(self):
        """A at initialisation ≈ WCM_A_INIT ± 0.01."""
        model = PINN(n_features=self.n_features)
        assert model.A.item() == pytest.approx(WCM_A_INIT, abs=0.01)

    def test_pinn_B_init_close_to_literature(self):
        """B at initialisation ≈ WCM_B_INIT ± 0.01."""
        model = PINN(n_features=self.n_features)
        assert model.B.item() == pytest.approx(WCM_B_INIT, abs=0.01)

    def test_pinn_output_shapes(self):
        """All output tensors have correct shapes."""
        outputs = self.model(self.X, self.ndvi, self.theta, self.vv_db)
        assert outputs["m_v_physics"].shape == (self.N,)
        assert outputs["delta_ml"].shape == (self.N,)
        assert outputs["m_v_final"].shape == (self.N,)
        assert outputs["sigma_wcm_db"].shape == (self.N,)
        assert outputs["epsilon"].shape == (self.N,)
        assert outputs["A_current"].shape == ()  # scalar
        assert outputs["B_current"].shape == ()  # scalar

    def test_pinn_no_nan_in_forward(self):
        """No NaN in any output tensor for typical inputs."""
        outputs = self.model(self.X, self.ndvi, self.theta, self.vv_db)
        for key, val in outputs.items():
            assert not torch.isnan(val).any(), f"NaN in PINN output '{key}'"

    def test_pinn_m_v_physics_bounded(self):
        """m_v_physics is in [0, PEAT_THETA_SAT] due to sigmoid activation."""
        outputs = self.model(self.X, self.ndvi, self.theta, self.vv_db)
        m_v_p = outputs["m_v_physics"]
        assert (m_v_p >= 0.0).all(), f"m_v_physics has negative values: min={m_v_p.min()}"
        assert (m_v_p <= PEAT_THETA_SAT).all(), f"m_v_physics exceeds θ_sat: max={m_v_p.max()}"

    def test_pinn_reproducible_with_seed(self):
        """Two runs with same seed produce identical weights."""
        torch.manual_seed(99)
        model_a = PINN(n_features=self.n_features)
        torch.manual_seed(99)
        model_b = PINN(n_features=self.n_features)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(pa.data, pb.data), "Weights differ with same seed"

    def test_pinn_gradient_flows_end_to_end(self):
        """Gradients flow through the entire PINN to all parameters.

        Uses a combined loss touching both m_v_final (for physics_net and
        correction_net params) and sigma_wcm_db (for A_raw and B_raw).
        """
        outputs = self.model(self.X, self.ndvi, self.theta, self.vv_db)
        # m_v_final.sum() → physics_net + correction_net params
        # sigma_wcm_db.sum() → A_raw, B_raw, physics_net params
        loss = outputs["m_v_final"].sum() + outputs["sigma_wcm_db"].sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ─── Composite loss function tests ──────────────────────────────────────────


class TestComputePINNLoss:
    """Tests for compute_pinn_loss — composite PINN loss function."""

    def setup_method(self):
        torch.manual_seed(42)
        self.n_features = 7
        self.N = 16
        self.model = PINN(n_features=self.n_features)
        self.X = torch.randn(self.N, self.n_features)
        self.ndvi = torch.rand(self.N) * 0.6 + 0.2
        self.theta = torch.deg2rad(torch.full((self.N,), 38.0))
        self.vv_db = torch.randn(self.N) * 2.0 - 12.0
        self.m_v_observed = torch.rand(self.N) * 0.5 + 0.2  # [0.2, 0.7]

    def _get_outputs(self):
        return self.model(self.X, self.ndvi, self.theta, self.vv_db)

    def test_loss_l_data_nonnegative(self):
        """L_data ≥ 0 always (MSE is non-negative)."""
        outputs = self._get_outputs()
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        assert loss["l_data"].item() >= 0.0

    def test_loss_l_physics_nonnegative(self):
        """L_physics ≥ 0 always (MSE is non-negative)."""
        outputs = self._get_outputs()
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        assert loss["l_physics"].item() >= 0.0

    def test_loss_l_monotonic_nonnegative(self):
        """L_monotonic ≥ 0 always (ReLU output is non-negative)."""
        outputs = self._get_outputs()
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        assert loss["l_monotonic"].item() >= 0.0

    def test_loss_l_bounds_nonnegative(self):
        """L_bounds ≥ 0 always (ReLU output is non-negative)."""
        outputs = self._get_outputs()
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        assert loss["l_bounds"].item() >= 0.0

    def test_loss_l_monotonic_zero_for_dobson(self):
        """
        L_monotonic ≈ 0 for Dobson (always increasing).

        Dobson ε(m_v) = ε_dry + (ε_water-1)·m_v^α with α=1.4 > 0,
        so dε/dm_v > 0 for all m_v > 0. The finite difference probe
        should always find positive gradients.
        """
        outputs = self._get_outputs()
        loss = compute_pinn_loss(
            outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1,
            dielectric_model=DobsonDielectric(),
        )
        assert loss["l_monotonic"].item() == pytest.approx(0.0, abs=1e-4)

    def test_loss_l_bounds_zero_within_range(self):
        """L_bounds = 0 when all m_v_final in [0, PEAT_THETA_SAT]."""
        # Force m_v_final into valid range
        outputs = self._get_outputs()
        outputs["m_v_final"] = torch.rand(self.N) * PEAT_THETA_SAT
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        assert loss["l_bounds"].item() == pytest.approx(0.0, abs=1e-7)

    def test_loss_l_bounds_positive_outside_range(self):
        """L_bounds > 0 when m_v_final < 0 or m_v_final > PEAT_THETA_SAT."""
        outputs = self._get_outputs()
        # Force some values outside bounds
        outputs["m_v_final"] = torch.tensor(
            [-0.1, 0.5, PEAT_THETA_SAT + 0.1] + [0.4] * (self.N - 3)
        )
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        assert loss["l_bounds"].item() > 0.0

    def test_loss_total_is_weighted_sum(self):
        """Total = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds."""
        outputs = self._get_outputs()
        l1, l2, l3 = 0.2, 0.05, 0.3
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, l1, l2, l3)
        expected_total = (
            loss["l_data"]
            + l1 * loss["l_physics"]
            + l2 * loss["l_monotonic"]
            + l3 * loss["l_bounds"]
        )
        assert loss["total"].item() == pytest.approx(expected_total.item(), abs=1e-6)

    def test_loss_returns_all_keys(self):
        """Loss dict contains all 5 required keys."""
        outputs = self._get_outputs()
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        required_keys = {"total", "l_data", "l_physics", "l_monotonic", "l_bounds"}
        assert set(loss.keys()) == required_keys

    def test_loss_gradient_flows_to_model(self):
        """Total loss backward() produces gradients in all model parameters."""
        outputs = self._get_outputs()
        loss = compute_pinn_loss(outputs, self.m_v_observed, self.vv_db, 0.1, 0.1, 0.1)
        loss["total"].backward()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_vv_db_used_unnormalised_in_l_physics(self):
        """
        L_physics compares WCM output (dB) to raw observed VV (dB).

        The WCM forward model produces σ° in dB. The vv_db_observed
        passed to compute_pinn_loss must be the raw (unnormalised) dB values,
        not the z-scored feature values.
        """
        outputs = self._get_outputs()
        raw_vv = torch.tensor([-12.5] * self.N)
        # L_physics should compare sigma_wcm_db to raw_vv
        loss = compute_pinn_loss(outputs, self.m_v_observed, raw_vv, 0.1, 0.1, 0.1)
        # Recompute expected L_physics manually
        expected = torch.nn.functional.mse_loss(outputs["sigma_wcm_db"], raw_vv)
        assert loss["l_physics"].item() == pytest.approx(expected.item(), abs=1e-6)


# ─── Residual ratio computation test ────────────────────────────────────────


class TestResidualRatio:
    """Tests for residual ratio computation: std(δ) / std(m_v_physics)."""

    def test_residual_ratio_computation(self):
        """std(delta_ml) / std(m_v_physics) computed correctly."""
        torch.manual_seed(42)
        model = PINN(n_features=7)
        N = 32
        X = torch.randn(N, 7)
        ndvi = torch.rand(N) * 0.6 + 0.2
        theta = torch.deg2rad(torch.full((N,), 38.0))
        vv_db = torch.randn(N) * 2.0 - 12.0

        outputs = model(X, ndvi, theta, vv_db)
        m_v_physics = outputs["m_v_physics"].detach()
        delta_ml = outputs["delta_ml"].detach()

        ratio = delta_ml.std() / (m_v_physics.std() + 1e-8)
        assert ratio.item() >= 0.0, "Residual ratio must be non-negative"
        assert torch.isfinite(ratio), "Residual ratio must be finite"
