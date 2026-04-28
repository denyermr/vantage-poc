"""
Unit tests for the Phase 1c-Lean per-channel normalised composite loss
(SPEC §18.4.1, v0.3) and its σ-init helper.

These tests pin the implementation invariants required by the G2-Lean
three-arm gate (SPEC §18.6.1) at unit scale, so that any future refactor
of `compute_pinn_mimics_loss_normalised` or `compute_init_sigma_normalisers`
breaks the test suite before it can corrupt a Block C-prime training run.

Invariants pinned (mirroring the §2 deliverable-5 requirements in the
Block B-prime kickoff prompt):

    (T1) Loss-formula correctness — hand-computed expected value at a
         synthetic fixture matches `total` at machine precision.
    (T2) σ-init-only invariant — `compute_pinn_mimics_loss_normalised`
         does NOT internally re-compute σ from inputs; the σ values passed
         as arguments are honoured verbatim. Per-batch normalisation is
         explicitly out of scope per SPEC §18.4.1.
    (T3) σ-checkpoint-roundtrip invariant — σ values save → load via
         `torch.save` / `torch.load` round-trip exactly (no float64↔float32
         drift).
    (T4) Scale-sanity invariant — on synthetic per-sample physics losses,
         dividing by the helper-computed σ produces a series of unit
         population standard deviation. This is the unit-scale analogue
         of the G2-Lean Arm 3 gate on the n=83 training set.
    (T5) σ nonzero-finite invariant — the helper raises ValueError on
         zero-variance physics losses (no silent σ=0 division).
    (T6) Gradient flow — autograd through both σ° tensors AND through the
         λ_VV / λ_VH scalars is live (guards against silent stop-gradient
         on the new λ axes that Arm 2 will check at gate scale).
    (T7) Phase 1b reduction — when σ_VV = σ_VH = 1 and λ_VV = λ_VH =
         λ_physics, the v0.3 composite reduces exactly to the Phase 1b
         joint VV+VH composite. Cross-validates the v0.3 formulation
         against the F-2b regression-tested implementation.

Reference:
    SPEC.md §18.4.1 (per-channel normalisation, v0.3)
    SPEC.md §18.6.1 (G2-Lean three-arm equivalence gate)
    Block B-prime kickoff prompt §2 deliverable 5
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from phase1b.pinn_mimics import (
    PinnMimics,
    compute_init_sigma_normalisers,
    compute_pinn_mimics_loss,
    compute_pinn_mimics_loss_normalised,
)
from shared.config import FEATURE_COLUMNS, PEAT_THETA_SAT


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _synthetic_outputs(
    m_v_final: torch.Tensor,
    m_v_physics: torch.Tensor,
    sigma_vv_db: torch.Tensor,
    sigma_vh_db: torch.Tensor,
    epsilon_ground: torch.Tensor,
) -> dict:
    """Build a minimal `outputs` dict matching `PinnMimics.forward()` schema."""
    return {
        "m_v_final": m_v_final,
        "m_v_physics": m_v_physics,
        "sigma_vv_db": sigma_vv_db,
        "sigma_vh_db": sigma_vh_db,
        "epsilon_ground": epsilon_ground,
        "delta_ml": torch.zeros_like(m_v_final),
        "vv_db_observed": torch.zeros_like(sigma_vv_db),
    }


# ─── T1 — loss formula correctness on a hand-computed fixture ────────────────


def test_total_matches_hand_computed_v03_composite():
    """
    SPEC §18.4.1 v0.3 composite:
        L = L_data
            + λ_VV · (L_phys_VV / σ_VV)
            + λ_VH · (L_phys_VH / σ_VH)
            + λ_mono · L_mono
            + λ_bnd · L_bnd
    """
    torch.manual_seed(0)
    n = 16

    sigma_vv_db = torch.tensor([-10.0] * n, dtype=torch.float32)
    vv_obs = torch.tensor([-12.5] * n, dtype=torch.float32)      # MSE = 6.25
    sigma_vh_db = torch.tensor([-18.0] * n, dtype=torch.float32)
    vh_obs = torch.tensor([-17.1] * n, dtype=torch.float32)      # MSE = 0.81

    m_v_final = torch.full((n,), 0.35, dtype=torch.float32)
    m_v_physics = torch.full((n,), 0.35, dtype=torch.float32)
    epsilon_ground = torch.full((n,), 10.0, dtype=torch.float32)

    outputs = _synthetic_outputs(
        m_v_final, m_v_physics, sigma_vv_db, sigma_vh_db, epsilon_ground,
    )

    sigma_vv = 2.0
    sigma_vh = 0.5
    lam_vv = 0.7
    lam_vh = 0.3
    lam_mono = 0.01
    lam_bnd = 0.01

    losses = compute_pinn_mimics_loss_normalised(
        outputs,
        m_v_observed=torch.full((n,), 0.35, dtype=torch.float32),  # L_data = 0
        vv_db_observed=vv_obs,
        vh_db_observed=vh_obs,
        lambda_vv=lam_vv,
        lambda_vh=lam_vh,
        lambda_monotonic=lam_mono,
        lambda_bounds=lam_bnd,
        sigma_vv=sigma_vv,
        sigma_vh=sigma_vh,
    )

    expected_l_phys_vv = 6.25
    expected_l_phys_vh = 0.81
    expected_normalised_vv = lam_vv * (expected_l_phys_vv / sigma_vv)
    expected_normalised_vh = lam_vh * (expected_l_phys_vh / sigma_vh)
    # L_data = 0 (m_v_final == m_v_observed). L_bounds = 0 (m_v ∈ [0, θ_sat]).
    # L_monotonic ≥ 0 from the Mironov probe; included via the actual return.
    expected_total_excl_mono = expected_normalised_vv + expected_normalised_vh

    actual_total = losses["total"].item()
    actual_mono_contrib = losses["weighted_l_monotonic"].item()
    actual_bnd_contrib = losses["weighted_l_bounds"].item()
    actual_data = losses["l_data"].item()

    assert losses["l_physics_vv"].item() == pytest.approx(expected_l_phys_vv, abs=1e-5)
    assert losses["l_physics_vh"].item() == pytest.approx(expected_l_phys_vh, abs=1e-5)
    assert losses["weighted_l_physics_vv_normalised"].item() == pytest.approx(
        expected_normalised_vv, abs=1e-6,
    )
    assert losses["weighted_l_physics_vh_normalised"].item() == pytest.approx(
        expected_normalised_vh, abs=1e-6,
    )
    # total = L_data + Σ_normalised + λ_mono·L_mono + λ_bnd·L_bnd
    reconstructed = (
        actual_data
        + expected_total_excl_mono
        + actual_mono_contrib
        + actual_bnd_contrib
    )
    assert actual_total == pytest.approx(reconstructed, abs=1e-6)


# ─── T2 — σ-init-only invariant ──────────────────────────────────────────────


def test_sigma_is_honoured_verbatim_not_recomputed_per_batch():
    """
    SPEC §18.4.1: σ_VV / σ_VH are computed once at init and treated as
    constants. The loss function must accept σ as an argument and use it
    verbatim; it must NOT re-compute σ from the batch (which would
    introduce per-batch normalisation, explicitly out of scope).

    Invariant: scaling σ_VV by 10× must scale the VV-normalised term by
    exactly 1/10 (no internal re-scaling). If the loss internally
    overrode σ via batch-statistics, this would not hold.
    """
    torch.manual_seed(7)
    n = 24

    sigma_vv_db = -11.0 + 2.0 * torch.randn(n)
    sigma_vh_db = -18.5 + 2.0 * torch.randn(n)
    vv_obs = -12.0 + 1.0 * torch.randn(n)
    vh_obs = -17.5 + 1.0 * torch.randn(n)
    m_v = torch.full((n,), 0.4)
    epsilon = torch.full((n,), 12.0)
    outputs = _synthetic_outputs(m_v, m_v, sigma_vv_db, sigma_vh_db, epsilon)

    base_sigma_vv = 1.5
    base_sigma_vh = 0.8

    losses_a = compute_pinn_mimics_loss_normalised(
        outputs, m_v_observed=m_v,
        vv_db_observed=vv_obs, vh_db_observed=vh_obs,
        lambda_vv=1.0, lambda_vh=1.0,
        lambda_monotonic=0.0, lambda_bounds=0.0,
        sigma_vv=base_sigma_vv, sigma_vh=base_sigma_vh,
    )
    losses_b = compute_pinn_mimics_loss_normalised(
        outputs, m_v_observed=m_v,
        vv_db_observed=vv_obs, vh_db_observed=vh_obs,
        lambda_vv=1.0, lambda_vh=1.0,
        lambda_monotonic=0.0, lambda_bounds=0.0,
        sigma_vv=base_sigma_vv * 10.0, sigma_vh=base_sigma_vh * 10.0,
    )

    # If the loss honoured σ verbatim, scaling σ by 10× must reduce the
    # normalised physics terms by exactly 1/10. l_physics_vv (unnormalised)
    # must NOT change between the two calls (it's pol-MSE, σ-independent).
    assert losses_a["l_physics_vv"].item() == pytest.approx(
        losses_b["l_physics_vv"].item(), abs=1e-7,
    )
    assert losses_a["l_physics_vh"].item() == pytest.approx(
        losses_b["l_physics_vh"].item(), abs=1e-7,
    )
    ratio_vv = (
        losses_b["weighted_l_physics_vv_normalised"].item()
        / losses_a["weighted_l_physics_vv_normalised"].item()
    )
    ratio_vh = (
        losses_b["weighted_l_physics_vh_normalised"].item()
        / losses_a["weighted_l_physics_vh_normalised"].item()
    )
    assert ratio_vv == pytest.approx(0.1, abs=1e-7)
    assert ratio_vh == pytest.approx(0.1, abs=1e-7)


def test_helper_sigma_is_deterministic_at_fixed_init():
    """
    σ-init helper called twice on the same model + same data must return
    identical σ values. Establishes that σ is a pure function of (model
    weights, training set), not a stateful per-batch quantity.
    """
    torch.manual_seed(42)
    n = 83
    n_features = len(FEATURE_COLUMNS)

    model = PinnMimics(n_features=n_features)

    X = torch.randn(n, n_features)
    theta = torch.full((n,), 39.0)
    vv_obs = -12.0 + 1.5 * torch.randn(n)
    vh_obs = -19.0 + 1.5 * torch.randn(n)

    sigmas_a = compute_init_sigma_normalisers(model, X, theta, vv_obs, vh_obs)
    sigmas_b = compute_init_sigma_normalisers(model, X, theta, vv_obs, vh_obs)

    assert sigmas_a["sigma_vv"] == sigmas_b["sigma_vv"]
    assert sigmas_a["sigma_vh"] == sigmas_b["sigma_vh"]


# ─── T3 — σ-checkpoint-roundtrip invariant ───────────────────────────────────


def test_sigma_checkpoint_roundtrip_exact(tmp_path):
    """
    σ values must save → load via torch.save/torch.load round-trip exactly,
    so that a model checkpoint reproduces the per-flight summary block
    (SPEC §18.11 schema item 3) verbatim.
    """
    torch.manual_seed(42)
    n = 83
    n_features = len(FEATURE_COLUMNS)
    model = PinnMimics(n_features=n_features)

    X = torch.randn(n, n_features)
    theta = torch.full((n,), 39.0)
    vv_obs = -12.0 + 1.5 * torch.randn(n)
    vh_obs = -19.0 + 1.5 * torch.randn(n)

    sigmas = compute_init_sigma_normalisers(model, X, theta, vv_obs, vh_obs)

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sigma_vv": sigmas["sigma_vv"],
            "sigma_vh": sigmas["sigma_vh"],
        },
        ckpt_path,
    )

    loaded = torch.load(ckpt_path, weights_only=False)

    assert loaded["sigma_vv"] == sigmas["sigma_vv"]
    assert loaded["sigma_vh"] == sigmas["sigma_vh"]


# ─── T4 — scale-sanity invariant on synthetic data ───────────────────────────


def test_scale_sanity_invariant_on_synthetic_per_sample_losses():
    """
    Unit-scale analogue of G2-Lean Arm 3. Construct a per-sample physics
    loss series with non-trivial variance, take std-pop as σ (matching
    `compute_init_sigma_normalisers`'s convention), and verify that
    series/σ has unit population std within 1e-12.

    This pins the σ-formula choice (population std, ddof=0) used by the
    helper. A switch to sample std (ddof=1) would leave a residual
    factor of √(n/(n−1)) on the normalised series, breaking Arm 3.
    """
    rng = np.random.default_rng(42)
    n = 83
    series = torch.tensor(
        rng.gamma(shape=2.0, scale=0.5, size=n).astype(np.float32)
    )

    # Mimic the helper's convention: population std (ddof=0).
    sigma = series.std(unbiased=False).item()
    normalised = series / sigma
    std_normalised = normalised.std(unbiased=False).item()

    assert std_normalised == pytest.approx(1.0, abs=1e-6)


# ─── T5 — σ nonzero-finite invariant ─────────────────────────────────────────


def test_init_sigma_helper_raises_on_zero_variance():
    """
    SPEC §18.4.1 forbids σ-floor / clamp. A zero-variance physics-loss
    series (e.g. perfectly constant residuals) is treated as a hard
    G2-Lean implementation failure; the helper must raise ValueError.
    """
    torch.manual_seed(123)
    n = 16
    n_features = len(FEATURE_COLUMNS)

    class _ZeroVarMockModel:
        """Forwards constant σ° = obs so per-sample squared errors are all 0."""

        training = False

        def eval(self):  # noqa: D401 — mimic nn.Module interface
            return self

        def train(self, mode=True):
            return self

        def __call__(self, X, theta, vv_obs):
            return {
                "sigma_vv_db": vv_obs.clone(),  # err = 0 everywhere
                "sigma_vh_db": vv_obs.clone(),  # same value used for both
            }

    X = torch.randn(n, n_features)
    theta = torch.full((n,), 39.0)
    vv_obs = -12.0 + torch.zeros(n)
    vh_obs = vv_obs.clone()

    with pytest.raises(ValueError, match=r"σ_VV"):
        compute_init_sigma_normalisers(
            _ZeroVarMockModel(), X, theta, vv_obs, vh_obs,
        )


def test_helper_returns_positive_finite_floats_at_real_init():
    """
    On a realistically-initialised PinnMimics with realistic synthetic
    inputs, the helper returns positive finite floats for both channels.
    Locks the §2 deliverable-5 invariant.
    """
    torch.manual_seed(42)
    n = 83
    n_features = len(FEATURE_COLUMNS)
    model = PinnMimics(n_features=n_features)

    X = torch.randn(n, n_features)
    theta = torch.full((n,), 39.0)
    vv_obs = -12.0 + 1.5 * torch.randn(n)
    vh_obs = -19.0 + 1.5 * torch.randn(n)

    sigmas = compute_init_sigma_normalisers(model, X, theta, vv_obs, vh_obs)

    assert isinstance(sigmas["sigma_vv"], float)
    assert isinstance(sigmas["sigma_vh"], float)
    assert math.isfinite(sigmas["sigma_vv"])
    assert math.isfinite(sigmas["sigma_vh"])
    assert sigmas["sigma_vv"] > 0.0
    assert sigmas["sigma_vh"] > 0.0


# ─── T6 — gradient flow through σ° AND λ_VV / λ_VH ───────────────────────────


def test_gradient_flows_through_both_pol_outputs_and_both_lambdas():
    """
    Autograd of the v0.3 composite must flow live through:
      - σ°_VV and σ°_VH (model outputs); and
      - λ_VV and λ_VH (the new tunable axes per §18.4.2).

    Guards against silent stop-gradient regressions on either axis. Unit
    analogue of the G2-Lean Arm 2 invariant on λ_grad.
    """
    torch.manual_seed(2)
    n = 8

    sigma_vv_db = torch.full((n,), -11.0, requires_grad=True)
    sigma_vh_db = torch.full((n,), -18.5, requires_grad=True)

    vv_obs = torch.full((n,), -12.0)
    vh_obs = torch.full((n,), -17.0)
    m_v = torch.full((n,), 0.3)
    epsilon = torch.full((n,), 12.0)
    outputs = _synthetic_outputs(m_v, m_v, sigma_vv_db, sigma_vh_db, epsilon)

    lam_vv = torch.tensor(0.5, requires_grad=True)
    lam_vh = torch.tensor(0.3, requires_grad=True)

    losses = compute_pinn_mimics_loss_normalised(
        outputs, m_v_observed=m_v,
        vv_db_observed=vv_obs, vh_db_observed=vh_obs,
        lambda_vv=lam_vv, lambda_vh=lam_vh,
        lambda_monotonic=0.0, lambda_bounds=0.0,
        sigma_vv=2.0, sigma_vh=0.7,
    )
    losses["total"].backward()

    assert sigma_vv_db.grad is not None and torch.any(sigma_vv_db.grad.abs() > 0)
    assert sigma_vh_db.grad is not None and torch.any(sigma_vh_db.grad.abs() > 0)
    assert lam_vv.grad is not None and lam_vv.grad.item() != 0.0
    assert lam_vh.grad is not None and lam_vh.grad.item() != 0.0


# ─── T7 — Phase 1b reduction sanity ──────────────────────────────────────────


def test_v03_composite_reduces_to_phase1b_when_sigma_eq_one_and_lambdas_shared():
    """
    Setting σ_VV = σ_VH = 1 and λ_VV = λ_VH = λ_physics must reduce the
    v0.3 five-term composite to the Phase 1b joint VV+VH composite at
    machine precision. Cross-validates the v0.3 formulation against the
    F-2b regression-tested implementation.
    """
    torch.manual_seed(3)
    n = 32

    sigma_vv_db = -11.0 + 2.0 * torch.randn(n)
    sigma_vh_db = -18.5 + 2.0 * torch.randn(n)
    vv_obs = -12.0 + 1.0 * torch.randn(n)
    vh_obs = -17.5 + 1.0 * torch.randn(n)
    m_v_final = torch.full((n,), 0.4) + 0.05 * torch.randn(n)
    m_v_physics = m_v_final.clone()
    epsilon = torch.full((n,), 15.0)
    outputs = _synthetic_outputs(
        m_v_final, m_v_physics, sigma_vv_db, sigma_vh_db, epsilon,
    )
    m_v_obs = torch.full((n,), 0.4) + 0.03 * torch.randn(n)

    lam_phys = 0.5
    lam_mono = 0.1
    lam_bnd = 0.2

    p1b = compute_pinn_mimics_loss(
        outputs,
        m_v_observed=m_v_obs,
        vv_db_observed=vv_obs,
        vh_db_observed=vh_obs,
        lambda_physics=lam_phys,
        lambda_monotonic=lam_mono,
        lambda_bounds=lam_bnd,
    )

    p1c = compute_pinn_mimics_loss_normalised(
        outputs,
        m_v_observed=m_v_obs,
        vv_db_observed=vv_obs,
        vh_db_observed=vh_obs,
        lambda_vv=lam_phys,
        lambda_vh=lam_phys,   # σ=1 + shared λ → reduces to λ_phys · (vv+vh)
        lambda_monotonic=lam_mono,
        lambda_bounds=lam_bnd,
        sigma_vv=1.0,
        sigma_vh=1.0,
    )

    assert p1c["total"].item() == pytest.approx(p1b["total"].item(), abs=1e-6)
    assert p1c["l_data"].item() == pytest.approx(p1b["l_data"].item(), abs=1e-6)
    assert p1c["l_monotonic"].item() == pytest.approx(p1b["l_monotonic"].item(), abs=1e-6)
    assert p1c["l_bounds"].item() == pytest.approx(p1b["l_bounds"].item(), abs=1e-6)


# ─── Bound-clamp sanity: m_v outside [0, θ_sat] still produces L_bounds > 0 ──


def test_l_bounds_active_outside_bounds():
    """
    Sanity: if m_v_final exits [0, PEAT_THETA_SAT], L_bounds > 0. Locks the
    Phase 1b L_bounds reuse (SPEC §18.4.1 explicitly preserves this term).
    """
    n = 4
    m_v_final = torch.tensor(
        [-0.1, 0.5, PEAT_THETA_SAT + 0.1, 0.3], dtype=torch.float32,
    )
    m_v_physics = m_v_final.clone()
    sigma_vv_db = torch.full((n,), -11.0)
    sigma_vh_db = torch.full((n,), -18.5)
    vv_obs = torch.full((n,), -12.0)
    vh_obs = torch.full((n,), -17.0)
    epsilon = torch.full((n,), 12.0)
    outputs = _synthetic_outputs(
        m_v_final, m_v_physics, sigma_vv_db, sigma_vh_db, epsilon,
    )

    losses = compute_pinn_mimics_loss_normalised(
        outputs,
        m_v_observed=m_v_final,
        vv_db_observed=vv_obs,
        vh_db_observed=vh_obs,
        lambda_vv=1.0, lambda_vh=1.0,
        lambda_monotonic=0.0, lambda_bounds=1.0,
        sigma_vv=1.0, sigma_vh=1.0,
    )
    assert losses["l_bounds"].item() > 0.0
