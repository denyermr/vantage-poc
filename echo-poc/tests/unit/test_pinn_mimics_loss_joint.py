"""
Regression tests for the Phase 1b PINN-MIMICS composite loss — pinning the
DEV-1b-010 adjudication that L_physics is joint VV+VH per SPEC §8.

Per DEV-1b-010 (2026-04-20), the F-3 entry-check cross-reference caught a
VV-only implementation of `compute_pinn_mimics_loss.l_physics` that diverged
from SPEC §8's signed-text specification:

    L_physics = MSE(σ°_VV_pred, VV_obs) + MSE(σ°_VH_pred, VH_obs)
              — joint VV+VH. "This is the central change." (SPEC §8)

These tests pin four invariants:

    (T1) `compute_pinn_mimics_loss` returns `l_physics` equal to the sum of
         per-pol MSEs at machine precision on a synthetic input where the
         expected value is hand-computed.

    (T2) The returned dict contains `l_physics_vv` and `l_physics_vh` as
         diagnostic breakdowns, and they sum to `l_physics`.

    (T3) Gradient flow through the VH term is live — ∂l_physics/∂σ°_VH is
         non-zero (not silently stop-gradient'd). Same for VV. This guards
         against a silent regression where `l_physics_vh` is returned but
         detached from the autograd graph.

    (T4) The function signature requires `vh_db_observed` as a positional
         argument. A call missing it must raise TypeError. This guards
         against a silent-revert regression where the VV-only signature
         is restored without a new DEV entry.

If any of these tests fail, the fix is NOT to relax the test — it is to
restore SPEC §8 joint VV+VH to `phase1b/pinn_mimics.py::compute_pinn_mimics_loss`
or author a new DEV entry adjudicating a change to the signed SPEC §8.

Reference:
    SPEC.md §8 (joint VV+VH composite loss)
    phase1b/DEV-1b-010.md (F-3 entry-check halt + Resolution B adjudication)
"""

from __future__ import annotations

import inspect

import pytest
import torch

from phase1b.pinn_mimics import compute_pinn_mimics_loss
from shared.config import PEAT_THETA_SAT


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
        # forward-echoed passthroughs (not read by compute_pinn_mimics_loss)
        "delta_ml": torch.zeros_like(m_v_final),
        "vv_db_observed": torch.zeros_like(sigma_vv_db),
    }


# ─── T1 — machine-precision joint sum ────────────────────────────────────────


def test_l_physics_equals_mse_vv_plus_mse_vh_machine_precision():
    """SPEC §8: L_physics = MSE(σ°_VV, VV_obs) + MSE(σ°_VH, VH_obs)."""
    torch.manual_seed(0)
    n = 16

    # Construct known-value inputs so the expected MSEs are hand-computable.
    sigma_vv_db = torch.tensor([-10.0] * n, dtype=torch.float32)
    vv_obs = torch.tensor([-12.5] * n, dtype=torch.float32)      # Δ = +2.5 dB → MSE = 6.25
    sigma_vh_db = torch.tensor([-18.0] * n, dtype=torch.float32)
    vh_obs = torch.tensor([-17.1] * n, dtype=torch.float32)      # Δ = −0.9 dB → MSE = 0.81

    m_v_final = torch.full((n,), 0.35, dtype=torch.float32)
    m_v_physics = torch.full((n,), 0.35, dtype=torch.float32)
    # epsilon_ground value is irrelevant to L_physics (only to L_monotonic).
    epsilon_ground = torch.full((n,), 10.0, dtype=torch.float32)

    outputs = _synthetic_outputs(
        m_v_final, m_v_physics, sigma_vv_db, sigma_vh_db, epsilon_ground,
    )

    losses = compute_pinn_mimics_loss(
        outputs,
        m_v_observed=torch.full((n,), 0.35, dtype=torch.float32),  # L_data = 0
        vv_db_observed=vv_obs,
        vh_db_observed=vh_obs,
        lambda_physics=1.0,
        lambda_monotonic=1.0,
        lambda_bounds=1.0,
    )

    expected_vv = 2.5 ** 2          # 6.25
    expected_vh = 0.9 ** 2          # 0.81
    expected_joint = expected_vv + expected_vh  # 7.06

    assert losses["l_physics_vv"].item() == pytest.approx(expected_vv, abs=1e-5)
    assert losses["l_physics_vh"].item() == pytest.approx(expected_vh, abs=1e-5)
    assert losses["l_physics"].item() == pytest.approx(expected_joint, abs=1e-5)


# ─── T2 — breakdown terms sum to joint ────────────────────────────────────────


def test_l_physics_vv_plus_vh_equals_l_physics():
    """Diagnostic breakdown invariant — VV + VH components sum to joint."""
    torch.manual_seed(1)
    n = 32

    sigma_vv_db = -12.0 + 2.0 * torch.randn(n)
    sigma_vh_db = -19.0 + 2.0 * torch.randn(n)
    vv_obs = -12.0 + 2.0 * torch.randn(n)
    vh_obs = -19.0 + 2.0 * torch.randn(n)
    m_v = torch.full((n,), 0.4)
    epsilon = torch.full((n,), 15.0)

    outputs = _synthetic_outputs(m_v, m_v, sigma_vv_db, sigma_vh_db, epsilon)

    losses = compute_pinn_mimics_loss(
        outputs,
        m_v_observed=m_v,
        vv_db_observed=vv_obs,
        vh_db_observed=vh_obs,
        lambda_physics=0.1,
        lambda_monotonic=0.1,
        lambda_bounds=0.1,
    )

    breakdown_sum = losses["l_physics_vv"] + losses["l_physics_vh"]
    assert torch.isclose(breakdown_sum, losses["l_physics"], atol=1e-6)


# ─── T3 — gradient flow through both VV and VH ───────────────────────────────


def test_l_physics_gradient_flows_through_both_vv_and_vh():
    """
    Guards against a silent stop-gradient regression on the VH term.

    If `sigma_vh_db` were detached before entering `l_physics`, the gradient
    of `l_physics` with respect to it would be None. Same for VV.
    """
    torch.manual_seed(2)
    n = 8

    # Both σ° tensors are leaf tensors with requires_grad=True — playing
    # the role of the MIMICS-forward output tensors in training.
    sigma_vv_db = torch.full((n,), -11.0, requires_grad=True)
    sigma_vh_db = torch.full((n,), -18.5, requires_grad=True)

    vv_obs = torch.full((n,), -12.0)
    vh_obs = torch.full((n,), -17.0)
    m_v = torch.full((n,), 0.3)
    epsilon = torch.full((n,), 12.0)

    outputs = _synthetic_outputs(m_v, m_v, sigma_vv_db, sigma_vh_db, epsilon)

    losses = compute_pinn_mimics_loss(
        outputs,
        m_v_observed=m_v,
        vv_db_observed=vv_obs,
        vh_db_observed=vh_obs,
        lambda_physics=1.0,
        lambda_monotonic=0.0,
        lambda_bounds=0.0,
    )

    losses["l_physics"].backward()

    assert sigma_vv_db.grad is not None, "Gradient did not flow through VV term"
    assert sigma_vh_db.grad is not None, "Gradient did not flow through VH term (SPEC §8 violation)"
    assert torch.any(sigma_vv_db.grad.abs() > 0), "VV gradient is identically zero"
    assert torch.any(sigma_vh_db.grad.abs() > 0), "VH gradient is identically zero"


# ─── T4 — signature locks vh_db_observed as required ─────────────────────────


def test_compute_pinn_mimics_loss_requires_vh_db_observed_parameter():
    """
    SPEC §8 joint VV+VH requires `vh_db_observed`. A silent revert to the
    VV-only signature must fail loudly.
    """
    sig = inspect.signature(compute_pinn_mimics_loss)
    assert "vh_db_observed" in sig.parameters, (
        "compute_pinn_mimics_loss must accept vh_db_observed per SPEC §8 "
        "joint VV+VH formulation (DEV-1b-010). A signature without this "
        "parameter is a regression from DEV-1b-010 and requires a new DEV "
        "entry to justify."
    )

    # Calling without vh_db_observed must raise TypeError.
    n = 4
    m_v = torch.full((n,), 0.3)
    outputs = _synthetic_outputs(
        m_v, m_v,
        torch.full((n,), -10.0),
        torch.full((n,), -18.0),
        torch.full((n,), 10.0),
    )
    with pytest.raises(TypeError):
        compute_pinn_mimics_loss(  # type: ignore[call-arg]
            outputs,
            m_v_observed=m_v,
            vv_db_observed=torch.full((n,), -12.0),
            lambda_physics=1.0,
            lambda_monotonic=1.0,
            lambda_bounds=1.0,
        )
