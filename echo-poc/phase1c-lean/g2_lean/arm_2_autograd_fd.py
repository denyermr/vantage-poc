"""
G2-Lean Arm 2: autograd ↔ finite-difference on the v0.3 five-term per-channel
normalised composite gradient (SPEC §18.6.1 Arm 2).

Compares PyTorch autograd gradients against central finite-difference
numerical gradients for:

  (a) a sample of model parameters at a fixed random initialisation
      (SEED=42), with FD step h_param = 1e-5; and
  (b) the two tunable λ values (λ_VV, λ_VH) — the only differentiable
      coefficients in the v0.3 composite (λ_data, λ_monotonic, λ_bounds
      are fixed scalars per §18.4.1) — with FD step h_lambda = 1e-6.

Tolerance: 1e-3 relative (per SPEC §18.6.1: 0.003 dB equivalent → 1e-3
relative).

The model and all input tensors are cast to float64 for this arm so
central FD has enough precision to land within the 1e-3 tolerance. At
float32 (production training dtype), the composite-loss magnitude is
O(100) while h=1e-5 perturbations produce loss changes of O(1e-9) —
well below float32 epsilon (~1.2e-7) — so the FD signal would drown in
quantisation noise. PyTorch MIMICS forward (`phase1b/physics/mimics.py`)
infers dtype polymorphically from input tensors, so .double()-ing the
model and inputs propagates float64 through the entire chain.

The Arm 2 fixture is deliberately small (n=8) because central FD requires
2 forward passes per scalar perturbed — the cost scales O(N × n_params).
The 8-sample fixture is sufficient to surface any silent stop-gradient
or chain-rule break: the autograd / FD agreement is independent of N
(it would hold at n=1 in principle).

Reference:
    SPEC.md §18.6.1 Arm 2 (autograd ↔ finite-difference)
    Phase 1b DEV-1b-008 G2 gradient-arm (h_param = 1e-5 convention)
    Block B-prime kickoff prompt §5.2
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add repo root (echo-poc/) to sys.path so `phase1`, `phase1b`, `shared` resolve
# when this script is run directly (script-dir is `phase1c-lean/g2_lean/`).
_ECHO_POC_ROOT = Path(__file__).resolve().parents[2]
if str(_ECHO_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ECHO_POC_ROOT))

from phase1b.pinn_mimics import (
    PinnMimics,
    compute_init_sigma_normalisers,
    compute_pinn_mimics_loss_normalised,
)
from shared.config import FEATURE_COLUMNS, PEAT_THETA_SAT

# SPEC §18.4.1 v0.3 fixed coefficients.
LAMBDA_DATA = 1.0
LAMBDA_MONOTONIC_FIXED = 0.01
LAMBDA_BOUNDS_FIXED = 0.01

# Tunable λ values for the gradient evaluation point. Picked mid-grid so
# the gradient is non-trivial but well-conditioned. These are the
# evaluation points, NOT the §18.4.2 grid.
LAMBDA_VV_EVAL = 0.1
LAMBDA_VH_EVAL = 0.1

FIXTURE_SEED = 42
N_FIXTURE = 8                  # small n: FD is O(2 × n_params × N).
N_PARAMS_TO_TEST = 8           # sample of PhysicsNet+CorrectionNet weights.

H_PARAM = 1e-5
H_LAMBDA = 1e-6
TOLERANCE_REL = 1e-3

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = (
    Path(__file__).parent / "results" / "arm_2_result.json"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Could not resolve git hash: %s", e)
        return "unknown"


def _build_synthetic_fixture(seed: int = FIXTURE_SEED, n: int = N_FIXTURE) -> Dict:
    """
    Synthetic deterministic float64 fixture for the gradient arm. Inputs
    are sized to match the Phase 1c-Lean training-pool feature layout (7
    features, incidence-angle ~39° ± wobble) but with random values —
    the gradient arm is shape-of-the-chain-rule only; it does not depend
    on input realism beyond getting non-degenerate per-term losses.

    All tensors are float64 so MIMICS forward propagates float64 through
    autograd and FD branches identically (see module docstring).
    """
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_COLUMNS)
    X = torch.tensor(rng.standard_normal((n, n_features)), dtype=torch.float64)
    theta_inc_deg = torch.tensor(
        39.0 + 2.0 * rng.standard_normal(n), dtype=torch.float64,
    )
    vv_db = torch.tensor(-12.0 + 1.5 * rng.standard_normal(n), dtype=torch.float64)
    vh_db = torch.tensor(-19.0 + 1.5 * rng.standard_normal(n), dtype=torch.float64)
    m_v_obs = torch.tensor(
        np.clip(0.4 + 0.05 * rng.standard_normal(n), 0.0, PEAT_THETA_SAT),
        dtype=torch.float64,
    )
    return {
        "X": X,
        "theta_inc_deg": theta_inc_deg,
        "vv_db": vv_db,
        "vh_db": vh_db,
        "m_v_obs": m_v_obs,
    }


def _composite_total(
    model: PinnMimics,
    fix: Dict,
    lambda_vv: torch.Tensor,
    lambda_vh: torch.Tensor,
    sigma_vv: float,
    sigma_vh: float,
) -> torch.Tensor:
    outputs = model(fix["X"], fix["theta_inc_deg"], fix["vv_db"])
    losses = compute_pinn_mimics_loss_normalised(
        outputs,
        m_v_observed=fix["m_v_obs"],
        vv_db_observed=fix["vv_db"],
        vh_db_observed=fix["vh_db"],
        lambda_vv=lambda_vv,
        lambda_vh=lambda_vh,
        lambda_monotonic=LAMBDA_MONOTONIC_FIXED,
        lambda_bounds=LAMBDA_BOUNDS_FIXED,
        sigma_vv=sigma_vv,
        sigma_vh=sigma_vh,
    )
    return losses["total"]


def _select_test_params(model: PinnMimics, k: int, seed: int) -> List:
    """
    Sample k scalar parameter sites from PhysicsNet + CorrectionNet weight
    tensors. Avoids the 5 MIMICS structural sigmoids — their gradients
    flow through the orientation quadrature and have h-sensitive FD
    behaviour at the Phase 1b DEV-1b-008 precedent (autograd vs FD
    agreement ~1e-2 not 1e-3 there). The composite-loss arm here tests
    the chain rule through smoother NN weights for which 1e-3 is a tight
    invariant.
    """
    rng = np.random.default_rng(seed)
    nn_param_sites: List = []
    for name, p in model.named_parameters():
        if name.startswith("physics_net.") or name.startswith("correction_net."):
            n_scalars = p.numel()
            for flat_idx in range(n_scalars):
                nn_param_sites.append((name, p, flat_idx))
    if len(nn_param_sites) < k:
        raise RuntimeError(
            f"Requested {k} param sites; PhysicsNet+CorrectionNet only has "
            f"{len(nn_param_sites)}."
        )
    chosen = rng.choice(len(nn_param_sites), size=k, replace=False)
    return [nn_param_sites[i] for i in sorted(chosen.tolist())]


def _scalar_at(p: torch.Tensor, flat_idx: int) -> float:
    return p.detach().view(-1)[flat_idx].item()


def _set_scalar_at(p: torch.Tensor, flat_idx: int, value: float) -> None:
    with torch.no_grad():
        p.view(-1)[flat_idx] = value


def run() -> Dict:
    torch.manual_seed(FIXTURE_SEED)
    fix = _build_synthetic_fixture()

    n_features = len(FEATURE_COLUMNS)
    model = PinnMimics(n_features=n_features)
    # Promote model parameters to float64 so the entire chain (PhysicsNet +
    # CorrectionNet + MIMICS forward + Mironov dielectric + composite loss)
    # runs in float64. MIMICS is dtype-polymorphic via `_infer_device_dtype`
    # in `phase1b/physics/mimics.py`.
    model.double()
    model.eval()

    sigmas = compute_init_sigma_normalisers(
        model, fix["X"], fix["theta_inc_deg"], fix["vv_db"], fix["vh_db"],
    )
    sigma_vv = sigmas["sigma_vv"]
    sigma_vh = sigmas["sigma_vh"]

    # ── (a) Parameter-gradient arm ───────────────────────────────────────
    lambda_vv_t = torch.tensor(LAMBDA_VV_EVAL, dtype=torch.float64)
    lambda_vh_t = torch.tensor(LAMBDA_VH_EVAL, dtype=torch.float64)

    model.zero_grad(set_to_none=True)
    total = _composite_total(model, fix, lambda_vv_t, lambda_vh_t, sigma_vv, sigma_vh)
    total.backward()

    autograd_param_grads_per_site = {}
    for name, p, flat_idx in _select_test_params(
        model, N_PARAMS_TO_TEST, FIXTURE_SEED,
    ):
        if p.grad is None:
            raise RuntimeError(f"No gradient on {name} — autograd chain broken.")
        autograd_param_grads_per_site[(name, flat_idx)] = (
            p.grad.detach().view(-1)[flat_idx].item()
        )

    # FD: re-evaluate the composite at param ± h, central diff.
    fd_param_grads_per_site = {}
    for name, p, flat_idx in _select_test_params(
        model, N_PARAMS_TO_TEST, FIXTURE_SEED,
    ):
        x0 = _scalar_at(p, flat_idx)
        _set_scalar_at(p, flat_idx, x0 + H_PARAM)
        with torch.no_grad():
            t_plus = _composite_total(
                model, fix, lambda_vv_t, lambda_vh_t, sigma_vv, sigma_vh,
            ).item()
        _set_scalar_at(p, flat_idx, x0 - H_PARAM)
        with torch.no_grad():
            t_minus = _composite_total(
                model, fix, lambda_vv_t, lambda_vh_t, sigma_vv, sigma_vh,
            ).item()
        _set_scalar_at(p, flat_idx, x0)  # restore
        fd_param_grads_per_site[(name, flat_idx)] = (t_plus - t_minus) / (2.0 * H_PARAM)

    # Compute relative differences.
    param_rows = []
    param_max_rel = 0.0
    for key in autograd_param_grads_per_site:
        ag = autograd_param_grads_per_site[key]
        fd = fd_param_grads_per_site[key]
        denom = max(abs(ag), abs(fd), 1e-12)
        rel = abs(ag - fd) / denom
        param_max_rel = max(param_max_rel, rel)
        name, flat_idx = key
        param_rows.append({
            "param_name": name,
            "flat_idx": int(flat_idx),
            "autograd": ag,
            "fd_central": fd,
            "abs_diff": abs(ag - fd),
            "rel_diff": rel,
        })

    param_pass = bool(param_max_rel < TOLERANCE_REL)

    # ── (b) λ-gradient arm ───────────────────────────────────────────────
    # Re-do with λ values as differentiable tensors. Model and inputs are
    # already float64; ∂L/∂λ_VV / ∂L/∂λ_VH are float64 deterministic.
    model.zero_grad(set_to_none=True)
    lambda_vv_diff = torch.tensor(LAMBDA_VV_EVAL, dtype=torch.float64, requires_grad=True)
    lambda_vh_diff = torch.tensor(LAMBDA_VH_EVAL, dtype=torch.float64, requires_grad=True)

    outputs = model(fix["X"], fix["theta_inc_deg"], fix["vv_db"])
    losses = compute_pinn_mimics_loss_normalised(
        outputs,
        m_v_observed=fix["m_v_obs"],
        vv_db_observed=fix["vv_db"],
        vh_db_observed=fix["vh_db"],
        lambda_vv=lambda_vv_diff,
        lambda_vh=lambda_vh_diff,
        lambda_monotonic=LAMBDA_MONOTONIC_FIXED,
        lambda_bounds=LAMBDA_BOUNDS_FIXED,
        sigma_vv=sigma_vv,
        sigma_vh=sigma_vh,
    )
    losses["total"].backward()

    autograd_lambda_vv_grad = lambda_vv_diff.grad.item()
    autograd_lambda_vh_grad = lambda_vh_diff.grad.item()

    # FD on λ: re-evaluate composite at λ ± h, central diff. Use float
    # scalars to keep the model forward in float32 (same as production).
    def _eval_at_lambda(lvv: float, lvh: float) -> float:
        with torch.no_grad():
            outs = model(fix["X"], fix["theta_inc_deg"], fix["vv_db"])
            ls = compute_pinn_mimics_loss_normalised(
                outs,
                m_v_observed=fix["m_v_obs"],
                vv_db_observed=fix["vv_db"],
                vh_db_observed=fix["vh_db"],
                lambda_vv=lvv,
                lambda_vh=lvh,
                lambda_monotonic=LAMBDA_MONOTONIC_FIXED,
                lambda_bounds=LAMBDA_BOUNDS_FIXED,
                sigma_vv=sigma_vv,
                sigma_vh=sigma_vh,
            )
            return ls["total"].item()

    fd_vv = (
        _eval_at_lambda(LAMBDA_VV_EVAL + H_LAMBDA, LAMBDA_VH_EVAL)
        - _eval_at_lambda(LAMBDA_VV_EVAL - H_LAMBDA, LAMBDA_VH_EVAL)
    ) / (2.0 * H_LAMBDA)
    fd_vh = (
        _eval_at_lambda(LAMBDA_VV_EVAL, LAMBDA_VH_EVAL + H_LAMBDA)
        - _eval_at_lambda(LAMBDA_VV_EVAL, LAMBDA_VH_EVAL - H_LAMBDA)
    ) / (2.0 * H_LAMBDA)

    rel_vv = abs(autograd_lambda_vv_grad - fd_vv) / max(
        abs(autograd_lambda_vv_grad), abs(fd_vv), 1e-12,
    )
    rel_vh = abs(autograd_lambda_vh_grad - fd_vh) / max(
        abs(autograd_lambda_vh_grad), abs(fd_vh), 1e-12,
    )
    lambda_max_rel = max(rel_vv, rel_vh)
    lambda_pass = bool(lambda_max_rel < TOLERANCE_REL)

    overall_pass = param_pass and lambda_pass

    result = {
        "arm": "autograd_finite_difference",
        "spec_version": "v0.3",
        "loss_formulation": "v0.3_five_term_per_channel_normalised",
        "param_grad": {
            "n_params_tested": N_PARAMS_TO_TEST,
            "max_rel_diff": param_max_rel,
            "tolerance_rel": TOLERANCE_REL,
            "pass": param_pass,
            "rows": param_rows,
        },
        "lambda_grad": {
            "n_lambdas_tested": 2,
            "max_rel_diff": lambda_max_rel,
            "tolerance_rel": TOLERANCE_REL,
            "pass": lambda_pass,
            "lambdas_differentiated": ["lambda_vv", "lambda_vh"],
            "lambdas_fixed": ["lambda_data", "lambda_monotonic", "lambda_bounds"],
            "rows": [
                {
                    "lambda": "lambda_vv",
                    "autograd": autograd_lambda_vv_grad,
                    "fd_central": fd_vv,
                    "rel_diff": rel_vv,
                },
                {
                    "lambda": "lambda_vh",
                    "autograd": autograd_lambda_vh_grad,
                    "fd_central": fd_vh,
                    "rel_diff": rel_vh,
                },
            ],
        },
        "fixture_seed": FIXTURE_SEED,
        "fd_step_param": H_PARAM,
        "fd_step_lambda": H_LAMBDA,
        "tolerance_rationale": (
            "0.003 dB equivalent per SPEC §18.6.1; in loss-space ≈ 7e-4; "
            "operational tolerance 1e-3"
        ),
        "code_version_hash": _git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pass": overall_pass,
        "init_sigma_vv": sigma_vv,
        "init_sigma_vh": sigma_vh,
    }
    return result


def main() -> int:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = run()
    t1 = time.time()
    with open(RESULT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "Arm 2 autograd↔FD: param_max_rel=%.3e λ_max_rel=%.3e (tol=%.0e) → %s [%.1fs]",
        result["param_grad"]["max_rel_diff"],
        result["lambda_grad"]["max_rel_diff"],
        result["param_grad"]["tolerance_rel"],
        "PASS" if result["pass"] else "FAIL",
        t1 - t0,
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
