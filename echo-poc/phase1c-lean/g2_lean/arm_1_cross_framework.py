"""
G2-Lean Arm 1: numpy ↔ PyTorch cross-framework consistency on the v0.3
five-term per-channel normalised composite loss (SPEC §18.6.1 Arm 1).

Mirrors the Phase 1b DEV-1b-008 G2 Moderate Pass convention. Generates a
fixed random fixture of 64 tuples of pre-computed per-term scalars and
weight/sigma values, evaluates the v0.3 composite

    L_total = L_data
              + λ_VV · (L_phys_VV / σ_VV)
              + λ_VH · (L_phys_VH / σ_VH)
              + λ_monotonic · L_monotonic
              + λ_bounds   · L_bounds

independently in numpy (float64) and PyTorch (float64), and checks
agreement at machine precision (max abs diff < 1e-12). Fixed coefficients
λ_data = 1.0, λ_monotonic = 0.01, λ_bounds = 0.01 per SPEC §18.4.1 v0.3 are
applied; λ_VV and λ_VH are sampled per row over the §18.4.2 grid range
[1e-4, 1e1] in log space. σ_VV / σ_VH sampled positive on a realistic
range (~0.5, 5.0).

Tolerance: abs_diff < 1e-12 (machine-precision invariant per Phase 1b
DEV-1b-008 convention). Result JSON written to
`phase1c-lean/g2_lean/results/arm_1_result.json`.

Reference:
    SPEC.md §18.4.1 (v0.3 five-term composite)
    SPEC.md §18.4.2 (λ-grid range)
    SPEC.md §18.6.1 Arm 1 (cross-framework consistency)
    Phase 1b DEV-1b-008 (G2 Moderate Pass on MIMICS forward model)
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# Fixed coefficients (SPEC §18.4.1 v0.3).
LAMBDA_DATA = 1.0
LAMBDA_MONOTONIC = 0.01
LAMBDA_BOUNDS = 0.01

# Tunable axes ranges (SPEC §18.4.2): log-space [1e-4, 1e1].
LAMBDA_LOG10_LB = -4.0
LAMBDA_LOG10_UB = 1.0

N_FIXTURE = 64
FIXTURE_SEED = 42
TOLERANCE_ABS = 1e-12

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = (
    Path(__file__).parent / "results" / "arm_1_result.json"
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


def _build_fixture(seed: int = FIXTURE_SEED, n: int = N_FIXTURE) -> Dict[str, np.ndarray]:
    """
    Generate a fixed random fixture of 64 tuples of pre-computed per-term
    scalars. All per-term losses sampled positive (matching their definitions
    as MSE / hinge means). λ_VV, λ_VH sampled in log-space over [1e-4, 1e1].
    σ_VV, σ_VH sampled positive over [~0.5, 5].
    """
    rng = np.random.default_rng(seed)
    return {
        "l_data": rng.uniform(0.001, 0.05, size=n),
        "l_phys_vv": rng.uniform(0.5, 50.0, size=n),
        "l_phys_vh": rng.uniform(0.2, 25.0, size=n),
        "l_monotonic": rng.uniform(0.0, 5.0, size=n),
        "l_bounds": rng.uniform(0.0, 0.2, size=n),
        "lambda_vv": 10.0 ** rng.uniform(LAMBDA_LOG10_LB, LAMBDA_LOG10_UB, size=n),
        "lambda_vh": 10.0 ** rng.uniform(LAMBDA_LOG10_LB, LAMBDA_LOG10_UB, size=n),
        "sigma_vv": rng.uniform(0.5, 5.0, size=n),
        "sigma_vh": rng.uniform(0.5, 5.0, size=n),
    }


def _v03_composite_numpy(f: Dict[str, np.ndarray]) -> np.ndarray:
    """Independent numpy reference implementation of the v0.3 composite."""
    return (
        LAMBDA_DATA * f["l_data"]
        + f["lambda_vv"] * (f["l_phys_vv"] / f["sigma_vv"])
        + f["lambda_vh"] * (f["l_phys_vh"] / f["sigma_vh"])
        + LAMBDA_MONOTONIC * f["l_monotonic"]
        + LAMBDA_BOUNDS * f["l_bounds"]
    )


def _v03_composite_torch(f: Dict[str, np.ndarray]) -> np.ndarray:
    """PyTorch implementation, evaluated at float64 to match numpy default."""
    t = {k: torch.from_numpy(v.astype(np.float64)) for k, v in f.items()}
    total = (
        LAMBDA_DATA * t["l_data"]
        + t["lambda_vv"] * (t["l_phys_vv"] / t["sigma_vv"])
        + t["lambda_vh"] * (t["l_phys_vh"] / t["sigma_vh"])
        + LAMBDA_MONOTONIC * t["l_monotonic"]
        + LAMBDA_BOUNDS * t["l_bounds"]
    )
    return total.detach().cpu().numpy()


def run() -> Dict:
    fixture = _build_fixture()
    np_total = _v03_composite_numpy(fixture)
    pt_total = _v03_composite_torch(fixture)

    abs_diff = np.abs(np_total - pt_total)
    # Use np.maximum to avoid div-by-zero on degenerate (all-zero) entries.
    rel_diff = abs_diff / np.maximum(np.abs(np_total), 1e-300)

    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    passed = bool(max_abs < TOLERANCE_ABS)

    result = {
        "arm": "cross_framework_consistency",
        "spec_version": "v0.3",
        "loss_formulation": "v0.3_five_term_per_channel_normalised",
        "n_fixture_rows": N_FIXTURE,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "tolerance_abs": TOLERANCE_ABS,
        "pass": passed,
        "fixture_seed": FIXTURE_SEED,
        "fixed_coefficients": {
            "lambda_data": LAMBDA_DATA,
            "lambda_monotonic": LAMBDA_MONOTONIC,
            "lambda_bounds": LAMBDA_BOUNDS,
        },
        "code_version_hash": _git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return result


def main() -> int:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    with open(RESULT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "Arm 1 cross-framework: max_abs_diff=%.3e (tol=%.0e) → %s",
        result["max_abs_diff"], result["tolerance_abs"],
        "PASS" if result["pass"] else "FAIL",
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
