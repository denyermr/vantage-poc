"""
G2-Lean Arm 3: scale sanity on the actual Phase 1 training set (SPEC §18.6.1
Arm 3).

Computes σ_VV and σ_VH on the actual Phase 1c-Lean training pool (the
Phase 1b 100% config_000 train_indices, n=83) at a randomly-initialised
PinnMimics model (SEED=42), then verifies that after normalising the
per-sample squared backscatter errors by σ, both series have unit
population standard deviation within 1e-6 numerical tolerance.

The data path is resolved by the same `prepare_pinn_data` helper that
`phase1b/lambda_search/run_f2.py` uses, so the n=83 training set is
identical to F-2b. The Arm halts-and-flags if the resolved path differs
or if the train_indices length is not 83 (per SPEC §15 / §18.5
training-pool definition).

Tolerance: abs(std_normalised - 1.0) < 1e-6 per channel.

Reference:
    SPEC.md §18.4.1 (per-channel normalisation, v0.3)
    SPEC.md §18.6.1 Arm 3 (scale sanity invariant)
    Block B-prime kickoff prompt §5.3
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# Add repo root (echo-poc/) to sys.path so `phase1`, `phase1b`, `shared` resolve
# when this script is run directly (script-dir is `phase1c-lean/g2_lean/`).
_ECHO_POC_ROOT = Path(__file__).resolve().parents[2]
if str(_ECHO_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ECHO_POC_ROOT))

from phase1.lambda_search import prepare_pinn_data
from phase1b.pinn_mimics import PinnMimics, compute_init_sigma_normalisers
from shared import config
from shared.config import FEATURE_COLUMNS

# Phase 1c-Lean uses 100% config_000 — same n=83 training pool as F-2b
# (run_f2.py CONFIG_INDICES_100PCT = list(range(0, 10)); we anchor on the
# first one for σ-init).
CONFIG_IDX_FOR_SIGMA_INIT = 0
EXPECTED_N_TRAIN = 83

FIXTURE_SEED = 42
TOLERANCE_ABS = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = (
    Path(__file__).parent / "results" / "arm_3_result.json"
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


def _load_n83_training_pool() -> Dict:
    """
    Load the n=83 training pool via the F-2b convention.

    SPEC §18.6.2: "Training fraction: 100% only (N=83)." Phase 1c-Lean
    trains on the full training pool (no inner CV split during training;
    the §18.5 5-fold CV is for evaluation only). At F-2b config_000,
    `prepare_pinn_data` returns the pool pre-split into train_indices
    (n=66) + val_indices (n=17) — Phase 1b used this inner split for
    early-stopping. For Phase 1c-Lean σ-init at G3-Lean, we union the
    two back into the N=83 pool used for the actual training forward
    pass per SPEC §18.6.2.

    Halts-and-flags if the resolved data path differs from F-2b or if
    train ∪ val ≠ 83 (SPEC §15 / §18.5 invariant).
    """
    aligned_dataset_path = config.DATA_PROCESSED / "aligned_dataset.csv"
    splits_dir = config.DATA_SPLITS / "configs"
    test_indices_path = config.DATA_SPLITS / "test_indices.json"
    cfg_path = splits_dir / f"config_{CONFIG_IDX_FOR_SIGMA_INIT:03d}.json"

    if not aligned_dataset_path.exists():
        raise FileNotFoundError(
            f"aligned_dataset.csv not found at {aligned_dataset_path}; "
            "G2-Lean Arm 3 cannot proceed without the Phase 1 training data."
        )
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config split file not found at {cfg_path}; expected the F-2b "
            f"config_{CONFIG_IDX_FOR_SIGMA_INIT:03d}.json train_indices."
        )

    data = prepare_pinn_data(aligned_dataset_path, cfg_path, test_indices_path)

    cfg = data["config"]
    if abs(cfg["fraction"] - 1.0) > 1e-9:
        raise ValueError(
            f"Expected 100% training fraction at config_idx="
            f"{CONFIG_IDX_FOR_SIGMA_INIT}; got {cfg['fraction']}. SPEC §18.6.2 "
            "requires the n=83 training pool for Phase 1c-Lean σ-init."
        )

    n_inner_train = data["X_train"].shape[0]
    n_inner_val = data["X_val"].shape[0]
    n_pool = n_inner_train + n_inner_val
    if n_pool != EXPECTED_N_TRAIN:
        raise ValueError(
            f"Phase 1c-Lean Arm 3 expects training-pool size N="
            f"{EXPECTED_N_TRAIN} per SPEC §15 / §18.5; got "
            f"train({n_inner_train}) + val({n_inner_val}) = {n_pool}."
        )

    # Union the inner train + val back into the N=83 training pool used
    # for actual Phase 1c-Lean training forward pass per SPEC §18.6.2.
    X_pool = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    vv_db_pool = np.concatenate(
        [data["vv_db_train_raw"], data["vv_db_val_raw"]], axis=0,
    ).astype(np.float32)

    # MIMICS forward needs degrees, not radians; re-extract from raw frame.
    import pandas as pd
    df = pd.read_csv(aligned_dataset_path)
    theta_col_idx = FEATURE_COLUMNS.index("incidence_angle_mean")
    vh_col_idx = FEATURE_COLUMNS.index("vh_db")
    feat_vals = df[FEATURE_COLUMNS].values
    pool_idx = np.concatenate(
        [np.array(cfg["train_indices"]), np.array(cfg["val_indices"])]
    )
    theta_pool_deg = feat_vals[pool_idx][:, theta_col_idx].astype(np.float32)
    vh_db_pool = feat_vals[pool_idx][:, vh_col_idx].astype(np.float32)

    return {
        "X_train": X_pool,
        "theta_train_deg": theta_pool_deg,
        "vv_db_train_raw": vv_db_pool,
        "vh_db_train_raw": vh_db_pool,
        "data_source": str(aligned_dataset_path),
        "config_path": str(cfg_path),
        "pool_indices_source": (
            f"config_{CONFIG_IDX_FOR_SIGMA_INIT:03d}.json train_indices "
            f"({n_inner_train}) ∪ val_indices ({n_inner_val}) = {n_pool}"
        ),
    }


def run() -> Dict:
    pool = _load_n83_training_pool()

    torch.manual_seed(FIXTURE_SEED)
    n_features = len(FEATURE_COLUMNS)
    model = PinnMimics(n_features=n_features)
    model.eval()

    X_train = torch.tensor(pool["X_train"], dtype=torch.float32)
    theta_train_deg = torch.tensor(pool["theta_train_deg"], dtype=torch.float32)
    vv_db_train = torch.tensor(pool["vv_db_train_raw"], dtype=torch.float32)
    vh_db_train = torch.tensor(pool["vh_db_train_raw"], dtype=torch.float32)

    sigmas = compute_init_sigma_normalisers(
        model, X_train, theta_train_deg, vv_db_train, vh_db_train,
    )
    sigma_vv = sigmas["sigma_vv"]
    sigma_vh = sigmas["sigma_vh"]

    # Re-do the forward pass to get the per-sample squared errors and
    # apply the normalisation — the helper computed σ as a scalar but
    # didn't return the per-sample series.
    with torch.no_grad():
        outputs = model(X_train, theta_train_deg, vv_db_train)
        sq_err_vv = ((outputs["sigma_vv_db"] - vv_db_train) ** 2).cpu().numpy()
        sq_err_vh = ((outputs["sigma_vh_db"] - vh_db_train) ** 2).cpu().numpy()

    normalised_vv = sq_err_vv / sigma_vv
    normalised_vh = sq_err_vh / sigma_vh
    # Match the σ-helper convention: population std (ddof=0).
    std_normalised_vv = float(np.std(normalised_vv, ddof=0))
    std_normalised_vh = float(np.std(normalised_vh, ddof=0))

    pass_vv = bool(abs(std_normalised_vv - 1.0) < TOLERANCE_ABS)
    pass_vh = bool(abs(std_normalised_vh - 1.0) < TOLERANCE_ABS)
    overall_pass = pass_vv and pass_vh

    result = {
        "arm": "scale_sanity",
        "spec_version": "v0.3",
        "n_train": int(EXPECTED_N_TRAIN),
        "sigma_vv": sigma_vv,
        "sigma_vh": sigma_vh,
        "std_normalised_l_phys_vv": std_normalised_vv,
        "std_normalised_l_phys_vh": std_normalised_vh,
        "tolerance_abs": TOLERANCE_ABS,
        "pass_vv": pass_vv,
        "pass_vh": pass_vh,
        "fixture_seed": FIXTURE_SEED,
        "data_source": pool["data_source"],
        "config_path": pool["config_path"],
        "config_idx": CONFIG_IDX_FOR_SIGMA_INIT,
        "pool_indices_source": pool["pool_indices_source"],
        "code_version_hash": _git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pass": overall_pass,
    }
    return result


def main() -> int:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    with open(RESULT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "Arm 3 scale-sanity: σ_VV=%.6e σ_VH=%.6e std_norm_VV=%.9f std_norm_VH=%.9f → %s",
        result["sigma_vv"], result["sigma_vh"],
        result["std_normalised_l_phys_vv"],
        result["std_normalised_l_phys_vh"],
        "PASS" if result["pass"] else "FAIL",
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
