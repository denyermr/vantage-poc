"""
Phase 1c-Lean Block C-prime — Baselines reproduction script (G3-Lean deliverable 3).

Reproduces the two Phase 1c-Lean baselines per SPEC §18.6.2 v0.3.3:

  1. RF baseline at 100% training on the n=83 training pool with 5-fold outer CV
     (random_split). Hyperparameters from `shared/config.py:225-232` (RF_PARAM_GRID).
     Implementation matches `shared/baselines/random_forest.py:71-84` shuffled-KFold
     convention. Phase 1 sealed-test reference: 0.147 cm³/cm³ (informational only —
     not directly comparable to training-pool CV per the same training-pool-vs-
     sealed-test variability gap that motivated the v0.3.3 null amendment).

  2. Seasonal-climatological null on VWC over the n=83 training pool, **5-fold
     CV** with per-fold seasonal-mean computation (Method 3 per DEV-1c-lean-007).
     Methodologically symmetric with the RF baseline. Season definition
     `meteorological_DJF_MAM_JJA_SON` per `data/splits/split_manifest.json`.
     Phase 1 sealed-test reference: 0.178 cm³/cm³ (informational only —
     not reproduced at G3-Lean per sealed-set "used-once held-out" discipline).

Output: `phase1c-lean/g3_lean/baselines_locked.json` with both sub-objects, full
reproducibility metadata (RMSE, per-fold values, hyperparameters, code-version
hash, data SHA-256), and explicit informational-comparison annotations.
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from shared import config
from shared.splits import assign_season

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/aligned_dataset.csv")
OUTPUT_PATH = Path(__file__).parent / "baselines_locked.json"

PHASE1_RF_SEALED_TEST_REF = 0.147
PHASE1_NULL_SEALED_TEST_REF = 0.178


def get_git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_pool() -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Load the n=83 chronological training pool from aligned_dataset.csv."""
    df = pd.read_csv(DATA_PATH).sort_values("date").reset_index(drop=True)
    if len(df) != 119:
        raise ValueError(f"Expected n=119 aligned observations; got {len(df)}")
    train = df.iloc[:83].reset_index(drop=True)
    if len(train) != 83:
        raise ValueError(f"Expected n=83 training pool; got {len(train)}")
    y = train["vwc"]
    dates = pd.to_datetime(train["date"])
    return train, y, dates


def reproduce_rf_baseline(
    X_train: np.ndarray, y_train: np.ndarray, seed: int
) -> dict:
    """Outer 5-fold CV (random_split, fixed seed) on n=83 training pool."""
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    per_fold_rmse: list[float] = []
    per_fold_best_params: list[dict] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_train)):
        X_tr, X_te = X_train[tr_idx], X_train[te_idx]
        y_tr, y_te = y_train[tr_idx], y_train[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        inner_cv = KFold(
            n_splits=min(config.RF_CV_FOLDS, len(X_tr)),
            shuffle=True,
            random_state=seed,
        )
        grid = GridSearchCV(
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            config.RF_PARAM_GRID,
            scoring=config.RF_SCORING,
            cv=inner_cv,
            refit=True,
        )
        grid.fit(X_tr_s, y_tr)
        y_pred = grid.predict(X_te_s)
        rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
        per_fold_rmse.append(rmse)
        per_fold_best_params.append(grid.best_params_)
        logger.info(
            "  RF fold %d: RMSE=%.4f, best_params=%s",
            fold_idx, rmse, grid.best_params_,
        )

    return {
        "baseline_id": "rf_100pct_5fold_cv",
        "rmse_cm3_per_cm3": float(np.mean(per_fold_rmse)),
        "rmse_per_fold": per_fold_rmse,
        "rmse_std_across_folds": float(np.std(per_fold_rmse)),
        "n_train": int(len(X_train)),
        "n_outer_folds": 5,
        "n_inner_cv_folds": config.RF_CV_FOLDS,
        "cv_strategy": "random_split",
        "rf_param_grid": dict(config.RF_PARAM_GRID),
        "rf_scoring": config.RF_SCORING,
        "best_params_per_fold": per_fold_best_params,
        "seed": seed,
        "informational_comparison_phase1_sealed_test_rmse": PHASE1_RF_SEALED_TEST_REF,
        "comparison_note": (
            "Phase 1's 0.147 is sealed-test RMSE; this number is "
            "training-pool 5-fold CV RMSE; not directly comparable due to "
            "training-pool-vs-sealed-test variability gap. Comparison is "
            "informational only, not a G3-Lean gate criterion "
            "(per kickoff §3.3.1 / SPEC §18.6.2 v0.3.3)."
        ),
    }


def reproduce_null_baseline_5fold_cv(
    y_train: np.ndarray, dates: pd.DatetimeIndex, seed: int
) -> dict:
    """
    Method 3 per DEV-1c-lean-007: 5-fold CV on n=83 training pool. Each fold
    fits seasonal means on the 4-of-5 training portion and predicts on the
    held-out fifth. Symmetric with the RF baseline's 5-fold CV methodology.
    """
    seasons = assign_season(dates)
    if len(seasons) != len(y_train):
        raise ValueError("season vector / y_train length mismatch")

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    per_fold_rmse: list[float] = []
    per_fold_seasonal_means: list[dict] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(y_train)):
        train_seasons = seasons[tr_idx]
        train_y = y_train[tr_idx]
        means: dict[str, float] = {}
        for s in ("DJF", "MAM", "JJA", "SON"):
            mask = (train_seasons == s)
            if mask.any():
                means[s] = float(np.mean(train_y[mask]))
        global_mean = float(np.mean(train_y))

        pred_te = np.array([
            means.get(s, global_mean) for s in seasons[te_idx]
        ])
        rmse = float(np.sqrt(np.mean((pred_te - y_train[te_idx]) ** 2)))
        per_fold_rmse.append(rmse)
        per_fold_seasonal_means.append({"means": means, "global_mean": global_mean})
        logger.info(
            "  null fold %d: RMSE=%.4f, season_means=%s",
            fold_idx, rmse, {k: f"{v:.4f}" for k, v in means.items()},
        )

    season_counts = {
        s: int(np.sum(seasons == s)) for s in ("DJF", "MAM", "JJA", "SON")
    }

    return {
        "baseline_id": "seasonal_climatological_null_5fold_cv",
        "rmse_cm3_per_cm3": float(np.mean(per_fold_rmse)),
        "rmse_per_fold": per_fold_rmse,
        "rmse_std_across_folds": float(np.std(per_fold_rmse)),
        "n_train": int(len(y_train)),
        "n_folds": 5,
        "cv_strategy": "random_split",
        "season_definition": "meteorological_DJF_MAM_JJA_SON",
        "season_counts_full_pool": season_counts,
        "method": "method_3_5fold_cv_per_fold_seasonal_means",
        "per_fold_seasonal_means": per_fold_seasonal_means,
        "seed": seed,
        "informational_comparison_phase1_sealed_test_rmse": PHASE1_NULL_SEALED_TEST_REF,
        "comparison_note": (
            "Phase 1's 0.178 is sealed-test RMSE under Method 1 (training-pool "
            "seasonal means evaluated on n=36 sealed test). This number is "
            "training-pool 5-fold CV RMSE under Method 3 (per DEV-1c-lean-007); "
            "not directly comparable due to the training-pool-vs-sealed-test "
            "variability gap. Comparison is informational only, not a G3-Lean "
            "gate criterion (per SPEC §18.6.2 v0.3.3 / DEV-1c-lean-007)."
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Reproducing baselines per SPEC §18.6.2 v0.3.3 (Method 3 null)")

    train, y, dates = load_training_pool()
    feature_cols = config.FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Missing feature columns in training pool: {missing}")
    X = train[feature_cols].to_numpy()
    y_arr = y.to_numpy()

    logger.info("RF baseline: 5-fold outer CV on n=%d", len(X))
    rf = reproduce_rf_baseline(X, y_arr, seed=config.SEED)
    logger.info(
        "  → RF mean RMSE = %.4f (per-fold std = %.4f)",
        rf["rmse_cm3_per_cm3"], rf["rmse_std_across_folds"],
    )

    logger.info("Null baseline: Method 3 (5-fold CV) on n=%d", len(y_arr))
    null = reproduce_null_baseline_5fold_cv(y_arr, dates, seed=config.SEED)
    logger.info(
        "  → null mean RMSE = %.4f (per-fold std = %.4f)",
        null["rmse_cm3_per_cm3"], null["rmse_std_across_folds"],
    )

    payload = {
        "spec_version": "v0.3.3",
        "spec_section": "SPEC.md §2 line 87 + §18.6.2",
        "data_source": str(DATA_PATH),
        "data_sha256": sha256_file(DATA_PATH),
        "code_version_hash": get_git_hash(),
        "feature_columns": feature_cols,
        "n_train": int(len(X)),
        "rf_100pct_5fold_cv": rf,
        "seasonal_climatological_null": null,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("baselines_locked.json written.")


if __name__ == "__main__":
    main()
