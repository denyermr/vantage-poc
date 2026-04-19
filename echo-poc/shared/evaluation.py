"""
Evaluation harness for the ECHO PoC.

Provides metrics computation, aggregation across repetitions,
and statistical tests. Called identically for all models — no
model-specific logic.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from shared import config

logger = logging.getLogger(__name__)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute RMSE, R², mean bias.

    Args:
        y_pred: Predicted VWC, shape (N,). cm³/cm³.
        y_true: Observed VWC, shape (N,). cm³/cm³.

    Returns:
        {"rmse": float, "r_squared": float, "mean_bias": float}

    Raises:
        ValueError: If inputs have different shapes or contain NaN.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
        )
    if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_true)):
        raise ValueError("NaN values in predictions or targets")

    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mean_bias = float(np.mean(residuals))

    return {
        "rmse": rmse,
        "r_squared": r_squared,
        "mean_bias": mean_bias,
    }


def aggregate_metrics_across_reps(metrics_list: list[dict]) -> dict:
    """
    Aggregate metrics across N_REPS repetitions at a single training size.

    Args:
        metrics_list: List of dicts, each with "rmse", "r_squared", "mean_bias".

    Returns:
        Dict with median, q25, q75, mean, std for RMSE and R²,
        plus median bias and rep count.

    Raises:
        ValueError: If metrics_list is empty.
    """
    if not metrics_list:
        raise ValueError("Cannot aggregate empty metrics list")

    rmses = np.array([m["rmse"] for m in metrics_list])
    r2s = np.array([m["r_squared"] for m in metrics_list])
    biases = np.array([m["mean_bias"] for m in metrics_list])

    return {
        "rmse_median": float(np.median(rmses)),
        "rmse_q25": float(np.percentile(rmses, 25)),
        "rmse_q75": float(np.percentile(rmses, 75)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "r_squared_median": float(np.median(r2s)),
        "r_squared_q25": float(np.percentile(r2s, 25)),
        "r_squared_q75": float(np.percentile(r2s, 75)),
        "mean_bias_median": float(np.median(biases)),
        "n_reps": len(metrics_list),
    }


def wilcoxon_test(
    rmse_model_a: list[float],
    rmse_model_b: list[float],
) -> dict:
    """
    Paired Wilcoxon signed-rank test: H0 = distributions equal.

    Args:
        rmse_model_a: RMSE values for model A, length N_REPS.
        rmse_model_b: RMSE values for model B, length N_REPS.

    Returns:
        Dict with statistic, p_value, significance flags, and parameters.

    Raises:
        ValueError: If input lengths differ or are too short.
    """
    a = np.array(rmse_model_a)
    b = np.array(rmse_model_b)

    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 5:
        raise ValueError(f"Need at least 5 pairs for Wilcoxon test, got {len(a)}")

    # Handle case where all differences are zero
    diffs = a - b
    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value_uncorrected": 1.0,
            "significant_uncorrected": False,
            "significant_bonferroni": False,
            "alpha_uncorrected": config.ALPHA_UNCORRECTED,
            "alpha_bonferroni": config.ALPHA_BONFERRONI,
            "n_pairs": len(a),
        }

    result = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")

    p_val = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value_uncorrected": p_val,
        "significant_uncorrected": p_val < config.ALPHA_UNCORRECTED,
        "significant_bonferroni": p_val < config.ALPHA_BONFERRONI,
        "alpha_uncorrected": config.ALPHA_UNCORRECTED,
        "alpha_bonferroni": config.ALPHA_BONFERRONI,
        "n_pairs": len(a),
    }


def build_metrics_json(
    model_name: str,
    config_idx,
    fraction: float,
    fraction_label: str,
    rep: int,
    seed_used: int,
    n_train: int,
    n_val: int,
    n_test: int,
    metrics: dict,
    training_metadata: dict | None = None,
    warnings: list[str] | None = None,
) -> dict:
    """
    Build a standardised metrics JSON dict for saving.

    Args:
        model_name: Model identifier (e.g. "baseline_a").
        config_idx: Config index or "full_pool" for null model.
        fraction: Training fraction.
        fraction_label: Human-readable label.
        rep: Repetition index.
        seed_used: RNG seed used.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        metrics: Dict with rmse, r_squared, mean_bias.
        training_metadata: Model-specific metadata.
        warnings: List of warning strings.

    Returns:
        Complete metrics dict matching Phase 2 schema.
    """
    # Validate no NaN in metrics
    for key, val in metrics.items():
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            raise ValueError(f"Metric '{key}' is {val} — not allowed")

    return {
        "model": model_name,
        "config_idx": config_idx,
        "fraction": fraction,
        "fraction_label": fraction_label,
        "rep": rep,
        "seed_used": seed_used,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "feature_columns": config.FEATURE_COLUMNS,
        "metrics": metrics,
        "training_metadata": training_metadata or {
            "best_params": None,
            "stopped_at_epoch": None,
            "best_val_loss": None,
            "cv_warning": False,
            "stratification_used": None,
        },
        "warnings": warnings or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_metrics_json(metrics_dict: dict, output_path: Path) -> None:
    """
    Write metrics dict to JSON file.

    Args:
        metrics_dict: Validated metrics dict.
        output_path: Path to write JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Metrics saved: %s", output_path)
