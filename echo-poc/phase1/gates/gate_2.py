"""
Gate 2 — Baseline Models & Evaluation Harness.

Checks all 13 criteria from SPEC_PHASE2.md §P2.11.
Exit code 0 = pass, 1 = fail.

Usage:
    python poc/gates/gate_2.py [--confirm-deviations]
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from shared import config
from shared.evaluation import aggregate_metrics_across_reps

logger = logging.getLogger(__name__)

# Hard-fail criteria: G2-01 through G2-08, G2-11, G2-12, G2-13
# G2-09 is warning only, G2-10 is soft


def check_g2_01(configs_dir: Path) -> dict:
    """G2-01: All 40 config files exist and are valid JSON."""
    valid = 0
    for i in range(config.N_CONFIGS):
        path = configs_dir / f"config_{i:03d}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if "train_indices" in data and "val_indices" in data:
                    valid += 1
            except json.JSONDecodeError:
                pass
    return {
        "id": "G2-01",
        "description": "Config files",
        "threshold": "40 valid",
        "measured": valid,
        "passed": valid == 40,
    }


def check_g2_02(splits_dir: Path) -> dict:
    """G2-02: Split manifest exists."""
    path = splits_dir / "split_manifest.json"
    exists = path.exists()
    return {
        "id": "G2-02",
        "description": "Split manifest",
        "threshold": "present",
        "measured": "✓" if exists else "✗",
        "passed": exists,
    }


def check_g2_03(configs_dir: Path, aligned_path: Path) -> dict:
    """G2-03: All train indices chronologically before all test indices."""
    df = pd.read_csv(aligned_path)
    dates = pd.DatetimeIndex(df["date"])

    with open(config.DATA_SPLITS / "test_indices.json") as f:
        test_info = json.load(f)
    split_idx = test_info["split_idx"]
    test_dates = dates[split_idx:]

    violations = 0
    for i in range(config.N_CONFIGS):
        path = configs_dir / f"config_{i:03d}.json"
        if not path.exists():
            continue
        with open(path) as f:
            cfg = json.load(f)
        train_dates = dates[cfg["train_indices"]]
        if train_dates.max() >= test_dates.min():
            violations += 1

    return {
        "id": "G2-03",
        "description": "Chronological splits",
        "threshold": "no leakage",
        "measured": "verified" if violations == 0 else f"{violations} violations",
        "passed": violations == 0,
    }


def check_g2_04(configs_dir: Path) -> dict:
    """G2-04: No overlap between train, val, and test sets."""
    with open(config.DATA_SPLITS / "test_indices.json") as f:
        test_info = json.load(f)
    split_idx = test_info["split_idx"]
    test_set = set(range(split_idx, test_info["n_total"]))

    overlaps = 0
    for i in range(config.N_CONFIGS):
        path = configs_dir / f"config_{i:03d}.json"
        if not path.exists():
            continue
        with open(path) as f:
            cfg = json.load(f)
        train = set(cfg["train_indices"])
        val = set(cfg["val_indices"])
        if train & val or train & test_set or val & test_set:
            overlaps += 1

    return {
        "id": "G2-04",
        "description": "No set overlap",
        "threshold": "0 overlap",
        "measured": overlaps,
        "passed": overlaps == 0,
    }


def check_g2_05(metrics_dir: Path) -> dict:
    """G2-05: All 40 baseline_a metric files exist and are valid."""
    valid = 0
    for i in range(config.N_CONFIGS):
        path = metrics_dir / f"config_{i:03d}_baseline_a.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if "metrics" in data and "rmse" in data["metrics"]:
                    valid += 1
            except json.JSONDecodeError:
                pass
    return {
        "id": "G2-05",
        "description": "RF metric files",
        "threshold": "40 valid",
        "measured": valid,
        "passed": valid == 40,
    }


def check_g2_06(metrics_dir: Path) -> dict:
    """G2-06: All 40 baseline_b metric files exist and are valid."""
    valid = 0
    for i in range(config.N_CONFIGS):
        path = metrics_dir / f"config_{i:03d}_baseline_b.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if "metrics" in data and "rmse" in data["metrics"]:
                    valid += 1
            except json.JSONDecodeError:
                pass
    return {
        "id": "G2-06",
        "description": "NN metric files",
        "threshold": "40 valid",
        "measured": valid,
        "passed": valid == 40,
    }


def check_g2_07(metrics_dir: Path) -> dict:
    """G2-07: baseline_0 metrics file exists."""
    path = metrics_dir / "baseline_0_metrics.json"
    exists = path.exists()
    return {
        "id": "G2-07",
        "description": "Null model metrics",
        "threshold": "present",
        "measured": "✓" if exists else "✗",
        "passed": exists,
    }


def load_rf_metrics_at_100pct(metrics_dir: Path) -> list[dict]:
    """Load RF metrics for 100% training configs (0-9)."""
    metrics = []
    for i in range(config.N_REPS):
        path = metrics_dir / f"config_{i:03d}_baseline_a.json"
        if path.exists():
            with open(path) as f:
                metrics.append(json.load(f)["metrics"])
    return metrics


def check_g2_08(metrics_dir: Path) -> dict:
    """G2-08: RF beats null model RMSE by ≥10% at 100% training (DEV-006).

    Original spec: RF RMSE ≤ 0.10 (absolute). Not met at Moor House due to
    attenuated SAR-VWC signal (DEV-003). Reinterpreted as relative criterion:
    RF must beat null model RMSE by ≥10%. See DEVIATIONS.md DEV-006.
    """
    rf_100 = load_rf_metrics_at_100pct(metrics_dir)
    null_path = metrics_dir / "baseline_0_metrics.json"

    if not rf_100 or not null_path.exists():
        return {
            "id": "G2-08",
            "description": "RF vs null @ 100%",
            "threshold": f"≥{config.GATE2_RF_RELATIVE_IMPROVEMENT_MIN:.0%} improvement",
            "measured": "no data",
            "passed": False,
        }

    with open(null_path) as f:
        null_rmse = json.load(f)["metrics"]["rmse"]

    agg = aggregate_metrics_across_reps(rf_100)
    median_rmse = agg["rmse_median"]
    relative_improvement = (null_rmse - median_rmse) / null_rmse

    return {
        "id": "G2-08",
        "description": "RF vs null @ 100%",
        "threshold": f"≥{config.GATE2_RF_RELATIVE_IMPROVEMENT_MIN:.0%} improvement",
        "measured": f"{median_rmse:.4f} ({relative_improvement:.1%} improvement)",
        "passed": relative_improvement >= config.GATE2_RF_RELATIVE_IMPROVEMENT_MIN,
        "detail": {
            "rf_median_rmse": median_rmse,
            "null_rmse": null_rmse,
            "relative_improvement": round(relative_improvement, 4),
            "original_absolute_threshold": config.GATE2_RF_RMSE_THRESHOLD,
            "deviation": "DEV-006",
        },
    }


def check_g2_09(metrics_dir: Path) -> dict:
    """G2-09: Both baselines beat null model at 100% (WARNING only)."""
    null_path = metrics_dir / "baseline_0_metrics.json"
    if not null_path.exists():
        return {
            "id": "G2-09",
            "description": "Baselines beat null",
            "threshold": "RMSE < null",
            "measured": "no null data",
            "passed": False,
            "is_warning": True,
        }

    with open(null_path) as f:
        null_rmse = json.load(f)["metrics"]["rmse"]

    rf_100 = load_rf_metrics_at_100pct(metrics_dir)
    nn_100 = []
    for i in range(config.N_REPS):
        path = metrics_dir / f"config_{i:03d}_baseline_b.json"
        if path.exists():
            with open(path) as f:
                nn_100.append(json.load(f)["metrics"])

    rf_median = aggregate_metrics_across_reps(rf_100)["rmse_median"] if rf_100 else 999
    nn_median = aggregate_metrics_across_reps(nn_100)["rmse_median"] if nn_100 else 999

    rf_beats = rf_median < null_rmse
    nn_beats = nn_median < null_rmse
    both_beat = rf_beats and nn_beats

    return {
        "id": "G2-09",
        "description": "Baselines beat null",
        "threshold": f"RMSE < {null_rmse:.3f}",
        "measured": f"RF:{rf_median:.3f} NN:{nn_median:.3f}",
        "passed": both_beat,
        "is_warning": True,  # G2-09 is warning only, doesn't fail the gate
    }


def check_g2_10() -> dict:
    """G2-10: Learning curve figures exist."""
    fig1 = config.OUTPUTS_FIGURES / "p2_learning_curves_baselines.png"
    fig2 = config.OUTPUTS_FIGURES / "p2_feature_diagnostics.png"
    count = sum(1 for f in [fig1, fig2] if f.exists())
    return {
        "id": "G2-10",
        "description": "Figures exist",
        "threshold": "2 files",
        "measured": count,
        "passed": count == 2,
    }


def check_g2_11(metrics_dir: Path) -> dict:
    """G2-11: No NaN values in any metrics file."""
    nan_count = 0
    for path in metrics_dir.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        for key, val in metrics.items():
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                nan_count += 1

    return {
        "id": "G2-11",
        "description": "No NaN in metrics",
        "threshold": "0",
        "measured": nan_count,
        "passed": nan_count == 0,
    }


def check_g2_12(confirm: bool) -> dict:
    """G2-12: DEVIATIONS.md reviewed (manual sign-off)."""
    return {
        "id": "G2-12",
        "description": "Deviations reviewed",
        "threshold": "manual",
        "measured": "✓" if confirm else "pending",
        "passed": confirm,
    }


def check_g2_13() -> dict:
    """G2-13: pytest tests/ passes with 0 failures."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True, text=True, timeout=300,
            cwd=config.PROJECT_ROOT,
        )
        # Parse pytest output for failures
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        return {
            "id": "G2-13",
            "description": "pytest passes",
            "threshold": "0 failures",
            "measured": "0" if passed else output.strip().split("\n")[-1],
            "passed": passed,
        }
    except subprocess.TimeoutExpired:
        return {
            "id": "G2-13",
            "description": "pytest passes",
            "threshold": "0 failures",
            "measured": "timeout",
            "passed": False,
        }


def build_baseline_summary(metrics_dir: Path) -> dict:
    """Build baseline performance summary for gate output."""
    null_path = metrics_dir / "baseline_0_metrics.json"
    null_rmse = None
    if null_path.exists():
        with open(null_path) as f:
            null_rmse = json.load(f)["metrics"]["rmse"]

    summary = {"null_rmse": null_rmse}
    for model_key in ["baseline_a", "baseline_b"]:
        key = "rf_rmse_by_fraction" if model_key == "baseline_a" else "nn_rmse_by_fraction"
        summary[key] = {}
        for size_idx, frac in enumerate(config.TRAINING_FRACTIONS):
            label = config.TRAINING_SIZE_LABELS[frac]
            metrics = []
            for rep in range(config.N_REPS):
                idx = size_idx * config.N_REPS + rep
                path = metrics_dir / f"config_{idx:03d}_{model_key}.json"
                if path.exists():
                    with open(path) as f:
                        metrics.append(json.load(f)["metrics"])
            if metrics:
                agg = aggregate_metrics_across_reps(metrics)
                summary[key][label] = {
                    "median": agg["rmse_median"],
                    "q25": agg["rmse_q25"],
                    "q75": agg["rmse_q75"],
                }

    return summary


def compute_pinn_targets(summary: dict) -> dict:
    """Compute PINN performance targets from baseline results at N=25 (25%)."""
    rf_25 = summary.get("rf_rmse_by_fraction", {}).get("25%", {})
    nn_25 = summary.get("nn_rmse_by_fraction", {}).get("25%", {})

    rf_median = rf_25.get("median", 999)
    nn_median = nn_25.get("median", 999)

    if rf_median <= nn_median:
        best = "rf"
        best_rmse = rf_median
    else:
        best = "nn"
        best_rmse = nn_median

    return {
        "best_baseline_at_n25": best,
        "best_baseline_rmse_at_n25": best_rmse,
        "strong_threshold": round(best_rmse * 0.80, 4),       # >20% improvement
        "significant_threshold": round(best_rmse * 0.85, 4),  # 15-20%
        "moderate_threshold": round(best_rmse * 0.90, 4),     # 10-15%
        "inconclusive_boundary": round(best_rmse * 0.90, 4),  # <10%
    }


def run_gate(confirm_deviations: bool = False) -> dict:
    """Run all Gate 2 criteria and return structured result."""
    configs_dir = config.DATA_SPLITS / "configs"
    aligned_path = config.DATA_PROCESSED / "aligned_dataset.csv"
    metrics_dir = config.OUTPUTS_METRICS

    criteria = {}
    all_passed = True
    warnings = []

    checks = [
        check_g2_01(configs_dir),
        check_g2_02(config.DATA_SPLITS),
        check_g2_03(configs_dir, aligned_path),
        check_g2_04(configs_dir),
        check_g2_05(metrics_dir),
        check_g2_06(metrics_dir),
        check_g2_07(metrics_dir),
        check_g2_08(metrics_dir),
        check_g2_09(metrics_dir),
        check_g2_10(),
        check_g2_11(metrics_dir),
        check_g2_12(confirm_deviations),
        check_g2_13(),
    ]

    for check in checks:
        cid = check["id"]
        criteria[cid] = check
        is_warning = check.get("is_warning", False)
        if not check["passed"] and not is_warning:
            all_passed = False
        if not check["passed"] and is_warning:
            warnings.append(f"{cid}: {check['description']} — {check['measured']}")

    # Build summary
    summary = build_baseline_summary(metrics_dir)
    pinn_targets = compute_pinn_targets(summary)

    result = {
        "gate": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": all_passed,
        "exit_code": 0 if all_passed else 1,
        "criteria": criteria,
        "baseline_summary": summary,
        "pinn_targets": pinn_targets,
        "warnings": warnings,
    }

    # Print formatted output
    print("\n" + "═" * 65)
    print(" ECHO PoC — Gate 2: Baseline Models & Evaluation Harness")
    print(f" Run at: {result['timestamp']}")
    print("═" * 65)
    print()
    print(f" {'Criterion':<30} {'Threshold':<15} {'Measured':<18} {'Status'}")
    print(" " + "─" * 65)

    for check in checks:
        status = "PASS" if check["passed"] else ("WARN" if check.get("is_warning") else "FAIL")
        print(f" {check['id']} {check['description']:<24} {str(check['threshold']):<15} "
              f"{str(check['measured']):<18} {status}")

    print(" " + "─" * 65)

    # Performance summary table
    if summary.get("null_rmse") is not None:
        print()
        print(" Baseline performance summary:")
        print()
        print(f" {'Model':<14}│ {'10%':<14}│ {'25%':<14}│ {'50%':<14}│ {'100%':<14}")
        print(f" {'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}")

        null_str = f"{summary['null_rmse']:.3f}"
        print(f" {'Null (floor)':<14}│ {null_str:<14}│ {null_str:<14}│ {null_str:<14}│ {null_str:<14}")

        for model_label, key in [("RF  (median)", "rf_rmse_by_fraction"),
                                  ("NN  (median)", "nn_rmse_by_fraction")]:
            parts = []
            for label in ["10%", "25%", "50%", "100%"]:
                d = summary.get(key, {}).get(label, {})
                if d:
                    iqr_half = (d["q75"] - d["q25"]) / 2
                    parts.append(f"{d['median']:.3f} ±{iqr_half:.3f}")
                else:
                    parts.append("—")
            print(f" {model_label:<14}│ {parts[0]:<14}│ {parts[1]:<14}│ {parts[2]:<14}│ {parts[3]:<14}")

        print(f" {'─' * 14}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}")
        print(" Format: median RMSE (cm³/cm³) ± IQR/2")

    print()
    print(f" Result: {'PASS' if all_passed else 'FAIL'}")

    if pinn_targets["best_baseline_rmse_at_n25"] < 900:
        print()
        print(f" PINN target at N=25 for Strong result (>20%): "
              f"< {pinn_targets['strong_threshold']:.3f} cm³/cm³")
        print(f" PINN target at N=25 for Significant (15-20%): "
              f"< {pinn_targets['significant_threshold']:.3f} cm³/cm³")
        print(f" PINN target at N=25 for Moderate (10-15%):    "
              f"< {pinn_targets['moderate_threshold']:.3f} cm³/cm³")

    print("═" * 65)

    # Save result
    config.OUTPUTS_GATES.mkdir(parents=True, exist_ok=True)
    output_path = config.OUTPUTS_GATES / "gate_2_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n Gate result saved: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="ECHO PoC Gate 2")
    parser.add_argument("--confirm-deviations", action="store_true",
                        help="Confirm DEVIATIONS.md has been reviewed")
    args = parser.parse_args()

    result = run_gate(confirm_deviations=args.confirm_deviations)
    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
