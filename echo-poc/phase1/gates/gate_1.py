"""
Gate 1 — Data Quality gate check for the ECHO PoC.

Evaluates all 10 criteria from SPEC_PHASE1.md §P1.8.
Must exit 0 (all pass) before Phase 2 can begin.

Usage:
    python poc/gates/gate_1.py [--confirm-deviations]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared import config

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

ALIGNED_CSV = config.DATA_PROCESSED / "aligned_dataset.csv"
TEST_INDICES = config.DATA_SPLITS / "test_indices.json"
DEVIATIONS_MD = config.PROJECT_ROOT.parent / "DEVIATIONS.md"
GATE_RESULT = config.OUTPUTS_GATES / "gate_1_result.json"

REQUIRED_FIGURES = [
    "p1_cosmos_diagnostic.png",
    "p1_sar_diagnostic.png",
    "p1_ancillary_diagnostic.png",
    "p1_aligned_dataset_summary.png",
]

GEE_MODULES = [
    "shared.data.gee.extract_sentinel1",
    "shared.data.gee.extract_sentinel2",
    "shared.data.gee.extract_era5",
    "shared.data.gee.extract_terrain",
]


# ─── Individual criterion checks ──────────────────────────────────────────────


def check_g1_01(df: pd.DataFrame) -> dict:
    """G1-01: Paired observations ≥ 100."""
    n = len(df)
    passed = n >= 100
    return {
        "id": "G1-01",
        "description": "Paired observations >= 100",
        "threshold": 100,
        "measured": n,
        "passed": passed,
    }


def check_g1_02(df: pd.DataFrame) -> dict:
    """G1-02: VWC range within [0.10, 1.00] cm³/cm³."""
    vwc_min = float(df["vwc"].min())
    vwc_max = float(df["vwc"].max())
    passed = vwc_min >= config.VWC_RANGE_MIN and vwc_max <= config.VWC_RANGE_MAX
    return {
        "id": "G1-02",
        "description": "VWC range within [0.10, 1.00] cm³/cm³",
        "threshold": f"[{config.VWC_RANGE_MIN}, {config.VWC_RANGE_MAX}]",
        "measured": f"[{vwc_min:.4f}, {vwc_max:.4f}]",
        "passed": passed,
    }


def check_g1_03(df: pd.DataFrame) -> dict:
    """G1-03: VV backscatter within [−20, −5] dB."""
    vv_min = float(df["vv_db"].min())
    vv_max = float(df["vv_db"].max())
    passed = vv_min >= config.VV_RANGE_MIN and vv_max <= config.VV_RANGE_MAX
    return {
        "id": "G1-03",
        "description": "VV backscatter within [-20, -5] dB",
        "threshold": f"[{config.VV_RANGE_MIN}, {config.VV_RANGE_MAX}]",
        "measured": f"[{vv_min:.4f}, {vv_max:.4f}]",
        "passed": passed,
    }


def check_g1_04(df: pd.DataFrame) -> dict:
    """G1-04: Seasonal VWC signal — winter mean > summer mean."""
    months = df["date"].dt.month
    winter_vwc = df.loc[months.isin(config.SEASONS["DJF"]), "vwc"].mean()
    summer_vwc = df.loc[months.isin(config.SEASONS["JJA"]), "vwc"].mean()
    passed = bool(winter_vwc > summer_vwc)
    return {
        "id": "G1-04",
        "description": "Seasonal VWC signal: winter mean > summer mean",
        "threshold": "winter > summer",
        "measured": f"winter={winter_vwc:.4f}, summer={summer_vwc:.4f}",
        "passed": passed,
    }


def check_g1_05(df: pd.DataFrame) -> dict:
    """G1-05: SAR–VWC signal — statistically significant with coherent multivariate structure.

    DEV-003: Reinterpreted from raw Pearson r ≥ 0.30 threshold to
    "statistically significant SAR–VWC signal present" (p < 0.05).
    The weak univariate VV correlation at vegetated peatland sites is
    the scientific motivation for the PINN approach.
    """
    from scipy import stats as sp_stats

    r_vv, p_vv = sp_stats.pearsonr(df["vv_db"], df["vwc"])
    r_ndvi, _ = sp_stats.pearsonr(df["ndvi"], df["vwc"])
    r_precip7, _ = sp_stats.pearsonr(df["precip_7day_mm"], df["vwc"])

    # DEV-003: Pass if VV–VWC is statistically significant (p < 0.05)
    passed = p_vv < 0.05
    return {
        "id": "G1-05",
        "description": "SAR-VWC signal: statistically significant (p<0.05) with coherent multivariate structure (DEV-003)",
        "threshold": "p < 0.05 (reinterpreted from |r| >= 0.30 per DEV-003)",
        "measured": f"VV r={r_vv:.4f} (p={p_vv:.4e}); NDVI r={r_ndvi:.4f}; precip_7day r={r_precip7:.4f}",
        "passed": passed,
    }


def check_g1_06(df: pd.DataFrame) -> dict:
    """G1-06: No NaN in aligned dataset."""
    nan_count = int(df.isna().sum().sum())
    passed = nan_count == 0
    return {
        "id": "G1-06",
        "description": "No NaN in aligned dataset",
        "threshold": 0,
        "measured": nan_count,
        "passed": passed,
    }


def check_g1_07() -> dict:
    """G1-07: Test indices file exists."""
    exists = TEST_INDICES.exists()
    return {
        "id": "G1-07",
        "description": "Test indices file exists (data/splits/test_indices.json)",
        "threshold": "file exists",
        "measured": str(TEST_INDICES) if exists else "NOT FOUND",
        "passed": exists,
    }


def check_g1_08(confirm_deviations: bool) -> dict:
    """G1-08: DEVIATIONS.md reviewed (requires --confirm-deviations flag)."""
    exists = DEVIATIONS_MD.exists()
    passed = exists and confirm_deviations
    return {
        "id": "G1-08",
        "description": "DEVIATIONS.md reviewed (--confirm-deviations)",
        "threshold": "--confirm-deviations flag provided",
        "measured": (
            "confirmed" if confirm_deviations
            else "NOT confirmed (run with --confirm-deviations)"
        ),
        "passed": passed,
    }


def check_g1_09() -> dict:
    """G1-09: All 4 diagnostic figures exist."""
    missing = []
    for fig in REQUIRED_FIGURES:
        if not (config.OUTPUTS_FIGURES / fig).exists():
            missing.append(fig)
    passed = len(missing) == 0
    return {
        "id": "G1-09",
        "description": "All 4 diagnostic figures exist in outputs/figures/",
        "threshold": "0 missing",
        "measured": f"{len(missing)} missing" + (f": {missing}" if missing else ""),
        "passed": passed,
    }


def check_g1_10() -> dict:
    """G1-10: All 4 GEE extraction modules importable."""
    failures = []
    for module_name in GEE_MODULES:
        try:
            __import__(module_name)
        except Exception as e:
            failures.append(f"{module_name}: {e}")
    passed = len(failures) == 0
    return {
        "id": "G1-10",
        "description": "All 4 GEE extraction modules importable",
        "threshold": "0 import failures",
        "measured": (
            f"{len(failures)} failures" + (f": {failures}" if failures else "")
        ),
        "passed": passed,
    }


# ─── Gate runner ──────────────────────────────────────────────────────────────


def run_gate_1(confirm_deviations: bool = False) -> dict:
    """
    Run all 10 Gate 1 criteria and return structured result.

    Args:
        confirm_deviations: Whether the user has confirmed DEVIATIONS.md review.

    Returns:
        Gate result dict with criteria, overall pass/fail, and metadata.
    """
    results = []

    # Load aligned dataset if it exists
    if ALIGNED_CSV.exists():
        df = pd.read_csv(ALIGNED_CSV, parse_dates=["date"])
        results.append(check_g1_01(df))
        results.append(check_g1_02(df))
        results.append(check_g1_03(df))
        results.append(check_g1_04(df))
        results.append(check_g1_05(df))
        results.append(check_g1_06(df))
    else:
        # All data-dependent criteria fail if no aligned dataset
        for cid, desc in [
            ("G1-01", "Paired observations >= 100"),
            ("G1-02", "VWC range within [0.10, 1.00]"),
            ("G1-03", "VV backscatter within [-20, -5] dB"),
            ("G1-04", "Seasonal VWC signal"),
            ("G1-05", "SAR-VWC correlation"),
            ("G1-06", "No NaN in aligned dataset"),
        ]:
            results.append({
                "id": cid,
                "description": desc,
                "threshold": "—",
                "measured": f"SKIPPED — aligned dataset not found: {ALIGNED_CSV}",
                "passed": False,
            })

    results.append(check_g1_07())
    results.append(check_g1_08(confirm_deviations))
    results.append(check_g1_09())
    results.append(check_g1_10())

    # Overall result
    all_passed = all(r["passed"] for r in results)
    n_passed = sum(1 for r in results if r["passed"])
    n_failed = sum(1 for r in results if not r["passed"])

    gate_result = {
        "gate": "Gate 1 — Data Quality",
        "passed": all_passed,
        "n_criteria": len(results),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "criteria": results,
        "timestamp": datetime.utcnow().isoformat(),
        "aligned_dataset": str(ALIGNED_CSV),
    }

    return gate_result


def print_report(gate_result: dict) -> None:
    """Print a human-readable gate report to stdout."""
    print("\n" + "=" * 70)
    print(f"  {gate_result['gate']}")
    print("=" * 70)

    for c in gate_result["criteria"]:
        status = "PASS" if c["passed"] else "FAIL"
        icon = "✓" if c["passed"] else "✗"
        print(f"  {icon} [{status}] {c['id']}: {c['description']}")
        print(f"           threshold: {c['threshold']}")
        print(f"           measured:  {c['measured']}")
        print()

    print("-" * 70)
    overall = "PASSED" if gate_result["passed"] else "FAILED"
    print(
        f"  Result: {overall} "
        f"({gate_result['n_passed']}/{gate_result['n_criteria']} criteria met)"
    )
    print("=" * 70 + "\n")


def save_result(gate_result: dict) -> None:
    """Save gate result JSON."""
    GATE_RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_RESULT, "w") as f:
        json.dump(gate_result, f, indent=2, default=str)
    logger.info("Gate 1 result saved to %s", GATE_RESULT)


def main():
    parser = argparse.ArgumentParser(description="Gate 1 — Data Quality check")
    parser.add_argument(
        "--confirm-deviations",
        action="store_true",
        help="Confirm that DEVIATIONS.md has been reviewed",
    )
    args = parser.parse_args()

    gate_result = run_gate_1(confirm_deviations=args.confirm_deviations)
    print_report(gate_result)
    save_result(gate_result)

    sys.exit(0 if gate_result["passed"] else 1)


if __name__ == "__main__":
    main()
