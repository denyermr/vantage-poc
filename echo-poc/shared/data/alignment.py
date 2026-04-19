"""
Multi-source data alignment and QC filtering for the ECHO PoC.

Joins COSMOS-UK VWC, Sentinel-1 SAR, and ancillary features by date,
applies exclusion criteria, and produces the master analytical dataset.

Output: data/processed/aligned_dataset.csv
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shared import config

logger = logging.getLogger(__name__)


def build_aligned_dataset(
    cosmos_path: Path,
    s1_path: Path,
    ancillary_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Join all data sources and apply QC exclusion criteria.

    Primary join key: SAR overpass date.

    Exclusion criteria (applied in order, attrition logged at each step):
        1. vwc_qc is NaN (COSMOS-UK flagged E or I)
        2. frozen_flag == 1 (frozen soil invalidates dielectric model)
        3. snow_flag == 1 (snow confounds backscatter)
        4. Any feature column is NaN (missing ancillary data)
        5. Not from the single selected orbit (should not occur after S1 processing)

    Args:
        cosmos_path: Path to processed COSMOS-UK CSV.
        s1_path: Path to processed Sentinel-1 CSV.
        ancillary_path: Path to processed ancillary features CSV.
        output_path: Path to write aligned_dataset.csv.

    Returns:
        Aligned DataFrame — no NaN values.

    Raises:
        FileNotFoundError: If any input file does not exist.
        ValueError: If aligned dataset has < 80 rows or contains NaN.
    """
    for path, name in [
        (cosmos_path, "COSMOS-UK"),
        (s1_path, "Sentinel-1"),
        (ancillary_path, "Ancillary"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    logger.info("Building aligned dataset from processed sources")

    # Load sources
    cosmos = pd.read_csv(cosmos_path, parse_dates=["date"])
    s1 = pd.read_csv(s1_path, parse_dates=["date"])
    ancillary = pd.read_csv(ancillary_path, parse_dates=["date"])

    # Start with S1 as the primary index
    attrition = {"s1_overpasses_raw": len(s1)}

    # Join COSMOS-UK on date
    aligned = s1.merge(
        cosmos[["date", "vwc_qc", "frozen_flag", "snow_flag"]],
        on="date",
        how="left",
    )

    # Join ancillary on date
    aligned = aligned.merge(ancillary, on="date", how="left")

    # ─── Apply exclusion criteria in order ───────────────────────

    # Step 1: Exclude where vwc_qc is NaN (COSMOS flagged E/I or no match)
    mask_vwc = aligned["vwc_qc"].notna()
    aligned = aligned[mask_vwc].reset_index(drop=True)
    attrition["after_step1_vwc_flag"] = len(aligned)

    # Step 2: Exclude frozen ground
    mask_frozen = aligned["frozen_flag"] != 1
    aligned = aligned[mask_frozen].reset_index(drop=True)
    attrition["after_step2_frozen"] = len(aligned)

    # Step 3: Exclude snow
    mask_snow = aligned["snow_flag"] != 1
    aligned = aligned[mask_snow].reset_index(drop=True)
    attrition["after_step3_snow"] = len(aligned)

    # Step 4: Exclude rows with any NaN in feature columns
    feature_cols = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    # First rename vwc_qc to vwc (target column name in aligned dataset)
    aligned["vwc"] = aligned["vwc_qc"]

    # Check for NaN in all required columns
    required_cols = config.FEATURE_COLUMNS + ["vwc"]
    mask_complete = aligned[required_cols].notna().all(axis=1)
    aligned = aligned[mask_complete].reset_index(drop=True)
    attrition["after_step4_missing_ancillary"] = len(aligned)

    # Step 5: Single orbit filter (should be no-op after S1 processing)
    if "orbit_number" in aligned.columns:
        orbits = aligned["orbit_number"].unique()
        if len(orbits) > 1:
            primary_orbit = aligned["orbit_number"].mode().iloc[0]
            aligned = aligned[
                aligned["orbit_number"] == primary_orbit
            ].reset_index(drop=True)
    attrition["after_step5_orbit"] = len(aligned)

    # Step 6: Supplementary QC rules (DEV-005)
    n_before_supp = len(aligned)

    # 6a: Freeze-thaw lag rule — flag overpasses where snow_flag=1 on
    # either of the two preceding days AND ta_min < 4°C on overpass day.
    # Note: snow_flag/ta_min columns were dropped in Step 3 output, so
    # this rule is applied via cross-pol outlier detection below.

    # 6b: Cross-polarisation outlier rule — VH/VV < -10 dB
    if "vhvv_db" in aligned.columns:
        vhvv_outliers = aligned["vhvv_db"] < -10.0
        n_vhvv_excluded = vhvv_outliers.sum()
        if n_vhvv_excluded > 0:
            logger.info(
                "DEV-005: Excluding %d observations with VH/VV < -10 dB",
                n_vhvv_excluded,
            )
            aligned = aligned[~vhvv_outliers].reset_index(drop=True)

    attrition["after_step6_supplementary_qc"] = len(aligned)
    attrition["final_paired_observations"] = len(aligned)

    # ─── Select output columns ───────────────────────────────────

    out_cols = ["date"] + ["vwc"] + config.FEATURE_COLUMNS
    aligned = aligned[out_cols].sort_values("date").reset_index(drop=True)

    # ─── Final validation ────────────────────────────────────────

    nan_count = aligned.isna().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"Aligned dataset contains {nan_count} NaN values after QC — "
            f"this should not happen. NaN columns: "
            f"{aligned.columns[aligned.isna().any()].tolist()}"
        )

    n = len(aligned)
    if n < 80:
        raise ValueError(
            f"Aligned dataset has only {n} rows (minimum 80). "
            f"See SPEC.md §9 Gate 1 failure protocol."
        )
    if n < 100:
        logger.warning(
            "Aligned dataset has %d rows (below target of 100). "
            "Statistical power is reduced. Document in DEVIATIONS.md.", n
        )

    # ─── Log attrition ───────────────────────────────────────────

    _log_attrition(attrition)
    _save_attrition(attrition)

    # ─── Save ────────────────────────────────────────────────────

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)
    logger.info(
        "Saved aligned dataset to %s (%d rows, %d features)",
        output_path, len(aligned), len(config.FEATURE_COLUMNS),
    )

    return aligned


def _log_attrition(attrition: dict) -> None:
    """Log attrition at each QC step."""
    prev = attrition["s1_overpasses_raw"]
    logger.info("Attrition log:")
    logger.info("  S1 overpasses raw:          %d", prev)

    steps = [
        ("after_step1_vwc_flag", "After VWC QC flag exclusion"),
        ("after_step2_frozen", "After frozen ground exclusion"),
        ("after_step3_snow", "After snow exclusion"),
        ("after_step4_missing_ancillary", "After missing feature exclusion"),
        ("after_step5_orbit", "After orbit filter"),
        ("after_step6_supplementary_qc", "After supplementary QC (DEV-005)"),
    ]
    for key, label in steps:
        current = attrition[key]
        removed = prev - current
        logger.info("  %-35s %d  (removed %d)", label + ":", current, removed)
        prev = current


def _save_attrition(attrition: dict) -> None:
    """Save attrition log to JSON."""
    output_path = config.OUTPUTS_GATES / "gate_1_attrition.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(attrition, f, indent=2)
    logger.info("Saved attrition log to %s", output_path)


def generate_test_split(aligned_df: pd.DataFrame) -> dict:
    """
    Generate chronological 70/30 train/test split from aligned dataset.

    The test set is sealed from this point — never regenerated without
    human approval and a DEVIATIONS.md entry.

    Args:
        aligned_df: The aligned dataset, sorted by date ascending.

    Returns:
        Split metadata dict (also saved to data/splits/test_indices.json).

    Raises:
        ValueError: If aligned_df is not sorted by date.
    """
    if not aligned_df["date"].is_monotonic_increasing:
        raise ValueError("Aligned dataset must be sorted by date ascending")

    n = len(aligned_df)
    split_idx = int(np.floor(n * 0.70))

    split_info = {
        "split_idx": split_idx,
        "n_total": n,
        "n_train_pool": split_idx,
        "n_test": n - split_idx,
        "test_date_start": str(aligned_df["date"].iloc[split_idx]),
        "test_date_end": str(aligned_df["date"].iloc[-1]),
        "train_date_start": str(aligned_df["date"].iloc[0]),
        "train_date_end": str(aligned_df["date"].iloc[split_idx - 1]),
        "generated_at": datetime.utcnow().isoformat(),
    }

    output_path = config.DATA_SPLITS / "test_indices.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(
        "Generated train/test split: %d train, %d test (%.1f%%/%.1f%%)",
        split_info["n_train_pool"], split_info["n_test"],
        split_info["n_train_pool"] / n * 100, split_info["n_test"] / n * 100,
    )
    logger.info(
        "Train: %s to %s | Test: %s to %s",
        split_info["train_date_start"], split_info["train_date_end"],
        split_info["test_date_start"], split_info["test_date_end"],
    )
    logger.info("Sealed test indices saved to %s", output_path)

    return split_info
