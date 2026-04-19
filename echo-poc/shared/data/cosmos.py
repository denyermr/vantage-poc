"""
COSMOS-UK data loading and QC for Moor House station (MOORH).

Loads the daily product CSV, converts units, applies QC flags,
and writes the processed output.

Deviation DEV-001: Daily product used instead of hourly (see DEVIATIONS.md).
"""

import logging
from pathlib import Path

import pandas as pd

from shared import config

logger = logging.getLogger(__name__)


def load_cosmos(path: Path) -> pd.DataFrame:
    """
    Load and QC the COSMOS-UK daily product CSV.

    The COSMOS-UK daily CSV has a multi-row metadata header:
        Row 0-4: metadata
        Row 5:   column names (parameter-id row)
        Row 6:   parameter-name (skip)
        Row 7:   parameter-units (skip)
        Row 8+:  data

    Args:
        path: Path to raw COSMOS-UK daily CSV.

    Returns:
        DataFrame with columns: date, vwc_raw, vwc_qc, cosmos_vwc_flag,
        ta_min, snow, frozen_flag, snow_flag.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing, VWC out of range,
                    dates not monotonic, or seasonal check fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"COSMOS-UK CSV not found: {path}")

    logger.info("Loading COSMOS-UK daily product from %s", path)

    # Parse: skip 5 metadata rows, use row 5 as header,
    # then drop rows 0-1 (parameter-name and parameter-units)
    df = pd.read_csv(path, skiprows=5, header=0)
    df = df.iloc[2:].reset_index(drop=True)

    # Rename first column to 'date'
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})

    # Check required columns exist in raw data
    required_raw = ["date", "cosmos_vwc", "cosmos_vwc_flag", "ta_min", "snow"]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(
            f"COSMOS-UK CSV missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse date — strip timezone for consistency
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Convert numeric columns
    df["cosmos_vwc"] = pd.to_numeric(df["cosmos_vwc"], errors="coerce")
    df["ta_min"] = pd.to_numeric(df["ta_min"], errors="coerce")
    df["snow"] = pd.to_numeric(df["snow"], errors="coerce")

    # Convert VWC from % to cm³/cm³
    df["vwc_raw"] = df["cosmos_vwc"] / 100.0

    # Apply QC flags — NaN where flag is E (gap-filled) or I (interpolated)
    qc_pass = ~df["cosmos_vwc_flag"].isin(config.COSMOS_EXCLUDE_FLAGS)
    df["vwc_qc"] = df["vwc_raw"].where(qc_pass)

    # Frozen and snow flags
    df["frozen_flag"] = (df["ta_min"] < 0).astype(int)
    df["snow_flag"] = (df["snow"] > 0).astype(float).fillna(0).astype(int)

    # Select output columns
    out_cols = [
        "date", "vwc_raw", "vwc_qc", "cosmos_vwc_flag",
        "ta_min", "snow", "frozen_flag", "snow_flag",
    ]
    df = df[out_cols].copy()

    # ─── Validation ──────────────────────────────────────────────────────

    _validate_cosmos(df)

    n_excluded = qc_pass.value_counts().get(False, 0)
    logger.info(
        "COSMOS-UK loaded: %d days, %d excluded by QC flags (E/I), "
        "%d valid VWC observations",
        len(df), n_excluded, df["vwc_qc"].notna().sum(),
    )

    return df


def _validate_cosmos(df: pd.DataFrame) -> None:
    """
    Validate the processed COSMOS-UK DataFrame.

    Raises ValueError if any validation criterion fails.
    """
    # 1. VWC range check (non-null values)
    vwc_valid = df["vwc_qc"].dropna()
    if len(vwc_valid) == 0:
        raise ValueError("No valid VWC observations after QC — all flagged E or I")

    out_of_range = vwc_valid[
        ~vwc_valid.between(config.VWC_RANGE_MIN, config.VWC_RANGE_MAX)
    ]
    if len(out_of_range) > 0:
        raise ValueError(
            f"vwc_qc contains {len(out_of_range)} values outside "
            f"[{config.VWC_RANGE_MIN}, {config.VWC_RANGE_MAX}] cm³/cm³. "
            f"Range found: [{vwc_valid.min():.4f}, {vwc_valid.max():.4f}]. "
            f"Check units (should be cm³/cm³, not %)"
        )

    # 2. Date monotonicity — no duplicates
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Date index is not monotonically increasing")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in COSMOS-UK data")

    # 3. Date range check
    start = pd.Timestamp(config.STUDY_START)
    end = pd.Timestamp(config.STUDY_END)
    if df["date"].min() > start:
        raise ValueError(
            f"Date range starts at {df['date'].min()}, "
            f"expected to include {config.STUDY_START}"
        )
    if df["date"].max() < end:
        raise ValueError(
            f"Date range ends at {df['date'].max()}, "
            f"expected to include {config.STUDY_END}"
        )

    # 4. Seasonal check: winter VWC > summer VWC (wetter winter at Moor House)
    months = df["date"].dt.month
    winter_mask = months.isin(config.SEASONS["DJF"])
    summer_mask = months.isin(config.SEASONS["JJA"])

    winter_mean = df.loc[winter_mask, "vwc_qc"].mean()
    summer_mean = df.loc[summer_mask, "vwc_qc"].mean()

    if pd.isna(winter_mean) or pd.isna(summer_mean):
        raise ValueError(
            "Cannot compute seasonal means — insufficient valid VWC "
            "data in winter or summer"
        )

    if winter_mean <= summer_mean:
        raise ValueError(
            f"Seasonal check failed: winter mean VWC ({winter_mean:.4f}) "
            f"should be > summer mean ({summer_mean:.4f}) at Moor House. "
            f"This peatland site is wetter in winter."
        )

    logger.info(
        "Seasonal check passed: winter mean=%.4f > summer mean=%.4f",
        winter_mean, summer_mean,
    )


def save_cosmos(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save processed COSMOS-UK data to CSV.

    Args:
        df: Processed COSMOS-UK DataFrame from load_cosmos().
        output_path: Path to write the CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed COSMOS-UK data to %s (%d rows)", output_path, len(df))
