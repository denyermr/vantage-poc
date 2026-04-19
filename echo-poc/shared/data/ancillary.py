"""
Ancillary feature assembly for the ECHO PoC.

Assembles NDVI (interpolated from S2 monthly composites), precipitation
(ERA5-Land daily + 7-day antecedent), and terrain (SRTM static) features
aligned to SAR overpass dates.

Output: data/processed/ancillary_features.csv
"""

import logging
from pathlib import Path

import pandas as pd

from shared import config
from shared.data.gee.extract_sentinel2 import interpolate_ndvi, process_raw as process_s2
from shared.data.gee.extract_era5 import process_raw as process_era5
from shared.data.gee.extract_terrain import load_terrain

logger = logging.getLogger(__name__)


def build_ancillary(
    s1_dates: pd.DatetimeIndex,
    ndvi_monthly_path: Path,
    era5_raw_path: Path,
    terrain_raw_path: Path,
) -> pd.DataFrame:
    """
    Assemble ancillary_features.csv aligned to SAR overpass dates.

    All rows correspond 1:1 with Sentinel-1 overpass dates.

    Args:
        s1_dates: DatetimeIndex of SAR overpass dates.
        ndvi_monthly_path: Path to raw S2 NDVI monthly CSV.
        era5_raw_path: Path to raw ERA5-Land precipitation CSV.
        terrain_raw_path: Path to raw terrain CSV.

    Returns:
        DataFrame with columns: date, ndvi, precip_mm, precip_7day_mm,
        slope_deg, aspect_sin, aspect_cos, twi.

    Raises:
        FileNotFoundError: If any source file does not exist.
        ValueError: If validation fails.
    """
    logger.info(
        "Building ancillary features for %d SAR overpass dates", len(s1_dates)
    )

    # ─── NDVI (S2 monthly → interpolated to SAR dates) ───────────
    ndvi_monthly = process_s2(ndvi_monthly_path)
    ndvi_series = interpolate_ndvi(ndvi_monthly, s1_dates)

    # ─── Precipitation (ERA5-Land daily → 7-day antecedent) ──────
    precip_df = process_era5(era5_raw_path, s1_dates)

    # ─── Terrain (SRTM static → broadcast to all dates) ─────────
    terrain_df = load_terrain(terrain_raw_path, s1_dates)

    # ─── Combine ─────────────────────────────────────────────────
    result = pd.DataFrame({"date": s1_dates})
    result["ndvi"] = ndvi_series.values

    # Merge precipitation by date
    result = result.merge(
        precip_df[["date", "precip_mm", "precip_7day_mm"]],
        on="date",
        how="left",
    )

    # Merge terrain by date
    result = result.merge(
        terrain_df[["date", "slope_deg", "aspect_sin", "aspect_cos", "twi"]],
        on="date",
        how="left",
    )

    logger.info(
        "Ancillary features assembled: %d rows, %d NaN cells total",
        len(result), result.isna().sum().sum(),
    )

    return result


def save_ancillary(df: pd.DataFrame, output_path: Path) -> None:
    """Save ancillary features to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved ancillary features to %s (%d rows)", output_path, len(df))
