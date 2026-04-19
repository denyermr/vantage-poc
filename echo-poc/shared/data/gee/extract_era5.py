"""
ERA5-Land daily precipitation extraction via Google Earth Engine.

Extracts daily total precipitation at the Moor House site coordinates,
2021–2024 (with 8-day buffer at start for antecedent index).

Usage:
    python poc/data/gee/extract_era5.py [--dry-run]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared import config

logger = logging.getLogger(__name__)


def submit_extraction(dry_run: bool = False) -> str | None:
    """
    Submit a GEE export task for ERA5-Land daily precipitation.

    Extracts 8 days before STUDY_START to allow 7-day antecedent
    index computation for the first observations.

    Args:
        dry_run: If True, print metadata but do not submit.

    Returns:
        GEE task ID string, or None if dry_run.
    """
    import ee

    if not config.GEE_PROJECT:
        raise ValueError("GEE_PROJECT not set in config.py")

    ee.Initialize(project=config.GEE_PROJECT)
    logger.info("GEE initialised for ERA5-Land precipitation extraction")

    point = ee.Geometry.Point([config.SITE_LON, config.SITE_LAT])

    # 8-day buffer before study start for antecedent index
    buffer_start = (
        pd.Timestamp(config.STUDY_START) - pd.Timedelta(days=8)
    ).strftime("%Y-%m-%d")

    era5 = (
        ee.ImageCollection(config.ERA5_COLLECTION)
        .filterDate(buffer_start, config.STUDY_END)
        .select("total_precipitation_sum")
    )

    n_images = era5.size().getInfo()
    logger.info("ERA5-Land daily images: %d", n_images)

    if dry_run:
        print(f"GEE authenticated. ERA5-Land images: {n_images}")
        return None

    # Extract point value for each day
    def extract_precip(image):
        value = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=11132,  # ERA5-Land native ~9km, use ~0.1 degree
        )
        # Convert metres to mm
        precip_m = ee.Number(value.get("total_precipitation_sum"))
        precip_mm = precip_m.multiply(1000)

        return ee.Feature(None, {
            "system:time_start": image.get("system:time_start"),
            "precip_mm": precip_mm,
        })

    fc = era5.map(extract_precip)

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description="echo_era5_precip_moorh",
        folder=config.GEE_DRIVE_FOLDER,
        fileNamePrefix="era5_precip_raw",
        fileFormat="CSV",
    )
    task.start()

    print(f"Export task started: {task.id}")
    print(f"Monitor at: https://code.earthengine.google.com/tasks")
    print(f"Download to: {config.DATA_RAW_GEE / 'era5_precip_raw.csv'}")

    return task.id


def process_raw(
    raw_path: Path,
    sar_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Process raw GEE ERA5-Land precipitation CSV.

    Computes 7-day antecedent precipitation index and filters to SAR dates.

    Args:
        raw_path: Path to raw CSV downloaded from Google Drive.
        sar_dates: SAR overpass dates to filter to.

    Returns:
        DataFrame with columns: date, precip_mm, precip_7day_mm,
        indexed to SAR overpass dates.

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If validation fails.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw ERA5 CSV not found: {raw_path}")

    logger.info("Processing raw ERA5-Land precipitation: %s", raw_path)
    df = pd.read_csv(raw_path)

    # Drop GEE artefact columns
    drop_cols = [c for c in df.columns if c in (".geo", "system:index")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Parse date
    df["date"] = pd.to_datetime(df["system:time_start"], unit="ms").dt.normalize()
    df["date"] = df["date"].dt.tz_localize(None)
    df = df[["date", "precip_mm"]].sort_values("date").reset_index(drop=True)

    # Round
    df["precip_mm"] = df["precip_mm"].round(3)

    # Compute 7-day antecedent precipitation index
    # shift(1) excludes current day; rolling(7) sums prior 7 days
    df["precip_7day_mm"] = df["precip_mm"].shift(1).rolling(7).sum()

    # Validate before filtering
    _validate_era5(df)

    # Filter to SAR overpass dates
    df = df[df["date"].isin(sar_dates)].reset_index(drop=True)

    logger.info(
        "ERA5 precipitation: %d SAR dates matched, "
        "mean precip=%.1f mm/day, mean 7-day=%.1f mm",
        len(df), df["precip_mm"].mean(), df["precip_7day_mm"].mean(),
    )

    return df


def _validate_era5(df: pd.DataFrame) -> None:
    """Validate ERA5 precipitation data."""
    # Precip must be non-negative
    if (df["precip_mm"] < 0).any():
        raise ValueError("Negative precipitation values found")

    # Annual mean check (Moor House ~1800–2500 mm/year)
    annual_precip = df.groupby(df["date"].dt.year)["precip_mm"].sum()
    for year, total in annual_precip.items():
        if not (1000 <= total <= 3500):
            logger.warning(
                "Year %d annual precipitation %.0f mm "
                "(expected ~1800–2500 mm for Moor House)", year, total,
            )

    logger.info("ERA5 validation passed: %d daily records", len(df))


def main():
    parser = argparse.ArgumentParser(
        description="Extract ERA5-Land precipitation via GEE"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--process", type=str, default=None)
    args = parser.parse_args()

    if args.process:
        # Need SAR dates — load from processed S1 file
        s1_path = config.DATA_PROCESSED / "sentinel1_extractions.csv"
        if not s1_path.exists():
            raise FileNotFoundError(
                f"Process S1 data first: {s1_path} not found"
            )
        s1 = pd.read_csv(s1_path, parse_dates=["date"])
        process_raw(Path(args.process), pd.DatetimeIndex(s1["date"]))
    else:
        submit_extraction(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
