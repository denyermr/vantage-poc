"""
Sentinel-2 NDVI extraction via Google Earth Engine.

Produces monthly maximum-NDVI composites within the COSMOS-UK footprint
using cloud-masked pixels (SCL filter), 2021–2024.

Usage:
    python poc/data/gee/extract_sentinel2.py [--dry-run]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared import config

logger = logging.getLogger(__name__)


def submit_extraction(dry_run: bool = False) -> str | None:
    """
    Submit a GEE export task for Sentinel-2 monthly NDVI composites.

    Args:
        dry_run: If True, print metadata but do not submit.

    Returns:
        GEE task ID string, or None if dry_run.
    """
    import ee

    if not config.GEE_PROJECT:
        raise ValueError("GEE_PROJECT not set in config.py")

    ee.Initialize(project=config.GEE_PROJECT)
    logger.info("GEE initialised for Sentinel-2 NDVI extraction")

    point = ee.Geometry.Point([config.SITE_LON, config.SITE_LAT])
    footprint = point.buffer(config.SITE_RADIUS_M)

    # SCL classes to keep (vegetation, bare soil, water, unclassified)
    GOOD_SCL = [4, 5, 6, 7]

    def mask_clouds(image):
        scl = image.select("SCL")
        mask = scl.remap(GOOD_SCL, [1] * len(GOOD_SCL), defaultValue=0).eq(1)
        return image.updateMask(mask)

    def compute_ndvi(image):
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return image.addBands(ndvi)

    # Load, filter, mask, compute NDVI
    s2 = (
        ee.ImageCollection(config.S2_COLLECTION)
        .filterDate(config.STUDY_START, config.STUDY_END)
        .filterBounds(footprint)
        .map(mask_clouds)
        .map(compute_ndvi)
        .select("NDVI")
    )

    n_images = s2.size().getInfo()
    logger.info("S2 cloud-masked images: %d", n_images)

    if dry_run:
        print(f"GEE authenticated. S2 images over Moor House: {n_images}")
        return None

    # Generate year-month list
    start = pd.Timestamp(config.STUDY_START)
    end = pd.Timestamp(config.STUDY_END)
    months = pd.date_range(start, end, freq="MS")

    features = []
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
        ym = month_start.strftime("%Y-%m")

        # Filter to this month
        monthly = s2.filterDate(
            month_start.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d"),
        )

        # Max-NDVI composite
        composite = monthly.qualityMosaic("NDVI")

        # Spatial mean NDVI
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=footprint,
            scale=10,
            maxPixels=1e6,
        )

        feature = ee.Feature(None, {
            "year_month": ym,
            "ndvi_mean": stats.get("NDVI_mean"),
            "n_clear_pixels": stats.get("NDVI_count"),
            "composite_date": month_start.strftime("%Y-%m-15"),
        })
        features.append(feature)

    fc = ee.FeatureCollection(features)

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description="echo_sentinel2_ndvi_moorh",
        folder=config.GEE_DRIVE_FOLDER,
        fileNamePrefix="sentinel2_ndvi_raw",
        fileFormat="CSV",
    )
    task.start()

    print(f"Export task started: {task.id}")
    print(f"Monitor at: https://code.earthengine.google.com/tasks")
    print(f"Download to: {config.DATA_RAW_GEE / 'sentinel2_ndvi_raw.csv'}")

    return task.id


def process_raw(raw_path: Path) -> pd.DataFrame:
    """
    Process raw GEE Sentinel-2 NDVI CSV.

    Args:
        raw_path: Path to raw CSV downloaded from Google Drive.

    Returns:
        Monthly NDVI DataFrame with columns: composite_date, ndvi_mean,
        n_clear_pixels, year_month.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw S2 NDVI CSV not found: {raw_path}")

    logger.info("Processing raw S2 NDVI CSV: %s", raw_path)
    df = pd.read_csv(raw_path)

    # Drop GEE artefact columns
    drop_cols = [c for c in df.columns if c in (".geo", "system:index")]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["composite_date"] = pd.to_datetime(df["composite_date"])

    # Validate
    _validate_s2(df)

    return df


def interpolate_ndvi(
    ndvi_monthly: pd.DataFrame,
    sar_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Linearly interpolate monthly NDVI composites to SAR overpass dates.

    Args:
        ndvi_monthly: Monthly NDVI DataFrame with composite_date and ndvi_mean.
        sar_dates: DatetimeIndex of SAR overpass dates to interpolate to.

    Returns:
        pd.Series of interpolated NDVI values, indexed by sar_dates.
        NaN where SAR date falls outside the range of available composites.
    """
    ts = ndvi_monthly.set_index("composite_date")["ndvi_mean"].sort_index()
    ts.index = pd.DatetimeIndex(ts.index)

    # Reindex to daily, interpolate
    daily_index = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    ts_daily = ts.reindex(daily_index).interpolate(
        method="time", limit_direction="both"
    )

    # Extract at SAR dates
    result = ts_daily.reindex(sar_dates)

    n_nan = result.isna().sum()
    if n_nan > 0:
        nan_frac = n_nan / len(result)
        msg = f"NDVI interpolation: {n_nan}/{len(result)} SAR dates have NaN ({nan_frac:.1%})"
        if nan_frac > 0.10:
            logger.warning(msg + " — exceeds 10% threshold")
        else:
            logger.info(msg)

    return result


def _validate_s2(df: pd.DataFrame) -> None:
    """Validate processed S2 NDVI DataFrame."""
    # At least one composite per quarter across all 4 years
    df_check = df.dropna(subset=["ndvi_mean"])
    for year in range(2021, 2025):
        for q_months in [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]:
            mask = (
                (df_check["composite_date"].dt.year == year)
                & (df_check["composite_date"].dt.month.isin(q_months))
            )
            if mask.sum() == 0:
                logger.warning(
                    "No valid NDVI composite in %d Q%d",
                    year, (q_months[0] - 1) // 3 + 1,
                )

    # NDVI range
    valid_ndvi = df["ndvi_mean"].dropna()
    out_of_range = valid_ndvi[~valid_ndvi.between(-0.2, 1.0)]
    if len(out_of_range) > 0:
        raise ValueError(
            f"ndvi_mean has {len(out_of_range)} values outside [-0.2, 1.0]"
        )

    logger.info("S2 NDVI validation passed: %d monthly composites", len(df))


def main():
    parser = argparse.ArgumentParser(
        description="Extract Sentinel-2 NDVI via GEE"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--process", type=str, default=None)
    args = parser.parse_args()

    if args.process:
        process_raw(Path(args.process))
    else:
        submit_extraction(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
