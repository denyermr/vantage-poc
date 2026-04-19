"""
Terrain feature extraction via Google Earth Engine (SRTM + MERIT Hydro).

Computes static terrain features (slope, aspect, TWI) for the COSMOS-UK
footprint at Moor House. Single-row output — same values for all dates.

Usage:
    python poc/data/gee/extract_terrain.py [--dry-run]
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared import config

logger = logging.getLogger(__name__)


def submit_extraction(dry_run: bool = False) -> str | None:
    """
    Submit a GEE export task for terrain features.

    Computes slope, circular-mean aspect (sin/cos), and TWI from
    SRTM DEM and MERIT Hydro upstream area.

    Args:
        dry_run: If True, print metadata but do not submit.

    Returns:
        GEE task ID string, or None if dry_run.
    """
    import ee

    if not config.GEE_PROJECT:
        raise ValueError("GEE_PROJECT not set in config.py")

    ee.Initialize(project=config.GEE_PROJECT)
    logger.info("GEE initialised for terrain extraction")

    point = ee.Geometry.Point([config.SITE_LON, config.SITE_LAT])
    footprint = point.buffer(config.SITE_RADIUS_M)

    # ─── Slope and aspect from SRTM ─────────────────────────────
    dem = ee.Image(config.SRTM_COLLECTION)
    terrain = ee.Terrain.products(dem)

    # Mean slope over footprint
    slope_stats = terrain.select("slope").reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=footprint,
        scale=30,
        maxPixels=1e6,
    )

    # Circular mean of aspect:
    # Convert to radians, take mean of sin and cos, reconstruct
    aspect_rad = terrain.select("aspect").multiply(math.pi / 180)
    aspect_sin = aspect_rad.sin()
    aspect_cos = aspect_rad.cos()

    sin_stats = aspect_sin.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=footprint, scale=30, maxPixels=1e6,
    )
    cos_stats = aspect_cos.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=footprint, scale=30, maxPixels=1e6,
    )

    # ─── TWI from MERIT Hydro ────────────────────────────────────
    merit = ee.Image(config.MERIT_COLLECTION)
    upa = merit.select("upa")  # upstream drainage area in km²

    upa_stats = upa.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=footprint,
        scale=90,  # MERIT Hydro native ~90m
        maxPixels=1e6,
    )

    if dry_run:
        print(f"GEE authenticated. Terrain extraction configured for Moor House")
        print(f"SRTM scale: 30m, MERIT Hydro scale: 90m")
        return None

    # Compute TWI server-side
    slope_mean = ee.Number(slope_stats.get("slope"))
    slope_rad_ee = slope_mean.multiply(math.pi / 180)
    tan_slope = slope_rad_ee.tan().max(0.001)  # floor to avoid log(inf)
    upa_m2 = ee.Number(upa_stats.get("upa")).multiply(1e6)
    twi = upa_m2.divide(tan_slope).log()  # natural log

    terrain_feature = ee.Feature(None, {
        "slope_deg": slope_mean,
        "aspect_sin": sin_stats.get("aspect"),
        "aspect_cos": cos_stats.get("aspect"),
        "twi": twi,
    })

    fc = ee.FeatureCollection([terrain_feature])

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description="echo_terrain_moorh",
        folder=config.GEE_DRIVE_FOLDER,
        fileNamePrefix="terrain_static_raw",
        fileFormat="CSV",
    )
    task.start()

    print(f"Export task started: {task.id}")
    print(f"Monitor at: https://code.earthengine.google.com/tasks")
    print(f"Download to: {config.DATA_RAW_GEE / 'terrain_static_raw.csv'}")

    return task.id


def load_terrain(
    raw_path: Path,
    sar_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Load terrain features from raw GEE export and broadcast to SAR dates.

    The raw CSV is a single row of static terrain values. These are
    validated and then repeated for every SAR overpass date.

    Args:
        raw_path: Path to raw terrain CSV downloaded from Google Drive.
        sar_dates: SAR overpass dates to broadcast terrain values to.

    Returns:
        DataFrame with columns: date, slope_deg, aspect_sin, aspect_cos, twi.

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If terrain values are out of expected range.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw terrain CSV not found: {raw_path}")

    logger.info("Loading terrain features from %s", raw_path)
    df = pd.read_csv(raw_path)

    # Drop GEE artefact columns
    drop_cols = [c for c in df.columns if c in (".geo", "system:index")]
    df = df.drop(columns=drop_cols, errors="ignore")

    row = df.iloc[0]
    terrain = {
        "slope_deg": float(row["slope_deg"]),
        "aspect_sin": float(row["aspect_sin"]),
        "aspect_cos": float(row["aspect_cos"]),
        "twi": float(row["twi"]),
    }

    # Validate
    if not (0 <= terrain["slope_deg"] <= 45):
        raise ValueError(
            f"slope_deg={terrain['slope_deg']:.2f} outside expected range [0, 45]"
        )
    if not (-1 <= terrain["aspect_sin"] <= 1):
        raise ValueError(
            f"aspect_sin={terrain['aspect_sin']:.4f} outside [-1, 1]"
        )
    if not (-1 <= terrain["aspect_cos"] <= 1):
        raise ValueError(
            f"aspect_cos={terrain['aspect_cos']:.4f} outside [-1, 1]"
        )
    if terrain["twi"] <= 0:
        raise ValueError(f"twi={terrain['twi']:.2f} must be > 0")

    logger.info(
        "Terrain: slope=%.2f°, aspect_sin=%.3f, aspect_cos=%.3f, TWI=%.2f",
        terrain["slope_deg"], terrain["aspect_sin"],
        terrain["aspect_cos"], terrain["twi"],
    )

    # Broadcast to all SAR dates
    result = pd.DataFrame(
        [terrain] * len(sar_dates),
        index=range(len(sar_dates)),
    )
    result["date"] = sar_dates.values

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract terrain features via GEE"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--process", type=str, default=None)
    args = parser.parse_args()

    if args.process:
        s1_path = config.DATA_PROCESSED / "sentinel1_extractions.csv"
        if not s1_path.exists():
            raise FileNotFoundError(f"Process S1 data first: {s1_path}")
        s1 = pd.read_csv(s1_path, parse_dates=["date"])
        load_terrain(Path(args.process), pd.DatetimeIndex(s1["date"]))
    else:
        submit_extraction(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
