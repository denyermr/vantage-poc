"""
Sentinel-1 SAR backscatter extraction via Google Earth Engine.

Extracts spatial-mean VV/VH backscatter within the COSMOS-UK footprint
for all descending IW overpasses, single relative orbit, 2021–2024.

Usage:
    python poc/data/gee/extract_sentinel1.py [--orbit ORBIT_NUMBER] [--dry-run]

The script submits a GEE export task to Google Drive. It does NOT wait for
completion. Download the CSV from Drive manually, then call process_raw()
to produce the processed output.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as script or import as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared import config

logger = logging.getLogger(__name__)


def _check_gee_project() -> None:
    """Raise ValueError if GEE_PROJECT is not configured."""
    if not config.GEE_PROJECT:
        raise ValueError(
            "GEE_PROJECT not set in config.py. "
            "Set config.GEE_PROJECT to your GEE Cloud project ID "
            "before running GEE extraction scripts."
        )


def submit_extraction(orbit_number: int | None = None, dry_run: bool = False) -> str | None:
    """
    Submit a GEE export task for Sentinel-1 SAR backscatter.

    Args:
        orbit_number: Relative orbit number to filter. If None, auto-selects
                      the most frequent descending orbit over the site.
        dry_run: If True, print metadata but do not submit the export task.

    Returns:
        GEE task ID string, or None if dry_run.

    Raises:
        ValueError: If GEE_PROJECT not set or no images found.
    """
    import ee

    _check_gee_project()
    ee.Initialize(project=config.GEE_PROJECT)
    logger.info("GEE initialised with project: %s", config.GEE_PROJECT)

    # Define footprint
    point = ee.Geometry.Point([config.SITE_LON, config.SITE_LAT])
    footprint = point.buffer(config.SITE_RADIUS_M)

    # Load and filter S1 collection
    s1 = (
        ee.ImageCollection(config.S1_COLLECTION)
        .filterDate(config.STUDY_START, config.STUDY_END)
        .filter(ee.Filter.eq("instrumentMode", config.S1_MODE))
        .filter(ee.Filter.eq("orbitProperties_pass", config.S1_PASS))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filterBounds(footprint)
    )

    n_images = s1.size().getInfo()
    logger.info("S1 descending IW images over Moor House: %d", n_images)
    if n_images == 0:
        raise ValueError("No Sentinel-1 images found matching filters")

    # Auto-select orbit if not specified
    if orbit_number is None:
        # Count overpasses per relative orbit
        orbit_counts = (
            s1.aggregate_histogram("relativeOrbitNumber_start").getInfo()
        )
        logger.info("Orbit counts: %s", orbit_counts)
        orbit_number = int(float(max(orbit_counts, key=orbit_counts.get)))
        logger.info("Auto-selected most frequent orbit: %d", orbit_number)

    # Filter to selected orbit
    s1 = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit_number))
    n_orbit = s1.size().getInfo()
    logger.info("Images from orbit %d: %d", orbit_number, n_orbit)

    if dry_run:
        print(f"GEE authenticated. Moor House S1 scenes available: ~{n_orbit}")
        print(f"Selected orbit: {orbit_number}")
        return None

    # Map over collection to extract stats per image
    def extract_stats(image):
        # Compute spatial means within footprint
        stats = image.select(["VV", "VH", "angle"]).reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=footprint,
            scale=10,
            maxPixels=1e6,
        )

        vv_db = stats.get("VV_mean")
        vh_db = stats.get("VH_mean")
        inc_angle = stats.get("angle_mean")
        n_pixels = stats.get("VV_count")

        return ee.Feature(None, {
            "system:time_start": image.get("system:time_start"),
            "vv_db": vv_db,
            "vh_db": vh_db,
            "incidence_angle_mean": inc_angle,
            "n_pixels": n_pixels,
            "orbit_number": image.get("relativeOrbitNumber_start"),
        })

    fc = s1.map(extract_stats)

    # Export to Google Drive
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description="echo_sentinel1_moorh",
        folder=config.GEE_DRIVE_FOLDER,
        fileNamePrefix="sentinel1_raw",
        fileFormat="CSV",
    )
    task.start()

    print(f"Export task started: {task.id}")
    print(f"Monitor at: https://code.earthengine.google.com/tasks")
    print(f"Download to: {config.DATA_RAW_GEE / 'sentinel1_raw.csv'}")

    return task.id


def process_raw(
    raw_path: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Process a raw GEE Sentinel-1 CSV export into the standard schema.

    Cleans GEE artefact columns (.geo, system:index), parses dates,
    computes VH/VV ratio, and validates.

    Args:
        raw_path: Path to raw CSV downloaded from Google Drive.
        output_path: If provided, save processed CSV here.

    Returns:
        Processed DataFrame with standard schema.

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If validation fails.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw S1 CSV not found: {raw_path}")

    logger.info("Processing raw Sentinel-1 CSV: %s", raw_path)
    df = pd.read_csv(raw_path)

    # Drop GEE artefact columns
    drop_cols = [c for c in df.columns if c in (".geo", "system:index")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Parse date from system:time_start (ms since epoch)
    df["date"] = pd.to_datetime(df["system:time_start"], unit="ms").dt.normalize()
    df["date"] = df["date"].dt.tz_localize(None)

    # Compute VH/VV ratio in dB (subtraction in log space)
    df["vhvv_db"] = df["vh_db"] - df["vv_db"]

    # Convert types
    df["orbit_number"] = df["orbit_number"].astype(int)
    df["n_pixels"] = df["n_pixels"].astype(int)

    # Round floats
    for col in ["vv_db", "vh_db", "vhvv_db", "incidence_angle_mean"]:
        df[col] = df[col].round(4)

    # Select and order output columns
    out_cols = [
        "date", "vv_db", "vh_db", "vhvv_db",
        "orbit_number", "n_pixels", "incidence_angle_mean",
    ]
    df = df[out_cols].sort_values("date").reset_index(drop=True)

    # Validate
    _validate_s1(df)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved processed S1 data to %s (%d rows)", output_path, len(df))

    return df


def _validate_s1(df: pd.DataFrame) -> None:
    """
    Validate the processed Sentinel-1 DataFrame.

    Raises ValueError for hard failures. Logs warnings for soft issues.
    """
    # 1. Single orbit number
    orbits = df["orbit_number"].unique()
    if len(orbits) != 1:
        raise ValueError(
            f"Expected single orbit number, found {len(orbits)}: {orbits}"
        )

    # 2. VV range check
    vv_out = df[~df["vv_db"].between(config.VV_RANGE_MIN, config.VV_RANGE_MAX)]
    if len(vv_out) > 0:
        raise ValueError(
            f"vv_db has {len(vv_out)} values outside [{config.VV_RANGE_MIN}, "
            f"{config.VV_RANGE_MAX}] dB. Range: [{df['vv_db'].min():.2f}, "
            f"{df['vv_db'].max():.2f}]"
        )

    # 3. Minimum pixel count
    low_pixels = df[df["n_pixels"] <= config.S1_MIN_PIXELS]
    if len(low_pixels) > 0:
        raise ValueError(
            f"{len(low_pixels)} overpasses have n_pixels <= {config.S1_MIN_PIXELS}"
        )

    # 4. Annual overpass counts (warn, don't fail)
    annual_counts = df.groupby(df["date"].dt.year).size()
    for year, count in annual_counts.items():
        if not (45 <= count <= 75):
            logger.warning(
                "Year %d has %d overpasses (expected 45–75)", year, count
            )

    # 5. No duplicate dates
    if df["date"].duplicated().any():
        raise ValueError("Duplicate overpass dates found")

    # 6. All 4 years represented
    years = set(df["date"].dt.year)
    expected_years = {2021, 2022, 2023, 2024}
    missing_years = expected_years - years
    if missing_years:
        raise ValueError(f"Missing observations in years: {missing_years}")

    logger.info(
        "S1 validation passed: %d overpasses, orbit %d, "
        "VV range [%.2f, %.2f] dB",
        len(df), df["orbit_number"].iloc[0],
        df["vv_db"].min(), df["vv_db"].max(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract Sentinel-1 SAR backscatter via GEE"
    )
    parser.add_argument(
        "--orbit", type=int, default=None,
        help="Relative orbit number (auto-selects most frequent if omitted)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check GEE access and print metadata without submitting export"
    )
    parser.add_argument(
        "--process", type=str, default=None,
        help="Path to raw CSV to process (skip GEE export, just process)"
    )
    args = parser.parse_args()

    if args.process:
        raw_path = Path(args.process)
        output_path = config.DATA_PROCESSED / "sentinel1_extractions.csv"
        process_raw(raw_path, output_path)
    else:
        submit_extraction(orbit_number=args.orbit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
