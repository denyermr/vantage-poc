"""
Shared test fixtures and configuration for the ECHO PoC test suite.
"""

import pandas as pd
import numpy as np
import pytest

from shared import config


def pytest_addoption(parser):
    """Add custom CLI options for integration tests."""
    parser.addoption(
        "--require-data",
        action="store_true",
        default=False,
        help="Run integration tests that require data/processed/ to exist",
    )


@pytest.fixture
def sample_cosmos_df() -> pd.DataFrame:
    """
    Synthetic COSMOS-UK DataFrame for unit tests.

    50 days of plausible Moor House data with a mix of QC flags,
    frozen days, and snow days. VWC follows a simple seasonal pattern
    (higher in winter, lower in summer).
    """
    rng = np.random.default_rng(config.SEED)
    dates = pd.date_range("2021-06-01", periods=50, freq="7D")

    # Seasonal VWC pattern: higher in winter (DJF), lower in summer (JJA)
    doy = dates.dayofyear
    vwc_base = 0.65 + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365)
    vwc_raw = vwc_base + rng.normal(0, 0.03, size=50)
    vwc_raw = np.clip(vwc_raw, 0.20, 0.95)

    # Temperature — some freezing days in winter months
    ta_min = 8.0 - 12.0 * np.cos(2 * np.pi * (doy - 15) / 365)
    ta_min += rng.normal(0, 2, size=50)

    # Snow — some snow in winter
    snow = np.zeros(50)
    winter_mask = np.isin(dates.month, [12, 1, 2])
    snow[winter_mask] = rng.choice([0, 0, 0, 2.0, 5.0], size=winter_mask.sum())

    # QC flags — mostly good, a few E and I
    flags = ["" if rng.random() > 0.05 else rng.choice(["E", "I"]) for _ in range(50)]

    return pd.DataFrame({
        "date": dates,
        "vwc_raw": vwc_raw,
        "vwc_qc": [v if f not in ("E", "I") else np.nan for v, f in zip(vwc_raw, flags)],
        "cosmos_vwc_flag": flags,
        "ta_min": ta_min,
        "snow": snow,
        "frozen_flag": (ta_min < 0).astype(int),
        "snow_flag": (snow > 0).astype(int),
    })


@pytest.fixture
def sample_s1_df() -> pd.DataFrame:
    """
    Synthetic Sentinel-1 DataFrame for unit tests.

    Roughly 60 descending overpasses per year across 2021-2024.
    Single orbit, plausible VV/VH values for wet peat.
    """
    rng = np.random.default_rng(config.SEED + 1)
    # ~60 overpasses per year, 6-day revisit
    dates = pd.date_range("2021-01-03", "2024-12-28", freq="6D")

    n = len(dates)
    vv_db = rng.uniform(-16, -9, size=n)
    vh_db = vv_db - rng.uniform(5, 8, size=n)

    return pd.DataFrame({
        "date": dates,
        "vv_db": np.round(vv_db, 4),
        "vh_db": np.round(vh_db, 4),
        "vhvv_db": np.round(vh_db - vv_db, 4),
        "orbit_number": 81,
        "n_pixels": rng.integers(50, 120, size=n),
        "incidence_angle_mean": np.round(rng.uniform(38, 42, size=n), 4),
    })


@pytest.fixture
def sample_aligned_df(sample_s1_df, sample_cosmos_df) -> pd.DataFrame:
    """
    Synthetic aligned dataset for unit tests.

    Merges S1 and COSMOS synthetic data, adds ancillary features,
    filters to valid rows. Mimics the output of alignment.py.
    """
    rng = np.random.default_rng(config.SEED + 2)

    # Find overlapping dates
    common_dates = sorted(
        set(sample_s1_df["date"]) & set(sample_cosmos_df["date"])
    )

    if len(common_dates) < 20:
        # Generate enough data with matching dates
        dates = pd.date_range("2021-01-01", "2024-12-31", freq="6D")
        n = len(dates)
    else:
        dates = pd.DatetimeIndex(common_dates)
        n = len(dates)

    # Create a plausible aligned dataset
    doy = dates.dayofyear
    vwc = 0.65 + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365)
    vwc += rng.normal(0, 0.03, size=n)
    vwc = np.clip(vwc, 0.20, 0.95)

    vv_db = rng.uniform(-16, -9, size=n)
    vh_db = vv_db - rng.uniform(5, 8, size=n)

    # DEV-004: Terrain features excluded from aligned dataset (zero variance
    # at single site). Only 7 dynamic features remain.
    df = pd.DataFrame({
        "date": dates,
        "vwc": np.round(vwc, 4),
        "vv_db": np.round(vv_db, 4),
        "vh_db": np.round(vh_db, 4),
        "vhvv_db": np.round(vh_db - vv_db, 4),
        "ndvi": np.round(rng.uniform(0.1, 0.6, size=n), 4),
        "precip_mm": np.round(rng.exponential(3, size=n), 3),
        "precip_7day_mm": np.round(rng.exponential(20, size=n), 3),
        "incidence_angle_mean": np.round(rng.uniform(38, 42, size=n), 4),
    })

    return df.sort_values("date").reset_index(drop=True)
