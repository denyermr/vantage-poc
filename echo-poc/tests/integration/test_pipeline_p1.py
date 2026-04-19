"""
Phase 1 integration tests — require processed data files to exist.

Run with: pytest tests/integration/ --require-data

These tests validate the actual data pipeline output, not synthetic fixtures.
They are skipped unless --require-data is passed.
"""

import pandas as pd
import numpy as np
import pytest
from scipy import stats

from shared import config

# ─── Skip unless --require-data flag is provided ─────────────────────────────

ALIGNED_CSV = config.DATA_PROCESSED / "aligned_dataset.csv"


def _require_data(request):
    """Skip test if --require-data not provided or data file missing."""
    if not request.config.getoption("--require-data"):
        pytest.skip("Integration tests require --require-data flag")
    if not ALIGNED_CSV.exists():
        pytest.skip(f"Aligned dataset not found: {ALIGNED_CSV}")


@pytest.fixture
def aligned_df(request) -> pd.DataFrame:
    """Load the real aligned dataset, skipping if unavailable."""
    _require_data(request)
    return pd.read_csv(ALIGNED_CSV, parse_dates=["date"])


# ─── Integration tests ───────────────────────────────────────────────────────


class TestPipelineP1:
    """Phase 1 integration tests — validate real pipeline output."""

    def test_schema_validation(self, aligned_df):
        """Aligned dataset has all required columns in correct order."""
        expected_cols = ["date", "vwc"] + config.FEATURE_COLUMNS
        actual_cols = list(aligned_df.columns)
        assert actual_cols == expected_cols, (
            f"Column mismatch.\n"
            f"Expected: {expected_cols}\n"
            f"Actual:   {actual_cols}"
        )

    def test_zero_nan(self, aligned_df):
        """Aligned dataset has zero NaN values."""
        nan_count = aligned_df.isna().sum().sum()
        assert nan_count == 0, (
            f"Found {nan_count} NaN values. "
            f"NaN columns: {aligned_df.columns[aligned_df.isna().any()].tolist()}"
        )

    def test_date_range(self, aligned_df):
        """Dates span 2021–2024 study period."""
        years = set(aligned_df["date"].dt.year)
        expected_years = {2021, 2022, 2023, 2024}
        missing = expected_years - years
        assert len(missing) == 0, f"Missing years in aligned data: {missing}"

    def test_vwc_range_plausibility(self, aligned_df):
        """VWC values within physically plausible range for peat."""
        vwc = aligned_df["vwc"]
        assert vwc.min() >= config.VWC_RANGE_MIN, (
            f"VWC min {vwc.min():.4f} below {config.VWC_RANGE_MIN}"
        )
        assert vwc.max() <= config.VWC_RANGE_MAX, (
            f"VWC max {vwc.max():.4f} above {config.VWC_RANGE_MAX}"
        )

    def test_sar_vwc_correlation(self, aligned_df):
        """SAR–VWC Pearson |r| >= 0.30 (Gate 1 criterion G1-05)."""
        r, p = stats.pearsonr(aligned_df["vv_db"], aligned_df["vwc"])
        assert abs(r) >= 0.30, (
            f"SAR-VWC |Pearson r| = {abs(r):.4f} < 0.30. "
            f"This is a Gate 1 criterion (G1-05). "
            f"r={r:.4f}, p={p:.4e}"
        )
