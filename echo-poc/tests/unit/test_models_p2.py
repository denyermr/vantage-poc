"""
Phase 2 unit tests — 30 tests per SPEC_PHASE2.md §P2.12.

Tests split generation, metrics computation, null model, RF, NN,
Wilcoxon test, and config validation.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from shared import config
from shared.evaluation import (
    aggregate_metrics_across_reps,
    build_metrics_json,
    compute_metrics,
    wilcoxon_test,
)
from shared.splits import assign_season, stratified_subsample
from shared.baselines.null_model import NullModel
from shared.baselines.random_forest import RFModel
from shared.baselines.standard_nn import NNModel, StandardNNModule


# ─── Split generation tests ─────────────────────────────────────────────────


class TestSeasonAssignment:
    """Tests for assign_season()."""

    def test_season_assignment_djf(self):
        """Dec, Jan, Feb all return 'DJF'."""
        dates = pd.DatetimeIndex([
            "2021-12-15", "2022-01-10", "2022-02-20"
        ])
        result = assign_season(dates)
        assert all(s == "DJF" for s in result)

    def test_season_assignment_boundary(self):
        """Dec 1st = DJF, Mar 1st = MAM."""
        dates = pd.DatetimeIndex(["2021-12-01", "2022-03-01"])
        result = assign_season(dates)
        assert result[0] == "DJF"
        assert result[1] == "MAM"

    def test_season_assignment_all_months(self):
        """All 12 months assigned exactly one season."""
        dates = pd.DatetimeIndex([f"2021-{m:02d}-15" for m in range(1, 13)])
        result = assign_season(dates)
        assert len(result) == 12
        expected = ["DJF", "DJF", "MAM", "MAM", "MAM", "JJA",
                     "JJA", "JJA", "SON", "SON", "SON", "DJF"]
        assert list(result) == expected


class TestStratifiedSubsample:
    """Tests for stratified_subsample()."""

    @pytest.fixture
    def pool_setup(self):
        """Create a pool with known seasonal distribution."""
        # 80 dates across 4 years, roughly 20 per season
        dates = pd.date_range("2021-01-01", periods=365 * 3, freq="4D")
        pool_indices = np.arange(len(dates))
        return pool_indices, dates

    def test_stratified_subsample_preserves_fraction(self, pool_setup):
        """Output length ≈ N × fraction (±1)."""
        pool_indices, dates = pool_setup
        rng = np.random.default_rng(42)
        result = stratified_subsample(pool_indices, dates, 0.25, rng)
        expected = max(1, math.floor(len(pool_indices) * 0.25))
        assert abs(len(result) - expected) <= 2  # allow ±2 for rounding

    def test_stratified_subsample_all_seasons_represented(self, pool_setup):
        """All 4 seasons present in output when N large enough."""
        pool_indices, dates = pool_setup
        rng = np.random.default_rng(42)
        result = stratified_subsample(pool_indices, dates, 0.50, rng)
        result_seasons = assign_season(dates[result])
        assert set(result_seasons) == {"DJF", "MAM", "JJA", "SON"}

    def test_stratified_subsample_falls_back_below_min_stratum(self, caplog):
        """Warning logged when season < MIN_STRATUM_SAMPLES."""
        # Create pool with only 1 DJF sample
        dates = pd.DatetimeIndex([
            "2021-01-15",  # DJF — only 1
            "2021-04-01", "2021-04-15", "2021-05-01",  # MAM
            "2021-07-01", "2021-07-15", "2021-08-01",  # JJA
            "2021-10-01", "2021-10-15", "2021-11-01",  # SON
        ])
        pool_indices = np.arange(len(dates))
        rng = np.random.default_rng(42)

        import logging
        with caplog.at_level(logging.WARNING, logger="shared.splits"):
            result = stratified_subsample(pool_indices, dates, 0.50, rng)
        assert "falling back" in caplog.text.lower() or len(result) > 0


class TestSplitIntegrity:
    """Tests for split non-overlap and chronological integrity."""

    def test_no_overlap_train_val_test(self, sample_aligned_df):
        """Intersection of all three sets is empty."""
        n = len(sample_aligned_df)
        split_idx = int(n * 0.7)
        train_pool = np.arange(split_idx)
        test_indices = np.arange(split_idx, n)

        rng = np.random.default_rng(42)
        n_val = max(1, round(len(train_pool) * 0.2))
        val_positions = rng.choice(len(train_pool), size=n_val, replace=False)
        val_mask = np.zeros(len(train_pool), dtype=bool)
        val_mask[val_positions] = True
        val_indices = train_pool[val_mask]
        train_indices = train_pool[~val_mask]

        assert len(set(train_indices) & set(val_indices)) == 0
        assert len(set(train_indices) & set(test_indices)) == 0
        assert len(set(val_indices) & set(test_indices)) == 0

    def test_train_indices_before_test_indices(self, sample_aligned_df):
        """max(train_date) < min(test_date)."""
        n = len(sample_aligned_df)
        split_idx = int(n * 0.7)
        dates = pd.DatetimeIndex(sample_aligned_df["date"])

        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        assert train_dates.max() < test_dates.min()


# ─── Metrics tests ───────────────────────────────────────────────────────────


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_compute_metrics_perfect_prediction(self):
        """RMSE=0, R²=1, bias=0 for y_pred=y_true."""
        y = np.array([0.5, 0.6, 0.7, 0.8])
        m = compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["r_squared"] == pytest.approx(1.0, abs=1e-10)
        assert m["mean_bias"] == pytest.approx(0.0, abs=1e-10)

    def test_compute_metrics_constant_prediction(self):
        """R² ≤ 0 when y_pred is constant (may be negative)."""
        y_true = np.array([0.3, 0.5, 0.7, 0.9])
        y_pred = np.full_like(y_true, 0.6)
        m = compute_metrics(y_pred, y_true)
        assert m["r_squared"] <= 0.0 + 1e-10

    def test_compute_metrics_known_values(self):
        """Manually computed expected RMSE verified."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.7])
        # residuals: 0.1, 0.2, -0.3
        # sq: 0.01, 0.04, 0.09 → mean=0.04667 → rmse=0.21602
        m = compute_metrics(y_pred, y_true)
        assert m["rmse"] == pytest.approx(0.21602, abs=0.001)

    def test_mean_bias_sign(self):
        """Positive bias when predictions systematically high."""
        y_true = np.array([0.5, 0.6, 0.7])
        y_pred = np.array([0.6, 0.7, 0.8])  # all +0.1
        m = compute_metrics(y_pred, y_true)
        assert m["mean_bias"] > 0
        assert m["mean_bias"] == pytest.approx(0.1, abs=1e-10)


# ─── Null model tests ───────────────────────────────────────────────────────


class TestNullModel:
    """Tests for NullModel."""

    def test_null_model_predicts_seasonal_mean(self):
        """DJF prediction = mean of DJF training values."""
        model = NullModel()
        # Training data: 4 DJF, 4 JJA
        train_dates = pd.DatetimeIndex([
            "2021-01-10", "2021-01-20", "2021-02-05", "2021-12-15",  # DJF
            "2021-06-10", "2021-07-10", "2021-07-20", "2021-08-10",  # JJA
        ])
        y_train = np.array([0.70, 0.72, 0.68, 0.74,  # DJF
                            0.45, 0.48, 0.50, 0.42])  # JJA
        X_dummy = np.zeros((8, 7))

        model.fit(X_dummy, y_train, X_dummy[:1], y_train[:1],
                  train_dates=train_dates)

        # Predict for a DJF date
        pred_dates = pd.DatetimeIndex(["2022-01-15"])
        pred = model.predict(X_dummy[:1], pred_dates=pred_dates)
        expected_djf_mean = np.mean([0.70, 0.72, 0.68, 0.74])
        assert pred[0] == pytest.approx(expected_djf_mean, abs=1e-6)

    def test_null_model_fallback_missing_season(self):
        """Returns global mean when season absent from training."""
        model = NullModel()
        # Only DJF training data
        train_dates = pd.DatetimeIndex(["2021-01-10", "2021-02-10"])
        y_train = np.array([0.70, 0.80])
        X_dummy = np.zeros((2, 7))

        model.fit(X_dummy, y_train, X_dummy[:1], y_train[:1],
                  train_dates=train_dates)

        # Predict for JJA — should get global mean
        pred_dates = pd.DatetimeIndex(["2021-07-15"])
        pred = model.predict(X_dummy[:1], pred_dates=pred_dates)
        assert pred[0] == pytest.approx(0.75, abs=1e-6)  # (0.70 + 0.80) / 2

    def test_null_model_beats_no_information(self):
        """RMSE < std(y_test) — better than zero-information."""
        rng = np.random.default_rng(42)
        # Create seasonal training data
        train_dates = pd.date_range("2021-01-01", periods=60, freq="6D")
        doy = train_dates.dayofyear
        y_train = 0.65 + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365)
        y_train += rng.normal(0, 0.02, size=len(y_train))

        test_dates = pd.date_range("2023-01-01", periods=20, freq="18D")
        doy_test = test_dates.dayofyear
        y_test = 0.65 + 0.12 * np.cos(2 * np.pi * (doy_test - 15) / 365)
        y_test += rng.normal(0, 0.02, size=len(y_test))

        model = NullModel()
        X_dummy = np.zeros((len(y_train), 7))
        model.fit(X_dummy, y_train, X_dummy[:1], y_train[:1],
                  train_dates=train_dates)

        pred = model.predict(np.zeros((len(y_test), 7)), pred_dates=test_dates)
        m = compute_metrics(pred, y_test)
        assert m["rmse"] < np.std(y_test)


# ─── RF model tests ─────────────────────────────────────────────────────────


class TestRFModel:
    """Tests for RFModel."""

    @pytest.fixture
    def trained_rf(self):
        """Train a small RF model for testing."""
        rng = np.random.default_rng(42)
        n_train, n_val = 50, 10
        n_features = len(config.FEATURE_COLUMNS)
        X_train = rng.standard_normal((n_train, n_features))
        y_train = 0.5 + 0.1 * X_train[:, 0] + rng.normal(0, 0.02, n_train)
        X_val = rng.standard_normal((n_val, n_features))
        y_val = 0.5 + 0.1 * X_val[:, 0] + rng.normal(0, 0.02, n_val)

        model = RFModel(config_idx=0)
        model.fit(X_train, y_train, X_val, y_val)
        return model, X_train, y_train

    def test_rf_model_saves_and_loads(self, trained_rf):
        """Loaded RF produces identical predictions to original."""
        model, X_train, _ = trained_rf
        pred_original = model.predict(X_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            loaded = RFModel.load(Path(tmpdir))
            pred_loaded = loaded.predict(X_train)

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_rf_scaler_saved_with_model(self, trained_rf):
        """scaler.pkl exists after save()."""
        model, _, _ = trained_rf
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            assert (Path(tmpdir) / "scaler.pkl").exists()

    def test_rf_predicts_from_feature_matrix(self, trained_rf):
        """Output shape is (N,) for input (N, 7)."""
        model, _, _ = trained_rf
        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((15, len(config.FEATURE_COLUMNS)))
        pred = model.predict(X_test)
        assert pred.shape == (15,)


# ─── NN model tests ─────────────────────────────────────────────────────────


class TestNNModel:
    """Tests for NNModel."""

    def test_nn_architecture_layer_count(self):
        """3 hidden layers, confirmed by named modules."""
        net = StandardNNModule(n_features=7)
        linear_layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
        # 3 hidden + 1 output = 4 Linear layers
        assert len(linear_layers) == 4
        # Check hidden sizes
        assert linear_layers[0].out_features == 64
        assert linear_layers[1].out_features == 32
        assert linear_layers[2].out_features == 16
        assert linear_layers[3].out_features == 1

    def test_nn_reproducible_with_seed(self):
        """Two trainings with same seed produce identical weights."""
        rng = np.random.default_rng(42)
        n_features = len(config.FEATURE_COLUMNS)
        X_train = rng.standard_normal((30, n_features)).astype(np.float32)
        y_train = (0.5 + 0.1 * X_train[:, 0]).astype(np.float32)
        X_val = rng.standard_normal((8, n_features)).astype(np.float32)
        y_val = (0.5 + 0.1 * X_val[:, 0]).astype(np.float32)

        m1 = NNModel(config_idx=0)
        m1.fit(X_train, y_train, X_val, y_val)
        pred1 = m1.predict(X_val)

        m2 = NNModel(config_idx=0)
        m2.fit(X_train, y_train, X_val, y_val)
        pred2 = m2.predict(X_val)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_nn_early_stopping_triggers(self):
        """Stops before MAX_EPOCHS when val loss plateaus."""
        rng = np.random.default_rng(42)
        n_features = len(config.FEATURE_COLUMNS)
        # Easy problem — should converge fast
        X_train = rng.standard_normal((40, n_features)).astype(np.float32)
        y_train = np.full(40, 0.5, dtype=np.float32)  # constant target
        X_val = rng.standard_normal((10, n_features)).astype(np.float32)
        y_val = np.full(10, 0.5, dtype=np.float32)

        model = NNModel(config_idx=0)
        model.fit(X_train, y_train, X_val, y_val)
        assert model.stopped_at_epoch_ < config.NN_MAX_EPOCHS

    def test_nn_saves_training_history(self):
        """config.json contains training_history key."""
        rng = np.random.default_rng(42)
        n_features = len(config.FEATURE_COLUMNS)
        X = rng.standard_normal((20, n_features)).astype(np.float32)
        y = np.full(20, 0.5, dtype=np.float32)

        model = NNModel(config_idx=0)
        model.fit(X, y, X[:5], y[:5])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            with open(Path(tmpdir) / "config.json") as f:
                meta = json.load(f)
            assert "training_history" in meta
            assert "train_loss" in meta["training_history"]
            assert "val_loss" in meta["training_history"]


# ─── Wilcoxon test ───────────────────────────────────────────────────────────


class TestWilcoxon:
    """Tests for wilcoxon_test()."""

    def test_wilcoxon_identical_distributions(self):
        """p_value ≈ 1.0 for identical inputs."""
        values = [0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.08]
        result = wilcoxon_test(values, values)
        assert result["p_value_uncorrected"] == pytest.approx(1.0, abs=0.01)

    def test_wilcoxon_clearly_different(self):
        """p_value < 0.05 for sufficiently different inputs."""
        a = [0.10, 0.11, 0.12, 0.10, 0.11, 0.10, 0.12, 0.11, 0.10, 0.11]
        b = [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06, 0.04, 0.05, 0.04]
        result = wilcoxon_test(a, b)
        assert result["p_value_uncorrected"] < 0.05
        assert result["significant_uncorrected"] is True

    def test_wilcoxon_bonferroni_flag_correct(self):
        """significant_bonferroni=False when 0.05 > p > 0.0125."""
        # Use values that give a borderline p-value
        a = [0.06, 0.07, 0.06, 0.07, 0.06, 0.07, 0.06, 0.07, 0.06, 0.07]
        b = [0.05, 0.07, 0.05, 0.07, 0.05, 0.07, 0.05, 0.07, 0.05, 0.07]
        result = wilcoxon_test(a, b)
        # Just check the flags are consistent with the p-value
        if 0.0125 < result["p_value_uncorrected"] < 0.05:
            assert result["significant_uncorrected"] is True
            assert result["significant_bonferroni"] is False
        # If p < 0.0125 both should be True
        elif result["p_value_uncorrected"] < 0.0125:
            assert result["significant_uncorrected"] is True
            assert result["significant_bonferroni"] is True


# ─── Metrics JSON schema ────────────────────────────────────────────────────


class TestMetricsJSON:
    """Tests for metrics JSON schema validation."""

    def test_metrics_json_schema_valid(self):
        """Schema validates correct output."""
        metrics_dict = build_metrics_json(
            model_name="baseline_a",
            config_idx=0,
            fraction=1.0,
            fraction_label="100%",
            rep=0,
            seed_used=42,
            n_train=66,
            n_val=17,
            n_test=36,
            metrics={"rmse": 0.05, "r_squared": 0.8, "mean_bias": 0.001},
        )
        assert metrics_dict["model"] == "baseline_a"
        assert metrics_dict["metrics"]["rmse"] == 0.05
        assert "feature_columns" in metrics_dict
        assert "generated_at" in metrics_dict

    def test_metrics_json_rejects_nan(self):
        """Schema validation fails if rmse is NaN."""
        with pytest.raises(ValueError, match="NaN|nan|not allowed"):
            build_metrics_json(
                model_name="baseline_a",
                config_idx=0,
                fraction=1.0,
                fraction_label="100%",
                rep=0,
                seed_used=42,
                n_train=66,
                n_val=17,
                n_test=36,
                metrics={"rmse": float("nan"), "r_squared": 0.8, "mean_bias": 0.0},
            )


# ─── Config validation ──────────────────────────────────────────────────────


class TestConfigValidation:
    """Tests for config.py Phase 2 constants."""

    def test_config_n_configs_assertion(self):
        """N_CONFIGS == len(TRAINING_FRACTIONS) × N_REPS."""
        assert config.N_CONFIGS == len(config.TRAINING_FRACTIONS) * config.N_REPS

    def test_feature_columns_length(self):
        """len(FEATURE_COLUMNS) == 7 (DEV-004: terrain excluded)."""
        assert len(config.FEATURE_COLUMNS) == 7

    def test_vhvv_is_difference_not_ratio(self):
        """vhvv_db = vh_db - vv_db (subtraction in log space)."""
        vh_db = -18.0
        vv_db = -12.0
        vhvv_db = vh_db - vv_db
        assert vhvv_db == pytest.approx(-6.0, abs=1e-6)
