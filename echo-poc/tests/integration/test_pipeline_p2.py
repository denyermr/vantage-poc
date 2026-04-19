"""
Phase 2 integration tests — 6 tests per SPEC_PHASE2.md §P2.12.

Require data/processed/aligned_dataset.csv and split configs to exist.
"""

import json

import numpy as np
import pandas as pd
import pytest

from shared import config
from shared.evaluation import compute_metrics
from shared.splits import load_config
from shared.baselines.null_model import NullModel
from shared.baselines.random_forest import RFModel
from shared.baselines.standard_nn import NNModel

pytestmark = pytest.mark.skipif(
    not (config.DATA_PROCESSED / "aligned_dataset.csv").exists(),
    reason="Requires data/processed/aligned_dataset.csv",
)


@pytest.fixture
def aligned_df():
    """Load aligned dataset."""
    return pd.read_csv(config.DATA_PROCESSED / "aligned_dataset.csv")


@pytest.fixture
def test_info():
    """Load test indices info."""
    with open(config.DATA_SPLITS / "test_indices.json") as f:
        return json.load(f)


def test_all_40_configs_loadable():
    """All config JSON files parse without error."""
    configs_dir = config.DATA_SPLITS / "configs"
    for i in range(config.N_CONFIGS):
        cfg = load_config(configs_dir / f"config_{i:03d}.json")
        assert "train_indices" in cfg
        assert "val_indices" in cfg


def test_no_data_leakage_across_configs(test_info):
    """Test set never appears in any train or val set."""
    split_idx = test_info["split_idx"]
    test_set = set(range(split_idx, test_info["n_total"]))
    configs_dir = config.DATA_SPLITS / "configs"

    for i in range(config.N_CONFIGS):
        cfg = load_config(configs_dir / f"config_{i:03d}.json")
        train = set(cfg["train_indices"])
        val = set(cfg["val_indices"])
        assert len(train & test_set) == 0, f"Config {i}: train overlaps test"
        assert len(val & test_set) == 0, f"Config {i}: val overlaps test"


def test_baseline_a_runs_on_config_000(aligned_df, test_info):
    """RF trains and produces metrics without error on config 000."""
    cfg = load_config(config.DATA_SPLITS / "configs" / "config_000.json")
    split_idx = test_info["split_idx"]

    X = aligned_df[config.FEATURE_COLUMNS].values
    y = aligned_df[config.TARGET_COLUMN].values

    X_train = X[cfg["train_indices"]]
    y_train = y[cfg["train_indices"]]
    X_val = X[cfg["val_indices"]]
    y_val = y[cfg["val_indices"]]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    model = RFModel(config_idx=0)
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)

    assert y_pred.shape == y_test.shape
    m = compute_metrics(y_pred, y_test)
    assert not np.isnan(m["rmse"])


def test_baseline_b_runs_on_config_000(aligned_df, test_info):
    """NN trains and produces metrics without error on config 000."""
    cfg = load_config(config.DATA_SPLITS / "configs" / "config_000.json")
    split_idx = test_info["split_idx"]

    X = aligned_df[config.FEATURE_COLUMNS].values
    y = aligned_df[config.TARGET_COLUMN].values

    X_train = X[cfg["train_indices"]].astype(np.float32)
    y_train = y[cfg["train_indices"]].astype(np.float32)
    X_val = X[cfg["val_indices"]].astype(np.float32)
    y_val = y[cfg["val_indices"]].astype(np.float32)
    X_test = X[split_idx:].astype(np.float32)
    y_test = y[split_idx:].astype(np.float32)

    model = NNModel(config_idx=0)
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)

    assert y_pred.shape == y_test.shape
    m = compute_metrics(y_pred, y_test)
    assert not np.isnan(m["rmse"])


def test_null_model_runs_on_full_pool(aligned_df, test_info):
    """Null model trains and produces metrics on full training pool."""
    split_idx = test_info["split_idx"]
    dates = pd.DatetimeIndex(aligned_df["date"])

    X = aligned_df[config.FEATURE_COLUMNS].values
    y = aligned_df[config.TARGET_COLUMN].values

    model = NullModel()
    model.fit(
        X[:split_idx], y[:split_idx],
        X[:1], y[:1],
        train_dates=dates[:split_idx],
    )
    y_pred = model.predict(X[split_idx:], pred_dates=dates[split_idx:])

    m = compute_metrics(y_pred, y[split_idx:])
    assert not np.isnan(m["rmse"])
    assert m["rmse"] > 0


def test_rf_metrics_plausible_at_100pct(aligned_df, test_info):
    """RMSE in [0.01, 0.20] — not degenerate."""
    cfg = load_config(config.DATA_SPLITS / "configs" / "config_000.json")
    split_idx = test_info["split_idx"]

    X = aligned_df[config.FEATURE_COLUMNS].values
    y = aligned_df[config.TARGET_COLUMN].values

    model = RFModel(config_idx=0)
    model.fit(X[cfg["train_indices"]], y[cfg["train_indices"]],
              X[cfg["val_indices"]], y[cfg["val_indices"]])
    y_pred = model.predict(X[split_idx:])

    m = compute_metrics(y_pred, y[split_idx:])
    assert 0.01 <= m["rmse"] <= 0.20, f"RMSE={m['rmse']:.4f} outside plausible range"
