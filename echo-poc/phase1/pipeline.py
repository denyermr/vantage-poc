"""
ECHO PoC — Phase 1 data pipeline orchestrator.

Processes raw data files into the aligned analytical dataset,
generates the train/test split, and produces diagnostic figures.

Usage:
    python poc/pipeline.py --phase1
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import config

logger = logging.getLogger(__name__)


def _load_cosmos_preprocessed(path: Path) -> pd.DataFrame:
    """
    Load a pre-processed COSMOS-UK CSV that already has clean headers
    and VWC in cm³/cm³ (not the raw portal format with metadata rows).

    This handles the case where the COSMOS CSV was pre-processed before
    being placed in data/raw/cosmos/.

    Args:
        path: Path to COSMOS CSV.

    Returns:
        DataFrame with standard COSMOS schema.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing or validation fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"COSMOS-UK CSV not found: {path}")

    logger.info("Loading pre-processed COSMOS-UK CSV: %s", path)
    df = pd.read_csv(path)

    # Check which format we have
    if "cosmos_vwc" in df.columns:
        # Raw portal format — use the standard loader
        from shared.data.cosmos import load_cosmos
        return load_cosmos(path)

    # Pre-processed format: already has clean headers
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Identify the VWC column
    if "vwc_mean" in df.columns:
        df["vwc_raw"] = df["vwc_mean"]
    elif "vwc_raw" in df.columns:
        pass  # already has vwc_raw
    else:
        raise ValueError(
            f"Cannot find VWC column. Available: {list(df.columns)}"
        )

    # Ensure vwc_qc exists
    if "vwc_qc" not in df.columns:
        df["vwc_qc"] = df["vwc_raw"]

    # Ensure flags exist
    if "frozen_flag" not in df.columns:
        if "ta_min" in df.columns:
            df["frozen_flag"] = (pd.to_numeric(df["ta_min"], errors="coerce") < 0).astype(int)
        else:
            df["frozen_flag"] = 0

    if "snow_flag" not in df.columns:
        if "snow" in df.columns:
            df["snow_flag"] = (pd.to_numeric(df["snow"], errors="coerce") > 0).astype(int)
        else:
            df["snow_flag"] = 0

    if "cosmos_vwc_flag" not in df.columns:
        df["cosmos_vwc_flag"] = ""

    # Apply QC flags if present
    if df["cosmos_vwc_flag"].notna().any():
        flagged = df["cosmos_vwc_flag"].isin(config.COSMOS_EXCLUDE_FLAGS)
        df.loc[flagged, "vwc_qc"] = pd.NA

    # Validate VWC range (cm³/cm³)
    vwc_valid = df["vwc_qc"].dropna()
    out_of_range = vwc_valid[
        ~vwc_valid.between(config.VWC_RANGE_MIN, config.VWC_RANGE_MAX)
    ]
    if len(out_of_range) > 0:
        raise ValueError(
            f"vwc_qc contains {len(out_of_range)} values outside "
            f"[{config.VWC_RANGE_MIN}, {config.VWC_RANGE_MAX}] cm³/cm³. "
            f"Range: [{vwc_valid.min():.4f}, {vwc_valid.max():.4f}]"
        )

    # Select output columns
    out_cols = [
        "date", "vwc_raw", "vwc_qc", "cosmos_vwc_flag",
        "ta_min", "snow", "frozen_flag", "snow_flag",
    ]
    available = [c for c in out_cols if c in df.columns]
    df = df[available].copy()

    logger.info(
        "COSMOS-UK loaded: %d days, %d valid VWC, VWC range [%.3f, %.3f]",
        len(df), df["vwc_qc"].notna().sum(),
        vwc_valid.min(), vwc_valid.max(),
    )

    return df


def run_phase1():
    """
    Run the full Phase 1 data pipeline.

    Steps:
        1. Process Sentinel-1 raw CSV
        2. Load COSMOS-UK data
        3. Build ancillary features (NDVI, precip, terrain)
        4. Build aligned dataset
        5. Generate train/test split
        6. Generate diagnostic figures
        7. Save all processed outputs
    """
    logger.info("=" * 60)
    logger.info("ECHO PoC — Phase 1 Data Pipeline")
    logger.info("=" * 60)

    # ─── Step 1: Process Sentinel-1 ─────────────────────────────
    logger.info("Step 1: Processing Sentinel-1 SAR data")

    from shared.data.gee.extract_sentinel1 import process_raw as process_s1

    s1_raw = config.DATA_RAW_GEE / "sentinel1_raw.csv"
    s1_output = config.DATA_PROCESSED / "sentinel1_extractions.csv"
    s1_df = process_s1(s1_raw, s1_output)
    sar_dates = pd.DatetimeIndex(s1_df["date"])
    logger.info("S1: %d overpasses processed", len(s1_df))

    # ─── Step 2: Load COSMOS-UK ─────────────────────────────────
    logger.info("Step 2: Loading COSMOS-UK data")

    # Try standard filename first, then search for any CSV
    cosmos_path = config.DATA_RAW_COSMOS / config.COSMOS_RAW_FILENAME
    if not cosmos_path.exists():
        # Search for any CSV in the cosmos directory
        cosmos_csvs = list(config.DATA_RAW_COSMOS.glob("*.csv"))
        if not cosmos_csvs:
            raise FileNotFoundError(
                f"No COSMOS-UK CSV found in {config.DATA_RAW_COSMOS}"
            )
        cosmos_path = cosmos_csvs[0]
        logger.info("Using COSMOS CSV: %s", cosmos_path.name)

    cosmos_df = _load_cosmos_preprocessed(cosmos_path)

    # Save processed COSMOS
    cosmos_output = config.DATA_PROCESSED / "cosmos_processed.csv"
    cosmos_output.parent.mkdir(parents=True, exist_ok=True)
    cosmos_df.to_csv(cosmos_output, index=False)
    logger.info("Saved processed COSMOS to %s", cosmos_output)

    # ─── Step 3: Build ancillary features ───────────────────────
    logger.info("Step 3: Building ancillary features")

    from shared.data.ancillary import build_ancillary, save_ancillary

    ancillary_df = build_ancillary(
        s1_dates=sar_dates,
        ndvi_monthly_path=config.DATA_RAW_GEE / "sentinel2_ndvi_raw.csv",
        era5_raw_path=config.DATA_RAW_GEE / "era5_precip_raw.csv",
        terrain_raw_path=config.DATA_RAW_GEE / "terrain_static_raw.csv",
    )
    ancillary_output = config.DATA_PROCESSED / "ancillary_features.csv"
    save_ancillary(ancillary_df, ancillary_output)

    # ─── Step 4: Build aligned dataset ──────────────────────────
    logger.info("Step 4: Building aligned dataset")

    from shared.data.alignment import build_aligned_dataset

    aligned_output = config.DATA_PROCESSED / "aligned_dataset.csv"
    aligned_df = build_aligned_dataset(
        cosmos_path=cosmos_output,
        s1_path=s1_output,
        ancillary_path=ancillary_output,
        output_path=aligned_output,
    )

    # ─── Step 5: Generate train/test split ──────────────────────
    logger.info("Step 5: Generating train/test split")

    from shared.data.alignment import generate_test_split

    split_info = generate_test_split(aligned_df)
    logger.info(
        "Split: %d train, %d test",
        split_info["n_train_pool"], split_info["n_test"],
    )

    # ─── Step 6: Generate diagnostic figures ────────────────────
    logger.info("Step 6: Generating diagnostic figures")

    from phase1.plots.phase1_figs import (
        plot_cosmos_diagnostic,
        plot_sar_diagnostic,
        plot_ancillary_diagnostic,
        plot_aligned_summary,
    )

    plot_cosmos_diagnostic(cosmos_df)
    plot_sar_diagnostic(s1_df, aligned_df)
    plot_ancillary_diagnostic(ancillary_df)
    plot_aligned_summary(aligned_df)

    # ─── Done ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 1 pipeline complete!")
    logger.info("  Aligned dataset: %d rows, %d features", len(aligned_df), len(config.FEATURE_COLUMNS))
    logger.info("  Train/test split: %d / %d", split_info["n_train_pool"], split_info["n_test"])
    logger.info("  Output: %s", aligned_output)
    logger.info("=" * 60)

    return aligned_df


def run_phase2():
    """
    Run the full Phase 2 baseline model training pipeline.

    Trains null model, RF, and NN across all 40 configurations.
    Generates diagnostic figures.
    """
    import json
    import time

    import numpy as np

    from shared.evaluation import (
        build_metrics_json,
        compute_metrics,
        save_metrics_json,
    )
    from phase1.plots.phase2 import plot_feature_diagnostics, plot_learning_curves
    from shared.splits import load_config
    from shared.baselines.null_model import NullModel
    from shared.baselines.random_forest import RFModel
    from shared.baselines.standard_nn import NNModel

    logger.info("=" * 60)
    logger.info("ECHO PoC — Phase 2: Baseline Models")
    logger.info("=" * 60)

    # Load data
    aligned_path = config.DATA_PROCESSED / "aligned_dataset.csv"
    df = pd.read_csv(aligned_path)
    dates = pd.DatetimeIndex(df["date"])
    X = df[config.FEATURE_COLUMNS].values
    y = df[config.TARGET_COLUMN].values

    with open(config.DATA_SPLITS / "test_indices.json") as f:
        test_info = json.load(f)
    split_idx = test_info["split_idx"]
    n_test = test_info["n_test"]

    X_test = X[split_idx:]
    y_test = y[split_idx:]
    test_dates = dates[split_idx:]

    logger.info("Dataset: %d rows, %d features, split_idx=%d, n_test=%d",
                len(df), len(config.FEATURE_COLUMNS), split_idx, n_test)

    # ── Null model (single run on full training pool) ──
    logger.info("Training null model on full training pool...")
    null_model = NullModel()
    null_model.fit(
        X[:split_idx], y[:split_idx],
        X[:1], y[:1],
        train_dates=dates[:split_idx],
    )
    null_pred = null_model.predict(X_test, pred_dates=test_dates)
    null_metrics = compute_metrics(null_pred, y_test)

    null_json = {
        "model": "null_baseline",
        "config_idx": "full_pool",
        "training_fraction": 1.0,
        "n_train": split_idx,
        "n_test": n_test,
        "metrics": null_metrics,
        "seasonal_means_train": null_model.seasonal_means_,
        "note": "Performance floor. Any model with RMSE > this value adds no predictive value beyond seasonal climatology.",
    }

    null_model_dir = config.OUTPUTS_MODELS / "baseline_0"
    null_model.save(null_model_dir)
    save_metrics_json(null_json, config.OUTPUTS_METRICS / "baseline_0_metrics.json")
    logger.info("Null model: RMSE=%.4f, R²=%.4f, bias=%.4f",
                null_metrics["rmse"], null_metrics["r_squared"], null_metrics["mean_bias"])

    # ── RF and NN across 40 configs ──
    configs_dir = config.DATA_SPLITS / "configs"

    for config_idx in range(config.N_CONFIGS):
        cfg = load_config(configs_dir / f"config_{config_idx:03d}.json")
        train_indices = cfg["train_indices"]
        val_indices = cfg["val_indices"]

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]

        fraction = cfg["fraction"]
        fraction_label = cfg["fraction_label"]
        rep = cfg["rep"]
        seed = cfg["seed_used"]

        # ── RF ──
        t0 = time.time()
        rf = RFModel(config_idx=config_idx)
        rf.fit(X_train, y_train, X_val, y_val)
        rf_pred = rf.predict(X_test)
        rf_metrics = compute_metrics(rf_pred, y_test)

        rf_model_dir = config.OUTPUTS_MODELS / "baseline_a" / f"config_{config_idx:03d}"
        rf.save(rf_model_dir)

        rf_json = build_metrics_json(
            model_name="baseline_a",
            config_idx=config_idx,
            fraction=fraction,
            fraction_label=fraction_label,
            rep=rep,
            seed_used=seed,
            n_train=len(train_indices),
            n_val=len(val_indices),
            n_test=n_test,
            metrics=rf_metrics,
            training_metadata={
                "best_params": rf.best_params_,
                "stopped_at_epoch": None,
                "best_val_loss": None,
                "cv_warning": rf.cv_warning_,
                "stratification_used": cfg["stratification_used"],
            },
        )
        save_metrics_json(rf_json, config.OUTPUTS_METRICS / f"config_{config_idx:03d}_baseline_a.json")
        rf_time = time.time() - t0

        # ── NN ──
        t0 = time.time()
        nn = NNModel(config_idx=config_idx)
        nn.fit(
            X_train.astype(np.float32), y_train.astype(np.float32),
            X_val.astype(np.float32), y_val.astype(np.float32),
        )
        nn_pred = nn.predict(X_test)
        nn_metrics = compute_metrics(nn_pred, y_test)

        nn_model_dir = config.OUTPUTS_MODELS / "baseline_b" / f"config_{config_idx:03d}"
        nn.save(nn_model_dir)

        nn_json = build_metrics_json(
            model_name="baseline_b",
            config_idx=config_idx,
            fraction=fraction,
            fraction_label=fraction_label,
            rep=rep,
            seed_used=seed,
            n_train=len(train_indices),
            n_val=len(val_indices),
            n_test=n_test,
            metrics=nn_metrics,
            training_metadata={
                "best_params": None,
                "stopped_at_epoch": nn.stopped_at_epoch_,
                "best_val_loss": nn.best_val_loss_,
                "cv_warning": False,
                "stratification_used": cfg["stratification_used"],
            },
        )
        save_metrics_json(nn_json, config.OUTPUTS_METRICS / f"config_{config_idx:03d}_baseline_b.json")
        nn_time = time.time() - t0

        logger.info(
            "Config %03d (%.0f%% rep=%d): RF RMSE=%.4f (%.1fs) | NN RMSE=%.4f (%.1fs)",
            config_idx, fraction * 100, rep,
            rf_metrics["rmse"], rf_time,
            nn_metrics["rmse"], nn_time,
        )

    # ── Figures ──
    logger.info("Generating figures...")
    plot_learning_curves(
        config.OUTPUTS_METRICS,
        config.OUTPUTS_METRICS / "baseline_0_metrics.json",
        config.OUTPUTS_FIGURES / "p2_learning_curves_baselines.png",
    )
    plot_feature_diagnostics(
        aligned_path,
        config.OUTPUTS_METRICS,
        config.OUTPUTS_GATES / "gate_1_result.json",
        config.OUTPUTS_FIGURES / "p2_feature_diagnostics.png",
    )

    logger.info("Phase 2 training complete.")


def main():
    parser = argparse.ArgumentParser(description="ECHO PoC Pipeline")
    parser.add_argument("--phase1", action="store_true", help="Run Phase 1 data pipeline")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 baseline models")
    args = parser.parse_args()

    if args.phase1:
        run_phase1()
    elif args.phase2:
        run_phase2()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
