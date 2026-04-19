"""
Split generation for the ECHO PoC experimental design.

Generates 40 train/val configurations (4 training sizes × 10 repetitions)
with season-stratified subsampling from the 83-sample training pool.
The sealed test set is loaded separately and never modified.

Called once at the start of Phase 2. Configs saved to data/splits/configs/.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from shared import config

logger = logging.getLogger(__name__)


def assign_season(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Assign meteorological season label to each date.

    Seasons (UK meteorological standard):
        DJF: Dec, Jan, Feb
        MAM: Mar, Apr, May
        JJA: Jun, Jul, Aug
        SON: Sep, Oct, Nov

    Args:
        dates: DatetimeIndex of observation dates.

    Returns:
        Array of season strings, same length as dates.
    """
    month_to_season = {}
    for season, months in config.SEASONS.items():
        for m in months:
            month_to_season[m] = season
    return np.array([month_to_season[d.month] for d in dates])


def stratified_subsample(
    pool_indices: np.ndarray,
    dates: pd.DatetimeIndex,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a stratified random subsample from pool_indices.

    Stratification by meteorological season. Each season contributes
    proportionally to its representation in the pool.

    If any season has fewer than MIN_STRATUM_SAMPLES samples in the pool,
    falls back to unstratified random sampling and logs a warning.

    Args:
        pool_indices: Integer indices into aligned_dataset (training pool only).
        dates:        Full aligned_dataset date index (all N rows).
        fraction:     Target fraction of pool to retain (e.g. 0.25).
        rng:          NumPy Generator with fixed seed.

    Returns:
        Subsampled indices, length = floor(len(pool_indices) * fraction).
        Minimum 1 sample guaranteed.

    Raises:
        ValueError: If fraction not in (0, 1].
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    n_pool = len(pool_indices)
    n_target = max(1, math.floor(n_pool * fraction))

    # fraction=1.0 means return all
    if n_target >= n_pool:
        return pool_indices.copy()

    seasons = assign_season(dates[pool_indices])
    unique_seasons = ["DJF", "MAM", "JJA", "SON"]

    # Check if stratification is feasible
    season_counts = {s: int(np.sum(seasons == s)) for s in unique_seasons}
    can_stratify = all(
        count >= config.MIN_STRATUM_SAMPLES
        for count in season_counts.values()
        if count > 0
    )

    if not can_stratify:
        logger.warning(
            "Season stratum < MIN_STRATUM_SAMPLES=%d; "
            "falling back to unstratified sampling. Counts: %s",
            config.MIN_STRATUM_SAMPLES,
            season_counts,
        )
        return rng.choice(pool_indices, size=n_target, replace=False)

    # Stratified sampling
    sampled = []
    season_targets = {}
    for s in unique_seasons:
        n_season_pool = season_counts[s]
        if n_season_pool == 0:
            season_targets[s] = 0
            continue
        n_season_target = max(1, round(n_target * n_season_pool / n_pool))
        season_targets[s] = min(n_season_target, n_season_pool)

    # Adjust to hit n_target exactly
    total_allocated = sum(season_targets.values())
    diff = n_target - total_allocated
    if diff != 0:
        # Adjust the largest stratum
        largest_season = max(
            (s for s in unique_seasons if season_counts[s] > 0),
            key=lambda s: season_targets[s],
        )
        season_targets[largest_season] += diff
        # Ensure we don't exceed pool count for that season
        season_targets[largest_season] = min(
            season_targets[largest_season], season_counts[largest_season]
        )
        season_targets[largest_season] = max(1, season_targets[largest_season])

    for s in unique_seasons:
        if season_targets[s] == 0:
            continue
        season_mask = seasons == s
        season_indices = pool_indices[season_mask]
        chosen = rng.choice(season_indices, size=season_targets[s], replace=False)
        sampled.append(chosen)

    result = np.concatenate(sampled)
    np.sort(result)  # maintain chronological order
    return np.sort(result)


def load_test_indices(test_indices_path: Path) -> dict:
    """
    Load sealed test indices from JSON.

    Args:
        test_indices_path: Path to test_indices.json.

    Returns:
        Dict with split_idx, n_train_pool, n_test, etc.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required keys are missing.
    """
    if not test_indices_path.exists():
        raise FileNotFoundError(f"Test indices not found: {test_indices_path}")

    with open(test_indices_path) as f:
        data = json.load(f)

    required_keys = ["split_idx", "n_total", "n_train_pool", "n_test"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"test_indices.json missing keys: {missing}")

    return data


def generate_all_configs(
    aligned_dataset: pd.DataFrame,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Generate all 40 train/val split configurations and write to output_dir.

    Config index assignment (deterministic):
        config_idx = size_idx * N_REPS + rep_idx
        where size_idx in {0,1,2,3} for fractions [1.0, 0.5, 0.25, 0.1]
        and   rep_idx  in {0..9}

    Args:
        aligned_dataset: Full aligned DataFrame with 'date' column.
        test_indices_path: Path to sealed test_indices.json.
        output_dir: Directory to write config JSON files.

    Raises:
        ValueError: If dataset shape doesn't match test indices.
    """
    test_info = load_test_indices(test_indices_path)
    split_idx = test_info["split_idx"]
    n_total = len(aligned_dataset)

    if n_total != test_info["n_total"]:
        raise ValueError(
            f"Dataset has {n_total} rows but test_indices.json says {test_info['n_total']}"
        )

    train_pool = np.arange(split_idx)
    dates = pd.DatetimeIndex(aligned_dataset["date"])

    output_dir.mkdir(parents=True, exist_ok=True)

    configs_summary = []

    for size_idx, fraction in enumerate(config.TRAINING_FRACTIONS):
        for rep_idx in range(config.N_REPS):
            config_idx = size_idx * config.N_REPS + rep_idx
            seed = config.SEED + config_idx
            rng = np.random.default_rng(seed)

            # Subsample training pool
            if fraction >= 1.0:
                train_subsample = train_pool.copy()
            else:
                train_subsample = stratified_subsample(
                    train_pool, dates, fraction, rng
                )

            # Carve validation set from subsampled training set
            n_subsample = len(train_subsample)
            n_val = max(1, round(n_subsample * config.VAL_FRACTION))

            rng_val = np.random.default_rng(seed + 1000)
            val_positions = rng_val.choice(
                n_subsample, size=n_val, replace=False
            )
            val_mask = np.zeros(n_subsample, dtype=bool)
            val_mask[val_positions] = True

            val_indices = train_subsample[val_mask].tolist()
            train_indices = train_subsample[~val_mask].tolist()

            # Season counts for train
            train_seasons = assign_season(dates[train_indices])
            season_counts = {
                s: int(np.sum(train_seasons == s))
                for s in ["DJF", "MAM", "JJA", "SON"]
            }

            # Determine if stratification was used
            stratification_used = (
                fraction < 1.0
                and all(
                    season_counts.get(s, 0) >= config.MIN_STRATUM_SAMPLES
                    for s in ["DJF", "MAM", "JJA", "SON"]
                    if assign_season(dates[train_pool])[
                        np.isin(train_pool, train_pool)
                    ].tolist().count(s) > 0
                )
            )
            # Simpler: if fraction < 1.0, check if any season was too small
            pool_seasons = assign_season(dates[train_pool])
            pool_season_counts = {
                s: int(np.sum(pool_seasons == s))
                for s in ["DJF", "MAM", "JJA", "SON"]
            }
            stratification_used = fraction < 1.0 and all(
                c >= config.MIN_STRATUM_SAMPLES
                for c in pool_season_counts.values()
                if c > 0
            )
            # fraction=1.0 doesn't subsample, so stratification is N/A
            if fraction >= 1.0:
                stratification_used = False

            fraction_label = config.TRAINING_SIZE_LABELS[fraction]

            config_data = {
                "config_idx": config_idx,
                "fraction": fraction,
                "fraction_label": fraction_label,
                "rep": rep_idx,
                "seed_used": seed,
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "train_indices": sorted(train_indices),
                "val_indices": sorted(val_indices),
                "season_counts_train": season_counts,
                "stratification_used": stratification_used,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            config_path = output_dir / f"config_{config_idx:03d}.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            configs_summary.append({
                "config_idx": config_idx,
                "fraction": fraction,
                "rep": rep_idx,
                "n_train": len(train_indices),
                "n_val": len(val_indices),
            })

            logger.info(
                "Config %03d: frac=%.2f rep=%d n_train=%d n_val=%d seed=%d",
                config_idx, fraction, rep_idx,
                len(train_indices), len(val_indices), seed,
            )

    # Write split manifest
    manifest = {
        "n_configs": config.N_CONFIGS,
        "n_test": n_total - split_idx,
        "training_pool_size": split_idx,
        "fractions": config.TRAINING_FRACTIONS,
        "n_reps": config.N_REPS,
        "seed_base": config.SEED,
        "stratification": "meteorological_DJF_MAM_JJA_SON",
        "val_fraction": config.VAL_FRACTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configs_summary": configs_summary,
    }

    manifest_path = output_dir.parent / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Generated %d configs, manifest at %s",
        config.N_CONFIGS, manifest_path,
    )


def load_config(config_path: Path) -> dict:
    """
    Load a single split configuration JSON.

    Args:
        config_path: Path to config_NNN.json.

    Returns:
        Config dict with train_indices, val_indices, etc.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required keys are missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        data = json.load(f)

    required = ["config_idx", "fraction", "train_indices", "val_indices"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Config {config_path.name} missing keys: {missing}")

    return data
