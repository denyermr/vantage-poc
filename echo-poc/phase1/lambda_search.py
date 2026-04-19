"""
λ hyperparameter search for the PINN composite loss.

Searches over LAMBDA_GRID^3 = 64 combinations of (λ₁, λ₂, λ₃) using
the 10 × 100% training configs (config_idx 0–9). Selects the λ triple
with lowest median validation loss subject to the dominance constraint
(L_data must be the largest single term).

Also provides the core PINN training function used by both the λ search
and the full 40-config experiment (P3.5).

Reference:
    SPEC_PHASE3.md §P3.7 (λ search procedure)
    SPEC_PHASE3.md §P3.8 (PINN training procedure)
"""

import itertools
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from shared import config
from shared.splits import load_config, load_test_indices
from phase1.physics.dielectric import DielectricModel, DobsonDielectric
from phase1.physics.wcm import PINN, compute_pinn_loss

logger = logging.getLogger(__name__)


# ─── PINN training function ─────────────────────────────────────────────────


def train_pinn_single_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vv_db_train_raw: np.ndarray,
    vv_db_val_raw: np.ndarray,
    ndvi_train: np.ndarray,
    ndvi_val: np.ndarray,
    theta_train_rad: np.ndarray,
    theta_val_rad: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    config_idx: int,
    dielectric_model: DielectricModel | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Train a single PINN instance with early stopping.

    Args:
        X_train:         Normalised feature matrix, shape (N_train, n_features).
        y_train:         Target VWC, shape (N_train,). cm³/cm³.
        X_val:           Normalised validation features, shape (N_val, n_features).
        y_val:           Validation targets, shape (N_val,). cm³/cm³.
        vv_db_train_raw: Raw VV backscatter for training (dB, unnormalised).
        vv_db_val_raw:   Raw VV backscatter for validation (dB, unnormalised).
        ndvi_train:      NDVI values for training, shape (N_train,).
        ndvi_val:        NDVI values for validation, shape (N_val,).
        theta_train_rad: Incidence angles for training (radians), shape (N_train,).
        theta_val_rad:   Incidence angles for validation (radians), shape (N_val,).
        lambda1:         Weight for L_physics.
        lambda2:         Weight for L_monotonic.
        lambda3:         Weight for L_bounds.
        config_idx:      Config index (used for seeding).
        dielectric_model: DielectricModel instance. Default: DobsonDielectric().
        device:          Torch device. Default: config.get_torch_device().

    Returns:
        Dict with keys:
            model:              Trained PINN (best checkpoint loaded).
            best_val_loss:      Best validation loss (scalar).
            stopped_at_epoch:   Epoch at which training stopped.
            dominance_violated: Whether L_data was not the dominant term at best epoch.
            training_history:   Per-epoch logs (losses, A, B, residual_ratio).
    """
    if dielectric_model is None:
        dielectric_model = DobsonDielectric()
    if device is None:
        device = config.get_torch_device()

    seed = config.SEED + config_idx
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    n_features = X_train.shape[1]
    model = PINN(n_features=n_features, dielectric_model=dielectric_model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.NN_LR,
        weight_decay=config.NN_WEIGHT_DECAY,
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    vv_train_t = torch.tensor(vv_db_train_raw, dtype=torch.float32).to(device)
    vv_val_t = torch.tensor(vv_db_val_raw, dtype=torch.float32).to(device)
    ndvi_train_t = torch.tensor(ndvi_train, dtype=torch.float32).to(device)
    ndvi_val_t = torch.tensor(ndvi_val, dtype=torch.float32).to(device)
    theta_train_t = torch.tensor(theta_train_rad, dtype=torch.float32).to(device)
    theta_val_t = torch.tensor(theta_val_rad, dtype=torch.float32).to(device)

    batch_size = min(config.NN_BATCH_SIZE, len(X_train))
    train_ds = TensorDataset(
        X_train_t, y_train_t, vv_train_t, ndvi_train_t, theta_train_t
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None
    best_epoch_dominance_violated = False

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "A": [],
        "B": [],
        "residual_ratio": [],
        "l_data": [],
        "l_physics": [],
        "l_monotonic": [],
        "l_bounds": [],
    }

    for epoch in range(config.NN_MAX_EPOCHS):
        # ── Train step ──
        model.train()
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=g
        )

        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b, vv_b, ndvi_b, theta_b in loader:
            optimizer.zero_grad()
            outputs = model(X_b, ndvi_b, theta_b, vv_b)
            loss_dict = compute_pinn_loss(
                outputs, y_b, vv_b, lambda1, lambda2, lambda3,
                dielectric_model=dielectric_model,
            )
            loss_dict["total"].backward()
            optimizer.step()
            epoch_loss += loss_dict["total"].item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation step ──
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t, ndvi_val_t, theta_val_t, vv_val_t)
            val_loss_dict = compute_pinn_loss(
                val_outputs, y_val_t, vv_val_t, lambda1, lambda2, lambda3,
                dielectric_model=dielectric_model,
            )
            val_loss = val_loss_dict["total"].item()

            # Per-epoch diagnostics
            residual_ratio = (
                val_outputs["delta_ml"].std()
                / (val_outputs["m_v_physics"].std() + 1e-8)
            ).item()

        # Check dominance constraint at this epoch
        physics_sum = (
            lambda1 * val_loss_dict["l_physics"].item()
            + lambda2 * val_loss_dict["l_monotonic"].item()
            + lambda3 * val_loss_dict["l_bounds"].item()
        )
        epoch_dominance_violated = physics_sum > val_loss_dict["l_data"].item()

        # Log history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["A"].append(model.A.item())
        history["B"].append(model.B.item())
        history["residual_ratio"].append(residual_ratio)
        history["l_data"].append(val_loss_dict["l_data"].item())
        history["l_physics"].append(val_loss_dict["l_physics"].item())
        history["l_monotonic"].append(val_loss_dict["l_monotonic"].item())
        history["l_bounds"].append(val_loss_dict["l_bounds"].item())

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {
                k: v.clone().cpu() for k, v in model.state_dict().items()
            }
            best_epoch_dominance_violated = epoch_dominance_violated
        else:
            patience_counter += 1

        if patience_counter >= config.NN_PATIENCE:
            logger.info(
                "PINN config_%03d: early stopping at epoch %d (val_loss=%.6f)",
                config_idx, epoch + 1, best_val_loss,
            )
            break

    # Load best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    stopped_at = epoch + 1

    logger.info(
        "PINN config_%03d: stopped=%d best_val=%.6f A=%.4f B=%.4f",
        config_idx, stopped_at, best_val_loss, model.A.item(), model.B.item(),
    )

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "stopped_at_epoch": stopped_at,
        "dominance_violated": best_epoch_dominance_violated,
        "training_history": history,
    }


# ─── Data preparation helpers ───────────────────────────────────────────────


def prepare_pinn_data(
    aligned_dataset_path: Path,
    config_path: Path,
    test_indices_path: Path,
    scaler: StandardScaler | None = None,
) -> dict:
    """
    Load and prepare data for a single PINN training run.

    Fits StandardScaler on training data, extracts physics inputs
    (NDVI, incidence angle in radians, raw VV dB) separately.

    Args:
        aligned_dataset_path: Path to aligned_dataset.csv.
        config_path:          Path to config_NNN.json.
        test_indices_path:    Path to test_indices.json.
        scaler:               Pre-fitted scaler (if None, fits on train).

    Returns:
        Dict with all arrays needed for train_pinn_single_config() plus
        scaler, config metadata, and test set arrays.
    """
    import pandas as pd

    df = pd.read_csv(aligned_dataset_path)
    cfg = load_config(config_path)
    test_info = load_test_indices(test_indices_path)

    train_idx = np.array(cfg["train_indices"])
    val_idx = np.array(cfg["val_indices"])
    split_idx = test_info["split_idx"]
    test_idx = np.arange(split_idx, len(df))

    features = config.FEATURE_COLUMNS
    target = config.TARGET_COLUMN

    X_all = df[features].values
    y_all = df[target].values

    X_train_raw = X_all[train_idx]
    X_val_raw = X_all[val_idx]
    X_test_raw = X_all[test_idx]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    # Extract raw physics inputs BEFORE normalisation
    ndvi_col = features.index("ndvi")
    theta_col = features.index("incidence_angle_mean")
    vv_col = features.index("vv_db")

    ndvi_train = X_train_raw[:, ndvi_col].copy()
    ndvi_val = X_val_raw[:, ndvi_col].copy()
    ndvi_test = X_test_raw[:, ndvi_col].copy()

    theta_train_rad = np.deg2rad(X_train_raw[:, theta_col])
    theta_val_rad = np.deg2rad(X_val_raw[:, theta_col])
    theta_test_rad = np.deg2rad(X_test_raw[:, theta_col])

    vv_db_train_raw = X_train_raw[:, vv_col].copy()
    vv_db_val_raw = X_val_raw[:, vv_col].copy()
    vv_db_test_raw = X_test_raw[:, vv_col].copy()

    # Normalise features
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
    else:
        X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "vv_db_train_raw": vv_db_train_raw,
        "vv_db_val_raw": vv_db_val_raw,
        "vv_db_test_raw": vv_db_test_raw,
        "ndvi_train": ndvi_train,
        "ndvi_val": ndvi_val,
        "ndvi_test": ndvi_test,
        "theta_train_rad": theta_train_rad,
        "theta_val_rad": theta_val_rad,
        "theta_test_rad": theta_test_rad,
        "scaler": scaler,
        "config": cfg,
        "test_indices": test_idx,
        "train_indices": train_idx,
        "val_indices": val_idx,
    }


# ─── λ grid search ──────────────────────────────────────────────────────────


def run_lambda_search(
    aligned_dataset_path: Path,
    splits_dir: Path,
    test_indices_path: Path,
    output_path: Path,
    dielectric_model: DielectricModel | None = None,
) -> dict:
    """
    Search over LAMBDA_GRID^3 combinations to find optimal λ triple.

    Procedure (SPEC_PHASE3.md §P3.7):
        1. Load configs 000–009 (100% training, 10 reps)
        2. For each of 64 λ combos, train 10 PINNs, compute median val loss
        3. Select best λ triple subject to dominance constraint
        4. Write lambda_search_result.json

    Args:
        aligned_dataset_path: Path to aligned_dataset.csv.
        splits_dir:           Path to data/splits/configs/.
        test_indices_path:    Path to test_indices.json.
        output_path:          Path to write lambda_search_result.json.
        dielectric_model:     DielectricModel. Default: DobsonDielectric().

    Returns:
        Result dict (same as written to JSON).
    """
    if dielectric_model is None:
        dielectric_model = DobsonDielectric()

    start_idx, end_idx = config.LAMBDA_SEARCH_CONFIG_RANGE
    config_indices = list(range(start_idx, end_idx + 1))
    logger.info(
        "λ search: %d configs (idx %d–%d), %d combinations",
        len(config_indices), start_idx, end_idx,
        len(config.LAMBDA_GRID) ** 3,
    )

    # Pre-load all config data
    config_data_list = []
    for idx in config_indices:
        cfg_path = splits_dir / f"config_{idx:03d}.json"
        data = prepare_pinn_data(aligned_dataset_path, cfg_path, test_indices_path)
        config_data_list.append(data)

    # Generate all λ combinations
    lambda_combos = list(itertools.product(
        config.LAMBDA_GRID, config.LAMBDA_GRID, config.LAMBDA_GRID
    ))

    results = []
    device = config.get_torch_device()

    for combo_idx, (l1, l2, l3) in enumerate(lambda_combos):
        val_losses = []
        n_violations = 0

        for data in config_data_list:
            cfg = data["config"]
            result = train_pinn_single_config(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                vv_db_train_raw=data["vv_db_train_raw"],
                vv_db_val_raw=data["vv_db_val_raw"],
                ndvi_train=data["ndvi_train"],
                ndvi_val=data["ndvi_val"],
                theta_train_rad=data["theta_train_rad"],
                theta_val_rad=data["theta_val_rad"],
                lambda1=l1,
                lambda2=l2,
                lambda3=l3,
                config_idx=cfg["config_idx"],
                dielectric_model=dielectric_model,
                device=device,
            )
            val_losses.append(result["best_val_loss"])
            if result["dominance_violated"]:
                n_violations += 1

        median_val = float(np.median(val_losses))
        dominance_ok = n_violations == 0

        results.append({
            "lambda1": l1,
            "lambda2": l2,
            "lambda3": l3,
            "median_val_loss": median_val,
            "dominance_satisfied": dominance_ok,
            "n_violations": n_violations,
        })

        logger.info(
            "λ search [%d/%d]: (%.2f, %.2f, %.2f) median_val=%.6f dom_ok=%s",
            combo_idx + 1, len(lambda_combos), l1, l2, l3,
            median_val, dominance_ok,
        )

    # Select best: prefer dominance-satisfying, then lowest median val loss
    satisfying = [r for r in results if r["dominance_satisfied"]]
    if satisfying:
        best = min(satisfying, key=lambda r: r["median_val_loss"])
        all_violating = False
    else:
        logger.warning(
            "All %d λ combinations violate dominance constraint! "
            "Selecting lowest median val_loss regardless.",
            len(results),
        )
        best = min(results, key=lambda r: r["median_val_loss"])
        all_violating = True

    # Top 5 candidates (from dominance-satisfying pool if available)
    pool = satisfying if satisfying else results
    top5 = sorted(pool, key=lambda r: r["median_val_loss"])[:5]
    top5_clean = [
        {
            "lambda1": r["lambda1"],
            "lambda2": r["lambda2"],
            "lambda3": r["lambda3"],
            "median_val_loss": r["median_val_loss"],
        }
        for r in top5
    ]

    n_violating = sum(1 for r in results if not r["dominance_satisfied"])

    result_json = {
        "selected": {
            "lambda1": best["lambda1"],
            "lambda2": best["lambda2"],
            "lambda3": best["lambda3"],
        },
        "median_val_loss": best["median_val_loss"],
        "dominance_constraint_satisfied": best["dominance_satisfied"],
        "all_combinations_violated": all_violating,
        "search_configs_used": config_indices,
        "top5_candidates": top5_clean,
        "n_combinations_searched": len(lambda_combos),
        "n_violating_dominance": n_violating,
        "design_note": (
            "λ searched once on 100% training configs, fixed for all 40 configs. "
            "See SPEC_PHASE3.md §P3.7."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)

    logger.info(
        "λ search complete. Selected: (%.2f, %.2f, %.2f) median_val=%.6f",
        best["lambda1"], best["lambda2"], best["lambda3"],
        best["median_val_loss"],
    )
    logger.info("Top 5 candidates:")
    for i, c in enumerate(top5_clean):
        logger.info(
            "  %d. (%.2f, %.2f, %.2f) median_val=%.6f",
            i + 1, c["lambda1"], c["lambda2"], c["lambda3"],
            c["median_val_loss"],
        )

    return result_json


# ─── CLI entry point ─────────────────────────────────────────────────────────


def main():
    """Run λ search from command line."""
    result = run_lambda_search(
        aligned_dataset_path=config.DATA_PROCESSED / "aligned_dataset.csv",
        splits_dir=config.DATA_SPLITS / "configs",
        test_indices_path=config.DATA_SPLITS / "test_indices.json",
        output_path=config.OUTPUTS_MODELS / "pinn" / "lambda_search_result.json",
    )
    print(f"\nλ search result: {json.dumps(result['selected'], indent=2)}")
    print(f"Median val loss: {result['median_val_loss']:.6f}")
    print(f"Dominance OK: {result['dominance_constraint_satisfied']}")


if __name__ == "__main__":
    main()
