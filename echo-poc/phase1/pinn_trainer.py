"""
PINN training runner for all 40 experimental configurations.

Loads the selected λ triple from lambda_search_result.json, then trains
a PINN on each of the 40 split configurations. Saves model weights,
metrics JSON, and test predictions for each config.

Reference:
    SPEC_PHASE3.md §P3.8 (PINN training procedure)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

from shared import config
from shared.evaluation import build_metrics_json, compute_metrics, save_metrics_json
from phase1.lambda_search import prepare_pinn_data, train_pinn_single_config
from phase1.physics.dielectric import DielectricModel, DobsonDielectric

logger = logging.getLogger(__name__)


def load_lambda_result(lambda_path: Path) -> dict:
    """
    Load lambda_search_result.json.

    Args:
        lambda_path: Path to lambda_search_result.json.

    Returns:
        Dict with 'selected' key containing lambda1, lambda2, lambda3.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required keys are missing.
    """
    if not lambda_path.exists():
        raise FileNotFoundError(f"Lambda search result not found: {lambda_path}")

    with open(lambda_path) as f:
        data = json.load(f)

    if "selected" not in data:
        raise ValueError("lambda_search_result.json missing 'selected' key")
    for key in ["lambda1", "lambda2", "lambda3"]:
        if key not in data["selected"]:
            raise ValueError(f"selected missing '{key}'")

    return data


def train_and_evaluate_single(
    config_idx: int,
    aligned_dataset_path: Path,
    splits_dir: Path,
    test_indices_path: Path,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    model_output_dir: Path,
    metrics_output_dir: Path,
    dielectric_model: DielectricModel | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Train PINN on a single config, evaluate on test set, save all artefacts.

    Args:
        config_idx:          Config index (0–39).
        aligned_dataset_path: Path to aligned_dataset.csv.
        splits_dir:          Path to data/splits/configs/.
        test_indices_path:   Path to test_indices.json.
        lambda1, lambda2, lambda3: Fixed λ triple from search.
        model_output_dir:    Base dir for model saves (e.g. outputs/models/pinn/).
        metrics_output_dir:  Dir for metrics JSON (e.g. outputs/metrics/).
        dielectric_model:    DielectricModel. Default: DobsonDielectric().
        device:              Torch device. Default: config.get_torch_device().

    Returns:
        Metrics dict for this config.
    """
    if dielectric_model is None:
        dielectric_model = DobsonDielectric()

    cfg_path = splits_dir / f"config_{config_idx:03d}.json"
    data = prepare_pinn_data(aligned_dataset_path, cfg_path, test_indices_path)
    cfg = data["config"]

    # Train
    train_result = train_pinn_single_config(
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
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        config_idx=config_idx,
        dielectric_model=dielectric_model,
        device=device,
    )

    model = train_result["model"]

    # Evaluate on test set
    if device is None:
        device = config.get_torch_device()
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(data["X_test"], dtype=torch.float32).to(device)
        ndvi_test_t = torch.tensor(data["ndvi_test"], dtype=torch.float32).to(device)
        theta_test_t = torch.tensor(data["theta_test_rad"], dtype=torch.float32).to(device)
        vv_test_t = torch.tensor(data["vv_db_test_raw"], dtype=torch.float32).to(device)

        test_outputs = model(X_test_t, ndvi_test_t, theta_test_t, vv_test_t)

        y_pred = test_outputs["m_v_final"].cpu().numpy()
        m_v_physics_test = test_outputs["m_v_physics"].cpu().numpy()
        delta_ml_test = test_outputs["delta_ml"].cpu().numpy()
        sigma_wcm_test = test_outputs["sigma_wcm_db"].cpu().numpy()

    y_test = data["y_test"]
    metrics = compute_metrics(y_pred, y_test)

    # Compute physics-only RMSE (m_v_physics vs observed)
    physics_metrics = compute_metrics(m_v_physics_test, y_test)

    # Final diagnostics
    final_residual_ratio = float(
        np.std(delta_ml_test) / (np.std(m_v_physics_test) + 1e-8)
    )
    wcm_forward_rmse = float(
        np.sqrt(np.mean((sigma_wcm_test - data["vv_db_test_raw"]) ** 2))
    )

    # Build metrics JSON (extends Phase 2 schema with physics_diagnostics)
    warnings = []
    if final_residual_ratio > config.RESIDUAL_RATIO_WARN:
        warnings.append(
            f"Residual ratio {final_residual_ratio:.3f} > {config.RESIDUAL_RATIO_WARN} "
            f"— ML branch may dominate physics."
        )

    metrics_dict = build_metrics_json(
        model_name="pinn",
        config_idx=config_idx,
        fraction=cfg["fraction"],
        fraction_label=cfg["fraction_label"],
        rep=cfg["rep"],
        seed_used=cfg["seed_used"],
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        n_test=len(y_test),
        metrics=metrics,
        training_metadata={
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "stopped_at_epoch": train_result["stopped_at_epoch"],
            "best_val_loss": train_result["best_val_loss"],
            "dominance_constraint_satisfied": not train_result["dominance_violated"],
            "stratification_used": cfg.get("stratification_used", False),
        },
        warnings=warnings,
    )

    # Add physics_diagnostics block
    metrics_dict["physics_diagnostics"] = {
        "final_A": model.A.item(),
        "final_B": model.B.item(),
        "final_residual_ratio": final_residual_ratio,
        "residual_ratio_gt1_warning": final_residual_ratio > config.RESIDUAL_RATIO_WARN,
        "wcm_forward_rmse_vs_observed": wcm_forward_rmse,
        "physics_only_rmse": physics_metrics["rmse"],
        "physics_only_r_squared": physics_metrics["r_squared"],
    }

    # Save model weights and scaler
    config_dir = model_output_dir / f"config_{config_idx:03d}"
    config_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, config_dir / "model_weights.pt")
    joblib.dump(data["scaler"], config_dir / "scaler.pkl")

    # Save training history
    with open(config_dir / "training_history.json", "w") as f:
        json.dump(train_result["training_history"], f, indent=2)

    # Save test predictions
    test_predictions = {
        "config_idx": config_idx,
        "m_v_final": y_pred.tolist(),
        "m_v_physics": m_v_physics_test.tolist(),
        "delta_ml": delta_ml_test.tolist(),
        "sigma_wcm_db": sigma_wcm_test.tolist(),
        "y_test": y_test.tolist(),
        "vv_db_test_raw": data["vv_db_test_raw"].tolist(),
        "test_indices": data["test_indices"].tolist(),
    }
    with open(config_dir / "test_predictions.json", "w") as f:
        json.dump(test_predictions, f, indent=2)

    # Save model config
    model_config = {
        "n_features": data["X_train"].shape[1],
        "dielectric_model": type(dielectric_model).__name__,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda3": lambda3,
        "config_idx": config_idx,
        "seed_used": cfg["seed_used"],
        "final_A": model.A.item(),
        "final_B": model.B.item(),
    }
    with open(config_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Save metrics
    metrics_path = metrics_output_dir / f"config_{config_idx:03d}_pinn.json"
    save_metrics_json(metrics_dict, metrics_path)

    logger.info(
        "Config %03d: RMSE=%.4f R²=%.4f A=%.4f B=%.4f residual_ratio=%.3f",
        config_idx, metrics["rmse"], metrics["r_squared"],
        model.A.item(), model.B.item(), final_residual_ratio,
    )

    return metrics_dict


def run_all_configs(
    aligned_dataset_path: Path,
    splits_dir: Path,
    test_indices_path: Path,
    lambda_result_path: Path,
    model_output_dir: Path,
    metrics_output_dir: Path,
    dielectric_model: DielectricModel | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    """
    Train PINN on all 40 configurations.

    Args:
        aligned_dataset_path: Path to aligned_dataset.csv.
        splits_dir:          Path to data/splits/configs/.
        test_indices_path:   Path to test_indices.json.
        lambda_result_path:  Path to lambda_search_result.json.
        model_output_dir:    Base dir for model saves.
        metrics_output_dir:  Dir for metrics JSON.
        dielectric_model:    DielectricModel. Default: DobsonDielectric().
        device:              Torch device. Default: CPU for small batches.

    Returns:
        List of 40 metrics dicts.
    """
    lambda_result = load_lambda_result(lambda_result_path)
    selected = lambda_result["selected"]
    l1, l2, l3 = selected["lambda1"], selected["lambda2"], selected["lambda3"]

    logger.info(
        "Training 40 PINN configs with λ=(%.2f, %.2f, %.2f)",
        l1, l2, l3,
    )

    all_metrics = []
    for idx in range(config.N_CONFIGS):
        metrics = train_and_evaluate_single(
            config_idx=idx,
            aligned_dataset_path=aligned_dataset_path,
            splits_dir=splits_dir,
            test_indices_path=test_indices_path,
            lambda1=l1,
            lambda2=l2,
            lambda3=l3,
            model_output_dir=model_output_dir,
            metrics_output_dir=metrics_output_dir,
            dielectric_model=dielectric_model,
            device=device,
        )
        all_metrics.append(metrics)

    logger.info("All %d PINN configs trained successfully.", config.N_CONFIGS)
    return all_metrics


# ─── CLI entry point ─────────────────────────────────────────────────────────


def main():
    """Run PINN training on all 40 configs from command line."""
    import time

    t0 = time.time()
    device = torch.device("cpu")  # CPU faster for small batches

    all_metrics = run_all_configs(
        aligned_dataset_path=config.DATA_PROCESSED / "aligned_dataset.csv",
        splits_dir=config.DATA_SPLITS / "configs",
        test_indices_path=config.DATA_SPLITS / "test_indices.json",
        lambda_result_path=config.OUTPUTS_MODELS / "pinn" / "lambda_search_result.json",
        model_output_dir=config.OUTPUTS_MODELS / "pinn",
        metrics_output_dir=config.OUTPUTS_METRICS,
        device=device,
    )

    elapsed = time.time() - t0
    rmses = [m["metrics"]["rmse"] for m in all_metrics]
    print(f"\nAll 40 PINN configs trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"RMSE: median={np.median(rmses):.4f} mean={np.mean(rmses):.4f}")


if __name__ == "__main__":
    main()
