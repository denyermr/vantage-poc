"""
Pre-P3.6 gradient flow diagnostic.

Verifies that the physics branch contributes meaningfully to the
computational graph under the selected λ triple (0.01, 0.01, 1.0).

This is a correctness check run BEFORE full 40-config training.
It does not affect gate criteria — it informs how results are described.

Output: plain-language conclusion (active / passive / disconnected)
        saved to outputs/diagnostics/p3_gradient_check.txt

Reference:
    SPEC_PHASE3.md §P3.8 — pre-training diagnostic
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from shared import config
from phase1.lambda_search import prepare_pinn_data
from phase1.physics.dielectric import DobsonDielectric
from phase1.physics.wcm import PINN, compute_pinn_loss

logger = logging.getLogger(__name__)


def run_gradient_check(
    aligned_dataset_path: Path,
    splits_dir: Path,
    test_indices_path: Path,
    lambda_result_path: Path,
    output_path: Path,
) -> dict:
    """
    Run gradient flow diagnostic on a small batch.

    Instantiates PINN with selected λ values, runs a single forward pass
    on the first 10 rows of config_000 training data, computes total loss,
    calls .backward(), and reports gradient magnitudes for each branch.

    Args:
        aligned_dataset_path: Path to aligned_dataset.csv.
        splits_dir:           Path to data/splits/configs/.
        test_indices_path:    Path to test_indices.json.
        lambda_result_path:   Path to lambda_search_result.json.
        output_path:          Path to write diagnostic output.

    Returns:
        Dict with gradient magnitudes, ratio, and conclusion.

    Raises:
        FileNotFoundError: If required files don't exist.
        ValueError: If lambda_search_result.json is malformed.
    """
    # Load selected λ values
    with open(lambda_result_path) as f:
        lambda_result = json.load(f)
    selected = lambda_result["selected"]
    l1 = selected["lambda1"]
    l2 = selected["lambda2"]
    l3 = selected["lambda3"]

    # Load data from config_000
    cfg_path = splits_dir / "config_000.json"
    data = prepare_pinn_data(aligned_dataset_path, cfg_path, test_indices_path)

    # Use first 10 training rows
    n_batch = min(10, len(data["X_train"]))
    device = torch.device("cpu")  # CPU for diagnostic

    torch.manual_seed(config.SEED)
    dielectric_model = DobsonDielectric()
    n_features = data["X_train"].shape[1]
    model = PINN(n_features=n_features, dielectric_model=dielectric_model).to(device)

    # Prepare small batch tensors
    X_batch = torch.tensor(data["X_train"][:n_batch], dtype=torch.float32)
    y_batch = torch.tensor(data["y_train"][:n_batch], dtype=torch.float32)
    ndvi_batch = torch.tensor(data["ndvi_train"][:n_batch], dtype=torch.float32)
    theta_batch = torch.tensor(data["theta_train_rad"][:n_batch], dtype=torch.float32)
    vv_batch = torch.tensor(data["vv_db_train_raw"][:n_batch], dtype=torch.float32)

    # Forward pass
    model.train()
    outputs = model(X_batch, ndvi_batch, theta_batch, vv_batch)
    loss_dict = compute_pinn_loss(
        outputs, y_batch, vv_batch, l1, l2, l3,
        dielectric_model=dielectric_model,
    )

    # Backward pass
    loss_dict["total"].backward()

    # Collect gradient magnitudes by branch
    physics_grad_magnitudes = []
    correction_grad_magnitudes = []
    wcm_grad_magnitudes = []

    for name, param in model.named_parameters():
        if param.grad is None:
            logger.warning("No gradient for parameter: %s", name)
            continue

        grad_mag = param.grad.abs().mean().item()

        if name.startswith("physics_net."):
            physics_grad_magnitudes.append(grad_mag)
        elif name.startswith("correction_net."):
            correction_grad_magnitudes.append(grad_mag)
        elif name in ("A_raw", "B_raw"):
            wcm_grad_magnitudes.append(grad_mag)

    physics_mean = float(np.mean(physics_grad_magnitudes)) if physics_grad_magnitudes else 0.0
    correction_mean = float(np.mean(correction_grad_magnitudes)) if correction_grad_magnitudes else 0.0
    wcm_mean = float(np.mean(wcm_grad_magnitudes)) if wcm_grad_magnitudes else 0.0

    ratio = physics_mean / (correction_mean + 1e-12)

    # Check architecture: does correction_net receive physics branch outputs?
    # By design (SPEC_PHASE3.md §P3.5), both branches receive the same X tensor.
    # The correction_net does NOT receive physics branch intermediate outputs.
    # Physics contributes through: (1) m_v_physics → m_v_final → L_data,
    # (2) WCM forward → L_physics, (3) A/B parameters → L_physics.
    correction_input_shape = X_batch.shape
    physics_outputs_in_correction_input = False  # By architecture design

    # Determine conclusion
    if physics_mean > 0.01 * correction_mean and physics_mean > 1e-8:
        conclusion = "Physics branch active"
        description = (
            "Physics branch receives meaningful gradient signal. "
            f"Ratio = {ratio:.4f} (> 0.01 threshold). "
            "Standard PINN description applies."
        )
    elif physics_mean > 1e-8:
        conclusion = "Physics branch passive"
        description = (
            "Physics branch receives gradient signal but at low magnitude relative to correction branch. "
            f"Ratio = {ratio:.6f} (<= 0.01 threshold). "
            "Physics contributes through architecture (sigmoid-bounded m_v_physics) "
            "and parameter constraints (A/B reparameterisation) rather than forward model gradient signal. "
            "'Physics-constrained architecture' is more accurate than 'physics-informed' for results reporting."
        )
    else:
        conclusion = "Physics branch disconnected"
        description = (
            "Physics branch receives NO gradient signal. "
            "This indicates an architectural bug — do not proceed to P3.6."
        )

    result = {
        "lambda_triple": {"lambda1": l1, "lambda2": l2, "lambda3": l3},
        "n_batch": n_batch,
        "physics_branch_mean_grad": physics_mean,
        "correction_branch_mean_grad": correction_mean,
        "wcm_params_mean_grad": wcm_mean,
        "physics_to_correction_ratio": ratio,
        "correction_net_input_shape": list(correction_input_shape),
        "physics_outputs_in_correction_input": physics_outputs_in_correction_input,
        "loss_components": {
            "total": loss_dict["total"].item(),
            "l_data": loss_dict["l_data"].item(),
            "l_physics": loss_dict["l_physics"].item(),
            "l_monotonic": loss_dict["l_monotonic"].item(),
            "l_bounds": loss_dict["l_bounds"].item(),
        },
        "conclusion": conclusion,
        "description": description,
        "individual_param_grads": {
            name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path = output_path.with_suffix(".txt")
    json_path = output_path.with_suffix(".json")

    # Plain text summary
    lines = [
        "=" * 70,
        "PRE-P3.6 GRADIENT FLOW DIAGNOSTIC",
        "=" * 70,
        "",
        f"Lambda triple: lambda1={l1}, lambda2={l2}, lambda3={l3}",
        f"Batch size: {n_batch} (first {n_batch} rows of config_000 training set)",
        "",
        "GRADIENT MAGNITUDES (mean |grad| across parameters):",
        f"  Physics branch (physics_net):    {physics_mean:.6e}",
        f"  Correction branch (correction_net): {correction_mean:.6e}",
        f"  WCM parameters (A_raw, B_raw):   {wcm_mean:.6e}",
        "",
        f"  Ratio (physics / correction):     {ratio:.6f}",
        "",
        "ARCHITECTURE CHECK:",
        f"  Correction net input shape: {list(correction_input_shape)}",
        f"  Physics outputs in correction input: {physics_outputs_in_correction_input}",
        "  (By design: both branches receive the same normalised feature matrix X.",
        "   Physics contributes through m_v_physics -> m_v_final -> L_data,",
        "   and through WCM forward -> sigma_wcm -> L_physics.)",
        "",
        "LOSS COMPONENTS:",
        f"  L_data:      {loss_dict['l_data'].item():.6f}",
        f"  L_physics:   {loss_dict['l_physics'].item():.6f}  (x lambda1={l1})",
        f"  L_monotonic: {loss_dict['l_monotonic'].item():.6f}  (x lambda2={l2})",
        f"  L_bounds:    {loss_dict['l_bounds'].item():.6f}  (x lambda3={l3})",
        f"  Total:       {loss_dict['total'].item():.6f}",
        "",
        "INDIVIDUAL PARAMETER GRADIENTS:",
    ]

    for name, param in model.named_parameters():
        if param.grad is not None:
            lines.append(f"  {name}: {param.grad.abs().mean().item():.6e}")
        else:
            lines.append(f"  {name}: NO GRADIENT")

    lines.extend([
        "",
        "=" * 70,
        f"CONCLUSION: {conclusion}",
        "=" * 70,
        "",
        description,
        "",
    ])

    text_output = "\n".join(lines)

    with open(txt_path, "w") as f:
        f.write(text_output)

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(text_output)

    return result


def main():
    """Run gradient diagnostic from command line."""
    result = run_gradient_check(
        aligned_dataset_path=config.DATA_PROCESSED / "aligned_dataset.csv",
        splits_dir=config.DATA_SPLITS / "configs",
        test_indices_path=config.DATA_SPLITS / "test_indices.json",
        lambda_result_path=config.OUTPUTS_MODELS / "pinn" / "lambda_search_result.json",
        output_path=config.PROJECT_ROOT / "outputs" / "diagnostics" / "p3_gradient_check",
    )
    return result


if __name__ == "__main__":
    main()
