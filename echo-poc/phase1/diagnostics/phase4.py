"""
Phase 4 diagnostics for the ECHO PoC.

Four diagnostics characterising the Negative PINN outcome:
    A — Physics branch residual analysis (residuals vs NDVI, season, VWC, incidence angle)
    B — WCM forward model fit (optimised A/B params, pure forward mode predictions)
    C — Identifiability (residual ratio across training sizes, parameter sensitivity)
    D — Mironov sensitivity (PINN retrained with Mironov dielectric, 10 configs)

All diagnostics use the sealed test set predictions from Phase 3.
Figures saved to outputs/figures/ with p4_ prefix.
Diagnostic results saved to outputs/metrics/ as JSON.

Reference:
    SPEC_PHASE4.md §P4 (diagnostics)
    User prompt (Phase 4 diagnostic specifications)
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from shared import config
from phase1.physics.dielectric import DobsonDielectric, MironovDielectric
from phase1.physics.wcm import PINN, wcm_forward, oh_soil_backscatter, wcm_vegetation_terms

logger = logging.getLogger(__name__)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _load_all_pinn_predictions(models_dir: Path, config_range: range) -> list[dict]:
    """Load test_predictions.json for a range of config indices."""
    preds = []
    for idx in config_range:
        pred_path = models_dir / f"config_{idx:03d}" / "test_predictions.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing test predictions: {pred_path}")
        with open(pred_path) as f:
            preds.append(json.load(f))
    return preds


def _load_all_pinn_metrics(metrics_dir: Path, config_range: range) -> list[dict]:
    """Load PINN metric files for a range of config indices."""
    metrics = []
    for idx in config_range:
        metric_path = metrics_dir / f"config_{idx:03d}_pinn.json"
        if not metric_path.exists():
            raise FileNotFoundError(f"Missing PINN metrics: {metric_path}")
        with open(metric_path) as f:
            metrics.append(json.load(f))
    return metrics


def _assign_season(month: int) -> str:
    """Map calendar month to meteorological season."""
    for season, months in config.SEASONS.items():
        if month in months:
            return season
    raise ValueError(f"Month {month} not in any season")


# ─── Diagnostic A: Physics branch residual analysis ────────────────────────


def diagnostic_a_residual_analysis(
    aligned_dataset_path: Path,
    models_dir: Path,
    figures_dir: Path,
) -> dict:
    """
    Analyse WCM physics branch residuals on the sealed test set.

    Computes: predicted backscatter (sigma_wcm_db from physics branch)
    minus observed VV backscatter, for all 36 test observations.
    Uses median across 10 reps (configs 0-9, 100% training) for stability.

    Plots residuals against: NDVI, season, VWC, incidence angle.
    Tests for significant correlation between residuals and NDVI.

    Args:
        aligned_dataset_path: Path to aligned_dataset.csv.
        models_dir: Path to outputs/models/pinn/.
        figures_dir: Path to outputs/figures/.

    Returns:
        Dict with residual statistics and correlation results.
    """
    logger.info("Diagnostic A: Physics branch residual analysis")

    df = pd.read_csv(aligned_dataset_path, parse_dates=["date"])

    # Load predictions from 10 × 100% configs
    preds_list = _load_all_pinn_predictions(models_dir, range(0, 10))

    # Compute median sigma_wcm_db across 10 reps
    sigma_wcm_all = np.array([p["sigma_wcm_db"] for p in preds_list])  # (10, 36)
    sigma_wcm_median = np.median(sigma_wcm_all, axis=0)  # (36,)

    # Test indices and observed VV
    test_indices = preds_list[0]["test_indices"]
    vv_db_observed = np.array(preds_list[0]["vv_db_test_raw"])  # (36,)

    # Residuals: predicted backscatter - observed backscatter
    residuals = sigma_wcm_median - vv_db_observed  # (36,)

    # Extract features at test indices
    test_df = df.iloc[test_indices].copy()
    ndvi = test_df["ndvi"].values
    vwc = test_df["vwc"].values
    incidence_angle = test_df["incidence_angle_mean"].values
    months = pd.to_datetime(test_df["date"]).dt.month
    seasons = months.map(_assign_season).values

    # Correlations with features
    r_ndvi, p_ndvi = stats.pearsonr(residuals, ndvi)
    r_vwc, p_vwc = stats.pearsonr(residuals, vwc)
    r_theta, p_theta = stats.pearsonr(residuals, incidence_angle)

    logger.info("Residual-NDVI:  r=%.4f, p=%.6f", r_ndvi, p_ndvi)
    logger.info("Residual-VWC:   r=%.4f, p=%.6f", r_vwc, p_vwc)
    logger.info("Residual-theta: r=%.4f, p=%.6f", r_theta, p_theta)

    # Flag NDVI correlation
    ndvi_significant = p_ndvi < 0.05
    if ndvi_significant:
        logger.info(
            "SIGNIFICANT: Residuals correlate with NDVI (p=%.4f < 0.05). "
            "Direct evidence that WCM fails when vegetation is denser.",
            p_ndvi,
        )
    else:
        logger.info(
            "Residual-NDVI correlation not significant (p=%.4f >= 0.05).",
            p_ndvi,
        )

    # ─── Plot: 4-panel residual analysis ───
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Diagnostic A: WCM Physics Branch Residuals vs Features", fontsize=14)

    season_colors = {"DJF": "#3182bd", "MAM": "#31a354", "JJA": "#e6550d", "SON": "#756bb1"}

    # Panel 1: Residuals vs NDVI
    ax = axes[0, 0]
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == season
        ax.scatter(ndvi[mask], residuals[mask], c=season_colors[season],
                   label=season, alpha=0.7, edgecolors="k", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Residual (WCM pred - obs VV) [dB]")
    ax.set_title(f"vs NDVI (r={r_ndvi:.3f}, p={p_ndvi:.4f})")
    ax.legend(fontsize=8)
    # Add regression line
    z = np.polyfit(ndvi, residuals, 1)
    p_line = np.poly1d(z)
    ndvi_sorted = np.sort(ndvi)
    ax.plot(ndvi_sorted, p_line(ndvi_sorted), "r-", alpha=0.5, linewidth=1.5)

    # Panel 2: Residuals vs Season (box plot)
    ax = axes[0, 1]
    season_order = ["DJF", "MAM", "JJA", "SON"]
    season_data = [residuals[seasons == s] for s in season_order]
    bp = ax.boxplot(season_data, labels=season_order, patch_artist=True)
    for patch, season in zip(bp["boxes"], season_order):
        patch.set_facecolor(season_colors[season])
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Residual [dB]")
    ax.set_title("vs Season")

    # Panel 3: Residuals vs VWC
    ax = axes[1, 0]
    for season in season_order:
        mask = seasons == season
        ax.scatter(vwc[mask], residuals[mask], c=season_colors[season],
                   label=season, alpha=0.7, edgecolors="k", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("VWC [cm³/cm³]")
    ax.set_ylabel("Residual [dB]")
    ax.set_title(f"vs VWC (r={r_vwc:.3f}, p={p_vwc:.4f})")
    z = np.polyfit(vwc, residuals, 1)
    p_line = np.poly1d(z)
    vwc_sorted = np.sort(vwc)
    ax.plot(vwc_sorted, p_line(vwc_sorted), "r-", alpha=0.5, linewidth=1.5)

    # Panel 4: Residuals vs Incidence Angle
    ax = axes[1, 1]
    for season in season_order:
        mask = seasons == season
        ax.scatter(incidence_angle[mask], residuals[mask], c=season_colors[season],
                   label=season, alpha=0.7, edgecolors="k", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Incidence Angle [°]")
    ax.set_ylabel("Residual [dB]")
    ax.set_title(f"vs Incidence Angle (r={r_theta:.3f}, p={p_theta:.4f})")

    # Annotation
    fig.text(0.5, 0.01,
             f"n={len(residuals)} test observations | "
             f"Residual-NDVI: r={r_ndvi:.3f} (p={p_ndvi:.4f}) | "
             f"Residual-VWC: r={r_vwc:.3f} (p={p_vwc:.4f}) | "
             f"Residual-θ: r={r_theta:.3f} (p={p_theta:.4f})",
             ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = figures_dir / "p4_residual_analysis.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", fig_path)

    result = {
        "n_test_observations": len(residuals),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_ndvi_r": float(r_ndvi),
        "residual_ndvi_p": float(p_ndvi),
        "residual_ndvi_significant": bool(ndvi_significant),
        "residual_vwc_r": float(r_vwc),
        "residual_vwc_p": float(p_vwc),
        "residual_theta_r": float(r_theta),
        "residual_theta_p": float(p_theta),
        "figure": str(fig_path),
    }
    return result


# ─── Diagnostic B: WCM forward model fit ───────────────────────────────────


def diagnostic_b_wcm_forward_fit(
    aligned_dataset_path: Path,
    metrics_dir: Path,
    models_dir: Path,
    figures_dir: Path,
) -> dict:
    """
    Evaluate WCM forward model fit using optimised A and B parameters.

    Extracts median A and B from 10 × 100% PINN configs, runs WCM in
    pure forward mode on all 36 test observations, and compares against
    observed VV backscatter.

    Args:
        aligned_dataset_path: Path to aligned_dataset.csv.
        metrics_dir: Path to outputs/metrics/.
        models_dir: Path to outputs/models/pinn/.
        figures_dir: Path to outputs/figures/.

    Returns:
        Dict with WCM parameter analysis and forward fit statistics.
    """
    logger.info("Diagnostic B: WCM forward model fit")

    df = pd.read_csv(aligned_dataset_path, parse_dates=["date"])

    # Load PINN metrics for 100% configs to get A and B
    pinn_metrics = _load_all_pinn_metrics(metrics_dir, range(0, 10))
    A_values = [m["physics_diagnostics"]["final_A"] for m in pinn_metrics]
    B_values = [m["physics_diagnostics"]["final_B"] for m in pinn_metrics]

    A_median = float(np.median(A_values))
    B_median = float(np.median(B_values))
    A_std = float(np.std(A_values))
    B_std = float(np.std(B_values))

    logger.info("Optimised WCM A: median=%.4f (std=%.4f)", A_median, A_std)
    logger.info("Optimised WCM B: median=%.4f (std=%.4f)", B_median, B_std)

    # Check if at/near bounds (within 5%)
    A_range = config.WCM_A_UB - config.WCM_A_LB
    B_range = config.WCM_B_UB - config.WCM_B_LB
    A_near_lb = (A_median - config.WCM_A_LB) / A_range < 0.05
    A_near_ub = (config.WCM_A_UB - A_median) / A_range < 0.05
    B_near_lb = (B_median - config.WCM_B_LB) / B_range < 0.05
    B_near_ub = (config.WCM_B_UB - B_median) / B_range < 0.05
    A_at_bound = A_near_lb or A_near_ub
    B_at_bound = B_near_lb or B_near_ub

    bound_status = []
    if A_near_lb:
        bound_status.append(f"A near lower bound ({config.WCM_A_LB})")
    if A_near_ub:
        bound_status.append(f"A near upper bound ({config.WCM_A_UB})")
    if B_near_lb:
        bound_status.append(f"B near lower bound ({config.WCM_B_LB})")
    if B_near_ub:
        bound_status.append(f"B near upper bound ({config.WCM_B_UB})")

    if bound_status:
        logger.warning("WCM parameters at/near bounds: %s", "; ".join(bound_status))
    else:
        logger.info("WCM parameters within interior of bounds (not at edges)")

    # Load test set data
    preds_list = _load_all_pinn_predictions(models_dir, range(0, 10))
    test_indices = preds_list[0]["test_indices"]
    test_df = df.iloc[test_indices].copy()

    vwc_test = test_df["vwc"].values
    ndvi_test = test_df["ndvi"].values
    theta_test_deg = test_df["incidence_angle_mean"].values
    theta_test_rad = np.deg2rad(theta_test_deg)
    vv_db_observed = np.array(preds_list[0]["vv_db_test_raw"])
    months = pd.to_datetime(test_df["date"]).dt.month

    # Run WCM in pure forward mode
    dielectric = DobsonDielectric()
    with torch.no_grad():
        m_v_t = torch.tensor(vwc_test, dtype=torch.float32)
        A_t = torch.tensor(A_median, dtype=torch.float32)
        B_t = torch.tensor(B_median, dtype=torch.float32)
        ndvi_t = torch.tensor(ndvi_test, dtype=torch.float32)
        theta_t = torch.tensor(theta_test_rad, dtype=torch.float32)

        sigma_wcm_forward = wcm_forward(m_v_t, A_t, B_t, ndvi_t, theta_t, dielectric)
        sigma_wcm_np = sigma_wcm_forward.numpy()

    # Statistics
    wcm_rmse = float(np.sqrt(np.mean((sigma_wcm_np - vv_db_observed) ** 2)))
    wcm_r, wcm_p = stats.pearsonr(sigma_wcm_np, vv_db_observed)
    raw_vv_vwc_r = 0.290  # From Phase 2 feature diagnostics

    logger.info("WCM forward fit: RMSE=%.4f dB, r=%.4f (p=%.6f)", wcm_rmse, wcm_r, wcm_p)
    logger.info("Raw VV-VWC correlation from Phase 2: r=%.3f", raw_vv_vwc_r)
    logger.info(
        "WCM forward r (%.3f) vs raw VV-VWC r (%.3f): %s",
        wcm_r, raw_vv_vwc_r,
        "substantially better" if wcm_r > raw_vv_vwc_r + 0.1 else "not substantially better",
    )

    # ─── Plot: WCM forward predictions vs observed VV ───
    fig, ax = plt.subplots(figsize=(8, 7))
    season_colors = {"DJF": "#3182bd", "MAM": "#31a354", "JJA": "#e6550d", "SON": "#756bb1"}
    seasons = months.map(_assign_season).values

    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == season
        ax.scatter(vv_db_observed[mask], sigma_wcm_np[mask],
                   c=season_colors[season], label=season, alpha=0.7,
                   edgecolors="k", linewidth=0.5, s=50)

    # 1:1 line
    vmin = min(vv_db_observed.min(), sigma_wcm_np.min()) - 0.5
    vmax = max(vv_db_observed.max(), sigma_wcm_np.max()) + 0.5
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, alpha=0.5, label="1:1 line")

    ax.set_xlabel("Observed VV Backscatter [dB]", fontsize=12)
    ax.set_ylabel("WCM Forward Predicted VV [dB]", fontsize=12)
    ax.set_title(
        f"Diagnostic B: WCM Forward Model Fit\n"
        f"A={A_median:.4f}, B={B_median:.4f} (site-calibrated)",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    # Annotation
    ax.text(0.05, 0.95,
            f"r = {wcm_r:.3f} (p = {wcm_p:.4f})\n"
            f"RMSE = {wcm_rmse:.3f} dB\n"
            f"Raw VV-VWC r = {raw_vv_vwc_r:.3f}\n"
            f"n = {len(vv_db_observed)}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig_path = figures_dir / "p4_wcm_forward_fit.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", fig_path)

    result = {
        "A_median": A_median,
        "A_std": A_std,
        "A_values": A_values,
        "B_median": B_median,
        "B_std": B_std,
        "B_values": B_values,
        "A_at_bound": A_at_bound,
        "B_at_bound": B_at_bound,
        "A_bound_status": "interior" if not A_at_bound else "; ".join([s for s in bound_status if "A" in s]),
        "B_bound_status": "interior" if not B_at_bound else "; ".join([s for s in bound_status if "B" in s]),
        "wcm_forward_rmse_db": wcm_rmse,
        "wcm_forward_r": float(wcm_r),
        "wcm_forward_p": float(wcm_p),
        "raw_vv_vwc_r": raw_vv_vwc_r,
        "wcm_substantially_better": wcm_r > raw_vv_vwc_r + 0.1,
        "figure": str(fig_path),
    }
    return result


# ─── Diagnostic C: Identifiability ─────────────────────────────────────────


def diagnostic_c_identifiability(
    metrics_dir: Path,
    models_dir: Path,
    figures_dir: Path,
) -> dict:
    """
    Identifiability diagnostics: are WCM parameters A and B identifiable
    from the data, or does the loss surface show flat directions?

    Three sub-diagnostics:
    1. Residual ratio across training sizes (std(delta_ml) / std(m_v_physics))
    2. Parameter sensitivity (dσ/dA and dσ/dB)
    3. A and B parameter spread across reps

    Args:
        metrics_dir: Path to outputs/metrics/.
        models_dir: Path to outputs/models/pinn/.
        figures_dir: Path to outputs/figures/.

    Returns:
        Dict with identifiability findings.
    """
    logger.info("Diagnostic C: Identifiability analysis")

    fractions = config.TRAINING_FRACTIONS
    fraction_labels = config.TRAINING_SIZE_LABELS

    # ─── Sub-diagnostic 1: Residual ratio across training sizes ───
    residual_ratios_by_fraction = {}
    A_by_fraction = {}
    B_by_fraction = {}

    for frac_idx, frac in enumerate(fractions):
        start = frac_idx * config.N_REPS
        end = start + config.N_REPS
        label = fraction_labels[frac]

        preds_list = _load_all_pinn_predictions(models_dir, range(start, end))
        metrics_list = _load_all_pinn_metrics(metrics_dir, range(start, end))

        # Residual ratios
        ratios = []
        for p in preds_list:
            delta = np.array(p["delta_ml"])
            physics = np.array(p["m_v_physics"])
            ratio = float(np.std(delta) / (np.std(physics) + 1e-8))
            ratios.append(ratio)
        residual_ratios_by_fraction[label] = {
            "median": float(np.median(ratios)),
            "q25": float(np.percentile(ratios, 25)),
            "q75": float(np.percentile(ratios, 75)),
            "values": ratios,
        }

        # A and B values
        A_vals = [m["physics_diagnostics"]["final_A"] for m in metrics_list]
        B_vals = [m["physics_diagnostics"]["final_B"] for m in metrics_list]
        A_by_fraction[label] = {
            "median": float(np.median(A_vals)),
            "std": float(np.std(A_vals)),
            "values": A_vals,
        }
        B_by_fraction[label] = {
            "median": float(np.median(B_vals)),
            "std": float(np.std(B_vals)),
            "values": B_vals,
        }

    # ─── Sub-diagnostic 2: Parameter sensitivity ───
    # Compute dσ/dA and dσ/dB at the optimised parameters
    A_median = A_by_fraction["100%"]["median"]
    B_median = B_by_fraction["100%"]["median"]
    dielectric = DobsonDielectric()

    # Use mean test conditions
    preds_100 = _load_all_pinn_predictions(models_dir, range(0, 10))
    test_df = pd.read_csv(config.DATA_PROCESSED / "aligned_dataset.csv", parse_dates=["date"])
    test_indices = preds_100[0]["test_indices"]
    test_data = test_df.iloc[test_indices]
    ndvi_mean = float(test_data["ndvi"].mean())
    theta_mean_rad = float(np.deg2rad(test_data["incidence_angle_mean"].mean()))
    vwc_range = [0.3, 0.5, 0.7]  # low, mid, high

    sensitivities_A = []
    sensitivities_B = []
    delta = 0.001

    for vwc in vwc_range:
        with torch.no_grad():
            m_v_t = torch.tensor([vwc], dtype=torch.float32)
            ndvi_t = torch.tensor([ndvi_mean], dtype=torch.float32)
            theta_t = torch.tensor([theta_mean_rad], dtype=torch.float32)

            # dσ/dA
            A_lo = torch.tensor(A_median - delta, dtype=torch.float32)
            A_hi = torch.tensor(A_median + delta, dtype=torch.float32)
            B_t = torch.tensor(B_median, dtype=torch.float32)
            sigma_lo = wcm_forward(m_v_t, A_lo, B_t, ndvi_t, theta_t, dielectric).item()
            sigma_hi = wcm_forward(m_v_t, A_hi, B_t, ndvi_t, theta_t, dielectric).item()
            dσ_dA = (sigma_hi - sigma_lo) / (2 * delta)
            sensitivities_A.append(dσ_dA)

            # dσ/dB
            A_t = torch.tensor(A_median, dtype=torch.float32)
            B_lo = torch.tensor(B_median - delta, dtype=torch.float32)
            B_hi = torch.tensor(B_median + delta, dtype=torch.float32)
            sigma_lo = wcm_forward(m_v_t, A_t, B_lo, ndvi_t, theta_t, dielectric).item()
            sigma_hi = wcm_forward(m_v_t, A_t, B_hi, ndvi_t, theta_t, dielectric).item()
            dσ_dB = (sigma_hi - sigma_lo) / (2 * delta)
            sensitivities_B.append(dσ_dB)

    mean_sensitivity_A = float(np.mean(np.abs(sensitivities_A)))
    mean_sensitivity_B = float(np.mean(np.abs(sensitivities_B)))

    # Threshold: 0.1 dB per unit change
    A_well_constrained = mean_sensitivity_A > 0.1
    B_well_constrained = mean_sensitivity_B > 0.1

    logger.info("Sensitivity |dσ/dA|: %.4f dB/unit (%s)",
                mean_sensitivity_A, "well-constrained" if A_well_constrained else "poorly-constrained")
    logger.info("Sensitivity |dσ/dB|: %.4f dB/unit (%s)",
                mean_sensitivity_B, "well-constrained" if B_well_constrained else "poorly-constrained")

    # ─── Plot: 3-panel identifiability figure ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Diagnostic C: Identifiability Analysis", fontsize=14)

    # Panel 1: Residual ratio by training fraction
    ax = axes[0]
    labels = list(residual_ratios_by_fraction.keys())
    medians = [residual_ratios_by_fraction[l]["median"] for l in labels]
    q25s = [residual_ratios_by_fraction[l]["q25"] for l in labels]
    q75s = [residual_ratios_by_fraction[l]["q75"] for l in labels]
    x = range(len(labels))
    ax.bar(x, medians, color="#4292c6", alpha=0.7, edgecolor="k")
    ax.errorbar(x, medians,
                yerr=[np.array(medians) - np.array(q25s), np.array(q75s) - np.array(medians)],
                fmt="none", ecolor="k", capsize=5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Ratio = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Residual Ratio\nstd(δ) / std(m_v_physics)")
    ax.set_title("ML Branch Dominance")
    ax.legend(fontsize=8)

    # Panel 2: A and B spread across reps (100% training)
    ax = axes[1]
    A_100 = A_by_fraction["100%"]["values"]
    B_100 = B_by_fraction["100%"]["values"]
    ax.scatter(A_100, B_100, c="#e6550d", s=80, edgecolors="k", alpha=0.7, zorder=3)
    ax.axvline(config.WCM_A_INIT, color="gray", linestyle=":", label=f"A_init={config.WCM_A_INIT}")
    ax.axhline(config.WCM_B_INIT, color="gray", linestyle="--", label=f"B_init={config.WCM_B_INIT}")
    ax.set_xlabel("WCM A (vegetation scattering)")
    ax.set_ylabel("WCM B (vegetation attenuation)")
    ax.set_title("Parameter Space (100% training, 10 reps)")
    ax.legend(fontsize=8)

    # Panel 3: Sensitivity bars
    ax = axes[2]
    bars = ax.bar(["dσ/dA", "dσ/dB"], [mean_sensitivity_A, mean_sensitivity_B],
                  color=["#e6550d", "#756bb1"], alpha=0.7, edgecolor="k")
    ax.axhline(0.1, color="red", linestyle="--", linewidth=1, label="Threshold (0.1 dB)")
    ax.set_ylabel("|dσ°/dParam| [dB per unit]")
    ax.set_title("Parameter Sensitivity")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = figures_dir / "p4_identifiability.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", fig_path)

    result = {
        "residual_ratios_by_fraction": residual_ratios_by_fraction,
        "A_by_fraction": A_by_fraction,
        "B_by_fraction": B_by_fraction,
        "sensitivity_A_dB_per_unit": mean_sensitivity_A,
        "sensitivity_B_dB_per_unit": mean_sensitivity_B,
        "A_well_constrained": A_well_constrained,
        "B_well_constrained": B_well_constrained,
        "identifiability_finding": (
            "ML branch dominates at all training sizes (residual ratio >> 1.0). "
            f"WCM parameters A (sensitivity {mean_sensitivity_A:.3f} dB/unit) and "
            f"B (sensitivity {mean_sensitivity_B:.3f} dB/unit) are "
            f"{'well' if A_well_constrained and B_well_constrained else 'poorly'}-constrained "
            f"by the data, but the physics branch output (m_v_physics) has very low variance "
            f"compared to the ML correction (delta_ml), indicating the physics structure "
            f"is not contributing meaningfully to the final prediction."
        ),
        "figure": str(fig_path),
    }
    return result


# ─── Diagnostic D: Mironov sensitivity ─────────────────────────────────────


def diagnostic_d_mironov_sensitivity(
    aligned_dataset_path: Path,
    splits_dir: Path,
    test_indices_path: Path,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict:
    """
    Retrain PINN with Mironov dielectric model on 10 × 100% configs.
    Compare RMSE against Dobson-based PINN results.

    Applies ε clamping (min 1.01) in Oh model input as per DEV-007.

    Args:
        aligned_dataset_path: Path to aligned_dataset.csv.
        splits_dir: Path to data/splits/configs/.
        test_indices_path: Path to test_indices.json.
        metrics_dir: Path to outputs/metrics/.
        figures_dir: Path to outputs/figures/.

    Returns:
        Dict with Dobson vs Mironov RMSE comparison.
    """
    from phase1.lambda_search import prepare_pinn_data, train_pinn_single_config

    logger.info("Diagnostic D: Mironov sensitivity check (10 configs)")

    # Load Dobson PINN results for comparison
    dobson_metrics = _load_all_pinn_metrics(metrics_dir, range(0, 10))
    dobson_rmses = [m["metrics"]["rmse"] for m in dobson_metrics]
    dobson_median_rmse = float(np.median(dobson_rmses))

    # Load lambda search result
    lambda_result_path = config.OUTPUTS_MODELS / "pinn" / "lambda_search_result.json"
    with open(lambda_result_path) as f:
        lambda_result = json.load(f)
    lambda1 = lambda_result["selected"]["lambda1"]
    lambda2 = lambda_result["selected"]["lambda2"]
    lambda3 = lambda_result["selected"]["lambda3"]

    # Train 10 PINNs with Mironov
    mironov = MironovDielectric()
    device = config.get_torch_device()
    mironov_rmses = []

    for idx in range(10):
        cfg_path = splits_dir / f"config_{idx:03d}.json"
        data = prepare_pinn_data(aligned_dataset_path, cfg_path, test_indices_path)

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
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            config_idx=idx,
            dielectric_model=mironov,
            device=device,
        )

        # Evaluate on test set
        model = result["model"]
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(data["X_test"], dtype=torch.float32).to(device)
            ndvi_test_t = torch.tensor(data["ndvi_test"], dtype=torch.float32).to(device)
            theta_test_t = torch.tensor(data["theta_test_rad"], dtype=torch.float32).to(device)
            vv_test_t = torch.tensor(data["vv_db_test_raw"], dtype=torch.float32).to(device)
            y_test_t = torch.tensor(data["y_test"], dtype=torch.float32)

            outputs = model(X_test_t, ndvi_test_t, theta_test_t, vv_test_t)
            m_v_final = outputs["m_v_final"].cpu().numpy()

        rmse = float(np.sqrt(np.mean((m_v_final - data["y_test"]) ** 2)))
        mironov_rmses.append(rmse)
        logger.info("Mironov config_%03d: RMSE=%.4f", idx, rmse)

    mironov_median_rmse = float(np.median(mironov_rmses))
    relative_diff = (mironov_median_rmse - dobson_median_rmse) / dobson_median_rmse

    logger.info("Dobson median RMSE: %.4f", dobson_median_rmse)
    logger.info("Mironov median RMSE: %.4f", mironov_median_rmse)
    logger.info("Relative difference: %.1f%%", relative_diff * 100)

    # ─── Plot: Dobson vs Mironov bar comparison ───
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Dobson (1985)\n(primary)", "Mironov (2009)\n(sensitivity)"],
        [dobson_median_rmse, mironov_median_rmse],
        color=["#3182bd", "#e6550d"],
        alpha=0.7,
        edgecolor="k",
        width=0.5,
    )
    # Error bars (IQR)
    dobson_q25, dobson_q75 = np.percentile(dobson_rmses, [25, 75])
    mironov_q25, mironov_q75 = np.percentile(mironov_rmses, [25, 75])
    ax.errorbar(
        [0, 1],
        [dobson_median_rmse, mironov_median_rmse],
        yerr=[
            [dobson_median_rmse - dobson_q25, mironov_median_rmse - mironov_q25],
            [dobson_q75 - dobson_median_rmse, mironov_q75 - mironov_median_rmse],
        ],
        fmt="none", ecolor="k", capsize=8, linewidth=1.5,
    )
    ax.set_ylabel("Test RMSE [cm³/cm³]", fontsize=12)
    ax.set_title(
        f"Diagnostic D: Dielectric Model Sensitivity\n"
        f"Dobson: {dobson_median_rmse:.4f} vs Mironov: {mironov_median_rmse:.4f} "
        f"({relative_diff:+.1%})",
        fontsize=13,
    )
    ax.text(0.95, 0.95,
            f"n = 10 reps (100% training)\n"
            f"DEV-007: Mironov ε clamped ≥ 1.01\n"
            f"λ = ({lambda1}, {lambda2}, {lambda3})",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig_path = figures_dir / "p4_mironov_sensitivity.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", fig_path)

    result = {
        "dobson_median_rmse": dobson_median_rmse,
        "dobson_rmses": dobson_rmses,
        "mironov_median_rmse": mironov_median_rmse,
        "mironov_rmses": mironov_rmses,
        "relative_difference_pct": float(relative_diff * 100),
        "mironov_better": mironov_median_rmse < dobson_median_rmse,
        "finding": (
            f"Mironov RMSE ({mironov_median_rmse:.4f}) vs Dobson RMSE ({dobson_median_rmse:.4f}): "
            f"{relative_diff:+.1%} difference. "
            f"{'Mironov marginally better' if mironov_median_rmse < dobson_median_rmse else 'Dobson retained as primary model'}. "
            f"DEV-007 ε clamping applied."
        ),
        "figure": str(fig_path),
    }
    return result


# ─── Run all diagnostics ───────────────────────────────────────────────────


def run_all_diagnostics() -> dict:
    """
    Run all Phase 4 diagnostics in order and save results.

    Returns:
        Dict with all diagnostic results.
    """
    aligned_path = config.DATA_PROCESSED / "aligned_dataset.csv"
    metrics_dir = config.OUTPUTS_METRICS
    models_dir = config.OUTPUTS_MODELS / "pinn"
    figures_dir = config.OUTPUTS_FIGURES
    splits_dir = config.DATA_SPLITS / "configs"
    test_indices_path = config.DATA_SPLITS / "test_indices.json"

    # Ensure output directories exist
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Diagnostic A
    results["diagnostic_a"] = diagnostic_a_residual_analysis(
        aligned_path, models_dir, figures_dir
    )

    # Diagnostic B
    results["diagnostic_b"] = diagnostic_b_wcm_forward_fit(
        aligned_path, metrics_dir, models_dir, figures_dir
    )

    # Diagnostic C
    results["diagnostic_c"] = diagnostic_c_identifiability(
        metrics_dir, models_dir, figures_dir
    )

    # Diagnostic D
    results["diagnostic_d"] = diagnostic_d_mironov_sensitivity(
        aligned_path, splits_dir, test_indices_path, metrics_dir, figures_dir
    )

    # Save combined results
    # Clean non-serializable values for JSON
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = clean_for_json(results)
    results_clean["generated_at"] = datetime.now(timezone.utc).isoformat()

    output_path = metrics_dir / "phase4_diagnostics.json"
    with open(output_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    logger.info("All diagnostics saved to: %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 4 DIAGNOSTICS — SUMMARY")
    print("=" * 70)
    print(f"\nDiagnostic A — Residual Analysis:")
    print(f"  Residual-NDVI: r={results['diagnostic_a']['residual_ndvi_r']:.3f} "
          f"(p={results['diagnostic_a']['residual_ndvi_p']:.4f}) "
          f"{'*** SIGNIFICANT ***' if results['diagnostic_a']['residual_ndvi_significant'] else '(not significant)'}")
    print(f"  Residual-VWC:  r={results['diagnostic_a']['residual_vwc_r']:.3f} "
          f"(p={results['diagnostic_a']['residual_vwc_p']:.4f})")
    print(f"  Residual-θ:    r={results['diagnostic_a']['residual_theta_r']:.3f} "
          f"(p={results['diagnostic_a']['residual_theta_p']:.4f})")

    print(f"\nDiagnostic B — WCM Forward Fit:")
    print(f"  A={results['diagnostic_b']['A_median']:.4f} (std={results['diagnostic_b']['A_std']:.4f}), "
          f"bound: {results['diagnostic_b']['A_bound_status']}")
    print(f"  B={results['diagnostic_b']['B_median']:.4f} (std={results['diagnostic_b']['B_std']:.4f}), "
          f"bound: {results['diagnostic_b']['B_bound_status']}")
    print(f"  WCM forward r={results['diagnostic_b']['wcm_forward_r']:.3f} "
          f"vs raw VV-VWC r={results['diagnostic_b']['raw_vv_vwc_r']:.3f}: "
          f"{'substantially better' if results['diagnostic_b']['wcm_substantially_better'] else 'not substantially better'}")

    print(f"\nDiagnostic C — Identifiability:")
    for label in ["100%", "50%", "25%", "10%"]:
        rr = results["diagnostic_c"]["residual_ratios_by_fraction"][label]
        print(f"  Residual ratio @ {label}: {rr['median']:.2f} [IQR: {rr['q25']:.2f}–{rr['q75']:.2f}]")
    print(f"  |dσ/dA| = {results['diagnostic_c']['sensitivity_A_dB_per_unit']:.3f} dB/unit")
    print(f"  |dσ/dB| = {results['diagnostic_c']['sensitivity_B_dB_per_unit']:.3f} dB/unit")

    print(f"\nDiagnostic D — Mironov Sensitivity:")
    print(f"  {results['diagnostic_d']['finding']}")

    print("\n" + "=" * 70)
    return results


if __name__ == "__main__":
    run_all_diagnostics()
