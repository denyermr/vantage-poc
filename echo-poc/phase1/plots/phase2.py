"""
Phase 2 diagnostic figures.

Figure 1: p2_learning_curves_baselines.png — RMSE and R² learning curves
Figure 2: p2_feature_diagnostics.png — VV vs VWC scatter + RF feature importances
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared import config
from shared.evaluation import aggregate_metrics_across_reps
from shared.splits import assign_season

logger = logging.getLogger(__name__)

# Vantage dark theme colours
COLOUR_RF = "#38bdf8"       # cyan
COLOUR_NN = "#a78bfa"       # purple
COLOUR_NULL = "#fbbf24"     # amber
BG_COLOUR = "#0f172a"       # dark slate
TEXT_COLOUR = "#e2e8f0"     # light slate
GRID_COLOUR = "#334155"     # muted slate

SEASON_COLOURS = {
    "DJF": "#3b82f6",   # blue
    "MAM": "#22c55e",   # green
    "JJA": "#f59e0b",   # amber
    "SON": "#f97316",    # orange
}


def _apply_dark_theme(fig, axes):
    """Apply Vantage dark theme to figure."""
    fig.patch.set_facecolor(BG_COLOUR)
    for ax in axes:
        ax.set_facecolor(BG_COLOUR)
        ax.tick_params(colors=TEXT_COLOUR)
        ax.xaxis.label.set_color(TEXT_COLOUR)
        ax.yaxis.label.set_color(TEXT_COLOUR)
        ax.title.set_color(TEXT_COLOUR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOUR)
        ax.grid(True, color=GRID_COLOUR, alpha=0.3)


def load_metrics_by_fraction(
    metrics_dir: Path,
    model_key: str,
) -> dict[str, list[dict]]:
    """
    Load all metrics for a model, grouped by fraction label.

    Returns:
        Dict mapping fraction_label → list of metric dicts.
    """
    result = {}
    for frac in config.TRAINING_FRACTIONS:
        label = config.TRAINING_SIZE_LABELS[frac]
        size_idx = config.TRAINING_FRACTIONS.index(frac)
        metrics_list = []
        for rep in range(config.N_REPS):
            idx = size_idx * config.N_REPS + rep
            path = metrics_dir / f"config_{idx:03d}_{model_key}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                metrics_list.append(data["metrics"])
        if metrics_list:
            result[label] = metrics_list
    return result


def plot_learning_curves(
    metrics_dir: Path,
    null_metrics_path: Path,
    output_path: Path,
) -> None:
    """
    Generate the two-panel learning curve figure.

    Panel 1: RMSE vs training size
    Panel 2: R² vs training size
    """
    # Load null model RMSE
    with open(null_metrics_path) as f:
        null_data = json.load(f)
    null_rmse = null_data["metrics"]["rmse"]
    null_r2 = null_data["metrics"]["r_squared"]

    # Load RF and NN metrics
    rf_by_frac = load_metrics_by_fraction(metrics_dir, "baseline_a")
    nn_by_frac = load_metrics_by_fraction(metrics_dir, "baseline_b")

    labels = ["10%", "25%", "50%", "100%"]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_theme(fig, [ax1, ax2])

    for ax, metric_key, ylabel, title, null_val in [
        (ax1, "rmse", "RMSE (cm³/cm³)", "Baseline Learning Curves — RMSE vs Training Size", null_rmse),
        (ax2, "r_squared", "R²", "Baseline Learning Curves — R² vs Training Size", null_r2),
    ]:
        for model_key, by_frac, colour, label in [
            ("baseline_a", rf_by_frac, COLOUR_RF, "RF"),
            ("baseline_b", nn_by_frac, COLOUR_NN, "NN"),
        ]:
            medians = []
            q25s = []
            q75s = []
            for frac_label in labels:
                if frac_label in by_frac:
                    agg = aggregate_metrics_across_reps(by_frac[frac_label])
                    medians.append(agg[f"{metric_key}_median"])
                    q25s.append(agg[f"{metric_key}_q25"])
                    q75s.append(agg[f"{metric_key}_q75"])
                else:
                    medians.append(np.nan)
                    q25s.append(np.nan)
                    q75s.append(np.nan)

            medians = np.array(medians)
            q25s = np.array(q25s)
            q75s = np.array(q75s)

            ax.plot(x, medians, color=colour, marker="o", label=label, linewidth=2)
            ax.fill_between(x, q25s, q75s, color=colour, alpha=0.2)

        # Null model horizontal line
        ax.axhline(null_val, color=COLOUR_NULL, linestyle="--",
                   linewidth=1.5, label="Null (seasonal)")

        # N=25 critical threshold line (index 1 = 25%)
        ax.axvline(x=1, color=TEXT_COLOUR, linestyle=":", alpha=0.5)
        ax.text(1.05, ax.get_ylim()[1] * 0.95, "N≈25",
                color=TEXT_COLOUR, fontsize=8, alpha=0.7)

        # Annotate N=25 values
        annotation_parts = []
        for model_label, by_frac in [("RF", rf_by_frac), ("NN", nn_by_frac)]:
            if "25%" in by_frac:
                agg = aggregate_metrics_across_reps(by_frac["25%"])
                annotation_parts.append(
                    f"{model_label}: {agg[f'{metric_key}_median']:.3f}"
                )
        if annotation_parts:
            ax.text(1.15, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
                    "  ".join(annotation_parts),
                    color=TEXT_COLOUR, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOUR,
                              edgecolor=GRID_COLOUR, alpha=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right" if metric_key == "rmse" else "lower right",
                  facecolor=BG_COLOUR, edgecolor=GRID_COLOUR,
                  labelcolor=TEXT_COLOUR)

    # Set y-axis ranges
    if not np.isnan(null_rmse):
        ax1.set_ylim(0, null_rmse * 1.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOUR, bbox_inches="tight")
    plt.close(fig)
    logger.info("Learning curves saved: %s", output_path)


def plot_feature_diagnostics(
    aligned_dataset_path: Path,
    metrics_dir: Path,
    gate1_result_path: Path,
    output_path: Path,
) -> None:
    """
    Generate two-panel feature diagnostics figure.

    Panel 1: VV vs VWC scatter coloured by season
    Panel 2: RF feature importances (100% training, median across 10 reps)
    """
    df = pd.read_csv(aligned_dataset_path)
    dates = pd.DatetimeIndex(df["date"])
    seasons = assign_season(dates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_theme(fig, [ax1, ax2])

    # Panel 1: VV vs VWC scatter
    for season, colour in SEASON_COLOURS.items():
        mask = seasons == season
        ax1.scatter(df.loc[mask, "vv_db"], df.loc[mask, "vwc"],
                    color=colour, label=season, alpha=0.7, s=30, edgecolors="none")

    # Load gate 1 result for annotation
    if gate1_result_path.exists():
        with open(gate1_result_path) as f:
            g1 = json.load(f)
        criteria = g1.get("criteria", [])
        # criteria may be a list of dicts or a dict keyed by ID
        g105 = {}
        if isinstance(criteria, list):
            for c in criteria:
                if c.get("id") == "G1-05":
                    g105 = c
                    break
        elif isinstance(criteria, dict):
            g105 = criteria.get("G1-05", {})
        measured = g105.get("measured", {})
        if isinstance(measured, dict):
            r_val = measured.get("pearson_r", "?")
            p_val = measured.get("p_value", "?")
        else:
            r_val = "?"
            p_val = "?"
        ax1.text(0.05, 0.95, f"r = {r_val:.3f}, p = {p_val:.4f}" if isinstance(r_val, float) else f"r={r_val}",
                 transform=ax1.transAxes, color=TEXT_COLOUR, fontsize=9,
                 va="top", bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor=BG_COLOUR, edgecolor=GRID_COLOUR))

    ax1.set_xlabel("VV Backscatter (dB)")
    ax1.set_ylabel("VWC (cm³/cm³)")
    ax1.set_title("SAR Backscatter vs Soil Moisture — Seasonal Context", fontsize=10)
    ax1.legend(facecolor=BG_COLOUR, edgecolor=GRID_COLOUR, labelcolor=TEXT_COLOUR)

    # Panel 2: RF feature importances
    importances_all = []
    for rep in range(config.N_REPS):
        imp_path = (config.OUTPUTS_MODELS / "baseline_a" /
                    f"config_{rep:03d}" / "feature_importances.json")
        if imp_path.exists():
            with open(imp_path) as f:
                importances_all.append(json.load(f))

    if importances_all:
        features = config.FEATURE_COLUMNS
        imp_matrix = np.array([[d[f] for f in features] for d in importances_all])
        medians = np.median(imp_matrix, axis=0)
        q25 = np.percentile(imp_matrix, 25, axis=0)
        q75 = np.percentile(imp_matrix, 75, axis=0)

        sorted_idx = np.argsort(medians)
        features_sorted = [features[i] for i in sorted_idx]
        medians_sorted = medians[sorted_idx]
        errors = np.array([medians_sorted - q25[sorted_idx],
                           q75[sorted_idx] - medians_sorted])

        y_pos = np.arange(len(features_sorted))
        ax2.barh(y_pos, medians_sorted, color=COLOUR_RF, alpha=0.8,
                 xerr=errors, capsize=3, error_kw={"color": TEXT_COLOUR, "alpha": 0.6})
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features_sorted, color=TEXT_COLOUR)
        ax2.set_xlabel("Importance")
        ax2.set_title("RF Feature Importances (100% training, n=10 reps)", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Feature importances\nnot computed",
                 transform=ax2.transAxes, ha="center", va="center",
                 color=TEXT_COLOUR, fontsize=14)
        ax2.set_title("RF Feature Importances", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOUR, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature diagnostics saved: %s", output_path)
