"""
Phase 1 diagnostic figures for the ECHO PoC.

Generates 4 required PNG figures per SPEC_PHASE1.md §P1.9:
    1. p1_cosmos_diagnostic.png   — 3-panel VWC diagnostic
    2. p1_sar_diagnostic.png      — 3-panel SAR diagnostic
    3. p1_ancillary_diagnostic.png — 2-panel ancillary diagnostic
    4. p1_aligned_dataset_summary.png — 2-panel summary

Style: Vantage dark theme (#0d1117 background, #111820 surface).
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from shared import config

logger = logging.getLogger(__name__)

# ─── Vantage dark theme colours ──────────────────────────────────────────────

BG_COLOR = "#0d1117"
SURFACE_COLOR = "#111820"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"
ACCENT_BLUE = "#58a6ff"
ACCENT_ORANGE = "#d29922"
ACCENT_RED = "#f85149"
ACCENT_GREEN = "#3fb950"
ACCENT_PURPLE = "#bc8cff"
ACCENT_CYAN = "#56d4dd"

SEASON_COLORS = {
    "DJF": ACCENT_BLUE,
    "MAM": ACCENT_GREEN,
    "JJA": ACCENT_ORANGE,
    "SON": ACCENT_RED,
}


def _apply_dark_theme(fig, axes):
    """Apply Vantage dark theme to figure and axes."""
    fig.patch.set_facecolor(BG_COLOR)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(SURFACE_COLOR)
        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)


def _save_figure(fig, name: str) -> Path:
    """Save figure to outputs/figures/ and return path."""
    output_path = config.OUTPUTS_FIGURES / name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)
    return output_path


# ─── Figure 1: COSMOS diagnostic ─────────────────────────────────────────────


def plot_cosmos_diagnostic(cosmos_df: pd.DataFrame) -> Path:
    """
    Generate p1_cosmos_diagnostic.png — 3-panel VWC diagnostic.

    Panel 1: 4-year full time series (blue=valid, orange=frozen, cyan=snow, red=flagged)
    Panel 2: Monthly mean ± 1 SD (non-frozen, non-snow only)
    Panel 3: DOY climatology with winter/summer mean lines

    Args:
        cosmos_df: Processed COSMOS-UK DataFrame from load_cosmos().

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    _apply_dark_theme(fig, axes)

    df = cosmos_df.copy()

    # ─── Panel 1: Full time series ────────────────────────────────
    ax = axes[0]

    # Valid VWC (not flagged, not frozen, not snow)
    valid_mask = (
        df["vwc_qc"].notna()
        & (df["frozen_flag"] == 0)
        & (df["snow_flag"] == 0)
    )
    ax.scatter(
        df.loc[valid_mask, "date"], df.loc[valid_mask, "vwc_raw"],
        s=3, c=ACCENT_BLUE, alpha=0.7, label="Valid VWC", zorder=3,
    )

    # Frozen days
    frozen_mask = df["frozen_flag"] == 1
    if frozen_mask.any():
        ax.scatter(
            df.loc[frozen_mask, "date"], df.loc[frozen_mask, "vwc_raw"],
            s=3, c=ACCENT_ORANGE, alpha=0.7, label="Frozen", zorder=2,
        )

    # Snow days
    snow_mask = df["snow_flag"] == 1
    if snow_mask.any():
        ax.scatter(
            df.loc[snow_mask, "date"], df.loc[snow_mask, "vwc_raw"],
            s=3, c=ACCENT_CYAN, alpha=0.7, label="Snow", zorder=2,
        )

    # Flagged E/I
    flagged_mask = df["cosmos_vwc_flag"].isin(config.COSMOS_EXCLUDE_FLAGS)
    if flagged_mask.any():
        ax.scatter(
            df.loc[flagged_mask, "date"], df.loc[flagged_mask, "vwc_raw"],
            s=3, c=ACCENT_RED, alpha=0.7, label="Flagged (E/I)", zorder=2,
        )

    ax.set_ylabel("VWC (cm³/cm³)")
    ax.set_title("COSMOS-UK Moor House — VWC Time Series (2021–2024)")
    ax.legend(loc="upper right", fontsize=8, facecolor=SURFACE_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ─── Panel 2: Monthly mean ± 1 SD ────────────────────────────
    ax = axes[1]

    clean = df.loc[valid_mask].copy()
    clean["month"] = clean["date"].dt.to_period("M")
    monthly = clean.groupby("month")["vwc_qc"].agg(["mean", "std"])
    monthly.index = monthly.index.to_timestamp()

    ax.plot(monthly.index, monthly["mean"], color=ACCENT_BLUE, linewidth=1.5)
    ax.fill_between(
        monthly.index,
        monthly["mean"] - monthly["std"],
        monthly["mean"] + monthly["std"],
        alpha=0.2, color=ACCENT_BLUE,
    )
    ax.set_ylabel("VWC (cm³/cm³)")
    ax.set_title("Monthly Mean ± 1 SD (valid observations only)")

    # ─── Panel 3: DOY climatology ─────────────────────────────────
    ax = axes[2]

    clean["doy"] = clean["date"].dt.dayofyear
    doy_clim = clean.groupby("doy")["vwc_qc"].agg(["mean", "std"])

    ax.plot(doy_clim.index, doy_clim["mean"], color=ACCENT_BLUE, linewidth=1)
    ax.fill_between(
        doy_clim.index,
        doy_clim["mean"] - doy_clim["std"],
        doy_clim["mean"] + doy_clim["std"],
        alpha=0.15, color=ACCENT_BLUE,
    )

    # Winter/summer mean lines
    winter_mean = clean.loc[
        clean["date"].dt.month.isin(config.SEASONS["DJF"]), "vwc_qc"
    ].mean()
    summer_mean = clean.loc[
        clean["date"].dt.month.isin(config.SEASONS["JJA"]), "vwc_qc"
    ].mean()

    ax.axhline(winter_mean, color=ACCENT_CYAN, linestyle="--", linewidth=1,
               label=f"Winter mean: {winter_mean:.3f}")
    ax.axhline(summer_mean, color=ACCENT_ORANGE, linestyle="--", linewidth=1,
               label=f"Summer mean: {summer_mean:.3f}")

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("VWC (cm³/cm³)")
    ax.set_title("DOY Climatology")
    ax.legend(loc="upper right", fontsize=8, facecolor=SURFACE_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.suptitle("Phase 1 — COSMOS-UK Diagnostic", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return _save_figure(fig, "p1_cosmos_diagnostic.png")


# ─── Figure 2: SAR diagnostic ────────────────────────────────────────────────


def plot_sar_diagnostic(s1_df: pd.DataFrame, aligned_df: pd.DataFrame) -> Path:
    """
    Generate p1_sar_diagnostic.png — 3-panel SAR diagnostic.

    Panel 1: VV (dB) time series, coloured by orbit
    Panel 2: VH/VV ratio (dB) time series
    Panel 3: Scatter VV vs VWC with Pearson r, coloured by season

    Args:
        s1_df: Processed Sentinel-1 DataFrame.
        aligned_df: Aligned dataset with VWC and SAR.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    _apply_dark_theme(fig, axes)

    # ─── Panel 1: VV time series ──────────────────────────────────
    ax = axes[0]

    if "orbit_number" in s1_df.columns:
        orbits = s1_df["orbit_number"].unique()
        colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED]
        for i, orbit in enumerate(orbits):
            mask = s1_df["orbit_number"] == orbit
            c = colors[i % len(colors)]
            ax.scatter(
                s1_df.loc[mask, "date"], s1_df.loc[mask, "vv_db"],
                s=8, c=c, alpha=0.7, label=f"Orbit {orbit}", zorder=3,
            )
        ax.legend(loc="upper right", fontsize=8, facecolor=SURFACE_COLOR,
                  edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    else:
        ax.scatter(s1_df["date"], s1_df["vv_db"], s=8, c=ACCENT_BLUE, alpha=0.7)

    ax.set_ylabel("VV (dB)")
    ax.set_title("Sentinel-1 VV Backscatter Time Series")

    # ─── Panel 2: VH/VV ratio time series ────────────────────────
    ax = axes[1]
    ax.scatter(s1_df["date"], s1_df["vhvv_db"], s=8, c=ACCENT_PURPLE, alpha=0.7)
    ax.set_ylabel("VH/VV (dB)")
    ax.set_title("VH/VV Cross-Polarisation Ratio")

    # ─── Panel 3: VV vs VWC scatter ──────────────────────────────
    ax = axes[2]

    df = aligned_df.copy()
    df["season"] = pd.cut(
        df["date"].dt.month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["DJF", "MAM", "JJA", "SON", "DJF"],
        ordered=False,
    )

    for season, color in SEASON_COLORS.items():
        mask = df["season"] == season
        if mask.any():
            ax.scatter(
                df.loc[mask, "vv_db"], df.loc[mask, "vwc"],
                s=15, c=color, alpha=0.6, label=season, zorder=3,
            )

    r, p = stats.pearsonr(df["vv_db"], df["vwc"])
    ax.set_xlabel("VV (dB)")
    ax.set_ylabel("VWC (cm³/cm³)")
    ax.set_title(f"VV vs VWC (Pearson r={r:.3f}, p={p:.2e})")
    ax.legend(loc="upper right", fontsize=8, facecolor=SURFACE_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.suptitle("Phase 1 — SAR Diagnostic", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return _save_figure(fig, "p1_sar_diagnostic.png")


# ─── Figure 3: Ancillary diagnostic ──────────────────────────────────────────


def plot_ancillary_diagnostic(ancillary_df: pd.DataFrame) -> Path:
    """
    Generate p1_ancillary_diagnostic.png — 2-panel ancillary diagnostic.

    Panel 1: NDVI at SAR dates (scatter)
    Panel 2: Daily precipitation (bars) + 7-day antecedent index (line)

    Args:
        ancillary_df: Ancillary features DataFrame (date, ndvi, precip_mm, precip_7day_mm).

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    _apply_dark_theme(fig, axes)

    df = ancillary_df.copy()

    # ─── Panel 1: NDVI ───────────────────────────────────────────
    ax = axes[0]
    ax.scatter(df["date"], df["ndvi"], s=12, c=ACCENT_GREEN, alpha=0.7, zorder=3)
    ax.set_ylabel("NDVI")
    ax.set_title("NDVI at SAR Overpass Dates (interpolated from S2 monthly)")

    # ─── Panel 2: Precipitation ──────────────────────────────────
    ax = axes[1]
    ax.bar(df["date"], df["precip_mm"], width=4, color=ACCENT_BLUE,
           alpha=0.5, label="Daily precip (mm)")
    ax.plot(df["date"], df["precip_7day_mm"], color=ACCENT_ORANGE,
            linewidth=1.2, label="7-day antecedent (mm)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title("ERA5-Land Precipitation")
    ax.legend(loc="upper right", fontsize=8, facecolor=SURFACE_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.suptitle("Phase 1 — Ancillary Features Diagnostic", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return _save_figure(fig, "p1_ancillary_diagnostic.png")


# ─── Figure 4: Aligned dataset summary ───────────────────────────────────────


def plot_aligned_summary(aligned_df: pd.DataFrame) -> Path:
    """
    Generate p1_aligned_dataset_summary.png — 2-panel summary.

    Panel 1: Attrition waterfall (row counts at each QC step)
    Panel 2: Feature correlation heatmap (Pearson r)

    Args:
        aligned_df: Final aligned dataset.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _apply_dark_theme(fig, axes)

    # ─── Panel 1: Attrition waterfall ─────────────────────────────
    ax = axes[0]

    attrition_path = config.OUTPUTS_GATES / "gate_1_attrition.json"
    if attrition_path.exists():
        with open(attrition_path) as f:
            attrition = json.load(f)

        labels = [
            "S1 raw",
            "VWC flag",
            "Frozen",
            "Snow",
            "Missing",
            "Orbit",
        ]
        keys = [
            "s1_overpasses_raw",
            "after_step1_vwc_flag",
            "after_step2_frozen",
            "after_step3_snow",
            "after_step4_missing_ancillary",
            "after_step5_orbit",
        ]
        values = [attrition.get(k, 0) for k in keys]

        bar_colors = []
        for i in range(len(values)):
            if i == 0:
                bar_colors.append(ACCENT_BLUE)
            elif i == len(values) - 1:
                bar_colors.append(ACCENT_GREEN)
            else:
                bar_colors.append(ACCENT_BLUE)

        ax.bar(labels, values, color=bar_colors, alpha=0.8, edgecolor=GRID_COLOR)

        # Annotate removed counts
        for i in range(1, len(values)):
            removed = values[i - 1] - values[i]
            if removed > 0:
                ax.annotate(
                    f"-{removed}",
                    xy=(i, values[i]),
                    ha="center", va="bottom",
                    color=ACCENT_RED, fontsize=9, fontweight="bold",
                )

        ax.set_ylabel("Row count")
        ax.set_title("QC Attrition Waterfall")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(
            0.5, 0.5, "Attrition log not found\n(run alignment first)",
            ha="center", va="center", color=TEXT_COLOR, fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_title("QC Attrition Waterfall")

    # ─── Panel 2: Feature correlation heatmap ─────────────────────
    ax = axes[1]

    feature_cols = config.FEATURE_COLUMNS + ["vwc"]
    available_cols = [c for c in feature_cols if c in aligned_df.columns]
    corr = aligned_df[available_cols].corr()

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(available_cols)))
    ax.set_yticks(range(len(available_cols)))
    ax.set_xticklabels(available_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(available_cols, fontsize=7)

    # Annotate cells
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            val = corr.values[i, j]
            text_col = "white" if abs(val) > 0.6 else TEXT_COLOR
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=text_col)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature Correlation Matrix")

    fig.suptitle("Phase 1 — Aligned Dataset Summary", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return _save_figure(fig, "p1_aligned_dataset_summary.png")


# ─── Convenience: generate all 4 figures ─────────────────────────────────────


def generate_all_phase1_figures() -> list[Path]:
    """
    Generate all 4 Phase 1 diagnostic figures from processed data files.

    Requires:
        - data/processed/cosmos_processed.csv
        - data/processed/sentinel1_extractions.csv
        - data/processed/ancillary_features.csv
        - data/processed/aligned_dataset.csv

    Returns:
        List of paths to saved figures.

    Raises:
        FileNotFoundError: If any required processed file is missing.
    """
    paths = {
        "cosmos": config.DATA_PROCESSED / "cosmos_processed.csv",
        "s1": config.DATA_PROCESSED / "sentinel1_extractions.csv",
        "ancillary": config.DATA_PROCESSED / "ancillary_features.csv",
        "aligned": config.DATA_PROCESSED / "aligned_dataset.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Processed {name} file not found: {path}. "
                f"Run the data pipeline first."
            )

    cosmos_df = pd.read_csv(paths["cosmos"], parse_dates=["date"])
    s1_df = pd.read_csv(paths["s1"], parse_dates=["date"])
    ancillary_df = pd.read_csv(paths["ancillary"], parse_dates=["date"])
    aligned_df = pd.read_csv(paths["aligned"], parse_dates=["date"])

    figures = []
    figures.append(plot_cosmos_diagnostic(cosmos_df))
    figures.append(plot_sar_diagnostic(s1_df, aligned_df))
    figures.append(plot_ancillary_diagnostic(ancillary_df))
    figures.append(plot_aligned_summary(aligned_df))

    logger.info("All 4 Phase 1 diagnostic figures generated")
    return figures
