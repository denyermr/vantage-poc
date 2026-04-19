"""
Phase 3 diagnostic figures.

This module generates all Phase 3 figures. During P3.1, only the dielectric
comparison plot is available. Remaining figures are added as later P3 tasks
complete.

All figures saved to outputs/figures/ with p3_ prefix.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from shared.config import (
    OUTPUTS_FIGURES,
    PEAT_THETA_SAT,
)
from phase1.physics.dielectric import DobsonDielectric, MironovDielectric

logger = logging.getLogger(__name__)


def plot_dielectric_comparison(
    m_v_min: float = 0.2,
    m_v_max: float = 0.8,
    n_points: int = 200,
    save: bool = True,
) -> None:
    """
    Plot Dobson vs Mironov dielectric constant over the observed VWC range.

    Generates p3_dielectric_comparison.png showing both models across the
    0.2–0.8 cm³/cm³ range observed at Moor House.

    Args:
        m_v_min: Lower VWC bound for plot (cm³/cm³).
        m_v_max: Upper VWC bound for plot (cm³/cm³).
        n_points: Number of evaluation points.
        save: If True, save to outputs/figures/.
    """
    logger.info("Generating dielectric comparison plot: Dobson vs Mironov")

    m_v = torch.linspace(m_v_min, m_v_max, n_points)

    dobson = DobsonDielectric()
    mironov = MironovDielectric()

    with torch.no_grad():
        eps_dobson = dobson(m_v).numpy()
        eps_mironov = mironov(m_v).numpy()

    m_v_np = m_v.numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Both models on same axes
    ax1.plot(m_v_np, eps_dobson, color="#00BCD4", linewidth=2, label="Dobson (1985)")
    ax1.plot(m_v_np, eps_mironov, color="#9C27B0", linewidth=2, label="Mironov (2009)")
    ax1.axvline(x=PEAT_THETA_SAT, color="gray", linestyle="--", alpha=0.5,
                label=f"θ_sat = {PEAT_THETA_SAT}")
    ax1.set_xlabel("Volumetric Water Content (cm³/cm³)")
    ax1.set_ylabel("Dielectric Constant ε (dimensionless)")
    ax1.set_title("Dielectric Constant vs VWC — Moor House Peat")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Relative difference
    rel_diff = (eps_dobson - eps_mironov) / eps_mironov * 100
    ax2.plot(m_v_np, rel_diff, color="#FF5722", linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Volumetric Water Content (cm³/cm³)")
    ax2.set_ylabel("Relative Difference (Dobson − Mironov) / Mironov (%)")
    ax2.set_title("Model Disagreement")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("P3.1 — Dielectric Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_FIGURES / "p3_dielectric_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    plot_dielectric_comparison()
