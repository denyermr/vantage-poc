"""
Phase 1b Session F-2 — λ hyperparameter search for PINN-MIMICS.

Runs the 64-combination λ grid (SPEC §9) over 10 reps at the 100%
training fraction, evaluating the pre-registered Phase 1b dominance
constraint per combination and selecting via the three-tier fallback
(SPEC §9, SUCCESS_CRITERIA.md §3).

Pre-registered grid (SPEC §9, SUCCESS_CRITERIA.md §5):
    (λ_physics, λ_monotonic, λ_bounds) ∈ {0.01, 0.1, 0.5, 1.0}³
    = 4³ = 64 combinations.
    L_data coefficient is fixed at 1.0 and is NOT a search axis.

Pre-registered dominance constraint (SPEC §9, SUCCESS_CRITERIA.md §3):
    Evaluated over the FINAL 10 EPOCHS of training, averaged over all
    10 reps at the 100% training fraction (SPEC §9 verbatim).

    - Primary:   L_data is the largest single term in the composite
                 loss — i.e. L_data > each of (λ₁·L_physics),
                 (λ₂·L_monotonic), (λ₃·L_bounds) individually.
    - Secondary: (λ₁·L_physics) / L_total > 0.10.

Pre-registered three-tier fallback (SPEC §9, SUCCESS_CRITERIA.md §3):
    Tier 1 (FULL_DOMINANCE):
        ≥1 combination satisfies both primary and secondary. Select
        the dominance-compliant combination with the lowest median
        validation loss across the 10 reps.
    Tier 2 (PRIMARY_ONLY):
        0 satisfy both, but ≥1 satisfies primary only. Select the
        primary-only-compliant combination with the lowest median
        validation loss. Flag as a Phase 1b finding (DEV-1b-009
        drafted at F-2 close — see checkpoint report).
    Tier 3 (HALT):
        0 combinations satisfy even the primary criterion. HALT the
        experiment — Phase 1b architectural failure per SPEC §9.
        The Phase 1 fallback (lowest median regardless of dominance)
        is NOT retained.

Reference:
    phase1b/SUCCESS_CRITERIA.md (v1.0, locked 2026-04-19)
    SPEC.md §8, §9
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from shared import config
from phase1.lambda_search import prepare_pinn_data
from phase1b.pinn_mimics import PinnMimics, compute_pinn_mimics_loss

logger = logging.getLogger(__name__)


# ─── Pre-registered constants (locked) ───────────────────────────────────────

# SPEC §9 / SUCCESS_CRITERIA.md §5: 4³ = 64 combinations over
# (λ_physics, λ_monotonic, λ_bounds). L_data coefficient fixed at 1.0.
LAMBDA_GRID: List[float] = [0.01, 0.1, 0.5, 1.0]

# SPEC §9: dominance assessed over the final 10 epochs of training.
DOMINANCE_FINAL_EPOCHS = 10

# SPEC §9 secondary threshold: weighted L_physics > 10% of total.
SECONDARY_PHYSICS_FRACTION_THRESHOLD = 0.10

# SUCCESS_CRITERIA.md §5 / SPEC §9: 10 reps at the 100% fraction.
# Phase 1 stores these as config_idx 0..9 (the 100% subsampling reps).
CONFIG_INDICES_100PCT: List[int] = list(range(0, 10))


# ─── Per-rep training ───────────────────────────────────────────────────────


def train_pinn_mimics_single_rep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vv_db_train_raw: np.ndarray,
    vv_db_val_raw: np.ndarray,
    vh_db_train_raw: np.ndarray,
    vh_db_val_raw: np.ndarray,
    theta_train_deg: np.ndarray,
    theta_val_deg: np.ndarray,
    lambda_physics: float,
    lambda_monotonic: float,
    lambda_bounds: float,
    config_idx: int,
    device: torch.device,
    max_epochs: int = config.NN_MAX_EPOCHS,
    patience: int = config.NN_PATIENCE,
    lr: float = config.NN_LR,
    batch_size: int = config.NN_BATCH_SIZE,
    final_window_epochs: int = DOMINANCE_FINAL_EPOCHS,
) -> Dict:
    """
    Train a single PINN-MIMICS instance with early stopping and record the
    loss components over the final `final_window_epochs` epochs trained.

    "Final 10 epochs" per SPEC §9 means the last 10 epochs actually trained
    (before / at early-stop), not the 10 epochs preceding the best-val
    checkpoint. This matches the SUCCESS_CRITERIA.md §3 definition.

    Returns:
        dict with keys:
            best_val_loss:             float
            stopped_at_epoch:          int (number of epochs trained)
            final_window_means:        {l_data, weighted_l_physics,
                                        weighted_l_monotonic,
                                        weighted_l_bounds, total,
                                        physics_fraction}
            primary_dominance:         bool (L_data > each weighted term
                                        on the final-window means).
            secondary_dominance:       bool (physics_fraction > 0.10).
            history:                   per-epoch list of loss components
                                        and learned-parameter values
                                        (diagnostics only).
    """
    seed = config.SEED + config_idx
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    n_features = X_train.shape[1]
    model = PinnMimics(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.NN_WEIGHT_DECAY,
    )

    # Tensorise once (training pool is small so device-resident).
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    vv_train_t = torch.tensor(vv_db_train_raw, dtype=torch.float32, device=device)
    vv_val_t = torch.tensor(vv_db_val_raw, dtype=torch.float32, device=device)
    vh_train_t = torch.tensor(vh_db_train_raw, dtype=torch.float32, device=device)
    vh_val_t = torch.tensor(vh_db_val_raw, dtype=torch.float32, device=device)
    theta_train_t = torch.tensor(theta_train_deg, dtype=torch.float32, device=device)
    theta_val_t = torch.tensor(theta_val_deg, dtype=torch.float32, device=device)

    bs = min(batch_size, len(X_train))
    train_ds = TensorDataset(
        X_train_t, y_train_t, vv_train_t, vh_train_t, theta_train_t,
    )

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    patience_counter = 0
    epoch = 0

    for epoch in range(max_epochs):
        # ── train ──
        model.train()
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)

        for X_b, y_b, vv_b, vh_b, theta_b in loader:
            optimizer.zero_grad()
            out = model(X_b, theta_b, vv_b)
            loss_dict = compute_pinn_mimics_loss(
                out, y_b, vv_b, vh_b,
                lambda_physics=lambda_physics,
                lambda_monotonic=lambda_monotonic,
                lambda_bounds=lambda_bounds,
            )
            loss_dict["total"].backward()
            optimizer.step()

        # ── validate and record on full training + validation pool ──
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t, theta_val_t, vv_val_t)
            val_loss = compute_pinn_mimics_loss(
                val_out, y_val_t, vv_val_t, vh_val_t,
                lambda_physics=lambda_physics,
                lambda_monotonic=lambda_monotonic,
                lambda_bounds=lambda_bounds,
            )
            total_val = val_loss["total"].item()

        # Snapshot per-epoch loss components on the VAL set — this is what
        # the dominance constraint is evaluated against (SPEC §9).
        history.append({
            "epoch": epoch + 1,
            "val_total": total_val,
            "val_l_data": val_loss["l_data"].item(),
            "val_weighted_l_physics": val_loss["weighted_l_physics"].item(),
            "val_weighted_l_monotonic": val_loss["weighted_l_monotonic"].item(),
            "val_weighted_l_bounds": val_loss["weighted_l_bounds"].item(),
            "val_l_physics_unweighted": val_loss["l_physics"].item(),
            "val_l_physics_vv": val_loss["l_physics_vv"].item(),
            "val_l_physics_vh": val_loss["l_physics_vh"].item(),
            "N_b": float(model.N_b.item()),
            "N_l": float(model.N_l.item()),
            "sigma_orient_deg": float(model.sigma_orient_deg.item()),
            "m_g": float(model.m_g.item()),
            "s_cm": float(model.s_cm.item()),
        })

        # Guard against NaN/Inf — per F-2 fallback clause, NaN triggers rollback.
        if not np.isfinite(total_val):
            logger.error(
                "Non-finite val_loss at config_idx=%d λ=(%.3f,%.3f,%.3f) "
                "epoch=%d — aborting rep.",
                config_idx, lambda_physics, lambda_monotonic, lambda_bounds,
                epoch + 1,
            )
            return {
                "best_val_loss": float("nan"),
                "stopped_at_epoch": epoch + 1,
                "final_window_means": None,
                "primary_dominance": False,
                "secondary_dominance": False,
                "history": history,
                "non_finite_abort": True,
            }

        # Early stopping
        if total_val < best_val_loss - 1e-6:
            best_val_loss = total_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    stopped_at = epoch + 1

    # ── final-window dominance evaluation (SPEC §9) ──
    window = history[-final_window_epochs:] if len(history) >= final_window_epochs else history
    n_window = len(window)
    means = {
        "l_data": float(np.mean([h["val_l_data"] for h in window])),
        "weighted_l_physics": float(np.mean([h["val_weighted_l_physics"] for h in window])),
        "weighted_l_monotonic": float(np.mean([h["val_weighted_l_monotonic"] for h in window])),
        "weighted_l_bounds": float(np.mean([h["val_weighted_l_bounds"] for h in window])),
        "total": float(np.mean([h["val_total"] for h in window])),
        "n_epochs_in_window": n_window,
    }
    total_mean = means["total"] if means["total"] > 0 else 1e-12
    means["physics_fraction"] = means["weighted_l_physics"] / total_mean

    primary = (
        means["l_data"] > means["weighted_l_physics"]
        and means["l_data"] > means["weighted_l_monotonic"]
        and means["l_data"] > means["weighted_l_bounds"]
    )
    secondary = means["physics_fraction"] > SECONDARY_PHYSICS_FRACTION_THRESHOLD

    return {
        "best_val_loss": float(best_val_loss),
        "stopped_at_epoch": stopped_at,
        "final_window_means": means,
        "primary_dominance": bool(primary),
        "secondary_dominance": bool(secondary),
        "history": history,
        "non_finite_abort": False,
    }


# ─── Progressive-save helper ────────────────────────────────────────────────


def _write_partial_result(
    output_dir: Path,
    results: List[Dict],
    combos: List[tuple],
    config_indices: List[int],
    device: torch.device,
    t_start: float,
) -> None:
    """Write an in-flight `lambda_search_f2_partial.json` after every combo."""
    partial = {
        "session": "F-2",
        "status": "in_progress",
        "n_combinations_completed": len(results),
        "n_combinations_total": len(combos),
        "grid": {
            "lambda_physics": LAMBDA_GRID,
            "lambda_monotonic": LAMBDA_GRID,
            "lambda_bounds": LAMBDA_GRID,
            "n_combinations": len(combos),
            "l_data_coefficient_fixed": 1.0,
        },
        "reps_per_combination": len(config_indices),
        "training_config_indices": config_indices,
        "combinations": results,
        "elapsed_s": time.time() - t_start,
        "elapsed_min": (time.time() - t_start) / 60.0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }
    tmp = output_dir / "lambda_search_f2_partial.json.tmp"
    final = output_dir / "lambda_search_f2_partial.json"
    with open(tmp, "w") as f:
        json.dump(partial, f, indent=2)
    tmp.replace(final)


# ─── Resume helper ──────────────────────────────────────────────────────────


def _reconstruct_combo_record_from_history(
    combo_idx: int,
    lp: float,
    lm: float,
    lb: float,
    existing: Dict,
) -> Dict:
    """
    Rebuild the per-combination record from a previously-saved
    `per_rep_histories/combo_NNN.json`. Used by the resume path so a
    run interrupted mid-search (e.g. by the machine sleeping) can
    pick up where it left off without recomputing completed combos.

    The reconstructed record matches the schema produced in the main
    loop exactly; any drift between the two schemas would show up as
    a key mismatch at serialisation time.
    """
    rep_list = existing["per_rep"]
    val_losses = [r["best_val_loss"] for r in rep_list]
    finite_losses = [v for v in val_losses if np.isfinite(v)]
    median_val = float(np.median(finite_losses)) if finite_losses else float("nan")

    primary_per_rep = [bool(r["primary_dominance"]) for r in rep_list]
    secondary_per_rep = [bool(r["secondary_dominance"]) for r in rep_list]
    n_non_finite = sum(1 for r in rep_list if r.get("non_finite_abort"))

    physics_fractions = [
        r["final_window_means"]["physics_fraction"]
        if r.get("final_window_means") is not None else float("nan")
        for r in rep_list
    ]
    mean_phys_frac = float(np.nanmean(physics_fractions)) if any(np.isfinite(physics_fractions)) else float("nan")

    def _col(key: str) -> List[float]:
        return [
            r["final_window_means"][key]
            if r.get("final_window_means") is not None else float("nan")
            for r in rep_list
        ]

    l_data_values = _col("l_data")
    mean_l_data = float(np.nanmean(l_data_values)) if any(np.isfinite(l_data_values)) else float("nan")
    mean_w_phys = float(np.nanmean(_col("weighted_l_physics"))) if any(np.isfinite(_col("weighted_l_physics"))) else float("nan")
    mean_w_mono = float(np.nanmean(_col("weighted_l_monotonic"))) if any(np.isfinite(_col("weighted_l_monotonic"))) else float("nan")
    mean_w_bnds = float(np.nanmean(_col("weighted_l_bounds"))) if any(np.isfinite(_col("weighted_l_bounds"))) else float("nan")
    primary_mean_across_reps = bool(
        np.isfinite(mean_l_data) and np.isfinite(mean_w_phys)
        and np.isfinite(mean_w_mono) and np.isfinite(mean_w_bnds)
        and mean_l_data > mean_w_phys
        and mean_l_data > mean_w_mono
        and mean_l_data > mean_w_bnds
    )
    secondary_mean_across_reps = bool(
        np.isfinite(mean_phys_frac)
        and mean_phys_frac > SECONDARY_PHYSICS_FRACTION_THRESHOLD
    )

    return {
        "combo_idx": combo_idx,
        "lambda_physics": lp,
        "lambda_monotonic": lm,
        "lambda_bounds": lb,
        "median_val_loss": median_val,
        "n_reps": len(rep_list),
        "n_non_finite_aborts": n_non_finite,
        "primary_dominance_per_rep": primary_per_rep,
        "secondary_dominance_per_rep": secondary_per_rep,
        "primary_dominance_all_reps": all(primary_per_rep),
        "secondary_dominance_all_reps": all(secondary_per_rep),
        "primary_dominance_fraction": float(np.mean(primary_per_rep)),
        "secondary_dominance_fraction": float(np.mean(secondary_per_rep)),
        "mean_physics_fraction": mean_phys_frac,
        "mean_l_data_final_window": mean_l_data,
        "mean_weighted_l_physics_final_window": mean_w_phys,
        "mean_weighted_l_monotonic_final_window": mean_w_mono,
        "mean_weighted_l_bounds_final_window": mean_w_bnds,
        "primary_dominance_mean_across_reps": primary_mean_across_reps,
        "secondary_dominance_mean_across_reps": secondary_mean_across_reps,
        "wall_time_s": float(existing.get("wall_time_s", 0.0)),
        "resumed_from_disk": True,
    }


# ─── λ search ────────────────────────────────────────────────────────────────


def run_lambda_search_f2(
    aligned_dataset_path: Path,
    splits_dir: Path,
    test_indices_path: Path,
    output_dir: Path,
    device: Optional[torch.device] = None,
    save_histories: bool = True,
) -> Dict:
    """
    Execute Phase 1b Session F-2 λ search.

    Procedure (SPEC §9, SUCCESS_CRITERIA.md §3, DEV-1b-009):
        1. Load the 10 × 100% training configs.
        2. For each of 64 combinations (λ_physics, λ_monotonic, λ_bounds),
           train 10 PINN-MIMICS instances and record BOTH per-rep dominance
           and the cross-rep mean of the final-window loss components.
        3. Evaluate combination-level dominance per the verbatim SPEC §9
           text: L_data is the largest single term in the cross-rep mean
           of the per-rep final-10-epoch-window means. Per DEV-1b-009
           (F-2 closure), the mean-across-reps reading binds. The strict
           all-reps-AND reading (Phase 1's `n_violations==0` convention)
           is also computed and recorded in the result JSON for
           transparency, but does NOT drive tier classification.
        4. Select per the three-tier fallback using the binding reading.

    Returns:
        Result dict, also written to
        `<output_dir>/lambda_search_f2_result.json`.
    """
    if device is None:
        device = config.get_torch_device()

    # Pre-load all 10 × 100% config data. Same data-prep function used by
    # Phase 1 — guarantees the feature vector, scaler, and sealed test
    # boundaries are identical to Phase 1.
    logger.info("Loading 10 × 100%% training configs...")
    config_data: List[Dict] = []
    for cfg_idx in CONFIG_INDICES_100PCT:
        cfg_path = splits_dir / f"config_{cfg_idx:03d}.json"
        data = prepare_pinn_data(aligned_dataset_path, cfg_path, test_indices_path)
        cfg = data["config"]
        if abs(cfg["fraction"] - 1.0) > 1e-9:
            raise ValueError(
                f"Expected 100%% training fraction at config_idx={cfg_idx}, "
                f"got {cfg['fraction']} — Phase 1b F-2 λ search requires the "
                "100%% training pool per SPEC §9."
            )
        # MIMICS uses degrees; prepare_pinn_data gives radians, so re-extract
        # the raw degree column. The raw feature vector retains the degree
        # values (config.FEATURE_COLUMNS includes 'incidence_angle_mean').
        # Also extract raw vh_db for the SPEC §8 joint VV+VH L_physics term
        # (DEV-1b-010); prepare_pinn_data pre-DEV-1b-010 only exposed vv_db.
        import pandas as pd
        df = pd.read_csv(aligned_dataset_path)
        theta_col = config.FEATURE_COLUMNS.index("incidence_angle_mean")
        vh_col = config.FEATURE_COLUMNS.index("vh_db")
        feat_vals = df[config.FEATURE_COLUMNS].values
        theta_train_deg = feat_vals[cfg["train_indices"]][:, theta_col].astype(np.float32)
        theta_val_deg = feat_vals[cfg["val_indices"]][:, theta_col].astype(np.float32)
        vh_db_train_raw = feat_vals[cfg["train_indices"]][:, vh_col].astype(np.float32)
        vh_db_val_raw = feat_vals[cfg["val_indices"]][:, vh_col].astype(np.float32)
        data["theta_train_deg"] = theta_train_deg
        data["theta_val_deg"] = theta_val_deg
        data["vh_db_train_raw"] = vh_db_train_raw
        data["vh_db_val_raw"] = vh_db_val_raw
        config_data.append(data)

    combos: List[tuple] = list(itertools.product(LAMBDA_GRID, LAMBDA_GRID, LAMBDA_GRID))
    assert len(combos) == 64, f"Expected 64 combos, got {len(combos)}"

    results: List[Dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = output_dir / "per_rep_histories"
    if save_histories:
        histories_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for combo_idx, (lp, lm, lb) in enumerate(combos):
        combo_t0 = time.time()

        # Resume support: if per_rep_histories/combo_NNN.json exists and
        # the stored λ triple matches, reconstruct the combo_record from
        # it and skip re-training. This lets an interrupted run (e.g.
        # overnight machine sleep) resume from the last committed
        # combination without recomputing the completed ones.
        hist_path_existing = histories_dir / f"combo_{combo_idx:03d}.json"
        if save_histories and hist_path_existing.exists():
            try:
                with open(hist_path_existing) as fh:
                    existing = json.load(fh)
                if (
                    abs(existing.get("lambda_physics", -1) - lp) < 1e-9
                    and abs(existing.get("lambda_monotonic", -1) - lm) < 1e-9
                    and abs(existing.get("lambda_bounds", -1) - lb) < 1e-9
                    and len(existing.get("per_rep", [])) == len(CONFIG_INDICES_100PCT)
                ):
                    combo_record = _reconstruct_combo_record_from_history(
                        combo_idx, lp, lm, lb, existing,
                    )
                    results.append(combo_record)
                    logger.info(
                        "F-2 [%d/64] λ=(%.2f,%.2f,%.2f) RESUMED from disk "
                        "med_val=%.4f prim=%d/10 sec=%d/10 phys_frac=%.3f",
                        combo_idx + 1, lp, lm, lb,
                        combo_record["median_val_loss"],
                        sum(combo_record["primary_dominance_per_rep"]),
                        sum(combo_record["secondary_dominance_per_rep"]),
                        combo_record["mean_physics_fraction"],
                    )
                    continue
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(
                    "F-2 combo %d: existing history at %s could not be "
                    "reused (%s); recomputing.",
                    combo_idx, hist_path_existing, e,
                )

        rep_results = []
        n_non_finite = 0

        for cfg_idx, data in zip(CONFIG_INDICES_100PCT, config_data):
            cfg = data["config"]
            rep = train_pinn_mimics_single_rep(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                vv_db_train_raw=data["vv_db_train_raw"],
                vv_db_val_raw=data["vv_db_val_raw"],
                vh_db_train_raw=data["vh_db_train_raw"],
                vh_db_val_raw=data["vh_db_val_raw"],
                theta_train_deg=data["theta_train_deg"],
                theta_val_deg=data["theta_val_deg"],
                lambda_physics=lp,
                lambda_monotonic=lm,
                lambda_bounds=lb,
                config_idx=cfg["config_idx"],
                device=device,
            )
            rep_results.append(rep)
            if rep["non_finite_abort"]:
                n_non_finite += 1

        val_losses = [r["best_val_loss"] for r in rep_results]
        # Excluding NaNs for aggregation but recording the count.
        finite_losses = [v for v in val_losses if np.isfinite(v)]
        median_val = float(np.median(finite_losses)) if finite_losses else float("nan")

        primary_per_rep = [r["primary_dominance"] for r in rep_results]
        secondary_per_rep = [r["secondary_dominance"] for r in rep_results]
        primary_frac = float(np.mean(primary_per_rep))
        secondary_frac = float(np.mean(secondary_per_rep))
        primary_all = all(primary_per_rep)
        secondary_all = all(secondary_per_rep)

        physics_fractions = [
            r["final_window_means"]["physics_fraction"]
            if r["final_window_means"] is not None else float("nan")
            for r in rep_results
        ]
        mean_phys_frac = float(np.nanmean(physics_fractions)) if any(np.isfinite(physics_fractions)) else float("nan")

        l_data_values = [
            r["final_window_means"]["l_data"]
            if r["final_window_means"] is not None else float("nan")
            for r in rep_results
        ]
        mean_l_data = float(np.nanmean(l_data_values)) if any(np.isfinite(l_data_values)) else float("nan")

        # Mean-across-reps dominance (the verbatim-text reading of SPEC §9
        # "averaged over all 10 reps"). Provided alongside the all-reps AND
        # reading (primary_dominance_all_reps / secondary_dominance_all_reps)
        # so the dual-reading audit trail is preserved. Per DEV-1b-009
        # (F-2 closure), the mean-across-reps reading is the BINDING
        # interpretation and is the selection rule for tier classification;
        # the all-reps AND reading is reported for transparency only.
        weighted_physics_values = [
            r["final_window_means"]["weighted_l_physics"]
            if r["final_window_means"] is not None else float("nan")
            for r in rep_results
        ]
        weighted_mono_values = [
            r["final_window_means"]["weighted_l_monotonic"]
            if r["final_window_means"] is not None else float("nan")
            for r in rep_results
        ]
        weighted_bounds_values = [
            r["final_window_means"]["weighted_l_bounds"]
            if r["final_window_means"] is not None else float("nan")
            for r in rep_results
        ]
        mean_w_phys = float(np.nanmean(weighted_physics_values)) if any(np.isfinite(weighted_physics_values)) else float("nan")
        mean_w_mono = float(np.nanmean(weighted_mono_values)) if any(np.isfinite(weighted_mono_values)) else float("nan")
        mean_w_bnds = float(np.nanmean(weighted_bounds_values)) if any(np.isfinite(weighted_bounds_values)) else float("nan")
        primary_mean_across_reps = bool(
            np.isfinite(mean_l_data) and np.isfinite(mean_w_phys)
            and np.isfinite(mean_w_mono) and np.isfinite(mean_w_bnds)
            and mean_l_data > mean_w_phys
            and mean_l_data > mean_w_mono
            and mean_l_data > mean_w_bnds
        )
        secondary_mean_across_reps = bool(
            np.isfinite(mean_phys_frac)
            and mean_phys_frac > SECONDARY_PHYSICS_FRACTION_THRESHOLD
        )

        combo_record = {
            "combo_idx": combo_idx,
            "lambda_physics": lp,
            "lambda_monotonic": lm,
            "lambda_bounds": lb,
            "median_val_loss": median_val,
            "n_reps": len(rep_results),
            "n_non_finite_aborts": n_non_finite,
            "primary_dominance_per_rep": primary_per_rep,
            "secondary_dominance_per_rep": secondary_per_rep,
            "primary_dominance_all_reps": primary_all,
            "secondary_dominance_all_reps": secondary_all,
            "primary_dominance_fraction": primary_frac,
            "secondary_dominance_fraction": secondary_frac,
            "mean_physics_fraction": mean_phys_frac,
            "mean_l_data_final_window": mean_l_data,
            "mean_weighted_l_physics_final_window": mean_w_phys,
            "mean_weighted_l_monotonic_final_window": mean_w_mono,
            "mean_weighted_l_bounds_final_window": mean_w_bnds,
            "primary_dominance_mean_across_reps": primary_mean_across_reps,
            "secondary_dominance_mean_across_reps": secondary_mean_across_reps,
            "wall_time_s": time.time() - combo_t0,
        }
        results.append(combo_record)

        if save_histories:
            hist_path = histories_dir / f"combo_{combo_idx:03d}.json"
            with open(hist_path, "w") as f:
                json.dump({
                    "combo_idx": combo_idx,
                    "lambda_physics": lp,
                    "lambda_monotonic": lm,
                    "lambda_bounds": lb,
                    "per_rep": [
                        {
                            "config_idx": cfg_idx,
                            "best_val_loss": r["best_val_loss"],
                            "stopped_at_epoch": r["stopped_at_epoch"],
                            "final_window_means": r["final_window_means"],
                            "primary_dominance": r["primary_dominance"],
                            "secondary_dominance": r["secondary_dominance"],
                            "non_finite_abort": r["non_finite_abort"],
                            "history": r["history"],
                        }
                        for cfg_idx, r in zip(CONFIG_INDICES_100PCT, rep_results)
                    ],
                }, f, indent=2)

        logger.info(
            "F-2 [%d/64] λ=(%.2f,%.2f,%.2f) med_val=%.4f "
            "prim=%d/10 sec=%d/10 phys_frac=%.3f dt=%.1fs",
            combo_idx + 1, lp, lm, lb, median_val,
            sum(primary_per_rep), sum(secondary_per_rep),
            mean_phys_frac, combo_record["wall_time_s"],
        )

        # Progressive checkpoint: write an in-flight result JSON after
        # every combination. On interruption the main result JSON
        # reflects all completed combos; resume picks up via the
        # per-rep histories.
        _write_partial_result(
            output_dir, results, combos, CONFIG_INDICES_100PCT, device, t_start,
        )

    # ── three-tier selection (SPEC §9 / SUCCESS_CRITERIA.md §3) ──
    # Per DEV-1b-009 (Phase 1b Session F-2 closure), the binding aggregation
    # rule for the SPEC §9 primary / secondary dominance criterion is the
    # cross-rep mean (the verbatim-text "averaged over all 10 reps" reading),
    # NOT the strict all-reps-AND reading that was inherited from Phase 1's
    # `phase1/lambda_search.py` (`n_violations == 0`).
    #
    # Both readings continue to be computed and recorded per combination in
    # the result JSON (`primary_dominance_all_reps` / `secondary_dominance_all_reps`
    # and `primary_dominance_mean_across_reps` / `secondary_dominance_mean_across_reps`),
    # preserving the dual-reading audit trail. The tier classification below
    # uses the mean-across-reps reading per DEV-1b-009 adjudication.
    #
    # The binding-going-forward principle (DEV-1b-009 "Binding for future
    # Phase 1b diagnostics") extends this aggregation rule to the Phase 4
    # diagnostic thresholds (D-1 through D-4 in SUCCESS_CRITERIA.md §6).

    full_dominance = [
        r for r in results
        if r["primary_dominance_mean_across_reps"]
        and r["secondary_dominance_mean_across_reps"]
    ]
    primary_only = [
        r for r in results
        if r["primary_dominance_mean_across_reps"]
        and not r["secondary_dominance_mean_across_reps"]
    ]
    any_primary = [r for r in results if r["primary_dominance_mean_across_reps"]]

    if full_dominance:
        tier = "FULL_DOMINANCE"
        selected = min(full_dominance, key=lambda r: r["median_val_loss"])
        halt = False
        fallback_dev_entry_recommended = False
    elif primary_only:
        tier = "PRIMARY_ONLY"
        selected = min(primary_only, key=lambda r: r["median_val_loss"])
        halt = False
        fallback_dev_entry_recommended = True
    else:
        tier = "HALT"
        selected = None
        halt = True
        fallback_dev_entry_recommended = False

    elapsed = time.time() - t_start

    result_json = {
        "session": "F-2",
        "pre_registration_tag": "phase1b-success-criteria-pre-registered",
        "grid": {
            "lambda_physics": LAMBDA_GRID,
            "lambda_monotonic": LAMBDA_GRID,
            "lambda_bounds": LAMBDA_GRID,
            "n_combinations": len(combos),
            "l_data_coefficient_fixed": 1.0,
        },
        "reps_per_combination": len(CONFIG_INDICES_100PCT),
        "training_config_indices": CONFIG_INDICES_100PCT,
        "dominance_constraint": {
            "primary": "L_data > each of (λ_physics·L_physics, λ_monotonic·L_monotonic, λ_bounds·L_bounds), evaluated on the cross-rep mean of the per-rep final-10-epoch-window means (SPEC §9 verbatim 'averaged over all 10 reps'). Per DEV-1b-009 the mean-across-reps reading binds; the strict all-reps-AND reading is reported alongside for transparency.",
            "secondary": "(λ_physics·L_physics) / L_total > 0.10, evaluated on the cross-rep mean physics fraction (same aggregation as primary per SPEC §9 'averaged as above').",
            "final_window_epochs": DOMINANCE_FINAL_EPOCHS,
            "secondary_threshold": SECONDARY_PHYSICS_FRACTION_THRESHOLD,
            "aggregation_rule_binding": "mean_across_reps",
            "aggregation_rule_rationale": "DEV-1b-009 (2026-04-20) adjudicated the verbatim pre-registered text of SPEC §9 line 317 against the implementation-inherited strict all-reps-AND reading. The cross-rep mean is the binding reading. Both readings continue to be computed per combination and recorded in this JSON.",
        },
        "three_tier_fallback_outcome": {
            "tier": tier,
            "halted": halt,
            "n_full_dominance": len(full_dominance),
            "n_primary_only": len(primary_only),
            "n_any_primary": len(any_primary),
            "n_neither": len(results) - len(any_primary),
            "phase_1b_dev_entry_recommended": fallback_dev_entry_recommended,
        },
        "selected": (
            None if selected is None
            else {
                "combo_idx": selected["combo_idx"],
                "lambda_physics": selected["lambda_physics"],
                "lambda_monotonic": selected["lambda_monotonic"],
                "lambda_bounds": selected["lambda_bounds"],
                "median_val_loss": selected["median_val_loss"],
                "mean_physics_fraction": selected["mean_physics_fraction"],
                "mean_l_data_final_window": selected["mean_l_data_final_window"],
                # Binding reading per DEV-1b-009 (F-2 closure).
                "primary_dominance_mean_across_reps": selected["primary_dominance_mean_across_reps"],
                "secondary_dominance_mean_across_reps": selected["secondary_dominance_mean_across_reps"],
                # Strict reading retained for transparency only.
                "primary_dominance_all_reps": selected["primary_dominance_all_reps"],
                "secondary_dominance_all_reps": selected["secondary_dominance_all_reps"],
                "tier_activated": tier,
                "aggregation_rule_binding": "mean_across_reps",
            }
        ),
        "combinations": results,
        "wall_time_s": elapsed,
        "wall_time_min": elapsed / 60.0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }

    out_path = output_dir / "lambda_search_f2_result.json"
    with open(out_path, "w") as f:
        json.dump(result_json, f, indent=2)

    logger.info(
        "F-2 λ search complete in %.1f min. Tier=%s. Output: %s",
        elapsed / 60.0, tier, out_path,
    )
    if selected is not None:
        logger.info(
            "Selected λ=(%.2f, %.2f, %.2f) median_val=%.4f phys_frac=%.3f",
            selected["lambda_physics"], selected["lambda_monotonic"],
            selected["lambda_bounds"], selected["median_val_loss"],
            selected["mean_physics_fraction"],
        )
    else:
        logger.warning(
            "F-2 HALT: no combination satisfies the primary dominance constraint. "
            "Phase 1 fallback is NOT retained per SPEC §9. F-3 authorisation required "
            "from the science agent with a re-scoping decision."
        )

    return result_json


# ─── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run F-2 λ search from the CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1b Session F-2 λ search")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "results",
        help="Directory to write result JSON and per-rep histories.",
    )
    parser.add_argument(
        "--no-histories", action="store_true",
        help="Skip saving per-rep histories (smaller output).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "Run only the first combination × first rep for plumbing / "
            "timing smoke test. Does not produce a result JSON."
        ),
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    run_lambda_search_f2(
        aligned_dataset_path=config.DATA_PROCESSED / "aligned_dataset.csv",
        splits_dir=config.DATA_SPLITS / "configs",
        test_indices_path=config.DATA_SPLITS / "test_indices.json",
        output_dir=args.output_dir,
        save_histories=not args.no_histories,
    )


def _run_smoke_test() -> None:
    """One-combo × one-rep sanity check."""
    device = config.get_torch_device()
    logger.info("F-2 smoke test on device=%s", device)

    cfg_path = config.DATA_SPLITS / "configs" / f"config_{CONFIG_INDICES_100PCT[0]:03d}.json"
    data = prepare_pinn_data(
        config.DATA_PROCESSED / "aligned_dataset.csv",
        cfg_path,
        config.DATA_SPLITS / "test_indices.json",
    )
    import pandas as pd
    df = pd.read_csv(config.DATA_PROCESSED / "aligned_dataset.csv")
    theta_col = config.FEATURE_COLUMNS.index("incidence_angle_mean")
    vh_col = config.FEATURE_COLUMNS.index("vh_db")
    feat_vals = df[config.FEATURE_COLUMNS].values
    theta_train_deg = feat_vals[data["config"]["train_indices"]][:, theta_col].astype(np.float32)
    theta_val_deg = feat_vals[data["config"]["val_indices"]][:, theta_col].astype(np.float32)
    vh_db_train_raw = feat_vals[data["config"]["train_indices"]][:, vh_col].astype(np.float32)
    vh_db_val_raw = feat_vals[data["config"]["val_indices"]][:, vh_col].astype(np.float32)

    t0 = time.time()
    result = train_pinn_mimics_single_rep(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        vv_db_train_raw=data["vv_db_train_raw"],
        vv_db_val_raw=data["vv_db_val_raw"],
        vh_db_train_raw=vh_db_train_raw,
        vh_db_val_raw=vh_db_val_raw,
        theta_train_deg=theta_train_deg,
        theta_val_deg=theta_val_deg,
        lambda_physics=LAMBDA_GRID[0],
        lambda_monotonic=LAMBDA_GRID[0],
        lambda_bounds=LAMBDA_GRID[0],
        config_idx=data["config"]["config_idx"],
        device=device,
    )
    dt = time.time() - t0

    print(f"\n=== F-2 smoke test ===")
    print(f"Device:              {device}")
    print(f"Epochs trained:      {result['stopped_at_epoch']}")
    print(f"Best val loss:       {result['best_val_loss']:.6f}")
    print(f"Primary dominance:   {result['primary_dominance']}")
    print(f"Secondary dominance: {result['secondary_dominance']}")
    print(f"Final-window means:")
    for k, v in (result["final_window_means"] or {}).items():
        if isinstance(v, (int, float)):
            print(f"  {k:28s} {v:.6f}")
        else:
            print(f"  {k:28s} {v}")
    print(f"Wall time:           {dt:.1f}s ({dt/60.0:.2f} min)")
    print(f"Non-finite abort:    {result['non_finite_abort']}")
    est_full = dt * 64 * 10
    print(
        f"\nExtrapolated full λ search: {est_full:.0f}s "
        f"= {est_full/60.0:.1f} min = {est_full/3600.0:.2f} h"
    )


if __name__ == "__main__":
    main()
