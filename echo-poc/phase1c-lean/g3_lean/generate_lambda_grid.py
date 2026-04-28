"""
Phase 1c-Lean Block C-prime — λ-grid generation script (G3-Lean deliverable 2).

Emits the locked 6×6 = 36-cell λ grid plus 3-rep cross-product = 108 entries
per SPEC §18.4.2 / §18.6.2 v0.3.2.

Canonical enumeration (SPEC §18.4.2 v0.3):
    LAMBDA_VV_VALUES = LAMBDA_VH_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    λ_VV outer loop, λ_VH inner loop.

Seed convention (SPEC §18.6.2 v0.3.2):
    config_idx = 18 * λ_VV_idx + 3 * λ_VH_idx + rep_idx
    SEED = 42 + config_idx
    Range: 108 unique seeds in [42, 149].

Output: phase1c-lean/g3_lean/lambda_grid.json with 108 entries, each carrying
    the full (cell_idx, λ_VV_idx, λ_VH_idx, rep_idx, λ_VV, λ_VH, config_idx, seed)
    tuple for downstream Block D-prime sweep self-documentation.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LAMBDA_VV_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
LAMBDA_VH_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
N_REPS = 3
SEED_BASE = 42

OUTPUT_PATH = Path(__file__).parent / "lambda_grid.json"


def generate_grid() -> list[dict]:
    """Emit the 108-entry (cell × rep) grid in canonical order."""
    if len(LAMBDA_VV_VALUES) != 6 or len(LAMBDA_VH_VALUES) != 6:
        raise ValueError(
            f"Expected 6×6 grid; got {len(LAMBDA_VV_VALUES)}×{len(LAMBDA_VH_VALUES)}"
        )
    entries = []
    for vv_idx, lambda_vv in enumerate(LAMBDA_VV_VALUES):
        for vh_idx, lambda_vh in enumerate(LAMBDA_VH_VALUES):
            cell_idx = 6 * vv_idx + vh_idx
            for rep_idx in range(N_REPS):
                config_idx = 18 * vv_idx + 3 * vh_idx + rep_idx
                seed = SEED_BASE + config_idx
                entries.append({
                    "cell_idx": cell_idx,
                    "lambda_vv_idx": vv_idx,
                    "lambda_vh_idx": vh_idx,
                    "rep_idx": rep_idx,
                    "lambda_vv": lambda_vv,
                    "lambda_vh": lambda_vh,
                    "config_idx": config_idx,
                    "seed": seed,
                })
    return entries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entries = generate_grid()

    expected_n = len(LAMBDA_VV_VALUES) * len(LAMBDA_VH_VALUES) * N_REPS
    if len(entries) != expected_n:
        raise ValueError(f"Expected {expected_n} entries; got {len(entries)}")

    config_indices = [e["config_idx"] for e in entries]
    if set(config_indices) != set(range(expected_n)):
        raise ValueError(
            f"config_idx must cover [0, {expected_n}); "
            f"got missing={set(range(expected_n)) - set(config_indices)}, "
            f"extra={set(config_indices) - set(range(expected_n))}"
        )
    if config_indices != sorted(config_indices):
        raise ValueError("config_idx must enumerate in canonical order")

    seeds = [e["seed"] for e in entries]
    if len(set(seeds)) != expected_n:
        raise ValueError(
            f"Expected {expected_n} unique seeds; got {len(set(seeds))} "
            f"(min={min(seeds)}, max={max(seeds)})"
        )

    payload = {
        "spec_version": "v0.3.3",
        "spec_section": "SPEC.md §18.4.2 (grid) + §18.6.2 (seed convention, locked at v0.3.2 amendment)",
        "lambda_vv_values": LAMBDA_VV_VALUES,
        "lambda_vh_values": LAMBDA_VH_VALUES,
        "n_cells": len(LAMBDA_VV_VALUES) * len(LAMBDA_VH_VALUES),
        "n_reps_per_cell": N_REPS,
        "n_total_runs": expected_n,
        "seed_base": SEED_BASE,
        "config_idx_formula": "config_idx = 18 * lambda_vv_idx + 3 * lambda_vh_idx + rep_idx",
        "seed_formula": "seed = 42 + config_idx",
        "config_idx_range": [0, expected_n - 1],
        "seed_range": [SEED_BASE, SEED_BASE + expected_n - 1],
        "enumeration_order": "lambda_vv outer, lambda_vh middle, rep_idx inner",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "lambda_grid.json: %d entries, %d cells × %d reps, seeds [%d..%d]",
        len(entries), payload["n_cells"], N_REPS,
        payload["seed_range"][0], payload["seed_range"][1],
    )
    logger.info("First entry: %s", entries[0])
    logger.info("Last entry:  %s", entries[-1])


if __name__ == "__main__":
    main()
