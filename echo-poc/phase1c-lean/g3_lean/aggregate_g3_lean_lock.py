"""
Phase 1c-Lean Block C-prime — G3-Lean lock aggregator (G3-Lean deliverable 1).

Aggregates the four locked items (λ-grid, baselines, sealed-set definition,
preflight schema) into a single `g3_lean_lock.json` per kickoff §2 deliverable 1.

Sources (must exist before this script runs):
- phase1c-lean/g3_lean/lambda_grid.json          (deliverable 2)
- phase1c-lean/g3_lean/baselines_locked.json     (deliverable 3)
- phase1c-lean/g3_lean/sealed_set_definition.json (deliverable 4)
- phase1c-lean/g3_lean/preflight_schema.json     (deliverable 5)
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
SOURCES = {
    "lambda_grid": ROOT / "lambda_grid.json",
    "baselines": ROOT / "baselines_locked.json",
    "sealed_set": ROOT / "sealed_set_definition.json",
    "preflight_schema": ROOT / "preflight_schema.json",
}
OUTPUT_PATH = ROOT / "g3_lean_lock.json"


def get_git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    artefacts = {}
    sha256_per_artefact = {}
    for key, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"G3-Lean source artefact missing: {path}. "
                f"Run the corresponding generator script before aggregation."
            )
        artefacts[key] = json.loads(path.read_text())
        sha256_per_artefact[key] = sha256_bytes(path.read_bytes())

    payload = {
        "lock_id": "phase1c_lean_g3_lean_lock_v0_3_3",
        "spec_version": "v0.3.3",
        "spec_section": "SPEC.md §18.6.2",
        "block": "Block C-prime",
        "predecessor_tag": "phase1c-lean-g2-lean-passed",
        "amendment_tags": [
            "phase1c-lean-spec-v0_3_1",
            "phase1c-lean-spec-v0_3_2",
            "phase1c-lean-spec-v0_3_3",
        ],
        "dev_entries": [
            "DEV-1c-lean-001 (sealed-set used-once acknowledgement)",
            "DEV-1c-lean-002 (per-channel normalisation implementation choice)",
            "DEV-1c-lean-003 (founder-only sign-off, scientific co-supervisor TBC)",
            "DEV-1c-lean-004 (v0.2 → v0.3 SPEC scope correction)",
            "DEV-1c-lean-005 (v0.3 → v0.3.1 loss-formulation string canonicalisation)",
            "DEV-1c-lean-006 (v0.3.1 → v0.3.2 seed convention + phantom-citation cleanup)",
            "DEV-1c-lean-007 (v0.3.2 → v0.3.3 null-methodology specification)",
        ],
        "locked_items": {
            "lambda_grid": {
                "summary": (
                    f"6×6 = 36 cells × 3 reps = 108 runs; seeds [42, 149]; "
                    f"config_idx = 18 * lambda_vv_idx + 3 * lambda_vh_idx + rep_idx"
                ),
                "n_total_runs": artefacts["lambda_grid"]["n_total_runs"],
                "first_entry": artefacts["lambda_grid"]["entries"][0],
                "last_entry": artefacts["lambda_grid"]["entries"][-1],
                "source_path": str(SOURCES["lambda_grid"].relative_to(ROOT.parent.parent)),
                "source_sha256": sha256_per_artefact["lambda_grid"],
            },
            "baselines": {
                "summary": (
                    f"RF 5-fold CV RMSE = "
                    f"{artefacts['baselines']['rf_100pct_5fold_cv']['rmse_cm3_per_cm3']:.4f}; "
                    f"seasonal-climatological null 5-fold CV RMSE = "
                    f"{artefacts['baselines']['seasonal_climatological_null']['rmse_cm3_per_cm3']:.4f}"
                ),
                "rf_rmse": artefacts["baselines"]["rf_100pct_5fold_cv"]["rmse_cm3_per_cm3"],
                "null_rmse": artefacts["baselines"]["seasonal_climatological_null"]["rmse_cm3_per_cm3"],
                "phase1_sealed_test_references": {
                    "rf": 0.147,
                    "null": 0.178,
                },
                "comparison_disposition": "informational only (training-pool-vs-sealed-test variability gap; not gate criteria at G3-Lean per SPEC §18.6.2 v0.3.3)",
                "source_path": str(SOURCES["baselines"].relative_to(ROOT.parent.parent)),
                "source_sha256": sha256_per_artefact["baselines"],
            },
            "sealed_set": {
                "summary": (
                    f"n={artefacts['sealed_set']['n_observations']}, "
                    f"{artefacts['sealed_set']['date_range'][0]}..{artefacts['sealed_set']['date_range'][1]}; "
                    f"sha256 {artefacts['sealed_set']['sha256'][:16]}..."
                ),
                "n_observations": artefacts["sealed_set"]["n_observations"],
                "date_range": artefacts["sealed_set"]["date_range"],
                "slice_sha256": artefacts["sealed_set"]["sha256"],
                "loaded_at_g3_lean": artefacts["sealed_set"]["loaded_at_g3_lean"],
                "source_path": str(SOURCES["sealed_set"].relative_to(ROOT.parent.parent)),
                "source_sha256": sha256_per_artefact["sealed_set"],
            },
            "preflight_schema": {
                "summary": (
                    f"per-run schema; loss_formulation constant = "
                    f"\"{artefacts['preflight_schema']['items']['loss_formulation']['constant_value']}\"; "
                    f"6 §18.11 audit items"
                ),
                "schema_id": artefacts["preflight_schema"]["schema_id"],
                "n_items": len(artefacts["preflight_schema"]["items"]),
                "source_path": str(SOURCES["preflight_schema"].relative_to(ROOT.parent.parent)),
                "source_sha256": sha256_per_artefact["preflight_schema"],
            },
        },
        "code_version_hash": get_git_hash(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_lock_policy": (
            "After lock, no element of the four locked items is modified before "
            "Block D-prime sweep close. Any required change halts and routes "
            "through a SPEC amendment cycle (DEV-1c-lean-NNN)."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("g3_lean_lock.json written; lock_id=%s", payload["lock_id"])


if __name__ == "__main__":
    main()
