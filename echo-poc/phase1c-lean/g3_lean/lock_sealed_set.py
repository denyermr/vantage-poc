"""
Phase 1c-Lean Block C-prime — Sealed-set definition lock script (G3-Lean deliverable 4).

Per SPEC §18.5 v0.3.3 / kickoff §3.4:
- Reads `data/splits/test_indices.json` to confirm n=36 and date range
  2023-07-25 to 2024-12-10.
- Computes SHA-256 of the underlying CSV slice corresponding to the 36 sealed
  observations (canonical CSV bytes).
- Emits `sealed_set_definition.json` per the schema in kickoff §3.4.

CRITICAL: The sealed set is **metadata-only at G3-Lean** per SPEC §18.5
v0.3.2 / DEV-1c-lean-001. The 36 dates are read for definition-lock purposes
only; no SAR/VWC values are loaded into memory beyond what is required for
the SHA-256 computation. The set is not used for prediction, evaluation, or
any Phase 1c-Lean operation at G3-Lean. Conditional Block E-prime unsealing
is the only authorised path to evaluation.
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/aligned_dataset.csv")
TEST_INDICES_PATH = Path("data/splits/test_indices.json")
OUTPUT_PATH = Path(__file__).parent / "sealed_set_definition.json"

EXPECTED_N = 36
EXPECTED_START = "2023-07-25"
EXPECTED_END = "2024-12-10"


def get_git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(TEST_INDICES_PATH) as f:
        idx = json.load(f)
    if idx["n_test"] != EXPECTED_N:
        raise ValueError(
            f"test_indices n_test {idx['n_test']} != expected {EXPECTED_N}"
        )
    actual_start = idx["test_date_start"][:10]
    actual_end = idx["test_date_end"][:10]
    if actual_start != EXPECTED_START or actual_end != EXPECTED_END:
        raise ValueError(
            f"sealed-set date range {actual_start}..{actual_end} != "
            f"expected {EXPECTED_START}..{EXPECTED_END} per SPEC §18.5"
        )

    df = pd.read_csv(DATA_PATH).sort_values("date").reset_index(drop=True)
    if len(df) != idx["n_total"]:
        raise ValueError(
            f"aligned_dataset n={len(df)} != test_indices n_total {idx['n_total']}"
        )
    sealed = df.iloc[idx["split_idx"]:].reset_index(drop=True)
    if len(sealed) != EXPECTED_N:
        raise ValueError(
            f"sealed slice has n={len(sealed)} after split_idx={idx['split_idx']}; "
            f"expected {EXPECTED_N}"
        )

    sealed_dates = sealed["date"].tolist()
    if sealed_dates[0] != EXPECTED_START or sealed_dates[-1] != EXPECTED_END:
        raise ValueError(
            f"sealed slice first/last dates {sealed_dates[0]}/{sealed_dates[-1]} != "
            f"expected {EXPECTED_START}/{EXPECTED_END}"
        )

    sealed_csv_bytes = sealed.to_csv(index=False).encode("utf-8")
    sealed_sha256 = hashlib.sha256(sealed_csv_bytes).hexdigest()
    logger.info("Sealed slice SHA-256: %s", sealed_sha256)

    payload = {
        "sealed_set_id": "phase1_sealed_n36_v1",
        "n_observations": EXPECTED_N,
        "date_range": [EXPECTED_START, EXPECTED_END],
        "observation_dates": sealed_dates,
        "sha256": sealed_sha256,
        "sha256_algorithm": "sha256_of_canonical_csv_slice_utf8",
        "data_source": str(DATA_PATH),
        "data_source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "split_indices_source": str(TEST_INDICES_PATH),
        "split_idx": idx["split_idx"],
        "spec_reference": (
            "SPEC §14 (Phase 1 pre-registration; sealed-set lock at "
            "data/splits/test_indices.json) and §18.5 (Phase 1c-Lean "
            "used-once held-out policy)"
        ),
        "phase1_reference": (
            "Phase 1 PINN RMSE 0.167; Phase 1 RF RMSE 0.155 at N≈25; "
            "Phase 1 RF RMSE 0.147 at 100% training (sealed-test evaluation)"
        ),
        "loaded_at_g3_lean": False,
        "loaded_at_block_d_prime_sweep": False,
        "loaded_at_block_e_prime_evaluation": (
            "conditional on Strong/Significant/Moderate per §18.5 / §18.7"
        ),
        "code_version_hash": get_git_hash(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("sealed_set_definition.json: n=%d, %s..%s",
                EXPECTED_N, EXPECTED_START, EXPECTED_END)


if __name__ == "__main__":
    main()
