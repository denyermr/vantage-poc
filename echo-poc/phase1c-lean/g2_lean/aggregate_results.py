"""
G2-Lean three-arm aggregator (SPEC §18.6.1).

Reads the per-arm result JSONs produced by arm_1_cross_framework.py,
arm_2_autograd_fd.py, and arm_3_scale_sanity.py and writes a combined
result at `results/g2_lean_equivalence_result.json` per the §5.4 schema
in the Block B-prime kickoff prompt.

Tag recommendation:
  `phase1c-lean-g2-lean-passed` if all three arms pass;
  `phase1c-lean-g2-lean-halt`   otherwise (HALT trigger per SPEC §18.8
                                first bullet).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / "results"
ARM_1_PATH = RESULTS_DIR / "arm_1_result.json"
ARM_2_PATH = RESULTS_DIR / "arm_2_result.json"
ARM_3_PATH = RESULTS_DIR / "arm_3_result.json"
AGGREGATE_PATH = RESULTS_DIR / "g2_lean_equivalence_result.json"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Could not resolve git hash: %s", e)
        return "unknown"


def _load(p: Path) -> Dict:
    if not p.exists():
        raise FileNotFoundError(
            f"Per-arm result missing at {p}. Run the corresponding arm script."
        )
    with open(p) as fh:
        return json.load(fh)


def aggregate() -> Dict:
    arm1 = _load(ARM_1_PATH)
    arm2 = _load(ARM_2_PATH)
    arm3 = _load(ARM_3_PATH)

    overall_pass = bool(arm1["pass"] and arm2["pass"] and arm3["pass"])
    tag = (
        "phase1c-lean-g2-lean-passed"
        if overall_pass
        else "phase1c-lean-g2-lean-halt"
    )

    return {
        "g2_lean_overall_pass": overall_pass,
        "spec_version": "v0.3",
        "spec_section": "SPEC.md §18.6.1",
        "phase1b_g2_predecessor": "DEV-1b-008 G2 Moderate Pass on MIMICS forward model",
        "arm_1_cross_framework": arm1,
        "arm_2_autograd_fd": arm2,
        "arm_3_scale_sanity": arm3,
        "tag_recommendation": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version_hash": _git_hash(),
    }


def main() -> int:
    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = aggregate()
    with open(AGGREGATE_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "G2-Lean aggregate: %s → tag=%s",
        "PASS" if result["g2_lean_overall_pass"] else "HALT",
        result["tag_recommendation"],
    )
    return 0 if result["g2_lean_overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
