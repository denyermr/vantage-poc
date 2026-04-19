"""
Phase 1b Gate G1 — Phase 1 baseline reproducibility check.

`SPEC.md` (Phase 1b) §2 requires the Phase 1 baseline RMSEs to be reproduced
within 0.005 cm³/cm³ before any MIMICS training begins. This script is the
authoritative G1 implementation: it aggregates the sealed Phase 1 per-config
metrics files in `outputs/metrics/` and compares the medians against the
published values in `outputs/write-up/poc_results.md`.

Two run modes:

    python phase1/run_baselines.py --confirm
        Default. Loads sealed per-config metric JSONs produced by Phase 1
        training. No retraining. Writes `phase1b/implementation_gate/results/
        g1_baseline_result.json` and exits 0 iff every baseline stays within
        tolerance.

    python phase1/run_baselines.py --retrain
        Reserved for future use. Currently not implemented — retraining
        requires the Phase 1 pipeline entry point (phase1/pipeline.py) and
        is explicitly out of scope for the G1 reproducibility gate, which
        is a check against the *sealed artefacts*, not a fresh training run.

Published Phase 1 numbers (from poc_results.md Table 1, re-derived on the
sealed test set n=36):

    Null (floor):  0.178 at every training fraction
    RF:            0.144 / 0.155 / 0.155 / 0.147 at 10% / 25% / 50% / 100%
    NN:            0.319 / 0.236 / 0.219 / 0.159 at 10% / 25% / 50% / 100%

SPEC.md §2 calls out four key values that gate Phase 1b sign-off:

    Null 0.178         (any fraction)
    RF   0.155         at N≈25  (25%)
    RF   0.147         at N=83  (100%)
    NN   0.159         at N=83  (100%)

Tolerance is ±0.005 cm³/cm³ in absolute RMSE. Any drift above this halts
Phase 1b. In practice a clean migration with no retraining should reproduce
to machine precision — drift can only come from reading the wrong files.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Allow running as script: `python phase1/run_baselines.py --confirm`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import OUTPUTS_METRICS, PROJECT_ROOT  # noqa: E402

logger = logging.getLogger(__name__)

TOLERANCE = 0.005  # cm³/cm³, per SPEC.md §2
METRICS_DIR = OUTPUTS_METRICS
G1_RESULT_PATH = (
    PROJECT_ROOT / "phase1b" / "implementation_gate" / "results" / "g1_baseline_result.json"
)

# Published Phase 1 reference numbers, per poc_results.md Table 1 / SPEC.md §2.
# Keys are (model, fraction_label). Values are RMSE in cm³/cm³.
PUBLISHED = {
    ("null", "10%"): 0.178,
    ("null", "25%"): 0.178,
    ("null", "50%"): 0.178,
    ("null", "100%"): 0.178,
    ("RF", "10%"): 0.144,
    ("RF", "25%"): 0.155,
    ("RF", "50%"): 0.155,
    ("RF", "100%"): 0.147,
    ("NN", "10%"): 0.319,
    ("NN", "25%"): 0.236,
    ("NN", "50%"): 0.219,
    ("NN", "100%"): 0.159,
}

# SPEC.md §2 calls these four values out explicitly as blocking for Phase 1b
# sign-off. Any of these drifting > TOLERANCE halts the experiment.
SIGN_OFF_ANCHORS = [
    ("null", "100%"),
    ("RF", "25%"),
    ("RF", "100%"),
    ("NN", "100%"),
]

# File-naming conventions in outputs/metrics/ (written by Phase 1):
#   config_{idx:03d}_baseline_a.json  → RandomForestRegressor  (40 files)
#   config_{idx:03d}_baseline_b.json  → StandardNN             (40 files)
#   baseline_0_metrics.json           → Null seasonal (single summary file)
BASELINE_FILE_MODEL = {
    "baseline_a": "RF",
    "baseline_b": "NN",
}


def _load_null_rmse() -> dict[str, float]:
    """Null baseline is fraction-independent; replicate the single value."""
    path = METRICS_DIR / "baseline_0_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"G1 expected null metrics at {path}")
    data = json.loads(path.read_text())
    rmse = float(data["metrics"]["rmse"])
    return {fraction: rmse for fraction in ("10%", "25%", "50%", "100%")}


def _load_per_config_rmse(suffix: str) -> dict[str, list[float]]:
    """
    Collect per-config RMSEs for the given model suffix, bucketed by
    fraction_label. Returns dict fraction_label → list of RMSEs (one per rep).
    """
    by_fraction: dict[str, list[float]] = {f: [] for f in ("10%", "25%", "50%", "100%")}
    pattern = f"config_*_{suffix}.json"
    files = sorted(METRICS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"G1 expected per-config metrics matching {pattern} in {METRICS_DIR}"
        )
    for f in files:
        data = json.loads(f.read_text())
        by_fraction[data["fraction_label"]].append(float(data["metrics"]["rmse"]))
    # Sanity: Phase 1 runs 10 reps × 4 fractions = 40 configs per model.
    counts = {k: len(v) for k, v in by_fraction.items()}
    if sum(counts.values()) != 40:
        raise ValueError(
            f"G1 expected 40 configs for suffix={suffix}; got counts={counts}"
        )
    return by_fraction


def _median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot compute median of empty sequence")
    return statistics.median(values)


def _check_row(
    model: str,
    fraction: str,
    observed: float,
    published: float,
    tolerance: float = TOLERANCE,
) -> dict:
    drift = observed - published
    passed = abs(drift) <= tolerance
    return {
        "model": model,
        "fraction": fraction,
        "published_rmse": published,
        "observed_rmse": round(observed, 6),
        "drift": round(drift, 6),
        "abs_drift": round(abs(drift), 6),
        "tolerance": tolerance,
        "pass": passed,
    }


def check_baselines(confirm: bool) -> dict:
    """Run the full G1 check and return a structured result."""
    if not confirm:
        raise NotImplementedError(
            "Retraining mode is not implemented — the G1 gate is defined as a "
            "check against sealed Phase 1 artefacts. Use --confirm."
        )

    null_rmse = _load_null_rmse()
    rf_rmse = _load_per_config_rmse("baseline_a")
    nn_rmse = _load_per_config_rmse("baseline_b")

    rows: list[dict] = []
    # Null model: single summary file, same value at every fraction.
    for fraction, rmse in null_rmse.items():
        rows.append(_check_row("null", fraction, rmse, PUBLISHED[("null", fraction)]))

    for fraction, rmses in rf_rmse.items():
        rows.append(_check_row("RF", fraction, _median(rmses), PUBLISHED[("RF", fraction)]))

    for fraction, rmses in nn_rmse.items():
        rows.append(_check_row("NN", fraction, _median(rmses), PUBLISHED[("NN", fraction)]))

    all_pass = all(r["pass"] for r in rows)

    # Isolate the four anchors SPEC.md §2 calls out as blocking.
    anchor_rows = [
        r for r in rows if (r["model"], r["fraction"]) in SIGN_OFF_ANCHORS
    ]
    anchors_pass = all(r["pass"] for r in anchor_rows)

    result = {
        "gate": "G1 — Phase 1 baseline reproducibility",
        "spec_reference": "SPEC.md (Phase 1b) §2",
        "tolerance_cm3_cm3": TOLERANCE,
        "pass": all_pass,
        "anchors_pass": anchors_pass,
        "n_configs_per_model": {"RF": sum(len(v) for v in rf_rmse.values()),
                                "NN": sum(len(v) for v in nn_rmse.values())},
        "rows": rows,
        "anchor_rows": anchor_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def _print_report(result: dict) -> None:
    header = "G1 baseline reproducibility report"
    print("\n" + "═" * len(header))
    print(header)
    print("═" * len(header))
    print(f"Tolerance: ±{result['tolerance_cm3_cm3']:.3f} cm³/cm³ (SPEC.md §2)")
    print(f"Metrics source: {METRICS_DIR}")
    print()

    # Pretty-print one row per (model, fraction).
    print(f"{'model':<6}  {'frac':<5}  {'published':>10}  {'observed':>10}  {'drift':>9}  pass")
    print("-" * 58)
    for r in result["rows"]:
        flag = "✓" if r["pass"] else "✗"
        print(
            f"{r['model']:<6}  {r['fraction']:<5}  "
            f"{r['published_rmse']:>10.4f}  {r['observed_rmse']:>10.4f}  "
            f"{r['drift']:>+9.4f}  {flag}"
        )
    print()
    print("Sign-off anchors (SPEC §2):")
    for r in result["anchor_rows"]:
        flag = "✓" if r["pass"] else "✗"
        print(
            f"  {r['model']} @ {r['fraction']}: "
            f"observed {r['observed_rmse']:.4f} vs published {r['published_rmse']:.4f} "
            f"(drift {r['drift']:+.4f}) {flag}"
        )
    print()
    if result["pass"]:
        print("G1 PASS — all baselines reproduce within tolerance.")
    else:
        fails = [r for r in result["rows"] if not r["pass"]]
        print(f"G1 FAIL — {len(fails)} row(s) outside tolerance:")
        for r in fails:
            print(f"  {r['model']} @ {r['fraction']}: drift {r['drift']:+.4f}")
    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required. Runs the check against sealed Phase 1 metrics without retraining.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Reserved. Not implemented — see module docstring.",
    )
    args = parser.parse_args()

    if args.retrain:
        logger.error("--retrain is reserved; use --confirm.")
        return 2
    if not args.confirm:
        parser.print_help()
        return 2

    result = check_baselines(confirm=True)

    G1_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    G1_RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {G1_RESULT_PATH}")

    _print_report(result)

    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
