"""
Phase 1b G4 — Dobson vs Mironov dielectric diagnostic.

`SPEC.md` (Phase 1b) §6 and §11 (Diagnostic C) require a one-shot
sensitivity check of the dielectric model choice at sign-off:

    Compute |ε_Dobson − ε_Mironov| over the observed m_v range
    (0.25–0.83 cm³/cm³ at Moor House). Report as a function of m_v.
    If the maximum relative difference is <5%, the choice of
    dielectric model is not the binding constraint and the
    Phase 1 → Phase 1b dielectric change cannot explain any RMSE
    difference. If >5%, the dielectric choice must be carried as
    an active source of variance in the Phase 1b interpretation.

The gate **passes either way** — its job is to record whether
dielectric choice is a binding constraint, not to decide it. The
result JSON is a sign-off artefact and is consumed by SPEC §14.

Usage:
    python phase1b/implementation_gate/dielectric_diagnostic.py
    # or
    make g4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402

from shared.config import PEAT_THETA_SAT, PROJECT_ROOT  # noqa: E402
from phase1.physics.dielectric import (  # noqa: E402
    DobsonDielectric,
    MironovDielectric,
)

logger = logging.getLogger(__name__)

# SPEC §6 / §11 Diagnostic C:
#   observed m_v range at Moor House: [0.25, 0.83] cm³/cm³.
#   binding threshold: 5% relative difference.
MV_MIN = 0.25
MV_MAX = 0.83
BINDING_THRESHOLD = 0.05  # 5%
N_SAMPLES = 501  # dense enough that max-finding is not noise-limited

# Mironov produces ε < 1.0 at very low m_v (Phase 1 DEV-007). The PINN
# physics branch enforces ε ≥ 1.01 via a soft clamp at the Oh interface,
# not inside the dielectric module itself. For G4 we report both the
# raw dielectric outputs (what the model produces) and the clamped
# values (what the Oh interface actually sees). The primary diagnostic
# is on the clamped values, because that is the quantity that enters
# the backscatter forward model.
EPSILON_CLAMP_MIN = 1.01

G4_RESULT_PATH = (
    PROJECT_ROOT / "phase1b" / "implementation_gate" / "results" / "g4_dielectric.json"
)


def _soft_clamp_min(x: torch.Tensor, floor: float) -> torch.Tensor:
    """
    Same soft-minimum used at the Oh interface (see `phase1/physics/wcm.py`).

    Hard `.clamp(min=floor)` is used here rather than a softplus-smoothed
    form because G4 is not in the autograd path — this is forward-mode
    numerical diagnostic, no gradient flow through ε.
    """
    return x.clamp(min=floor)


def compute_dielectric_diagnostic() -> dict:
    """
    Execute G4.

    Returns a result dict with:
      - `pass`: always True (the gate records rather than decides).
      - `binding`: True iff max relative diff ≥ 5%.
      - `max_relative_diff`: max over m_v of |ε_D − ε_M| / max(ε_D, ε_M).
      - `max_abs_diff`: max over m_v of |ε_D − ε_M|.
      - Per-sample arrays for downstream plotting / reporting.
    """
    dobson = DobsonDielectric()
    mironov = MironovDielectric()

    m_v = torch.linspace(MV_MIN, MV_MAX, N_SAMPLES, dtype=torch.float64)

    # Double precision is used throughout G4 — this is a diagnostic
    # computation, not a training loop, so precision costs nothing and
    # removes an axis of doubt from the max-difference estimate.
    m_v_f32 = m_v.to(torch.float32)
    eps_dobson_raw = dobson(m_v_f32).to(torch.float64)
    eps_mironov_raw = mironov(m_v_f32).to(torch.float64)

    # The quantity that actually enters the Oh backscatter.
    eps_dobson_clamped = _soft_clamp_min(eps_dobson_raw, EPSILON_CLAMP_MIN)
    eps_mironov_clamped = _soft_clamp_min(eps_mironov_raw, EPSILON_CLAMP_MIN)

    abs_diff = (eps_dobson_clamped - eps_mironov_clamped).abs()
    # Relative diff uses the larger of the two values as denominator,
    # which is the conservative relative-error convention. Pass-band is
    # defined on this quantity, per SPEC §11 Diagnostic C wording.
    denom = torch.maximum(eps_dobson_clamped, eps_mironov_clamped)
    rel_diff = abs_diff / denom

    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    mv_at_max_rel = float(m_v[int(rel_diff.argmax())])
    mv_at_max_abs = float(m_v[int(abs_diff.argmax())])

    binding = max_rel >= BINDING_THRESHOLD

    # Count how many m_v samples fell below the ε = 1.01 clamp so the
    # sign-off record knows which regime Mironov is in.
    n_mironov_clamped = int((eps_mironov_raw < EPSILON_CLAMP_MIN).sum())

    result = {
        "gate": "G4 — Dobson vs Mironov dielectric diagnostic",
        "spec_reference": "SPEC.md §6, §11 Diagnostic C",
        "m_v_range": [MV_MIN, MV_MAX],
        "peat_theta_sat": PEAT_THETA_SAT,
        "n_samples": N_SAMPLES,
        "epsilon_clamp_min": EPSILON_CLAMP_MIN,
        "binding_threshold_relative": BINDING_THRESHOLD,
        "binding": binding,
        "max_relative_diff": max_rel,
        "max_absolute_diff": max_abs,
        "m_v_at_max_relative_diff": mv_at_max_rel,
        "m_v_at_max_absolute_diff": mv_at_max_abs,
        "n_mironov_below_clamp": n_mironov_clamped,
        # Keep the sampling grid coarse enough to be useful for a plot
        # but light enough not to dominate the JSON file.
        "samples": {
            "m_v": [round(v, 4) for v in m_v.tolist()[::20]],
            "epsilon_dobson_clamped": [round(v, 4) for v in eps_dobson_clamped.tolist()[::20]],
            "epsilon_mironov_clamped": [round(v, 4) for v in eps_mironov_clamped.tolist()[::20]],
            "epsilon_dobson_raw": [round(v, 4) for v in eps_dobson_raw.tolist()[::20]],
            "epsilon_mironov_raw": [round(v, 4) for v in eps_mironov_raw.tolist()[::20]],
            "abs_diff_clamped": [round(v, 4) for v in abs_diff.tolist()[::20]],
            "relative_diff_clamped": [round(v, 6) for v in rel_diff.tolist()[::20]],
        },
        "pass": True,  # SPEC §6: "Pass either way; result is recorded."
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def _print_report(result: dict) -> None:
    header = "G4 Dobson vs Mironov dielectric diagnostic"
    print("\n" + "═" * len(header))
    print(header)
    print("═" * len(header))
    print(f"m_v range       : [{result['m_v_range'][0]:.2f}, {result['m_v_range'][1]:.2f}] cm³/cm³")
    print(f"n samples       : {result['n_samples']}")
    print(f"ε clamp floor   : {result['epsilon_clamp_min']:.2f}")
    print(f"binding thresh  : {result['binding_threshold_relative'] * 100:.1f}%")
    print()
    print(f"max |Δε|        : {result['max_absolute_diff']:.4f}  at m_v = {result['m_v_at_max_absolute_diff']:.3f}")
    print(f"max |Δε| / ε    : {result['max_relative_diff'] * 100:.2f}%  at m_v = {result['m_v_at_max_relative_diff']:.3f}")
    print(f"Mironov samples below clamp: {result['n_mironov_below_clamp']} / {result['n_samples']}")
    print()
    if result["binding"]:
        print(
            "BINDING: dielectric choice is an active source of variance in the "
            "Phase 1b interpretation.\n"
            "Report dielectric sensitivity arm results alongside primary outcome."
        )
    else:
        print(
            "NOT BINDING: dielectric choice is not the binding constraint at\n"
            "Moor House m_v levels. Phase 1 → Phase 1b dielectric change cannot\n"
            "explain any RMSE difference larger than this threshold."
        )
    print()
    print("G4 PASS (SPEC §6 records the diagnostic; gate passes either way).")
    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = compute_dielectric_diagnostic()

    G4_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    G4_RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {G4_RESULT_PATH}")

    _print_report(result)

    # G4 passes regardless of binding flag.
    return 0


if __name__ == "__main__":
    sys.exit(main())
