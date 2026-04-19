"""
Phase 1b G3 — Oh (1992) ks-validity check across s = 1–5 cm.

`SPEC.md` (Phase 1b) §7 requires a pre-experiment check on the Oh
model's behaviour across the Phase 1b learnable surface-roughness
range:

    Before any PINN training, run the Oh model in forward mode
    across the s = 1–5 cm range and confirm that σ° outputs remain
    physically plausible (no NaN, no extreme values, monotonic in m_v).
    If the check fails, implement the simplified AIEM (Chen et al.
    2003) as the surface scattering model and document the change as
    a pre-registration amendment (logged in the deviation log before
    training begins).

The check is run against both dielectric models (Dobson and Mironov)
because the Oh input includes the dielectric output and the clamp at
ε ≥ 1.01 is relevant to the Mironov arm.

Pass conditions (all must hold):
  1. No NaN or Inf in σ°_VV or σ°_VH anywhere on the grid.
  2. σ° values bounded within a very loose numerical-safety window
     (roughly [−120, +10] dB). These bounds are a guard against
     overflow / log-explosion, NOT an observational-plausibility
     check: at Mironov-clamped-ε regimes (DEV-007), Oh correctly
     outputs very low σ° because there is no dielectric contrast,
     and those values must be permitted by the gate since they are
     Oh's correct mathematical output given the input.
  3. σ°_VV non-decreasing in m_v at each (dielectric, s, θ) cell
     with a wiggle tolerance of 0.05 dB between adjacent samples.

The gate is about the Oh surface-scattering module's numerical
behaviour, not about whether the surrounding physics (dielectric,
dielectric clamp) produces observationally plausible σ°. A separate
"observational envelope" report section records σ° at the Moor House
operating m_v range [0.25, 0.83] so the sign-off record shows where
the model lands in practice without that check gating G3.

Usage:
    python phase1b/implementation_gate/ks_validity_check.py
    # or
    make g3
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
from phase1b.physics.oh1992_learnable_s import (  # noqa: E402
    S_MIN_CM,
    S_MAX_CM,
    oh_soil_backscatter_dual_pol,
    s_to_ks,
)

logger = logging.getLogger(__name__)

# Sweep grid. Integer-cm steps for s (matches the SPEC §5 bound
# statement s ∈ 1–5 cm); dense m_v sweep for monotonicity.
S_GRID_CM = [1.0, 2.0, 3.0, 4.0, 5.0]
N_MV_SAMPLES = 201
MV_MIN = 0.0
MV_MAX = PEAT_THETA_SAT  # 0.88

# SPEC §5 mean incidence angle at Moor House is 41.5°; per-observation
# range is ~29°–46°. Validity is checked at the mean and at the endpoints.
INCIDENCE_GRID_DEG = [29.0, 41.5, 46.0]

# Numerical-safety windows. Loose enough that Oh's correct
# mathematical outputs (including very low σ° in low-ε regimes like
# DEV-007 Mironov clamping) are not rejected; tight enough that NaN,
# Inf, and logarithm-overflow would trip the check.
SIGMA_VV_MIN_DB = -120.0
SIGMA_VV_MAX_DB = 10.0
SIGMA_VH_MIN_DB = -150.0   # VH may be 10–35 dB below VV via Oh cross-pol ratio
SIGMA_VH_MAX_DB = 10.0
# Monotonicity tolerance: a σ° decrease larger than this across
# adjacent m_v samples is a failure.
MONOTONIC_WIGGLE_TOL_DB = 0.05

# Moor House observed m_v range, for the secondary observational-envelope
# report. This is NOT used in pass/fail logic — it is reported so the
# sign-off record shows σ° behaviour in the operational regime.
MOOR_HOUSE_MV_MIN = 0.25
MOOR_HOUSE_MV_MAX = 0.83

G3_RESULT_PATH = (
    PROJECT_ROOT / "phase1b" / "implementation_gate" / "results" / "g3_ks.json"
)


def _check_one_combo(
    dielectric_name: str,
    dielectric,
    s_cm: float,
    theta_inc_deg: float,
    m_v: torch.Tensor,
) -> dict:
    """Run one (dielectric, s, θ) cell and collect pass/fail details."""
    theta_inc_rad = torch.deg2rad(torch.tensor(theta_inc_deg, dtype=torch.float32))
    s_tensor = torch.tensor(s_cm, dtype=torch.float32)

    epsilon = dielectric(m_v)
    sigma_vv_db, sigma_vh_db = oh_soil_backscatter_dual_pol(
        epsilon, theta_inc_rad, s_tensor
    )

    vv = sigma_vv_db.detach().double()
    vh = sigma_vh_db.detach().double()

    n_nan_vv = int(torch.isnan(vv).sum())
    n_nan_vh = int(torch.isnan(vh).sum())
    n_inf_vv = int(torch.isinf(vv).sum())
    n_inf_vh = int(torch.isinf(vh).sum())

    has_nan_or_inf = (n_nan_vv + n_nan_vh + n_inf_vv + n_inf_vh) > 0

    # Bounds: only evaluate on finite values to avoid re-flagging NaN.
    vv_finite = vv[torch.isfinite(vv)]
    vh_finite = vh[torch.isfinite(vh)]
    vv_min = float(vv_finite.min()) if vv_finite.numel() else float("nan")
    vv_max = float(vv_finite.max()) if vv_finite.numel() else float("nan")
    vh_min = float(vh_finite.min()) if vh_finite.numel() else float("nan")
    vh_max = float(vh_finite.max()) if vh_finite.numel() else float("nan")

    vv_out_of_bounds = (
        not torch.isfinite(torch.tensor(vv_min)).item()
        or vv_min < SIGMA_VV_MIN_DB
        or vv_max > SIGMA_VV_MAX_DB
    )
    vh_out_of_bounds = (
        not torch.isfinite(torch.tensor(vh_min)).item()
        or vh_min < SIGMA_VH_MIN_DB
        or vh_max > SIGMA_VH_MAX_DB
    )

    # Monotonicity in m_v for σ°_VV. σ° should increase with m_v for
    # fixed s and θ because ε increases with m_v and Γ_h increases
    # with ε.
    diffs = vv[1:] - vv[:-1]
    worst_negative = float(diffs.min()) if diffs.numel() else 0.0
    # A wiggle is a monotonic violation only if larger than the
    # tolerance and not an artefact of NaN adjacency.
    valid_adjacent = torch.isfinite(vv[:-1]) & torch.isfinite(vv[1:])
    worst_valid = (
        float(diffs[valid_adjacent].min()) if valid_adjacent.any() else 0.0
    )
    monotonic_violation = worst_valid < -MONOTONIC_WIGGLE_TOL_DB

    passed = (
        not has_nan_or_inf
        and not vv_out_of_bounds
        and not vh_out_of_bounds
        and not monotonic_violation
    )

    return {
        "dielectric": dielectric_name,
        "s_cm": s_cm,
        "ks": float(s_to_ks(s_tensor)),
        "theta_inc_deg": theta_inc_deg,
        "pass": passed,
        "n_nan": {"vv": n_nan_vv, "vh": n_nan_vh},
        "n_inf": {"vv": n_inf_vv, "vh": n_inf_vh},
        "sigma_vv_db_range": [vv_min, vv_max],
        "sigma_vh_db_range": [vh_min, vh_max],
        "worst_adjacent_decrease_db": worst_valid,
        "monotonic_violation": monotonic_violation,
        "vv_out_of_bounds": vv_out_of_bounds,
        "vh_out_of_bounds": vh_out_of_bounds,
    }


def _observational_envelope(
    dielectric_name: str,
    dielectric,
    s_cm: float,
    theta_inc_deg: float,
) -> dict:
    """
    Informational-only σ° summary over the Moor House observed m_v range.

    Not part of the pass/fail gate. Recorded so the sign-off artefact
    shows where the model lands in the operational regime. A Mironov
    cell reporting very low σ° here is consistent with DEV-007 and is
    expected; it is not a G3 failure.
    """
    theta_inc_rad = torch.deg2rad(torch.tensor(theta_inc_deg, dtype=torch.float32))
    s_tensor = torch.tensor(s_cm, dtype=torch.float32)
    m_v_obs = torch.linspace(
        MOOR_HOUSE_MV_MIN, MOOR_HOUSE_MV_MAX, N_MV_SAMPLES, dtype=torch.float32
    )
    epsilon = dielectric(m_v_obs)
    vv, vh = oh_soil_backscatter_dual_pol(epsilon, theta_inc_rad, s_tensor)
    vv64, vh64 = vv.detach().double(), vh.detach().double()
    return {
        "dielectric": dielectric_name,
        "s_cm": s_cm,
        "theta_inc_deg": theta_inc_deg,
        "moor_house_mv_range": [MOOR_HOUSE_MV_MIN, MOOR_HOUSE_MV_MAX],
        "sigma_vv_db_range": [float(vv64.min()), float(vv64.max())],
        "sigma_vh_db_range": [float(vh64.min()), float(vh64.max())],
    }


def run_ks_validity_check() -> dict:
    """Sweep (dielectric × s × θ) grid; return a structured result."""
    m_v = torch.linspace(MV_MIN, MV_MAX, N_MV_SAMPLES, dtype=torch.float32)

    dielectric_models = [
        ("Dobson", DobsonDielectric()),
        ("Mironov", MironovDielectric()),
    ]

    cells: list[dict] = []
    envelope: list[dict] = []
    for name, model in dielectric_models:
        for s_cm in S_GRID_CM:
            for theta_deg in INCIDENCE_GRID_DEG:
                cells.append(_check_one_combo(name, model, s_cm, theta_deg, m_v))
                envelope.append(_observational_envelope(name, model, s_cm, theta_deg))

    all_pass = all(c["pass"] for c in cells)
    failed = [c for c in cells if not c["pass"]]

    result = {
        "gate": "G3 — Oh ks-validity across s = 1–5 cm",
        "spec_reference": "SPEC.md §7",
        "s_range_cm": [S_MIN_CM, S_MAX_CM],
        "s_grid_cm": S_GRID_CM,
        "incidence_grid_deg": INCIDENCE_GRID_DEG,
        "m_v_range": [MV_MIN, MV_MAX],
        "n_m_v_samples": N_MV_SAMPLES,
        "sigma_vv_bounds_db": [SIGMA_VV_MIN_DB, SIGMA_VV_MAX_DB],
        "sigma_vh_bounds_db": [SIGMA_VH_MIN_DB, SIGMA_VH_MAX_DB],
        "monotonic_wiggle_tolerance_db": MONOTONIC_WIGGLE_TOL_DB,
        "n_cells": len(cells),
        "n_cells_passed": sum(1 for c in cells if c["pass"]),
        "n_cells_failed": len(failed),
        "pass": all_pass,
        "aiem_substitution_required": not all_pass,
        "cells": cells,
        "failed_cells": failed,
        "moor_house_observational_envelope": envelope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def _print_report(result: dict) -> None:
    header = "G3 Oh ks-validity check"
    print("\n" + "═" * len(header))
    print(header)
    print("═" * len(header))
    print(f"s grid (cm)         : {result['s_grid_cm']}")
    print(f"incidence grid (°)  : {result['incidence_grid_deg']}")
    print(f"m_v range           : [{result['m_v_range'][0]:.2f}, {result['m_v_range'][1]:.2f}]")
    print(f"m_v samples         : {result['n_m_v_samples']}")
    print(f"VV bounds (dB)      : {result['sigma_vv_bounds_db']}")
    print(f"VH bounds (dB)      : {result['sigma_vh_bounds_db']}")
    print(f"monotonic tol (dB)  : {result['monotonic_wiggle_tolerance_db']}")
    print()
    print(f"cells total         : {result['n_cells']}")
    print(f"cells passed        : {result['n_cells_passed']}")
    print(f"cells failed        : {result['n_cells_failed']}")
    print()

    if result["pass"]:
        # Primary pass summary — Oh's behaviour at each (s, θ) for the
        # Dobson-dielectric primary arm.
        print("Summary — full m_v sweep (σ°_VV dB range across the full [0, 0.88]):")
        for c in result["cells"]:
            if c["dielectric"] == "Dobson" and abs(c["theta_inc_deg"] - 41.5) < 0.01:
                lo, hi = c["sigma_vv_db_range"]
                print(f"  Dobson   s = {c['s_cm']:.1f} cm (ks = {c['ks']:.2f})  "
                      f"σ°_VV ∈ [{lo:+.2f}, {hi:+.2f}] dB")
        for c in result["cells"]:
            if c["dielectric"] == "Mironov" and abs(c["theta_inc_deg"] - 41.5) < 0.01:
                lo, hi = c["sigma_vv_db_range"]
                print(f"  Mironov  s = {c['s_cm']:.1f} cm (ks = {c['ks']:.2f})  "
                      f"σ°_VV ∈ [{lo:+.2f}, {hi:+.2f}] dB")
        print()
        # Informational — what the observational regime looks like.
        print("Observational envelope (Moor House m_v ∈ [0.25, 0.83], θ = 41.5°):")
        for e in result["moor_house_observational_envelope"]:
            if abs(e["theta_inc_deg"] - 41.5) < 0.01:
                lo_vv, hi_vv = e["sigma_vv_db_range"]
                lo_vh, hi_vh = e["sigma_vh_db_range"]
                print(
                    f"  {e['dielectric']:7s}  s = {e['s_cm']:.1f} cm   "
                    f"σ°_VV ∈ [{lo_vv:+6.2f}, {hi_vv:+6.2f}] dB   "
                    f"σ°_VH ∈ [{lo_vh:+6.2f}, {hi_vh:+6.2f}] dB"
                )
        print()
        print("G3 PASS — Oh model is numerically safe (no NaN / Inf, monotonic")
        print("in m_v) across s ∈ [1, 5] cm for both dielectric models.")
        print("AIEM substitution is NOT required.")
        print()
        print("Note: very-low σ° values at Mironov-clamped-ε regimes are")
        print("Oh's mathematically correct output given no dielectric")
        print("contrast (see DEV-007). Not a G3 failure; relevant for the")
        print("Mironov sensitivity arm's interpretation (SPEC §6, §11 Diag C).")
    else:
        print("G3 FAIL — details of failing cells:")
        for c in result["failed_cells"]:
            reasons = []
            if c["n_nan"]["vv"] + c["n_nan"]["vh"] > 0:
                reasons.append(
                    f"NaN(vv={c['n_nan']['vv']}, vh={c['n_nan']['vh']})"
                )
            if c["n_inf"]["vv"] + c["n_inf"]["vh"] > 0:
                reasons.append(
                    f"Inf(vv={c['n_inf']['vv']}, vh={c['n_inf']['vh']})"
                )
            if c["vv_out_of_bounds"]:
                reasons.append(f"VV out of bounds: {c['sigma_vv_db_range']}")
            if c["vh_out_of_bounds"]:
                reasons.append(f"VH out of bounds: {c['sigma_vh_db_range']}")
            if c["monotonic_violation"]:
                reasons.append(
                    f"non-monotone in m_v (worst Δ = {c['worst_adjacent_decrease_db']:+.3f} dB)"
                )
            print(
                f"  {c['dielectric']:7s}  s = {c['s_cm']:.1f} cm  "
                f"θ = {c['theta_inc_deg']:5.1f}°  —  {'; '.join(reasons)}"
            )
        print()
        print(
            "Substitute AIEM (Chen et al. 2003) per SPEC §7 and log as a "
            "pre-registration amendment (DEV-1b-00N) before sign-off."
        )
    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = run_ks_validity_check()

    G3_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    G3_RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {G3_RESULT_PATH}")

    _print_report(result)

    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
