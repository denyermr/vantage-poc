"""
Phase 1b pre-registration lock — regression test.

Enforces the technical commitment in `phase1b/SUCCESS_CRITERIA.md` §7
and `SPEC.md` §15: once Phase 1b training begins, the pre-registered
success criteria cannot be silently edited.

Two fail conditions:

  1. **Post-training edit:** if any Phase 1b training artefact exists
     and `phase1b/SUCCESS_CRITERIA.md` has an mtime *later than* the
     earliest training artefact's mtime, fail. (An edit after training
     began is a pre-registration violation that must go through a
     DEV-1b-NNN entry, the same way DEV-1b-001 / DEV-1b-002 handled
     the pre-sign-off prior withdrawals.)

  2. **SPEC §10 threshold drift:** if the SPEC §10 RMSE thresholds
     (Strong < 0.124; Significant < 0.131; Moderate < 0.139;
     Inconclusive lower bound 0.139, upper bound 0.155;
     Negative ≥ 0.155) cannot be matched verbatim against the current
     `SPEC.md`, fail. The thresholds are signed at SPEC §14 (2026-04-19);
     silent drift is forbidden.

The test is dormant for fail condition (1) until any of the
`PHASE1B_TRAINING_ARTEFACT_GLOBS` produce matches. At the
pre-registration tag (`phase1b-success-criteria-pre-registered`), no
training artefact exists, so condition (1) is a no-op pass. Once
Block 2 (Session F-2 onwards) produces training output, the test
becomes active.

Scope:
  - Filesystem-only (mtime-based). Does not consult git history; the
    intent is to make ordinary `git checkout` flows safe — checking
    out an old commit changes mtimes to `now`, which preserves
    SUCCESS_CRITERIA.md ≤ artefact ordering on a clean tree.
  - The test does NOT consult `phase1b/deviation_log.md`. A criterion
    change documented in a fresh DEV-1b-NNN entry will still fail this
    test until the test is updated in the same commit to acknowledge
    the deviation. This is intentional — same honest-gates pattern as
    DEV-1b-001 / DEV-1b-002.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Repository root resolves from this test file:
# echo-poc/tests/unit/test_pre_registration_lock.py
#   parents[0] = echo-poc/tests/unit/
#   parents[1] = echo-poc/tests/
#   parents[2] = echo-poc/
ECHO_POC_ROOT = Path(__file__).resolve().parents[2]

SUCCESS_CRITERIA_PATH = ECHO_POC_ROOT / "phase1b" / "SUCCESS_CRITERIA.md"
SPEC_PATH = ECHO_POC_ROOT / "SPEC.md"

# Globs that match Phase 1b training artefacts. Block 2 (Session F-2
# onwards) writes to these locations. Adding a new training-output
# location requires extending this list.
PHASE1B_TRAINING_ARTEFACT_GLOBS: tuple[tuple[str, str], ...] = (
    # (relative root from ECHO_POC_ROOT, glob pattern)
    ("outputs/models/pinn_mimics", "**/*"),
    ("phase1b/lambda_search/results", "**/*"),
    ("phase1b/results", "**/*.json"),
    ("outputs/metrics", "phase1b_*.json"),
)

# SPEC §10 RMSE thresholds verbatim from the signed sign-off block.
# These strings are matched against SPEC.md as substring searches; if
# the thresholds drift, the substring lookup fails and the test fails.
# Each entry is (label, expected substring).
SPEC_S10_THRESHOLD_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("Strong < 0.124", "PINN-MIMICS RMSE < 0.124"),
    ("Significant < 0.131", "PINN-MIMICS RMSE < 0.131"),
    ("Moderate < 0.139", "PINN-MIMICS RMSE < 0.139"),
    ("Inconclusive 0.139 ≤ ... < 0.155", "0.139 ≤ PINN-MIMICS RMSE < 0.155"),
    ("Negative ≥ 0.155", "PINN-MIMICS RMSE ≥ 0.155"),
)


def _iter_phase1b_artefacts() -> list[Path]:
    """Return every existing Phase 1b training artefact under ECHO_POC_ROOT."""
    artefacts: list[Path] = []
    for rel_root, pattern in PHASE1B_TRAINING_ARTEFACT_GLOBS:
        root = ECHO_POC_ROOT / rel_root
        if not root.exists():
            continue
        for p in root.glob(pattern):
            if p.is_file():
                artefacts.append(p)
    return artefacts


def test_success_criteria_file_exists() -> None:
    """SUCCESS_CRITERIA.md must exist for the pre-registration to be locked."""
    assert SUCCESS_CRITERIA_PATH.is_file(), (
        f"phase1b/SUCCESS_CRITERIA.md missing at {SUCCESS_CRITERIA_PATH}. "
        "This file is the formal Phase 1b pre-registration; without it, the "
        "pre-registration tag is meaningless."
    )


def test_spec_phase1b_pre_registration_section_present() -> None:
    """SPEC.md must contain §15 Phase 1b Pre-Registration document section."""
    assert SPEC_PATH.is_file(), f"SPEC.md missing at {SPEC_PATH}"
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    assert "## 15. Phase 1b Pre-Registration document" in spec_text, (
        "SPEC.md §15 'Phase 1b Pre-Registration document' missing. The "
        "pre-registration cross-reference (added 2026-04-19) is required "
        "to bind SUCCESS_CRITERIA.md to the signed SPEC."
    )
    # Cross-reference to SUCCESS_CRITERIA.md must be present.
    assert "phase1b/SUCCESS_CRITERIA.md" in spec_text, (
        "SPEC.md §15 must cross-reference phase1b/SUCCESS_CRITERIA.md."
    )


@pytest.mark.parametrize("label,fragment", SPEC_S10_THRESHOLD_FRAGMENTS)
def test_spec_s10_thresholds_unchanged(label: str, fragment: str) -> None:
    """
    SPEC §10 RMSE thresholds must match the signed values verbatim.

    Drift detection. If a threshold is silently edited in SPEC.md, the
    substring lookup fails and this test fails — even before any
    training artefact exists.
    """
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    assert fragment in spec_text, (
        f"SPEC §10 threshold '{label}' (expected substring '{fragment}') not "
        f"found in SPEC.md. The thresholds were signed at SPEC §14 on "
        f"2026-04-19; silent edits are forbidden. If a threshold change is "
        f"intentional, log a DEV-1b-NNN entry and update this test in the "
        f"same commit."
    )


def test_success_criteria_not_modified_after_training_began() -> None:
    """
    Pre-registration discipline test.

    Fails if any Phase 1b training artefact exists AND SUCCESS_CRITERIA.md
    has an mtime later than the earliest training artefact's mtime.

    Dormant until training begins (no artefacts → no-op pass). Once
    Block 2 produces training output, the test becomes active.
    """
    if not SUCCESS_CRITERIA_PATH.is_file():
        pytest.fail(
            "SUCCESS_CRITERIA.md missing; cannot evaluate pre-registration "
            "lock. See test_success_criteria_file_exists for the primary "
            "assertion."
        )

    artefacts = _iter_phase1b_artefacts()
    if not artefacts:
        # Dormant case: pre-Block-2. The test passes trivially. This is
        # the expected state at the pre-registration tag.
        return

    sc_mtime = SUCCESS_CRITERIA_PATH.stat().st_mtime
    earliest_artefact = min(artefacts, key=lambda p: p.stat().st_mtime)
    earliest_mtime = earliest_artefact.stat().st_mtime

    if sc_mtime > earliest_mtime:
        rel = earliest_artefact.relative_to(ECHO_POC_ROOT)
        pytest.fail(
            f"Pre-registration lock violated: phase1b/SUCCESS_CRITERIA.md "
            f"was modified after Phase 1b training began.\n\n"
            f"  SUCCESS_CRITERIA.md mtime: {sc_mtime}\n"
            f"  Earliest training artefact: {rel}\n"
            f"  Earliest artefact mtime:    {earliest_mtime}\n\n"
            f"If the criterion change is intentional, log a DEV-1b-NNN entry "
            f"in phase1b/deviation_log.md with full impact assessment and "
            f"update this test in the same commit (extend the dormancy gate "
            f"or amend the assertion). Do NOT touch the timestamps to bypass "
            f"this test."
        )


def test_success_criteria_lock_metadata_present() -> None:
    """The pre-registration lock metadata block must be intact."""
    text = SUCCESS_CRITERIA_PATH.read_text(encoding="utf-8")
    required_strings = (
        "phase1b-success-criteria-pre-registered",
        "Date locked",
        "2026-04-19",
        "Lead investigator",
        "Matthew Denyer",
    )
    missing = [s for s in required_strings if s not in text]
    assert not missing, (
        f"SUCCESS_CRITERIA.md missing required lock-metadata strings: "
        f"{missing!r}. The lock metadata is what makes the tag meaningful; "
        f"removing it is a pre-registration violation."
    )


def test_success_criteria_secondary_thresholds_present() -> None:
    """
    SUCCESS_CRITERIA.md §2 secondary numerical thresholds are the
    Session F-1 deliverable. They must be present verbatim in the locked
    document.
    """
    text = SUCCESS_CRITERIA_PATH.read_text(encoding="utf-8")
    # Secondary thresholds (§2 of SUCCESS_CRITERIA.md). The test matches
    # the bolded threshold form, which is the canonical place each value
    # is asserted in the document.
    required_thresholds = (
        ("Secondary 1: forward fit r > 0.3", "**r > 0.3**"),
        ("Secondary 2: residual ratio < 2.0", "**< 2.0**"),
        ("Secondary 3: VH r > 0.2", "**r > 0.2**"),
        ("Secondary 4: residual-NDVI |r| < 0.5", "**< 0.5**"),
        ("Dominance: L_physics > 10%", "**> 10%**"),
    )
    missing = [
        label for label, fragment in required_thresholds if fragment not in text
    ]
    assert not missing, (
        f"SUCCESS_CRITERIA.md missing required secondary/dominance "
        f"thresholds: {missing!r}. These are Session F-1 deliverables and "
        f"must be locked at the pre-registration tag."
    )
