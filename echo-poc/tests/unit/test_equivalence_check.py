"""
Unit tests for `phase1b/physics/equivalence_check.py` (G2 three-arm harness).

Scope (Session D):
  - Anchor reads JSON loads with the expected schema shape (5 sets,
    expected row counts per set).
  - Per-row delta computation.
  - Pass/fail exit code at the 0.5 dB threshold.
  - Result JSON schema matches g3_ks.json / g4_dielectric.json family
    (gate, spec_reference, arms, pass, generated_at).
  - Structural invariants on canonical_combinations.json
    (numpy_port count unchanged at 36; every row has citation fields).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_JSON = REPO_ROOT / "phase1b" / "refs" / "anchor_reads" / "anchor_reads_v1.json"
CANONICAL_JSON = (
    REPO_ROOT / "phase1b" / "physics" / "reference_mimics"
    / "canonical_combinations.json"
)
G2_RESULT_JSON = REPO_ROOT / "outputs" / "g2_equivalence_result.json"
SCRIPT = REPO_ROOT / "phase1b" / "physics" / "equivalence_check.py"


# ─── Anchor reads JSON schema ───────────────────────────────────────────────


class TestAnchorReadsSchema:
    """anchor_reads_v1.json must be loadable and have the expected shape."""

    def test_file_exists(self):
        assert ANCHOR_JSON.exists(), f"{ANCHOR_JSON} not staged"

    def test_loads_as_json(self):
        data = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_all_five_sets(self):
        data = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
        for key in ["set_A", "set_B", "set_C", "set_D", "set_E"]:
            assert key in data, f"{key} missing from anchor reads"

    def test_set_row_counts(self):
        """v0.2 spec: A=4, B=4, C=5 (C.3 dropped), D=4, E=5; total 22."""
        data = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
        assert len(data["set_A"]["rows"]) == 4
        assert len(data["set_B"]["rows"]) == 4
        assert len(data["set_D"]["rows"]) == 4
        assert len(data["set_E"]["rows"]) == 5
        # Set C has 6 entries but one is null (dropped); counted for 5 live.
        c_entries = data["set_C"]["human_reads_at_theta_30"]
        n_live = sum(1 for v in c_entries.values() if v["refined_dB"] is not None)
        assert n_live == 5
        assert len(c_entries) == 6  # all preserved, one dropped

    def test_all_rows_have_source_citation(self):
        """Every anchor row must cite its source (page/table/row)."""
        data = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
        for s_key in ["set_A", "set_B", "set_D", "set_E"]:
            for row in data[s_key]["rows"]:
                assert "source" in row or "confidence" in row, \
                    f"{s_key}/{row.get('row_id')} missing source/confidence"
        for row_id, row in data["set_C"]["human_reads_at_theta_30"].items():
            # Set C rows cite marker + mechanism + confidence, not "source" literally
            assert "marker" in row or "note" in row, \
                f"{row_id} missing marker/note"

    def test_summary_stats(self):
        data = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
        assert data["summary_stats"]["n_anchors_refined_v1"] == 22
        assert data["summary_stats"]["n_anchors_dropped"] == 1
        assert data["summary_stats"]["dropped_ids"] == ["C.3"]


# ─── canonical_combinations.json invariants ─────────────────────────────────


class TestCanonicalCombinationsInvariants:
    """numpy_port count unchanged at 36; published_table additions permitted."""

    def test_numpy_port_count_unchanged(self):
        data = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        numpy_port = [c for c in data["combinations"] if c["source"]["type"] == "numpy_port"]
        assert len(numpy_port) == 36, \
            f"numpy_port count expected 36, got {len(numpy_port)}"

    def test_schema_version_is_1(self):
        data = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        assert data.get("schema_version") == 1

    def test_all_numpy_port_entries_have_code_sha256(self):
        data = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        for c in data["combinations"]:
            if c["source"]["type"] == "numpy_port":
                sha = c["source"].get("code_sha256")
                assert isinstance(sha, str) and len(sha) == 64, \
                    f"numpy_port entry {c['id']} missing/invalid code_sha256"


# ─── equivalence_check.py — per-row delta and pass threshold ────────────────


class TestDeltaAndPassLogic:
    """Unit tests on the harness's delta-computation helper."""

    def test_delta_db_exact(self):
        from phase1b.physics.equivalence_check import _delta_db
        assert _delta_db(-10.0, -10.0) == 0.0
        assert _delta_db(-10.5, -10.0) == pytest.approx(0.5)
        assert _delta_db(-10.0, -10.5) == pytest.approx(0.5)
        assert _delta_db(-9.0, -10.0) == pytest.approx(1.0)

    def test_delta_db_matches_abs_subtraction(self):
        from phase1b.physics.equivalence_check import _delta_db
        import random
        random.seed(0)
        for _ in range(50):
            a = random.uniform(-50, 10)
            b = random.uniform(-50, 10)
            assert _delta_db(a, b) == abs(a - b)


# ─── equivalence_check.py — end-to-end run produces the expected JSON ───────


class TestHarnessEndToEnd:
    """Run the harness once and inspect the result JSON schema."""

    @pytest.fixture(scope="class")
    def result(self):
        # The harness is expected to exit non-zero in Session D scope
        # (published_table and gradient arms fail by design — real
        # physics disagreements between v0.1 mimics.py and the external
        # T94/M90 anchors). We still want the JSON written, so we run
        # via subprocess and tolerate any exit code.
        py = REPO_ROOT / ".venv" / "bin" / "python"
        subprocess.run(
            [str(py), str(SCRIPT)],
            capture_output=True, timeout=120, cwd=str(REPO_ROOT),
        )
        assert G2_RESULT_JSON.exists(), f"{G2_RESULT_JSON} not created"
        return json.loads(G2_RESULT_JSON.read_text(encoding="utf-8"))

    def test_top_level_schema(self, result):
        """Mirrors g3_ks.json / g4_dielectric.json top-level fields."""
        for key in ["gate", "spec_reference", "arms", "pass", "generated_at"]:
            assert key in result, f"missing top-level key '{key}'"

    def test_three_arms(self, result):
        arms = result["arms"]
        for arm_key in ["numpy_port", "published_table", "gradient"]:
            assert arm_key in arms
            assert "pass" in arms[arm_key]

    def test_numpy_port_arm_passes(self, result):
        """Expected to pass in Session D: PyTorch and numpy should agree."""
        assert result["arms"]["numpy_port"]["pass"] is True

    def test_published_table_has_four_sets(self, result):
        sets = result["arms"]["published_table"]["sets"]
        for s in ["A", "B", "C", "D"]:
            assert s in sets

    def test_gradient_arm_five_rows(self, result):
        assert result["arms"]["gradient"]["n_rows"] == 5

    def test_overall_pass_is_conjunction(self, result):
        """`pass` must be AND of the three arms."""
        a = result["arms"]["numpy_port"]["pass"]
        b = result["arms"]["published_table"]["pass"]
        c = result["arms"]["gradient"]["pass"]
        assert result["pass"] == (a and b and c)


# ─── Exit code test (binary threshold behaviour) ────────────────────────────


class TestExitCode:
    def test_script_exits_non_zero_when_any_arm_fails(self):
        py = REPO_ROOT / ".venv" / "bin" / "python"
        r = subprocess.run(
            [str(py), str(SCRIPT)],
            capture_output=True, timeout=120, cwd=str(REPO_ROOT),
        )
        result = json.loads(G2_RESULT_JSON.read_text(encoding="utf-8"))
        if result["pass"]:
            assert r.returncode == 0
        else:
            assert r.returncode != 0, "script must exit non-zero on G2 FAIL"


# ─── use_trunk_layer flag contract ─────────────────────────────────────────


class TestUseTrunkLayerFlag:
    """The flag must default to False and raise NotImplementedError when True."""

    def test_default_is_false(self):
        import inspect
        from phase1b.physics.mimics import mimics_toure_single_crown
        sig = inspect.signature(mimics_toure_single_crown)
        param = sig.parameters["use_trunk_layer"]
        assert param.default is False
        # And keyword-only (safety — cannot be accidentally set positionally)
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_true_raises_not_implemented(self):
        import torch
        from phase1b.physics.mimics import (
            MimicsToureParamsTorch, mimics_toure_single_crown,
        )
        params = MimicsToureParamsTorch()
        with pytest.raises(NotImplementedError, match="use_trunk_layer"):
            mimics_toure_single_crown(params, use_trunk_layer=True)

    def test_false_runs_normally(self):
        import torch
        from phase1b.physics.mimics import (
            MimicsToureParamsTorch, mimics_toure_single_crown,
        )
        params = MimicsToureParamsTorch()
        vv, vh = mimics_toure_single_crown(params)
        assert torch.isfinite(vv)
        assert torch.isfinite(vh)
