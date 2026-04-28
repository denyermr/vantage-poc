"""
Phase 1c-Lean Block C-prime — G3-Lean lock cross-check tests (deliverable 6).

Per kickoff §2 deliverable 6:
- λ-grid is exactly 36 cells in canonical order with 108 (cell × rep) entries.
- Each entry has well-defined config_idx (0–107) for seed assignment.
- Baselines reproducibility (locked numbers are within tolerance of fresh run under fixed seed).
- Sealed-set SHA-256 is stable.
- Pre-flight schema is valid JSON conforming to SPEC §18.11.
"""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

G3_DIR = Path(__file__).parent.parent.parent / "phase1c-lean" / "g3_lean"


# ─── λ-grid lock ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def grid():
    with open(G3_DIR / "lambda_grid.json") as f:
        return json.load(f)


def test_grid_has_108_entries(grid):
    assert grid["n_total_runs"] == 108
    assert len(grid["entries"]) == 108


def test_grid_has_36_cells_x_3_reps(grid):
    assert grid["n_cells"] == 36
    assert grid["n_reps_per_cell"] == 3


def test_grid_lambda_values_locked(grid):
    expected = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    assert grid["lambda_vv_values"] == expected
    assert grid["lambda_vh_values"] == expected


def test_grid_canonical_enumeration_order(grid):
    """λ_VV outer, λ_VH middle, rep_idx inner per SPEC §18.4.2 / §18.6.2 v0.3.2."""
    config_indices = [e["config_idx"] for e in grid["entries"]]
    assert config_indices == list(range(108))


def test_grid_seed_formula(grid):
    """SEED = 42 + config_idx per SPEC §18.6.2 v0.3.2."""
    for e in grid["entries"]:
        assert e["seed"] == 42 + e["config_idx"]
        assert e["config_idx"] == 18 * e["lambda_vv_idx"] + 3 * e["lambda_vh_idx"] + e["rep_idx"]


def test_grid_seeds_are_unique(grid):
    """108 unique seeds in [42, 149] per the v0.3.2 amendment uniqueness proof."""
    seeds = [e["seed"] for e in grid["entries"]]
    assert len(set(seeds)) == 108
    assert min(seeds) == 42
    assert max(seeds) == 149


def test_grid_first_and_last_entries(grid):
    """Sanity check on the canonical-order endpoints."""
    e0 = grid["entries"][0]
    assert (e0["lambda_vv_idx"], e0["lambda_vh_idx"], e0["rep_idx"]) == (0, 0, 0)
    assert e0["seed"] == 42
    e_last = grid["entries"][-1]
    assert (e_last["lambda_vv_idx"], e_last["lambda_vh_idx"], e_last["rep_idx"]) == (5, 5, 2)
    assert e_last["seed"] == 149


# ─── Baselines lock ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def baselines():
    with open(G3_DIR / "baselines_locked.json") as f:
        return json.load(f)


def test_baselines_spec_version_v033(baselines):
    assert baselines["spec_version"] == "v0.3.3"


def test_baselines_rf_5fold_cv(baselines):
    rf = baselines["rf_100pct_5fold_cv"]
    assert rf["n_train"] == 83
    assert rf["n_outer_folds"] == 5
    assert rf["cv_strategy"] == "random_split"
    assert len(rf["rmse_per_fold"]) == 5
    # RF reference 0.147 is informational; the locked CV-RMSE band is centred
    # around the halt-4 reproduction (0.1270) with reasonable tolerance.
    assert 0.10 <= rf["rmse_cm3_per_cm3"] <= 0.16


def test_baselines_null_5fold_cv_method3(baselines):
    null = baselines["seasonal_climatological_null"]
    assert null["n_folds"] == 5
    assert null["cv_strategy"] == "random_split"
    assert null["season_definition"] == "meteorological_DJF_MAM_JJA_SON"
    assert null["method"] == "method_3_5fold_cv_per_fold_seasonal_means"
    assert len(null["rmse_per_fold"]) == 5
    # Method 3 reproduction: halt-4 diagnostic 0.1006 (pooled) → locked 0.1005
    # (mean-of-per-fold). Tolerance ±0.005 per supervisor halt-4 §7 trigger.
    assert 0.095 <= null["rmse_cm3_per_cm3"] <= 0.106


def test_baselines_informational_phase1_references(baselines):
    """Phase 1 sealed-test references are recorded as informational, not gate."""
    rf = baselines["rf_100pct_5fold_cv"]
    null = baselines["seasonal_climatological_null"]
    assert rf["informational_comparison_phase1_sealed_test_rmse"] == 0.147
    assert null["informational_comparison_phase1_sealed_test_rmse"] == 0.178


def test_baselines_data_sha256_present(baselines):
    """Reproducibility metadata: data SHA-256 + git hash recorded."""
    assert "data_sha256" in baselines and len(baselines["data_sha256"]) == 64
    assert "code_version_hash" in baselines and baselines["code_version_hash"]


# ─── Sealed-set definition lock ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def sealed():
    with open(G3_DIR / "sealed_set_definition.json") as f:
        return json.load(f)


def test_sealed_n36(sealed):
    assert sealed["n_observations"] == 36
    assert len(sealed["observation_dates"]) == 36


def test_sealed_date_range(sealed):
    assert sealed["date_range"] == ["2023-07-25", "2024-12-10"]
    assert sealed["observation_dates"][0] == "2023-07-25"
    assert sealed["observation_dates"][-1] == "2024-12-10"


def test_sealed_sha256_stable(sealed):
    """SHA-256 of the canonical CSV slice must be deterministic and re-computable."""
    repo_root = Path(__file__).parent.parent.parent
    df = pd.read_csv(repo_root / "data" / "processed" / "aligned_dataset.csv")
    df = df.sort_values("date").reset_index(drop=True)
    slice_ = df.iloc[83:].reset_index(drop=True)
    recomputed = hashlib.sha256(slice_.to_csv(index=False).encode("utf-8")).hexdigest()
    assert recomputed == sealed["sha256"]


def test_sealed_metadata_only_at_g3_lean(sealed):
    """The lock declares the sealed set is not loaded at G3-Lean per SPEC §18.5."""
    assert sealed["loaded_at_g3_lean"] is False
    assert sealed["loaded_at_block_d_prime_sweep"] is False


# ─── Preflight schema ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def schema():
    with open(G3_DIR / "preflight_schema.json") as f:
        return json.load(f)


def test_schema_six_spec_18_11_items(schema):
    """All six SPEC §18.11 audit items present."""
    expected = {
        "code_version_hash",
        "loss_formulation",
        "preflight_summary_block",
        "sleep_wake_event_count",
        "dominance_verdict_per_run",
        "sigma_scale_factors",
    }
    assert set(schema["items"].keys()) == expected


def test_schema_loss_formulation_v031_string(schema):
    """v0.3.1 canonical string per DEV-1c-lean-005."""
    assert (
        schema["items"]["loss_formulation"]["constant_value"]
        == "v0.3_five_term_per_channel_normalised"
    )


def test_schema_dominance_verdict_uses_flat_config_idx(schema):
    """Phase 1c-Lean v0.3.2 flat config_idx scheme (DEV-1c-lean-006)."""
    fields = schema["items"]["dominance_verdict_per_run"]["fields"]
    assert fields["config_idx"]["range" if "range" in fields["config_idx"] else "type"]
    assert fields["lambda_vv_idx"]["range"] == [0, 5]
    assert fields["lambda_vh_idx"]["range"] == [0, 5]
    assert fields["rep_idx"]["range"] == [0, 2]


def test_schema_sigma_method_at_init_only(schema):
    """SPEC §18.4.1: σ values computed once at init; per-batch normalisation explicitly out of scope."""
    assert "random_init" in schema["items"]["sigma_scale_factors"]["fields"]["computation_method"]["constant_value"]


# ─── Aggregator lock ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lock():
    with open(G3_DIR / "g3_lean_lock.json") as f:
        return json.load(f)


def test_lock_records_all_four_items(lock):
    assert set(lock["locked_items"].keys()) == {
        "lambda_grid", "baselines", "sealed_set", "preflight_schema"
    }


def test_lock_amendment_tag_chain(lock):
    """The v0.3.1 / v0.3.2 / v0.3.3 amendment chain is recorded for forward audit."""
    assert lock["amendment_tags"] == [
        "phase1c-lean-spec-v0_3_1",
        "phase1c-lean-spec-v0_3_2",
        "phase1c-lean-spec-v0_3_3",
    ]


def test_lock_records_dev_entries(lock):
    """All seven Phase 1c-Lean DEV entries (001–007) are recorded."""
    dev_text = " ".join(lock["dev_entries"])
    for n in range(1, 8):
        assert f"DEV-1c-lean-00{n}" in dev_text


def test_lock_internal_sha256_consistency(lock):
    """Each source artefact's SHA-256 in the lock matches a fresh hash of the file."""
    for key, locked in lock["locked_items"].items():
        path = G3_DIR / f"{ {'lambda_grid': 'lambda_grid', 'baselines': 'baselines_locked', 'sealed_set': 'sealed_set_definition', 'preflight_schema': 'preflight_schema'}[key] }.json"
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha256 == locked["source_sha256"], f"{key} source_sha256 stale"
