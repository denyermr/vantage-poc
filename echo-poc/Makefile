# ECHO PoC — Phase 1 (frozen) + Phase 1b (active).
#
# Phase 1 targets are retained for reproducibility of the sealed results;
# they point at the new paths (shared/, phase1/). Phase 1b targets run the
# four pre-training gates listed in CLAUDE_3.md.

PY := .venv/bin/python

.PHONY: setup test test-unit test-integration test-cov \
        phase-1 phase-2 phase-3 \
        gate-1-orig gate-2-orig gate-3-orig gate-4-orig gate-5-orig \
        g1 g2 g3 g4 p1b-ready \
        clean

# ─── Environment ────────────────────────────────────────────────────────────

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# ─── Tests ──────────────────────────────────────────────────────────────────

test:
	$(PY) -m pytest tests/ -v

test-unit:
	$(PY) -m pytest tests/unit/ -v

test-integration:
	$(PY) -m pytest tests/integration/ -v --require-data

test-cov:
	$(PY) -m pytest tests/ --cov=shared --cov=phase1 --cov-report=term-missing

# ─── Phase 1 (frozen) ────────────────────────────────────────────────────────
# Retained for reproducibility. These re-run the original pipeline. They do
# NOT overwrite the sealed artefacts in outputs/metrics/ that the G1 gate
# reads — that output path is in the script.

phase-1:
	$(PY) phase1/pipeline.py --phase 1

phase-2:
	$(PY) phase1/pipeline.py --phase 2

phase-3:
	$(PY) phase1/pipeline.py --phase 3

gate-1-orig:
	$(PY) phase1/gates/gate_1.py

gate-2-orig:
	$(PY) phase1/gates/gate_2.py

# ─── Phase 1b pre-training gates (see CLAUDE_3.md) ──────────────────────────

g1:
	$(PY) phase1/run_baselines.py --confirm

g2:
	$(PY) phase1b/physics/equivalence_check.py

g3:
	$(PY) phase1b/implementation_gate/ks_validity_check.py

g4:
	$(PY) phase1b/implementation_gate/dielectric_diagnostic.py

p1b-ready: g1 g2 g3 g4
	@echo "All Phase 1b gates passed. Update SPEC.md §14 with hashes + signatures."

# ─── House-keeping ──────────────────────────────────────────────────────────

clean:
	rm -rf data/processed/*
	rm -rf outputs/models/*
	rm -rf outputs/figures/*
	rm -rf outputs/metrics/*
	rm -rf outputs/export/*
	rm -rf outputs/write-up/*
	@echo "Cleaned processed data and outputs. Raw data and splits preserved."
