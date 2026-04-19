# CLAUDE.md — Vantage ECHO Repository

This repository contains the Vantage ECHO PoC code and the Phase 1b PINN-MIMICS work that follows it. Read this file completely before making changes.

---

## What this project is

Vantage is a deep-tech venture building physics-informed satellite monitoring infrastructure for ecosystem carbon markets. ECHO is the Phase 1 proof-of-concept: a PINN architecture combining the Water Cloud Model and a neural correction branch, applied to Sentinel-1 C-band SAR for soil moisture retrieval at the Moor House blanket bog peatland. Phase 1 is complete; the result was Negative (the PINN did not outperform the Random Forest baseline) and the failure was diagnosed as structural inadequacy of the WCM at this site.

**Phase 1b** tests the diagnosis: if the WCM is the binding constraint, replacing it with a structurally richer physics model (MIMICS) — and changing nothing else — should recover the physics-informed advantage. This is the work currently active in the repository.

---

## Read these files first

In this order:

1. **`SPEC.md`** — the Phase 1b pre-registered test specification. The authoritative description of what experiment is being run and how. Sign-off in §14 must be complete before training begins.
2. **`ARCHITECTURE.md`** — repository layout, frozen-vs-active conventions, gate ordering. Read before making structural changes or before adding files anywhere new.
3. **`poc_results.md`** — the Phase 1 results document. Required reading because Phase 1b is a controlled-comparison re-run of this protocol; you need to know what's being mirrored.
4. **`phase1b/deviation_log.md`** — active log of any departure from `SPEC.md` during execution. Check for context before proposing changes.

---

## Phase status

| Phase | Status | Location | Modify? |
|---|---|---|---|
| Phase 1 (WCM PoC) | **Frozen** — published in `poc_results.md`, deviation log closed at 7 entries | `phase1/`, `data/`, frozen modules in `shared/` | **No.** Read-only by convention. |
| Phase 1b (MIMICS PoC) | **Active** — pre-registration v0.1 drafted, gates pending | `phase1b/`, growth in `shared/` only when justified | **Yes**, subject to spec and deviation logging. |
| Phase 2 (causal attribution) | Not yet started | — | — |

---

## Frozen-vs-active hierarchy (THIS IS A HARD RULE)

Three tiers. The rules differ by tier.

### Tier 1: Frozen — never modify

`phase1/`, `data/raw/`, `data/processed/`, `data/splits/`, and frozen modules in `shared/` (`baselines.py`, `evaluation.py`, `data_loader.py`, `qc_pipeline.py`, `seeds.py`).

If a change to any Tier 1 file feels necessary, **stop and ask the user before touching it**. Modifying these files invalidates Phase 1 reproducibility, which invalidates the controlled comparison Phase 1b depends on. Bugs in Tier 1 files are logged as Phase 1b deviations (`phase1b/deviation_log.md`), surfaced to the user, and only fixed with explicit approval and a follow-up G1 re-run to confirm Phase 1 baseline numbers still match.

### Tier 2: Shared — modify carefully

The rest of `shared/`. New utilities can be added here. Modifications to existing utilities require:

1. Explicit justification in the commit message.
2. A passing G1 re-run (`python phase1/run_baselines.py --confirm`) to confirm Phase 1 baselines still produce published numbers within 0.005 cm³/cm³.
3. A note in `phase1b/deviation_log.md` if the change relates to anything called out in `SPEC.md`.

### Tier 3: Active — work here

`phase1b/`. The working area. All new MIMICS code, λ search code, diagnostics, and results live here.

Even within Tier 3, changes that affect the four "Δ" elements or the seven "=" elements from `SPEC.md` §3 must be logged in `phase1b/deviation_log.md`. The deviation log is the audit trail; silent changes break pre-registration.

---

## Critical gates — must pass before training

These are sequential. Do not run training until all four have passed and §14 of `SPEC.md` is signed.

| # | Gate | Command (or location of result) | Tolerance |
|---|---|---|---|
| 1 | Baseline reproducibility (G1) | `python phase1/run_baselines.py --confirm` | RMSE drift < 0.005 cm³/cm³ vs published Phase 1 numbers |
| 2 | MIMICS forward equivalence | `python phase1b/implementation_gate/equivalence_check.py` | σ° within 0.5 dB of reference impl across canonical parameter combinations |
| 3 | Oh ks-validity | `python phase1b/implementation_gate/ks_validity_check.py` | No NaN, no extreme values, monotonic in m_v across s = 1–5 cm |
| 4 | Dobson vs Mironov diagnostic | `python phase1b/implementation_gate/dielectric_diagnostic.py` | Records max |ε_Dobson − ε_Mironov|. <5% means dielectric choice not binding; ≥5% means it is. Pass either way; result is recorded. |

If you (Claude Code) are starting a new session, your **first action** is to check `phase1b/implementation_gate/results/` and `SPEC.md` §14 to determine which gates have already passed and which are still pending.

If a gate fails, halt and surface to the user. Do not attempt to "fix forward" by adjusting parameters. Gate failures are diagnostic information.

---

## Reproducibility rules (non-negotiable)

- **Sealed test set:** n=36, 2023-07-25 to 2024-12-10. Never modified, never split differently, never accessed during training or λ search. The split file lives in `data/splits/` and is loaded read-only.
- **Seeds:** `SEED = 42 + config_idx`. Identical seeds across baselines and PINN-MIMICS. Do not introduce new seed conventions.
- **λ selection:** Once selected (per `SPEC.md` §9), the chosen λ is locked. Re-running the λ search after seeing test-set results is forbidden. If a re-run is genuinely needed, it's a deviation requiring explicit justification.
- **Pre-registered thresholds:** The five outcome categories in `SPEC.md` §10 (Strong / Significant / Moderate / Inconclusive / Negative) and their RMSE thresholds (0.124 / 0.131 / 0.139 / 0.155) are fixed. Do not adjust them post-hoc. If the G1 reproducibility check finds the Phase 1 baselines drift, the spec thresholds are re-derived **once**, before sign-off, and recorded as an amendment.

---

## Deviation logging — what counts

A deviation is any departure from what `SPEC.md` says will happen. Examples that must be logged:

- A baseline fails to reproduce within tolerance (G1 drift).
- The MIMICS equivalence check passes only at a relaxed tolerance.
- AIEM is substituted for Oh 1992.
- A learnable parameter is fixed (or vice versa) due to identifiability problems.
- The λ search fallback is invoked (per `SPEC.md` §9).
- A Phase 1 utility needs modification.

Examples that do NOT need logging (just normal commits):

- Refactoring within `phase1b/` that doesn't change behaviour.
- New diagnostics added to `phase1b/diagnostics/` that don't affect the primary outcome.
- Documentation updates.

Deviation log entries follow the template in `SPEC.md` §13: ID, summary, gate impact, resolution.

---

## Honest reporting (a methodological commitment, not a style note)

The Phase 1 result was Negative, and Phase 1 reported it as Negative — not "promising," not "directionally encouraging," not "informative for next steps" only. Phase 1b inherits this commitment.

If asked to summarise results, draft progress reports, or write up findings:

- Report against the pre-registered criteria, not against a re-shaped narrative.
- A Negative outcome remains Negative even if the secondary criteria pass — secondary criteria are diagnostic, not consolation.
- If diagnostics suggest a failure mode, name it explicitly using the decision tree in `SPEC.md` §11.
- Uncertainties are surfaced, not glossed.

This applies to internal documentation as well as external materials. The cross-document consistency role described in the project instructions depends on Phase 1b results being reported the same way Phase 1 results were.

---

## Cross-document consistency

When making changes that affect quantitative claims — credit prices, hectare targets, satellite specs, RMSE numbers, parameter counts, dataset sizes — flag the implications for the Vantage paper corpus (White Paper, Green Paper, Yellow Paper, Blue Paper, Spectrum Paper, Pitch Deck, Carbon Accounting Framework). The Phase 1b results, when they land, will need propagating. Don't update those documents from this repo (they live in the project knowledge), but **do** call out where claims will need updating.

---

## When in doubt

Ask. The cost of a clarifying question is much lower than the cost of an undocumented change to a frozen artefact or a silent departure from `SPEC.md`. The pre-registration framework only works if departures are visible.
