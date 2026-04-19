# Phase 1b — PINN-MIMICS at Moor House

Active working area for the Phase 1b controlled-comparison re-run of the
ECHO PoC. The pre-registration is [echo-poc/SPEC.md](../SPEC.md); the
governance rules are [echo-poc/CLAUDE_3.md](../CLAUDE_3.md); the repository
conventions are [echo-poc/ARCHITECTURE.md](../ARCHITECTURE.md).

## What Phase 1b is

Phase 1 (WCM-PINN) produced a Negative result at Moor House. The diagnosis
was that the Water Cloud Model is structurally inadequate for blanket bog
at C-band. Phase 1b tests that diagnosis: replace the WCM with a
Toure-style single-crown MIMICS, hold every other element of the Phase 1
protocol constant, and check whether the physics-informed advantage is
recovered at the pre-registered N≈25 evaluation point.

Everything else — data, splits, seeds, baselines, backbone, evaluation
harness — is inherited verbatim from Phase 1 via `shared/`. Three
elements change (physics module, dielectric primary, VV+VH joint loss)
plus one additive input (Sentinel-2 NDWI, extracted as a diagnostic-only
covariate). Two Sentinel-2-derived priors appeared in the v0.1-draft
specification and were withdrawn pre-sign-off: NDVI → LAI → N_l
following a citation audit ([`DEV-1b-001.md`](DEV-1b-001.md)), and
NDWI → m_g following a structural audit of the training-set-mean scalar
formulation ([`DEV-1b-002.md`](DEV-1b-002.md)). There are no
Sentinel-2-derived priors in the composite loss at v0.1 sign-off; the
§9 dominance constraint is the sole structural regulariser for the
MIMICS parameter space. Pre-registered v0.2 fallbacks in SPEC §5
cover both parameters if post-experiment non-identifiability is
observed.

## Directory layout

```
phase1b/
├── README.md                      (this file)
├── deviation_log.md               (active — populated during execution)
├── DEV-1b-001.md                  (full entry: withdrawal of N_l prior)
├── DEV-1b-002.md                  (full entry: withdrawal of m_g prior)
├── decisions/                     (pre-sign-off design decisions)
│   ├── U-3-ndwi-formulation.md    (Gao primary + McFeeters diagnostic;
│   │                                 re-scoped to diagnostic-only by DEV-1b-002)
│   └── U-4-ndwi-mg-mapping.md     (Option A — m_g prior withdrawn;
│                                     references DEV-1b-002)
├── physics/
│   ├── mimics.py                  (session C — differentiable PyTorch Toure
│   │                                 single-crown; mirrors reference_toure.py
│   │                                 at the physics level)
│   ├── mironov.py                 (NOT written — frozen Phase 1 dielectric
│   │                                 in `phase1/physics/dielectric.py`
│   │                                 reused directly per session A decision)
│   ├── oh1992_learnable_s.py      (session A — Oh 1992 with learnable s
│   │                                 ∈ 1–5 cm; reused inside mimics.py)
│   │                                 (s2_priors.py NOT written — both
│   │                                  S2-derived priors withdrawn
│   │                                  pre-sign-off per DEV-1b-001 and
│   │                                  DEV-1b-002)
│   └── reference_mimics/          (session B — numpy Toure reference
│                                     + canonical_combinations.json with 36
│                                     numpy_port entries; U-1 Option E)
├── implementation_gate/
│   ├── equivalence_check.py       (session D — to be written; G2 driver)
│   ├── ks_validity_check.py       (session A — G3 driver)
│   ├── dielectric_diagnostic.py   (session A — G4 driver)
│   └── results/
│       ├── g1_baseline_result.json    (written by phase1/run_baselines.py)
│       ├── g2_equivalence.json        (written by equivalence_check.py)
│       ├── g3_ks.json                 (written by ks_validity_check.py)
│       └── g4_dielectric.json         (written by dielectric_diagnostic.py)
├── lambda_search/
│   ├── search.py                  (to be written — §9; grid is
│   │                                 (λ₁, λ₂, λ₃) ∈ {0.01, 0.1, 0.5, 1.0}³
│   │                                 only; λ_prior removed per DEV-1b-002)
│   └── dominance_check.py         (to be written — stricter than Phase 1)
├── diagnostics/
│   ├── forward_fit.py             (VV + VH per §11 Diagnostic A)
│   ├── parameter_correlation.py   (5×5 matrix per §11 Diagnostic B)
│   ├── ndwi_mg_correlation.py     (Diagnostic D per §11 and DEV-1b-002;
│   │                                 post-experiment NDWI ↔ m_g correlation)
│   ├── residual_analysis.py       (inherited from Phase 1)
│   └── decision_tree.py           (Negative-outcome failure-mode decision tree)
├── train.py                       (PINN-MIMICS entry point; to be written)
└── results/                       (populated during execution)
```

## Four pre-training gates

From `CLAUDE_3.md` §"Critical gates":

| # | Gate                                       | Command                                                   | Tolerance                               |
|---|--------------------------------------------|-----------------------------------------------------------|-----------------------------------------|
| 1 | G1 — Phase 1 baseline reproducibility      | `python phase1/run_baselines.py --confirm`                | Drift < 0.005 cm³/cm³                   |
| 2 | G2 — MIMICS forward equivalence            | `python phase1b/implementation_gate/equivalence_check.py` | σ° within 0.5 dB of reference impl      |
| 3 | G3 — Oh ks-validity across s = 1–5 cm      | `python phase1b/implementation_gate/ks_validity_check.py` | No NaN; monotonic in m_v                |
| 4 | G4 — Dobson vs Mironov dielectric diag     | `python phase1b/implementation_gate/dielectric_diagnostic.py` | Records max \|Δε\|/ε (pass either way)  |

All four must pass and `SPEC.md` §14 must be signed before any training
begins.

## How to resume this work

[`SESSION_PLAN.md`](SESSION_PLAN.md) is the session entry point for
Phase 1b. Its session log at the bottom is the single source of truth
for what the last session shipped and what the next session is
committed to. (The repo-root [`PROGRESS.md`](../../PROGRESS.md) is a
frozen Phase 1 build-state snapshot and is not kept in sync with Phase
1b work.)

On every session start, read in this order:
1. [`CLAUDE_3.md`](../CLAUDE_3.md) — governance.
2. [`SESSION_PLAN.md`](SESSION_PLAN.md) — roadmap + session log.
3. [`SPEC.md`](../SPEC.md) — pre-registration (sections relevant to
   the current session, per the session log's "Reading before
   starting session X" line).
4. [`deviation_log.md`](deviation_log.md) and any open
   `DEV-1b-NNN.md` — active deviation state.
5. [`ARCHITECTURE.md`](../ARCHITECTURE.md) — only when making
   structural decisions.
