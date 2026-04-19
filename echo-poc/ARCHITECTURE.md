# Phase 1b Repository Architecture

**Vantage · Phase 1b · Repo Layout & Conventions · v0.1**

This document records the recommended repository layout for Phase 1b work and the reasoning behind it. Read this before making structural decisions about where new code or data lives.

---

## TL;DR

**Same repo as Phase 1, separate phase directory, single `CLAUDE.md`, shared utilities extracted into `shared/`.** The controlled-comparison logic of `SPEC.md` essentially forces this — the G1 baseline reproducibility check requires running Phase 1 baselines against Phase 1 data using Phase 1 code, and that's only clean if everything lives in one place.

---

## Why same-repo, separate-phase

### The G1 argument (decisive)

`SPEC.md` §2 requires re-running the Phase 1 baselines (Null 0.178, RF 0.147 at N=83, RF 0.155 at N≈25, NN 0.159 at N=83) within 0.005 cm³/cm³ tolerance **before** any MIMICS training begins. If Phase 1b lives in a fresh directory, there are three options for that check, all of them bad:

1. **Re-implement the baseline pipeline in the new repo** — drift risk, defeats the purpose of the check.
2. **Import across directories** — awkward, fragile, breaks reproducibility for anyone else who clones.
3. **Copy the Phase 1 code over** — now there are two copies that can silently diverge.

In a single repo with shared utilities, the G1 check is literally:

```bash
python phase1/run_baselines.py --confirm
```

Same data loader, same QC pipeline, same seeds, same train/test split file on disk. The check is trivially auditable.

### The audit-trail argument (also decisive, in the other direction)

Phase 1 is **done**. Pre-registered, results published in `poc_results.md`, deviation log closed at 7 entries. Phase 1b work must not modify any Phase 1 artefact, because that would invalidate the reproducibility claim.

The way to handle this is **convention plus tooling**, not a separate directory. See "Frozen-vs-active conventions" below.

### The exception case

If the existing Phase 1 codebase has accumulated significant drift since `poc_results.md` was written (i.e. you can't actually reproduce 0.147 at N=83 from the current `main` today), then a fresh directory built from the *frozen artefacts* makes more sense than untangling the old one. Confirm this by running the existing Phase 1 baseline pipeline before adopting the layout below.

---

## Recommended layout

```
vantage-echo/                          # existing repo
├── CLAUDE.md                          # updated — covers both phases
├── README.md
├── poc_results.md                     # frozen — Phase 1 results
├── SPEC.md                            # NEW — Phase 1b test specification
├── ARCHITECTURE.md                    # NEW — this file
│
├── data/                              # shared — single source of truth
│   ├── raw/                           # frozen — raw COSMOS-UK, S1, S2, ERA5 pulls
│   ├── processed/                     # frozen — N=119 paired observations
│   └── splits/                        # frozen — chronological 70/30 split file
│
├── shared/                            # NEW — utilities both phases use
│   ├── __init__.py
│   ├── data_loader.py                 # loads N=119 paired dataset
│   ├── qc_pipeline.py                 # frozen-ground, snow, freeze-thaw filters
│   ├── baselines.py                   # Null, RF, NN — frozen
│   ├── evaluation.py                  # RMSE, R², bias, Wilcoxon, Bonferroni
│   ├── pinn_backbone.py               # PhysicsNet + CorrectionNet skeleton
│   └── seeds.py                       # SEED = 42 + config_idx convention
│
├── phase1/                            # frozen — read-only by convention
│   ├── physics/
│   │   ├── wcm.py                     # Water Cloud Model
│   │   ├── dobson.py                  # Phase 1 dielectric
│   │   └── oh1992.py                  # surface scattering, ks=0.30 fixed
│   ├── train.py                       # the Phase 1 training entry point
│   ├── run_baselines.py               # G1 reproducibility check entry point
│   ├── results/                       # frozen — published Phase 1 results
│   │   ├── learning_curves.csv
│   │   ├── wilcoxon_results.csv
│   │   └── diagnostics/
│   └── deviation_log.md               # frozen — closed at 7 entries
│
└── phase1b/                           # NEW — active work
    ├── README.md                      # what Phase 1b is, in 200 words
    ├── deviation_log.md               # active — populated during execution
    ├── physics/
    │   ├── mimics.py                  # Toure-style single-crown MIMICS (PyTorch)
    │   ├── mironov.py                 # NOT WRITTEN — Phase 1b reuses the
    │   │                              #   frozen MironovDielectric class in
    │   │                              #   phase1/physics/dielectric.py per
    │   │                              #   the session A engineering decision
    │   │                              #   (a Phase 1b copy would double-
    │   │                              #   maintain a frozen artefact without
    │   │                              #   scientific benefit).
    │   ├── dobson_arm.py              # NOT WRITTEN — same reasoning as
    │   │                              #   mironov.py; DobsonDielectric is
    │   │                              #   reused from phase1/physics/dielectric.py
    │   │                              #   for the SPEC §11 sensitivity arm.
    │   ├── oh1992_learnable_s.py      # surface model with learnable s (PyTorch)
    │   └── reference_mimics/          # non-differentiable numpy reference
    │       ├── README.md              # U-1 sourcing decision (Option E)
    │       ├── reference_toure.py     # plain-numpy Toure 1994 single-crown
    │       ├── generate_numpy_port_combinations.py  # regenerator
    │       └── canonical_combinations.json          # G2 reference tuples
    ├── implementation_gate/
    │   ├── equivalence_check.py       # §4 forward-pass numerical check
    │   ├── ks_validity_check.py       # §7 Oh ks-validity check
    │   ├── dielectric_diagnostic.py   # §6 Dobson vs Mironov diagnostic
    │   └── results/                   # gate outcomes — required for sign-off
    ├── lambda_search/
    │   ├── search.py                  # 64-combo grid (§9)
    │   ├── dominance_check.py         # primary + secondary criteria (§9)
    │   └── results/
    ├── train.py                       # PINN-MIMICS training entry point
    ├── diagnostics/                   # §11 post-training diagnostics
    │   ├── forward_fit.py             # VV and VH (Diagnostic A)
    │   ├── parameter_correlation.py   # 5×5 matrix (Diagnostic B)
    │   ├── residual_analysis.py       # NDVI correlation, etc.
    │   └── decision_tree.py           # §11 Negative-outcome decision tree
    └── results/                       # populated during execution
```

---

## Frozen-vs-active conventions

Three tiers, with different rules:

### Tier 1: Frozen (`phase1/`, `data/`, frozen modules in `shared/`)

- Read-only by convention.
- No Phase 1b commit should touch these paths. If you find yourself wanting to, you've found something that needs to be promoted into `shared/` or replicated in `phase1b/` — not modified in place.
- **Enforcement:** consider a pre-commit hook that rejects any modification to `phase1/**`, `data/**`, or specific frozen files in `shared/`. At minimum, treat any diff to these paths as a code-review red flag.

### Tier 2: Shared (`shared/`)

- Used by both phases. Modifications must preserve Phase 1 baseline reproducibility — every change to `shared/` should be followed by re-running G1 to confirm Phase 1 baselines still produce the published numbers within tolerance.
- New utilities can be added to `shared/`. Modifications to existing frozen utilities (`baselines.py`, `evaluation.py`, `data_loader.py`) require explicit justification in the commit message and a passing G1 re-run.

### Tier 3: Active (`phase1b/`)

- The working area. All Phase 1b code goes here.
- Subject to the spec. Any change to anything in `phase1b/` that affects the four "Δ" elements or the seven "=" elements from `SPEC.md` §3 must be logged in `phase1b/deviation_log.md`.

---

## What goes in `shared/` vs what stays in a phase directory

The principle: **anything that should be identical between Phase 1 and Phase 1b lives in `shared/`. Anything that differs lives in the phase directory.**

| Concern | Location | Reason |
|---|---|---|
| Dataset loading (N=119) | `shared/data_loader.py` | Identical input data |
| QC pipeline | `shared/qc_pipeline.py` | Identical attrition |
| Train/test split logic | `shared/data_loader.py` | Identical chronological 70/30 |
| Seed generation | `shared/seeds.py` | Identical convention |
| Baseline models (Null, RF, NN) | `shared/baselines.py` | Identical baselines, both phases |
| Evaluation metrics | `shared/evaluation.py` | Identical metrics |
| Wilcoxon + Bonferroni | `shared/evaluation.py` | Identical statistical procedure |
| PINN backbone (PhysicsNet + CorrectionNet) | `shared/pinn_backbone.py` | Identical architecture skeleton |
| **WCM** | `phase1/physics/wcm.py` | Phase 1 only |
| **Dobson (Phase 1 calibration)** | `phase1/physics/dobson.py` | Phase 1 primary; Phase 1b sensitivity arm only |
| **Oh 1992 with ks=0.30 fixed** | `phase1/physics/oh1992.py` | Phase 1 calibration |
| **MIMICS** | `phase1b/physics/mimics.py` (PyTorch) + `phase1b/physics/reference_mimics/reference_toure.py` (numpy ref) | Phase 1b only |
| **Mironov GRMDM** | `phase1/physics/dielectric.py::MironovDielectric` | Phase 1b primary — reused as-is from Phase 1 per session A decision (no `phase1b/physics/mironov.py` created) |
| **Oh 1992 with learnable s** | `phase1b/physics/oh1992_learnable_s.py` | Phase 1b |
| Composite loss function | `phase1b/train.py` | Different loss (VV+VH joint) |
| λ search procedure | `phase1b/lambda_search/` | Different dominance criterion |

If the Phase 1 codebase already has utilities that could go into `shared/` but currently live in `phase1/`, the migration step is:

1. Move the file to `shared/`.
2. Update imports in `phase1/`.
3. Run `python phase1/run_baselines.py --confirm` to verify G1 still passes.
4. Commit with a clear message: "Migrate `<utility>` to shared/. G1 check confirmed."

Do this **before** starting Phase 1b work, not during.

---

## Critical gates — order of operations

Before any MIMICS training, Phase 1b must pass these gates in this order:

1. **G1 — Baseline reproducibility.** `python phase1/run_baselines.py --confirm`. Outputs Null, RF, NN RMSEs at all four training fractions. Compare to published values. If any drift > 0.005 cm³/cm³, halt and investigate before proceeding.

2. **Implementation gate — MIMICS equivalence.** `python phase1b/implementation_gate/equivalence_check.py`. Forward-pass numerical comparison of differentiable MIMICS implementation against reference (non-differentiable) MIMICS at canonical parameter combinations. Tolerance 0.5 dB. (`SPEC.md` §4.)

3. **Surface model gate — Oh ks-validity.** `python phase1b/implementation_gate/ks_validity_check.py`. Confirms Oh 1992 produces physically plausible σ° across s = 1–5 cm range. If failed, substitute AIEM and log as deviation. (`SPEC.md` §7.)

4. **Dielectric diagnostic — Dobson vs Mironov.** `python phase1b/implementation_gate/dielectric_diagnostic.py`. Computes max relative |ε_Dobson − ε_Mironov| over m_v range. Records whether dielectric choice is binding (>5%) or not. (`SPEC.md` §6.)

5. **Sign-off — `SPEC.md` §14.** Spec version, gate outcomes, and signatures recorded. Only then can λ search and training begin.

If you're a new Claude Code session opening this repo, your first task is to verify which of these gates have already passed. Check `phase1b/implementation_gate/results/` and the sign-off block in `SPEC.md`.

---

## Anti-patterns to avoid

- **Modifying Phase 1 files to "improve" them while doing Phase 1b work.** This invalidates G1. If a Phase 1 utility has a bug, log it as a Phase 1b deviation, fix it in `shared/` (with a G1 re-run to confirm Phase 1 baselines still match), and proceed.
- **Adding Phase 1b-specific logic to `shared/`.** `shared/` is the controlled-variable layer. Phase-specific logic belongs in the phase directory.
- **Skipping G1 because "the baselines haven't changed."** They might not have, but the runtime environment has, and the whole point of pre-registration is that you don't trust assumptions — you check them.
- **Re-running λ search after seeing results.** The selected λ is locked once `lambda_search/results/` is populated. Iterating undermines the controlled comparison.
- **Splitting the test set differently to "see what happens."** The sealed test set (n=36, 2023-07-25 to 2024-12-10) is fixed. Touching it is a reportable deviation regardless of motivation.

---

## When to update this document

- Major change to repo structure (new top-level directory, migration of phase boundaries).
- Phase 1b completes — add a `phase2/` section if/when Phase 2 starts.
- A v0.2 spec is drafted — note what changes structurally.
- Tooling changes (pre-commit hooks added, new gate scripts, etc.).

---

*Vantage · Phase 1b · Architecture v0.1 · April 2026*
