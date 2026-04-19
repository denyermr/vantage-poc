# CLAUDE.md — Vantage PoC Engineering Constitution

This file governs how Claude Code operates in this repository. It is not a suggestion — every rule here is a hard constraint. When in doubt, stop and ask rather than proceed with a judgement call.

---

## 1. Project Context

This repository implements the Vantage proof-of-concept experiment: a rigorous, reproducible comparison of physics-informed machine learning (PINN) against standard ML baselines for SAR-based soil moisture retrieval at COSMOS-UK Moor House. The results directly feed a Carbon13 venture builder submission and co-founder recruitment materials.

**The scientific credibility of the results is the product.** Code that produces plausible-looking but incorrect outputs is worse than no code at all. Every implementation decision that touches data quality, model training, or evaluation is a scientific decision.

Read `SPEC.md` completely before writing any code. The spec is the source of truth. If CLAUDE.md and SPEC.md conflict, flag the conflict and ask — do not resolve it unilaterally.

---

## 2. The Constitution — Non-Negotiable Engineering Rules

These rules cannot be waived, deferred, or worked around. They exist because this is scientific code where silent errors corrupt results.

### 2.1 No silent failures

- Every function that can fail must raise an informative exception, not return `None` or silently continue.
- Every data loading step must validate the shape, dtype, and key value ranges of its output before returning.
- If a validation fails, the program must stop with a clear error message describing what was expected and what was found.

```python
# BAD
def load_cosmos(path):
    df = pd.read_csv(path)
    return df

# GOOD
def load_cosmos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ['date', 'vwc_qc', 'frozen_flag', 'snow_flag']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"cosmos CSV missing required columns: {missing}")
    if df['vwc_qc'].dropna().between(0, 1).all() is False:
        raise ValueError("vwc_qc contains values outside [0, 1] — check units (should be cm³/cm³, not %)")
    return df
```

### 2.2 No placeholder implementations

- Do not write stub functions that return hardcoded values and mark them "TODO: implement".
- Do not write a function that documents what it *should* do but does something simpler instead.
- If you cannot implement something fully right now, stop and say so. Do not silently produce a placeholder that will produce wrong results when run.
- Exception: empty function stubs with `raise NotImplementedError("Phase N: description")` are permitted **only** in files for phases not yet started, and must be in a dedicated `stubs/` file clearly marked as unimplemented.

### 2.3 No deferred tests

- Tests for a function must be written in the same commit as the function, not afterwards.
- A phase is not complete if any test is marked `@pytest.mark.skip` or `@pytest.mark.xfail` without a documented, approved reason.
- Tests must test the actual behaviour, not just check the function runs without error.

```python
# BAD — does not test correctness
def test_compute_vhvv():
    result = compute_vhvv(vv=-12.0, vh=-18.0)
    assert result is not None

# GOOD — tests the actual physics
def test_compute_vhvv():
    # VH/VV ratio in dB is VH_dB - VV_dB (subtraction in log space)
    result = compute_vhvv(vv_db=-12.0, vh_db=-18.0)
    assert result == pytest.approx(-6.0, abs=1e-6)
```

### 2.4 Fixed seeds everywhere

Every stochastic operation must use an explicit, documented seed:

- NumPy: `rng = np.random.default_rng(SEED)` — never `np.random.seed()` at module level
- PyTorch: `torch.manual_seed(SEED)` + `torch.cuda.manual_seed_all(SEED)` before any model instantiation
- scikit-learn: `random_state=SEED` in every constructor call
- Train/test splits: generated once, saved to `splits/`, loaded thereafter — never regenerated on the fly

`SEED = 42` is defined once in `poc/config.py` and imported everywhere. It is never overridden except via explicit CLI argument with logging.

### 2.5 No hardcoded paths or magic numbers

- All file paths resolve from a single `PROJECT_ROOT` defined in `poc/config.py`.
- All physical constants, model hyperparameters, and thresholds are named constants in `poc/config.py` or passed as explicit arguments — never embedded in function bodies.
- If a number appears in code without a name, ask: does this have physical meaning? If yes, it must be a named constant with a comment citing its source.

```python
# BAD
theta_sat = 0.88  # somewhere in model code

# GOOD — in config.py
# Saturated volumetric water content for blanket bog peat
# Source: Bechtold et al. (2018), Table 1, near-natural sites
PEAT_THETA_SAT = 0.88
```

### 2.6 Chronological integrity

The train/test split is chronological: earliest ~70% is training, most recent ~30% is test. This rule exists to prevent temporal leakage.

- The test set indices are computed **once**, saved, and treated as sealed. They are never used for any model selection, hyperparameter tuning, or threshold decisions.
- Validation (for hyperparameter tuning) is carved from the training pool only.
- Any metric reported "on the test set" must come from the sealed test split. Any violation of this is a scientific error.

### 2.7 All deviations from spec must be logged

When the implementation differs from `SPEC.md` for any reason — a constraint discovered in practice, a data quality issue, a library limitation — it must be documented:

1. Add an entry to `DEVIATIONS.md` at the project root.
2. Include: what the spec said, what was done instead, why, and whether the change affects any pre-registered success criteria.
3. The deviation log is reviewed at each phase gate.

### 2.8 Reproducibility as a deliverable

The experiment must be fully reproducible by a third party from a clean checkout:

- `requirements.txt` (or `environment.yml`) is kept up to date and pinned to exact versions.
- A single command (`make poc` or `python run_experiment.py`) must be able to regenerate all results from the raw data files.
- Intermediate data products are saved to `data/processed/` and checked for existence before recomputation — but must be regeneratable if deleted.
- Model weights are saved with their config embedded so they can be reloaded without the original training script.

---

## 3. Repository Structure

This is a monorepo. `echo-poc/` is the Python science pipeline. `vantage-mvp/` is the Next.js dashboard. They are separate dependency stacks sharing a root.

```
echo-vantage/                          ← monorepo root
├── CLAUDE.md                          ← this file (governs echo-poc)
├── SPEC.md                            ← top-level specification
├── SPEC_PHASE1.md                     ← Phase 1 detail spec
├── SPEC_PHASE2.md                     ← Phase 2 detail spec
├── SPEC_PHASE3.md                     ← Phase 3 detail spec
├── SPEC_PHASE4.md                     ← Phase 4 detail spec (export + MVP)
├── SPEC_PHASE5.md                     ← Phase 5 detail spec (write-up)
├── DEVIATIONS.md                      ← append-only deviation log
├── PROGRESS.md                        ← build state tracker (updated each session)
├── README.md                          ← setup and reproduction instructions
├── Makefile                           ← targets: poc, export-fixtures, mvp, all
│
├── echo-poc/                          ← Python science pipeline
│   ├── requirements.txt               ← pinned Python dependencies
│   ├── Makefile                       ← targets: setup, test, gate-N, clean
│   │
│   ├── poc/
│   │   ├── config.py                  ← ALL constants, paths, seeds, hyperparameters
│   │   ├── data/
│   │   │   ├── cosmos.py              ← COSMOS-UK loading and QC
│   │   │   ├── gee.py                 ← GEE extraction scripts (site-agnostic)
│   │   │   ├── ancillary.py           ← precipitation, terrain feature assembly
│   │   │   └── alignment.py           ← multi-source join and QC filter
│   │   ├── models/
│   │   │   ├── base.py                ← abstract base model interface
│   │   │   ├── null_model.py          ← seasonal climatological baseline
│   │   │   ├── random_forest.py       ← RF with GridSearchCV
│   │   │   ├── standard_nn.py         ← 3-layer NN baseline
│   │   │   ├── pinn.py                ← physics-informed NN (physics + correction)
│   │   │   └── dielectric.py          ← Dobson + Mironov dielectric models
│   │   ├── evaluation/
│   │   │   ├── harness.py             ← metrics, Wilcoxon, aggregation
│   │   │   ├── splits.py              ← split generation (40 configs)
│   │   │   ├── plots.py               ← Phase 2 figures
│   │   │   ├── lambda_search.py       ← λ grid search (Phase 3)
│   │   │   ├── diagnostics.py         ← identifiability diagnostics (Phase 3)
│   │   │   ├── uncertainty.py         ← prediction interval calibration (Phase 3)
│   │   │   └── plots_p3.py            ← Phase 3 figures
│   │   ├── export/
│   │   │   ├── export_fixtures.py     ← writes MVP fixture JSON files (Phase 4)
│   │   │   └── schemas.py             ← Pydantic fixture schemas
│   │   ├── gates/
│   │   │   ├── gate_1.py
│   │   │   ├── gate_2.py
│   │   │   ├── gate_3.py
│   │   │   ├── gate_4.py
│   │   │   └── gate_5.py
│   │   └── pipeline.py                ← orchestrator: full experiment end-to-end
│   │
│   ├── data/
│   │   ├── raw/                       ← original downloads (never modified, not in git)
│   │   │   ├── cosmos/                ← COSMOS-UK daily CSV
│   │   │   ├── sentinel1/             ← GEE extraction outputs
│   │   │   ├── sentinel2/             ← NDVI composites
│   │   │   ├── precipitation/         ← Met Office / ERA5-Land
│   │   │   └── terrain/               ← EA LiDAR derivatives
│   │   ├── processed/                 ← QC'd and aligned datasets (regeneratable)
│   │   └── splits/                    ← train/test split indices (IN GIT)
│   │       ├── test_indices.json      ← sealed test set (never regenerated)
│   │       ├── split_manifest.json
│   │       └── configs/               ← 40 config JSON files
│   │
│   ├── outputs/
│   │   ├── models/                    ← saved weights + configs (not in git)
│   │   │   ├── baseline_a/            ← RF configs
│   │   │   ├── baseline_b/            ← NN configs
│   │   │   └── pinn/                  ← PINN configs + test_predictions.json
│   │   ├── metrics/                   ← JSON metrics per run
│   │   ├── figures/                   ← all diagnostic plots
│   │   ├── gates/                     ← gate result JSONs
│   │   ├── export/                    ← export_manifest.json
│   │   └── write-up/                  ← poc_results.md (Phase 5)
│   │
│   └── tests/
│       ├── unit/                      ← fast, no I/O
│       │   ├── test_data_p1.py
│       │   ├── test_models_p2.py
│       │   ├── test_models_p3.py
│       │   └── test_export_p4.py
│       ├── integration/               ← requires data/processed/ to exist
│       │   ├── test_pipeline_p1.py
│       │   ├── test_pipeline_p2.py
│       │   └── test_pipeline_p3.py
│       └── conftest.py
│
└── vantage-mvp/                       ← Next.js MVP dashboard
    ├── src/
    │   └── data/
    │       └── fixtures/              ← populated by export_fixtures.py
    ├── docs/sdd/                      ← VantageSpec SDD suite (00–06)
    ├── package.json
    └── pnpm-lock.yaml
```

### Cross-package path contract

The only place a cross-package path appears is in `echo-poc/poc/export/export_fixtures.py`:

```python
MONOREPO_ROOT = Path(__file__).parent.parent.parent.parent
MVP_FIXTURES  = MONOREPO_ROOT / "vantage-mvp" / "src" / "data" / "fixtures"
```

Do not hardcode cross-package paths anywhere else.

---

## 4. Phase Structure and Progression Gates

The project has five phases. **A phase cannot begin until its predecessor has passed its gate.** Gate passage requires all criteria to be met — there are no partial passes, no deferrals, and no "we'll fix it in the next phase."

### Phase overview

| Phase | Spec | Gate script | Governs |
|-------|------|-------------|---------|
| 1 — Data Acquisition | `SPEC_PHASE1.md` | `poc/gates/gate_1.py` | Data download, QC, alignment |
| 2 — Baseline Models | `SPEC_PHASE2.md` | `poc/gates/gate_2.py` | RF, NN, null model, evaluation harness |
| 3 — PINN & Evaluation | `SPEC_PHASE3.md` | `poc/gates/gate_3.py` | PINN, diagnostics, pre-registered outcome |
| 4 — Fixture Export & MVP | `SPEC_PHASE4.md` | `poc/gates/gate_4.py` | Export pipeline + full Next.js dashboard |
| 5 — Write-Up & Carbon13 | `SPEC_PHASE5.md` | `poc/gates/gate_5.py` | Results document, doc updates, pitch deck |

### How gate checks work

```bash
cd echo-poc && python poc/gates/gate_N.py --confirm-deviations
# or via Makefile:
make gate-1   # runs gate_1.py
```

The gate script evaluates all criteria programmatically where possible and prints a structured pass/fail report. It writes a result JSON to `outputs/gates/gate_N_result.json`. **The gate script must exit code 0 before the next phase begins. No exceptions.**

### Gate criteria

**Full gate criteria are in the phase spec, not here.** This prevents the constitution drifting out of sync with the specs. The authoritative source for each gate is:

- Gate 1: `SPEC_PHASE1.md §P1.8`
- Gate 2: `SPEC_PHASE2.md §P2` (Gate 2 script section)
- Gate 3: `SPEC_PHASE3.md §P3.14`
- Gate 4: `SPEC_PHASE4.md §P4.8`
- Gate 5: `SPEC_PHASE5.md §P5.4`

### Current build state

See `PROGRESS.md` for current phase, completed tasks, and gate status. Update PROGRESS.md at the end of every session.
| PINN trained on all 40 configurations | 40 | `metrics/` file count |
| Pre-registered success criterion evaluated at N=25 | recorded | `metrics/gate3_result.json` |
| Identifiability diagnostic computed (ML residual ratio) | recorded | `metrics/gate3_result.json` |
| Statistical tests (Wilcoxon) completed | p-values recorded | `metrics/` |
| `pytest tests/` passes with 0 failures | 0 failures | CI |

The gate **passes regardless of the result category** (Strong/Significant/Moderate/Inconclusive/Negative) — the gate tests whether the experiment was conducted correctly, not whether the hypothesis was confirmed. However, a Negative result triggers a mandatory review before Phase 4 begins.

### Phase 4 Gate — MVP Readiness (end of Week 7)

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| All PoC results loaded from DB (not hardcoded) | yes | code review |
| Physics↔ML decomposition chart renders correctly | yes | visual check |
| Uncertainty popover displays all 3 sources | yes | visual check |
| Lighthouse Performance score | ≥ 85 | Lighthouse audit |
| Demo walkthrough completed and timed | ≤ 8 min | manual |

---

## 5. Testing Standards

### Test categories

**Unit tests** (`tests/unit/`) — no file I/O, no network, no model training. Fast. Every function with non-trivial logic gets at least one unit test. Target: < 10 seconds total.

**Integration tests** (`tests/integration/`) — require processed data files to exist in `data/processed/`. Test data pipelines end-to-end with real data. Must be runnable with `pytest tests/integration/ --require-data`.

**Gate tests** (`poc/gates/gate_N.py`) — check phase completion criteria. Not pytest; standalone scripts with structured output.

### Coverage requirements

- Phase 1 (data): ≥ 80% line coverage on `poc/data/`
- Phase 2 (models): ≥ 80% line coverage on `poc/models/` and `poc/evaluation/`
- Phase 3 (PINN): ≥ 80% coverage on `poc/models/pinn.py`; additionally, all physics equations must have dedicated unit tests that verify against known-good values from the literature

### Physics equation tests

Every physics formula in `pinn.py` must have a corresponding test that checks the output against a manually-computed expected value derived from the original paper. This is non-negotiable because bugs in physical constants produce outputs that look plausible.

```python
# Example: Water Cloud Model forward pass
def test_wcm_forward_known_values():
    """
    Reference: Attema & Ulaby (1978), eq. 7
    With A=0.1, B=0.15, NDVI=0.4, theta_inc=39°, sigma_soil=-13 dB
    Expected total backscatter: computed manually from paper equations
    """
    sigma_total = wcm_forward(A=0.1, B=0.15, ndvi=0.4,
                              theta_inc_deg=39.0, sigma_soil_db=-13.0)
    assert sigma_total == pytest.approx(-13.8, abs=0.5)  # tolerance from model simplifications
```

---

## 6. Code Style and Documentation

### Docstrings

Every public function must have a docstring that includes:
- What it does (one line)
- Parameters with types and units
- Return value with type and units
- Any physical assumptions made
- Citation if the function implements a published equation

```python
def compute_vegetation_transmissivity(B: float, ndvi: float, theta_inc_deg: float) -> float:
    """
    Compute two-way vegetation transmissivity factor for the Water Cloud Model.

    τ² = exp(−2 * B * NDVI / cos(θ))

    Args:
        B: WCM vegetation attenuation coefficient (dimensionless, > 0)
        ndvi: Normalised Difference Vegetation Index (dimensionless, [−1, 1])
        theta_inc_deg: SAR incidence angle in degrees

    Returns:
        tau_squared: Two-way transmissivity factor (dimensionless, [0, 1])

    Reference:
        Attema & Ulaby (1978), Radio Science 13(2), eq. 5
    """
```

### Type hints

All function signatures must have type hints. Use `numpy.typing.NDArray` for array arguments.

### Logging

Use Python's `logging` module, not `print()`. Every major pipeline step must emit an `INFO` log entry at start and completion with key statistics (row counts, time elapsed).

---

## 7. Git Discipline

### Commit messages

Format: `[phase] verb: concise description`

Examples:
- `[p1] add: GEE Sentinel-1 extraction script with site-agnostic parameterisation`
- `[p1] fix: VWC unit conversion was returning % not cm³/cm³`
- `[p2] add: Random Forest training with grid search CV`
- `[p2] test: evaluation harness metric computation`
- `[deviation] log: daily product used instead of hourly — see DEVIATIONS.md`

### Branch strategy

- `main` — only merged into after a phase gate passes
- `phase-N/description` — feature branches for each phase
- No force-pushing to `main`

### What goes in git

- All source code
- `requirements.txt`, `CLAUDE.md`, `SPEC.md`, `DEVIATIONS.md`, `Makefile`
- Test fixtures (small synthetic data for unit tests)
- `data/splits/` — the saved train/test splits (essential for reproducibility)

**Not in git:**
- Raw data files (`data/raw/` — too large; listed in `.gitignore` with download instructions in README)
- Processed data files (`data/processed/` — regeneratable)
- Model weights (`outputs/models/` — regeneratable)
- GEE credentials or any API keys

---

## 8. Interaction Protocol

When working in this repository, Claude Code must:

1. **At the start of every session**, read in this order:
   - `CLAUDE.md` (this file — always)
   - `PROGRESS.md` (current build state — always)
   - `SPEC_PHASE{N}.md` for the current phase (never work from memory)
   - `DEVIATIONS.md` (check for any active deviations affecting this phase)
   
   Do not begin writing code until all four are read. Do not rely on memory of what a spec "probably says."

2. **State the phase and task** at the start of each coding session. Example: "Starting Phase 2, task: split generation (`poc/evaluation/splits.py`)."

3. **Before implementing anything that deviates from the phase spec**, stop and state: "This deviates from `SPEC_PHASE{N}.md §P{N}.{X}` because [reason]. Proposed change: [description]. Proceed?"

4. **After completing a task**, run `pytest tests/` and report the result. Do not commit failing tests.

5. **At the end of every session**, update `PROGRESS.md`:
   - Mark completed tasks
   - Record any blockers or open questions
   - Note the next task to pick up

6. **When a gate check fails**, report the exact failing criterion ID (e.g. G2-08) and its measured value. Do not suggest workarounds that bypass the gate criterion — suggest fixes that genuinely address the problem.

7. **Never modify the test set split** (`data/splits/test_indices.json`) under any circumstances, even if the results would look better.

8. **When uncertain about a physical model**, say so explicitly. Do not implement an approximation without flagging it as an approximation and proposing a DEVIATIONS.md entry.

---

## 9. Dependency Policy

Add a dependency only if:
- It is genuinely necessary (the task cannot be done reasonably without it)
- It has an active maintenance history
- It can be pinned to a specific version

New dependencies require updating `requirements.txt` with exact version and a one-line comment explaining why it was added.

Do not add deep learning frameworks beyond PyTorch. Do not add visualisation libraries beyond matplotlib and seaborn without discussion.

### PyTorch device selection (Apple Silicon)

All PyTorch training code must use the device constant from `config.py`:

```python
# In poc/config.py
import torch

# Apple Silicon MPS backend — automatic fallback to CPU if unavailable
# Provides ~2-4× speedup on M-series Macs with no code changes elsewhere
TORCH_DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
```

All model instantiation and tensor operations use `TORCH_DEVICE`. Seeds are always set on both CPU and MPS:

```python
torch.manual_seed(SEED + config_idx)
if TORCH_DEVICE.type == "mps":
    torch.mps.manual_seed(SEED + config_idx)
```

Do not hardcode `device="cpu"` or `device="cuda"` anywhere in model code. The device constant is the single point of control.

---

## 10. The DEVIATIONS.md Contract

`DEVIATIONS.md` is an append-only log. Entries are never edited or deleted after they are written. Format:

```markdown
## DEV-001 — [Date]

**Spec said:** [exact quote or description of what SPEC.md specifies]

**What was done:** [description of the actual implementation]

**Reason:** [why the deviation was necessary]

**Impact on results:** [none / minor / significant — and explanation]

**Impact on pre-registered criteria:** [none / affects Gate N criterion X — explanation]
```

If a deviation affects a pre-registered gate criterion, the criterion threshold may only be changed with explicit written agreement from the project lead (Matthew), documented in the same entry.
