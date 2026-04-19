# SPEC_PHASE4.md — Phase 4: Fixture Export & MVP Dashboard
# ECHO PoC — Vantage

**Prerequisite:** Phase 3 gate passed (`outputs/gates/gate_3_result.json` → `"passed": true`)  
**Version:** 1.0 — 06 March 2026  
**Goal:** Export verified PoC results into validated fixture files, then build the MVP dashboard to production quality per the VantageSpec SDD suite. Phase 4 is complete when the Carbon13 demo walkthrough passes end-to-end.

---

## P4. Overview and scope

Phase 4 has two distinct parts:

**Part A — Fixture Export Pipeline** (`echo-poc` side): A deterministic script reads the Phase 3 gate output and aligned dataset, aggregates the physics↔ML chart data, applies outcome-conditional narrative, and writes all fixture JSON files into `vantage-mvp/src/data/fixtures/`. This is the **only** mechanism by which PoC results enter the MVP. No manual transcription of numbers.

**Part B — MVP Build** (`vantage-mvp` side): The full frontend build per the VantageSpec SDD suite (documents `00-CONSTITUTION.md` through `06-DEPLOYMENT.md`). This spec does not respecify the frontend — the VantageSpec governs it completely. This spec adds only what the VantageSpec cannot know: the monorepo structure, the fixture source contract, and the outcome-conditional content rules.

---

## P4.1 Monorepo structure

The single repository contains both packages. The `echo-poc` Python pipeline and the `vantage-mvp` Next.js application share a root but have entirely separate dependency stacks.

```
echo-vantage/                          ← monorepo root
├── .github/
│   └── workflows/
│       ├── ci-poc.yml                 ← Python pipeline CI (phases 1–3)
│       └── ci-mvp.yml                 ← Next.js MVP CI (phase 4)
├── CLAUDE.md                          ← governs echo-poc work
├── SPEC.md                            ← original spec (superseded by phase specs)
├── SPEC_PHASE1.md
├── SPEC_PHASE2.md
├── SPEC_PHASE3.md
├── SPEC_PHASE4.md                     ← this document
├── DEVIATIONS.md
├── README.md                          ← root: setup for both packages
│
├── echo-poc/                          ← Python science pipeline
│   ├── data/
│   ├── poc/
│   │   ├── export/
│   │   │   └── export_fixtures.py     ← NEW: fixture export script (§P4.3)
│   ├── outputs/
│   ├── tests/
│   ├── requirements.txt
│   └── Makefile
│
└── vantage-mvp/                       ← Next.js MVP
    ├── src/
    │   └── data/
    │       └── fixtures/              ← populated by export_fixtures.py
    ├── docs/sdd/                      ← VantageSpec SDD suite (00–06)
    ├── package.json
    └── pnpm-lock.yaml
```

### Root Makefile targets

```makefile
# Run full science pipeline through Phase 3
poc:
	cd echo-poc && python poc/pipeline.py --from-raw

# Export fixtures from Phase 3 results into MVP
export-fixtures:
	cd echo-poc && python poc/export/export_fixtures.py

# Build MVP (requires fixtures to be populated first)
mvp:
	cd vantage-mvp && pnpm install && pnpm build

# Full end-to-end: science → fixtures → MVP build
all: poc export-fixtures mvp

# Run Phase 3 gate then export (most common workflow)
gate3-and-export:
	cd echo-poc && python poc/gates/gate_3.py --confirm-deviations
	cd echo-poc && python poc/export/export_fixtures.py
```

### Cross-package path contract

The export script resolves the MVP fixture directory relative to its own location:

```python
# In poc/export/export_fixtures.py
MONOREPO_ROOT   = Path(__file__).parent.parent.parent.parent
MVP_FIXTURES    = MONOREPO_ROOT / "vantage-mvp" / "src" / "data" / "fixtures"
```

This is the **only** place in the codebase where a cross-package path appears. If the monorepo layout ever changes, only this constant needs updating.

---

## P4.2 Fixture source contract

Every field in every MVP fixture file has exactly one canonical source. This table is the contract between Phase 3 and Phase 4. No fixture value may be invented, estimated, or copied from the VantageSpec placeholders once Phase 3 has completed.

### Legend

| Source tag | Meaning |
|-----------|---------|
| `G3` | Derived from `outputs/gates/gate_3_result.json` |
| `G2` | Derived from `outputs/gates/gate_2_result.json` |
| `G1` | Derived from `outputs/gates/gate_1_result.json` |
| `ALIGNED` | Derived from `data/processed/aligned_dataset.csv` |
| `SPLITS` | Derived from `data/splits/test_indices.json` or config files |
| `DIAG` | Derived from `outputs/metrics/diagnostics_*.json` |
| `UNCERT` | Derived from `outputs/metrics/uncertainty_calibration.json` |
| `STATIC` | Fixed value — does not change with PoC results |
| `COND` | Outcome-conditional — see §P4.4 |

### `moor-house/model-performance.json` — field mapping

| Field | Source | Derivation |
|-------|--------|-----------|
| `maturity` | `STATIC` | Always `"calibrated"` after Phase 3 |
| `trainingPairs` | `G1` | `gate_1_result.criteria.G1-01.measured` |
| `rSquared` | `G3` | `gate_3_result.pinn_vs_baselines["100%"].pinn` R² from best 100% config (median across 10 reps) |
| `rmse` | `G3` | `fixture_values_for_mvp.rmse` — rounded to 3 d.p. |
| `rmseUnit` | `STATIC` | `"cm³/cm³"` |
| `trainTestSplit.train` | `SPLITS` | `test_indices.n_train_pool` |
| `trainTestSplit.test` | `SPLITS` | `test_indices.n_test` |
| `dielectricModel` | `DIAG` | `"Dobson (organic)"` if `diagnostics_dobson_vs_mironov.primary_dielectric_model` = Dobson; `"Mironov (organic)"` if Mironov rerun triggered |
| `vegetationModel` | `STATIC` | `"WCM adapted"` |
| `meanBias` | `G3` | Median mean_bias across 100% configs — rounded to 3 d.p. |
| `meanBiasUnit` | `STATIC` | `"cm³/cm³"` |
| `physicsConsistency` | `G3` | `1 - (wcm_forward_rmse / std(vv_db_test))` — physics model's VV prediction quality, expressed as proportion. Rounded to 2 d.p. |

### `moor-house/physics-ml.json` — field mapping

| Field | Source | Derivation |
|-------|--------|-----------|
| `physicsEstimate` | `G3` | `fixture_values_for_mvp.physics_estimate` |
| `mlCorrection` | `G3` | `fixture_values_for_mvp.ml_correction` |
| `finalEstimate` | `G3` | `physicsEstimate + mlCorrection` |
| `unit` | `STATIC` | `"cm³/cm³"` |
| `chartData` | `ALIGNED` + `G3` | Monthly median aggregation — see §P4.3 |

### `moor-house/uncertainty.json` — field mapping

| Field | Source | Derivation |
|-------|--------|-----------|
| `sources[0].label` | `STATIC` | `"SAR input noise"` |
| `sources[0].value` | `STATIC` | `0.01` — radiometric calibration uncertainty (Sentinel-1 L1 specification) |
| `sources[0].notes` | `STATIC` | `"Radiometric calibration uncertainty"` |
| `sources[1].label` | `STATIC` | `"Model structural"` |
| `sources[1].value` | `UNCERT` | Computed as `sqrt(combinedRss² - sum(other_sources²))` — residual unexplained by other sources |
| `sources[1].notes` | `STATIC` | `"WCM simplification error"` |
| `sources[2].label` | `STATIC` | `"Parameter spread"` |
| `sources[2].value` | `DIAG` | `std(A_final_values_across_configs) × mean(dσ/dA)` — parameter uncertainty propagated through sensitivity. Rounded to 2 d.p. |
| `sources[2].notes` | `DIAG` | `"Dobson model parameter uncertainty"` or `"Mironov model parameter uncertainty"` |
| `sources[3].label` | `STATIC` | `"Temporal (Nd since)"` where N = days since most recent test observation |
| `sources[3].value` | `UNCERT` | `k_80pct × σ_epistemic_mean × temporal_decay_factor` — see §P4.3 |
| `sources[3].notes` | `STATIC` | `"Degrades between Sentinel-1 passes"` |
| `combinedRss` | `G3` | `fixture_values_for_mvp.uncertainty_k_80pct × median(σ_epistemic_test)` — rounded to 2 d.p. |

### `moor-house/observations.json` — field mapping

The 4 most recent test-set observations by date.

| Field | Source | Derivation |
|-------|--------|-----------|
| `date` | `ALIGNED` | Human-formatted date string: `"22 Feb 2026"` |
| `vvDb` | `ALIGNED` | Raw `vv_db` value for that overpass. Rounded to 1 d.p. |
| `vhDb` | `ALIGNED` | Raw `vh_db` value. Rounded to 1 d.p. |
| `vhVvDb` | `ALIGNED` | `vhvv_db` value. Rounded to 1 d.p. |
| `ndvi` | `ALIGNED` | NDVI for that overpass. Rounded to 2 d.p. |
| `estVwc` | `G3` | PINN `m_v_final` prediction for that observation (from best 100% config). Rounded to 2 d.p. |
| `vwcUncertainty` | `UNCERT` | `k_80pct × σ_epistemic` for that observation. Rounded to 2 d.p. |
| `vwcStatus` | `ALIGNED` + `UNCERT` | `"success"` if uncertainty ≤ 0.05; `"warning"` if 0.05 < uncertainty ≤ 0.08; `"error"` if > 0.08 |

### `moor-house/sar-config.json` — field mapping

| Field | Source | Derivation |
|-------|--------|-----------|
| `orbit` | `G1` | From `sentinel1_extractions.csv` orbit_number: `"Descending (rel. orbit {N})"` |
| `revisit` | `STATIC` | `"6 days"` |
| `mode` | `STATIC` | `"IW GRD (VV+VH)"` |
| `incidenceAngle` | `ALIGNED` | `mean(incidence_angle_mean)` across aligned dataset. `"~{N}° (mean)"` |
| `wcmA` | `DIAG` | `median(final_A across configs 000–009)` — rounded to 4 d.p. |
| `wcmALit` | `STATIC` | `0.0080` (Attema & Ulaby 1978 reference for heathland) |
| `wcmB` | `DIAG` | `median(final_B across configs 000–009)` — rounded to 3 d.p. |
| `wcmBLit` | `STATIC` | `0.130` (Singh et al. 2023 C-band reference) |
| `paramSource` | `G1` | `"Site-calibrated ({N} obs)"` where N = training pairs |

### `moor-house/data-quality.json` — field mapping

| Field | Source | Derivation |
|-------|--------|-----------|
| `frozen` | `G1` | `gate_1_attrition.after_step1_vwc_flag - gate_1_attrition.after_step2_frozen` |
| `snow` | `G1` | `gate_1_attrition.after_step2_frozen - gate_1_attrition.after_step3_snow` |
| `cosmosQcFail` | `G1` | `gate_1_attrition.s1_overpasses_raw - gate_1_attrition.after_step1_vwc_flag` |
| `cloudGaps` | `ALIGNED` | Count of NDVI interpolation gaps in test period |
| `totalExcluded` | `G1` | `s1_overpasses_raw - final_paired_observations` |
| `totalAcquired` | `G1` | `s1_overpasses_raw` |
| `usablePct` | `G1` | `final_paired_observations / s1_overpasses_raw × 100` — 1 d.p. |

### `moor-house/narrative.json` — field mapping

| Field | Source | Derivation |
|-------|--------|-----------|
| `siteDescription` | `STATIC` | Fixed prose (see §P4.4) |
| `monitoringObjectives` | `COND` | Outcome-conditional bullet list (see §P4.4) |
| `outcomeCategory` | `G3` | `gate_3_result.outcome.category` — written into fixture for UI display |
| `outcomeSummary` | `COND` | Single sentence for display in metric card (see §P4.4) |

### Static fixtures (not outcome-conditional)

These fixtures have no dependency on Phase 3 results. They are written by the export script from static values defined in the script itself — they do not change run-to-run.

| Fixture | Content |
|---------|---------|
| `sites.json` | Site list; Moor House `vwc`, `rSquared`, `observations` updated from `G3` |
| `moor-house/hydrology.json` | Mean VWC from ALIGNED; dielectric from DIAG; precip from ALIGNED |
| `moor-house/vegetation.json` | NDVI stats from ALIGNED; phenological phase is STATIC |
| `moor-house/risk-assessment.json` | STATIC — 4 site-specific risks (unchanged) |
| `moor-house/sensors.json` | STATIC — sensor inventory (unchanged) |
| `moor-house/alerts.json` | STATIC — 3 dashboard alerts + 2 site detail alerts |
| `dashboard-metrics.json` | `avgModelFit` updated from `G3`; rest STATIC |

---

## P4.3 Fixture export script (`echo-poc/poc/export/export_fixtures.py`)

### Invocation

```bash
python poc/export/export_fixtures.py [--dry-run] [--force]

Options:
  --dry-run   Print what would be written without writing any files.
              Validates all source data is present and computable.
  --force     Overwrite existing fixture files without confirmation prompt.
              Default: prompt if any target file already exists.
```

### Pre-flight checks (run before any file is written)

The script must verify all source files exist and are valid before writing a single fixture. If any check fails, the script exits with code 1 and prints a clear error — no partial writes.

```python
REQUIRED_SOURCES = [
    GATE3_RESULT,                      # outputs/gates/gate_3_result.json
    GATE2_RESULT,                      # outputs/gates/gate_2_result.json
    GATE1_RESULT,                      # outputs/gates/gate_1_result.json
    GATE1_ATTRITION,                   # outputs/gates/gate_1_attrition.json
    ALIGNED_DATASET,                   # data/processed/aligned_dataset.csv
    TEST_INDICES,                      # data/splits/test_indices.json
    UNCERTAINTY_CALIBRATION,           # outputs/metrics/uncertainty_calibration.json
    DIAGNOSTICS_RESIDUAL,             # outputs/metrics/diagnostics_residual_ratio.json
    DIAGNOSTICS_SENSITIVITY,          # outputs/metrics/diagnostics_parameter_sensitivity.json
    DIAGNOSTICS_DIELECTRIC,           # outputs/metrics/diagnostics_dobson_vs_mironov.json
    S1_EXTRACTIONS,                   # data/processed/sentinel1_extractions.csv
]
```

Additionally verify:
- `gate_3_result.json` → `"passed": true` — refuse to export if Phase 3 gate did not pass
- All 40 PINN metric files exist in `outputs/metrics/`
- `MVP_FIXTURES` directory exists in the monorepo

### Physics↔ML chart data aggregation

The chart in the MVP shows **monthly medians aggregated from the test set**. This matches the existing VantageSpec fixture structure (month label + three values) and makes the chart visually clean.

```python
def compute_physics_ml_chart_data(
    aligned_dataset: pd.DataFrame,
    test_indices: list[int],
    pinn_metrics_dir: Path,
) -> list[dict]:
    """
    Aggregate physics and fused predictions to monthly medians.

    Steps:
    1. Load test set observations from aligned_dataset at test_indices
    2. Load PINN predictions for all 10 × 100% configs (config_000 to config_009)
    3. For each test observation, compute:
       - m_v_physics_median: median of m_v_physics across 10 reps
       - m_v_final_median:   median of m_v_final across 10 reps
       - vwc_observed:       aligned_dataset['vwc'] at that index
    4. Group by calendar month (using observation date)
    5. For each month present in test set:
       - physics: median(m_v_physics_median values in that month)
       - fused:   median(m_v_final_median values in that month)
       - observed: median(vwc_observed values in that month)
       - month_label: abbreviated month name, e.g. "Jan", "Feb"
    6. Sort chronologically
    7. Return list of dicts matching VantageSpec chartData schema

    Minimum months: if test set spans < 3 calendar months, raise ValueError.
    Maximum months: all months present in test set (no cap).

    Critical visual requirement (from VantageSpec §7.2 and 04-SITE-DETAIL.md):
        The fused series MUST track closer to observed than the physics series.
        Verify: mean(|fused - observed|) < mean(|physics - observed|)
        If this is not satisfied: log a warning and write the data as-is.
        Do NOT fabricate or adjust values to satisfy this requirement.
        The chart shows the honest result. If physics tracks as well as the
        fused model, the chart will show it.

    Returns:
        [
            {"month": "Oct", "physics": 0.71, "fused": 0.74, "observed": 0.73},
            {"month": "Nov", "physics": 0.68, "fused": 0.72, "observed": 0.71},
            ...
        ]
    """
```

**Note on PINN predictions for chart data:** The export script needs the actual per-observation predictions from each model, not just the aggregate metrics. The PINN training loop (Phase 3, §P3.8 step 9) must save these alongside the metrics JSON:

Add to each PINN config artefact directory:
```
outputs/models/pinn/config_000/
    test_predictions.json     ← NEW: per-observation predictions on test set
```

```json
{
    "config_idx": 0,
    "test_indices": [99, 100, 101, ...],
    "dates": ["2024-01-05", "2024-01-11", ...],
    "m_v_physics": [0.68, 0.65, 0.71, ...],
    "delta_ml": [0.04, 0.03, 0.05, ...],
    "m_v_final": [0.72, 0.68, 0.76, ...]
}
```

This must be added to the Phase 3 spec as a required output. Add to `SPEC_PHASE3.md §P3.8` step 10 and to the Phase 3 file manifest.

### Temporal uncertainty decay

The `sources[3].value` for uncertainty.json (temporal degradation since last pass) is computed as:

```python
def temporal_uncertainty(
    k_80pct: float,
    sigma_epistemic_mean: float,
    days_since_last_pass: int,
    revisit_days: int = 6,
) -> float:
    """
    Uncertainty grows between Sentinel-1 passes.
    Linear interpolation from σ_epistemic at pass day to
    2 × σ_epistemic at next pass day (6 days later).

    decay_factor = 1.0 + (days_since / revisit_days)
    temporal_component = k_80pct × σ_epistemic_mean × decay_factor
    """
    decay_factor = 1.0 + (days_since_last_pass / revisit_days)
    return round(k_80pct * sigma_epistemic_mean * decay_factor, 2)
```

`days_since_last_pass` is computed as: today's date minus the date of the most recent test-set observation. If this is > 6 days (the Sentinel-1 revisit), use 6 (the uncertainty has reached its maximum between-pass value).

### Export procedure

```python
def run_export(dry_run: bool = False, force: bool = False) -> None:
    """
    Atomic fixture export. Either all fixtures are written successfully,
    or none are (write to temp dir, then rename atomically).

    Order of operations:
    1. Pre-flight checks (all source files present and valid)
    2. Load all source data into memory
    3. Compute all derived values
    4. Apply outcome-conditional content (§P4.4)
    5. Validate all computed fixtures against their Zod schemas
       (using Python equivalent: pydantic models matching the TS schemas)
    6. If --dry-run: print summary and exit
    7. If fixtures already exist and not --force: prompt for confirmation
    8. Write all fixtures atomically to MVP_FIXTURES
    9. Print export manifest (list of written files + key values)
    10. Write export provenance record (§P4.5)
    """
```

---

## P4.4 Outcome-conditional content

When the export script runs, it reads `gate_3_result.outcome.category` and selects content from the lookup below. This content is written into `moor-house/narrative.json` and `sites.json`.

### Monitoring objectives (bullet list in Site Narrative card)

**Strong:**
```
- Validate physics-informed ML outperforms standard ML on peatland SAR retrieval (>20% RMSE reduction at N=25 confirmed)
- Demonstrate data-efficiency advantage at N=25 training samples
- Establish causal attribution baseline for grip-blocking interventions
- Calibrate Water Cloud Model parameters for blanket bog vegetation
```

**Significant:**
```
- Validate physics-informed ML advantage on peatland SAR retrieval (statistically significant at N=25)
- Demonstrate data-efficiency advantage at limited training data volumes
- Establish causal attribution baseline for grip-blocking interventions
- Calibrate Water Cloud Model parameters for blanket bog vegetation
```

**Moderate:**
```
- Validate directional physics-informed ML advantage on peatland SAR retrieval
- Characterise data-efficiency behaviour across training data volumes
- Establish continuous monitoring baseline for future attribution analysis
- Calibrate Water Cloud Model parameters for blanket bog vegetation
```

**Inconclusive:**
```
- Establish continuous, uncertainty-quantified peatland soil moisture monitoring
- Characterise SAR-moisture relationships at Moor House across seasons
- Build calibrated monitoring baseline for future attribution analysis
- Investigate physics model parameterisation for blanket bog at C-band
```

**Negative:**
```
- Establish continuous, uncertainty-quantified peatland soil moisture monitoring
- Document SAR-moisture retrieval performance bounds at this site and data volume
- Build monitoring baseline demonstrating platform capability independent of physics advantage
- Inform PhD programme on physics model requirements for blanket bog SAR
```

### Outcome summary (single sentence displayed in narrative card header)

**Strong:** `"Physics-informed ML delivers >20% accuracy improvement at N=25 — core thesis strongly supported."`

**Significant:** `"Physics-informed ML delivers a statistically significant accuracy advantage at limited data volumes."`

**Moderate:** `"Physics-informed ML shows a directional accuracy advantage at N=25, consistent across training sizes."`

**Inconclusive:** `"Platform demonstrates continuous, uncertainty-quantified monitoring — physics advantage inconclusive at this data volume."`

**Negative:** `"Platform demonstrates continuous monitoring capability; physics-ML fusion performance under investigation."`

### Chart annotation text (written into `physics-ml.json` as `chartAnnotation`)

**Strong:** `"Physics RMSE: {physics_rmse:.3f}  Fused RMSE: {fused_rmse:.3f}  Improvement: {pct:.0f}% at N=25"`

**Significant:** `"Physics RMSE: {physics_rmse:.3f}  Fused RMSE: {fused_rmse:.3f}  Improvement: {pct:.0f}% (p={p:.3f})"`

**Moderate:** `"Physics RMSE: {physics_rmse:.3f}  Fused RMSE: {fused_rmse:.3f}  Directional improvement: {pct:.0f}%"`

**Inconclusive:** `"Physics RMSE: {physics_rmse:.3f}  Fused RMSE: {fused_rmse:.3f}  No consistent advantage at N=25"`

**Negative:** `"Physics RMSE: {physics_rmse:.3f}  Fused RMSE: {fused_rmse:.3f}  Standard ML competitive at this data volume"`

### `sites.json` Moor House `rSquared` and `vwc`

`rSquared`: always from `G3` (actual result).  
`vwc`: the most recent test-set observation's `m_v_final` prediction. Not the dataset mean — the most recent estimate, matching what "current state" means operationally.

---

## P4.5 Export provenance record

Written to `echo-poc/outputs/export/export_manifest.json` after each successful export run. This is the audit trail proving the MVP fixture values are traceable to the experiment.

```json
{
    "exported_at": "2026-03-15T14:22:00Z",
    "gate3_result_timestamp": "2026-03-14T18:11:23Z",
    "outcome_category": "Significant",
    "monorepo_root": "/Users/md/code/echo-vantage",
    "mvp_fixtures_dir": "vantage-mvp/src/data/fixtures",
    "files_written": [
        "moor-house/model-performance.json",
        "moor-house/physics-ml.json",
        "moor-house/uncertainty.json",
        "moor-house/observations.json",
        "moor-house/sar-config.json",
        "moor-house/data-quality.json",
        "moor-house/narrative.json",
        "moor-house/hydrology.json",
        "moor-house/vegetation.json",
        "sites.json",
        "dashboard-metrics.json"
    ],
    "key_values": {
        "training_pairs": 142,
        "rmse_100pct": 0.041,
        "r_squared_100pct": 0.871,
        "rmse_n25": 0.059,
        "best_baseline_rmse_n25": 0.071,
        "relative_reduction_n25": 0.169,
        "physics_estimate_current": 0.68,
        "ml_correction_current": 0.04,
        "combined_uncertainty": 0.04,
        "wcm_a_final": 0.091,
        "wcm_b_final": 0.163,
        "dielectric_model": "Dobson (1985)",
        "chart_data_months": 7,
        "fused_tracks_closer_to_observed": true
    },
    "warnings": []
}
```

---

## P4.6 Fixture schema validation in Python

Before writing any file, the export script validates each computed fixture dict against a Pydantic model that mirrors the TypeScript Zod schema. This catches type errors before they reach the frontend.

```python
# echo-poc/poc/export/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class ModelPerformance(BaseModel):
    maturity: Literal["calibrated", "transferred", "onboarding"]
    trainingPairs: int = Field(gt=0)
    rSquared: float = Field(ge=0.0, le=1.0)
    rmse: float = Field(ge=0.0, le=0.5)
    rmseUnit: Literal["cm³/cm³"]
    trainTestSplit: dict  # {train: int, test: int}
    dielectricModel: str
    vegetationModel: str
    meanBias: float = Field(ge=-0.5, le=0.5)
    meanBiasUnit: Literal["cm³/cm³"]
    physicsConsistency: float = Field(ge=0.0, le=1.0)

class PhysicsMlChartPoint(BaseModel):
    month: str
    physics: float = Field(ge=0.0, le=1.0)
    fused: float = Field(ge=0.0, le=1.0)
    observed: float = Field(ge=0.0, le=1.0)

class PhysicsMl(BaseModel):
    physicsEstimate: float = Field(ge=0.0, le=1.0)
    mlCorrection: float = Field(ge=-0.5, le=0.5)
    finalEstimate: float = Field(ge=0.0, le=1.0)
    unit: Literal["cm³/cm³"]
    chartAnnotation: str
    chartData: list[PhysicsMlChartPoint] = Field(min_length=3)

# ... (one Pydantic model per fixture file)
```

If any schema validation fails, the export script prints the field name, expected type, and actual value — then exits without writing. This is the equivalent of `pnpm validate:data` on the TypeScript side, run in Python before the files reach the frontend.

---

## P4.7 MVP build — VantageSpec governs

The frontend build follows the VantageSpec SDD suite exactly:

| VantageSpec phase | What it builds | Gate |
|-------------------|---------------|------|
| Phase 0 — Scaffold | Next.js 15, TypeScript strict, Tailwind, shadcn/ui, CI | Gate 0 |
| Phase 1 — Data & Design System | Fixtures validated, all UI primitives | Gate 1 |
| Phase 2 — Dashboard & Navigation | Dashboard, routing, tabs | Gate 2 |
| Phase 3 — Site Detail | State Estimation, Sensors, Phase-lock pages | Gate 3 |
| Phase 4 — Interactions & Polish | Uncertainty popover, theme toggle, a11y | Gate 4 |
| Phase 5 — Deployment & Demo | Static export, Vercel, demo E2E | Gate 5 |

**This spec adds three rules to the VantageSpec that it cannot specify itself:**

### Rule A — Fixtures must be exported before Phase 1 (VantageSpec) begins

`pnpm validate:data` (VantageSpec Phase 1 gate criterion) will fail if any fixture file is missing or contains placeholder values. The export script must be run and verified before any VantageSpec phase begins.

Verification command:
```bash
# From monorepo root
make export-fixtures
cd vantage-mvp && pnpm validate:data
```

Both must exit 0 before VantageSpec Phase 0 scaffold work begins.

### Rule B — The chart visual requirement is outcome-conditional

VantageSpec `04-SITE-DETAIL.md` states:

> "CRITICAL visual requirement: green line tracks closer to cyan dots than amber dashed line. This proves the core thesis visually."

This requirement is **conditional on the outcome category**:

- **Strong / Significant / Moderate:** The chart must satisfy this visual requirement. If it does not (i.e. the physics model tracks as well as the fused model), this is a scientific finding to report honestly — adjust the chart annotation to reflect it, do not adjust the data.
- **Inconclusive / Negative:** The chart annotation text (set by the export script per §P4.4) already describes the honest result. The VantageSpec visual requirement does not apply. The chart shows what the data shows.

Claude Code building the MVP must read `moor-house/narrative.json → outcomeCategory` and apply this rule. The chart annotation text is the single source of truth for what the chart means.

### Rule C — Fixture values are read-only after export

Once `export_fixtures.py` has run and `export_manifest.json` exists, fixture files in `vantage-mvp/src/data/fixtures/` are **read-only** from the MVP build's perspective. They must not be edited manually, adjusted for aesthetics, or overridden to show more favourable numbers.

The VantageSpec constitution Law 6 states: "All displayed data traces to documented PoC outputs or to demonstration values explicitly marked as `demo: true` in fixture files."

Phase 4 strengthens this: **no fixture value may be marked `demo: true`** once Phase 3 has completed. The `demo: true` flag was appropriate during VantageSpec development before real results existed. After export, every value is real.

---

## P4.8 Gate 4 criteria

Gate 4 combines the ECHO PoC export gate and the VantageSpec final gate (06-DEPLOYMENT.md Gate 5). Both must pass.

**Run with:** `python echo-poc/poc/gates/gate_4.py && cd vantage-mvp && pnpm gate-check`  
**Exit code:** 0 = both pass, 1 = either fails

### Part A — Export gate (Python)

| ID | Criterion | Threshold | Auto-checkable |
|----|-----------|-----------|----------------|
| G4-01 | Export manifest exists | file present | Yes |
| G4-02 | Gate 3 was passed before export | `gate_3_result.passed == true` | Yes |
| G4-03 | All 11 fixture files written | 11 files | Yes |
| G4-04 | All fixtures pass Pydantic validation | 0 schema errors | Yes |
| G4-05 | No placeholder values remain | no `demo: true` keys | Yes |
| G4-06 | Chart data spans ≥ 3 months | `len(chartData) >= 3` | Yes |
| G4-07 | Outcome category matches narrative | consistent across files | Yes |
| G4-08 | Export provenance record complete | all key_values present | Yes |

### Part B — MVP gate (TypeScript — per VantageSpec 06-DEPLOYMENT.md)

| ID | Criterion | Threshold |
|----|-----------|-----------|
| G4-09 | `pnpm validate:data` passes | 0 schema errors |
| G4-10 | Static export builds | `pnpm build` exits 0 |
| G4-11 | Deployed site loads at URL | HTTP 200 |
| G4-12 | Demo flow E2E test passes | all 8 scenes pass |
| G4-13 | All accumulated tests pass | 0 failures, 0 skipped |
| G4-14 | Cumulative test count | ≥ 250 |
| G4-15 | Lighthouse Performance | ≥ 85 |
| G4-16 | Lighthouse Accessibility | ≥ 90 |
| G4-17 | No TypeScript errors | 0 |
| G4-18 | No ESLint errors/warnings | 0 |
| G4-19 | No browser console errors | 0 |
| G4-20 | Visual regression: 0 unexpected diffs | 0 |
| G4-21 | DEVIATIONS.md reviewed | manual sign-off |

---

## P4.9 Phase 4 test requirements

### Export script tests (`echo-poc/tests/unit/test_export_p4.py`)

| Test | What it checks |
|------|---------------|
| `test_preflight_fails_if_gate3_not_passed` | Script exits 1 if gate_3 passed=false |
| `test_preflight_fails_if_source_missing` | Script exits 1 if any required source file absent |
| `test_chart_aggregation_monthly_median` | Monthly medians computed correctly from known test data |
| `test_chart_minimum_3_months` | ValueError raised if test set spans < 3 months |
| `test_chart_chronological_order` | Output months sorted ascending |
| `test_outcome_strong_content` | Strong outcome → correct bullet text |
| `test_outcome_negative_content` | Negative outcome → correct pivot text |
| `test_all_outcome_categories_covered` | All 5 categories have narrative content defined |
| `test_temporal_uncertainty_at_zero_days` | Returns base uncertainty at day 0 |
| `test_temporal_uncertainty_caps_at_revisit` | Days > 6 treated as 6 |
| `test_pydantic_model_performance_valid` | Valid fixture passes |
| `test_pydantic_model_performance_rmse_out_of_range` | rmse > 0.5 raises ValidationError |
| `test_no_demo_true_in_exported_fixtures` | No fixture key has value `"demo: true"` |
| `test_export_provenance_written` | export_manifest.json exists after run |
| `test_dry_run_writes_nothing` | --dry-run does not create any files |
| `test_atomic_write_no_partial` | Simulate write failure mid-export — no partial files left |

### VantageSpec tests

Per the VantageSpec constitution, cumulative test target ≥ 250 at Phase 5 gate. Distribution across VantageSpec phases: ≥ 80 unit (data + schemas), ≥ 120 component (RTL), ≤ 40 E2E (Playwright). Export tests above count towards the unit total.

---

## P4.10 Phase 4 file manifest

### echo-poc additions
```
echo-poc/poc/export/__init__.py
echo-poc/poc/export/export_fixtures.py
echo-poc/poc/export/schemas.py
echo-poc/poc/gates/gate_4.py
echo-poc/outputs/export/export_manifest.json   ← generated by export script
echo-poc/outputs/models/pinn/config_*/test_predictions.json  (40 files — Phase 3 addition)
echo-poc/tests/unit/test_export_p4.py
```

### vantage-mvp (all populated by export script)
```
vantage-mvp/src/data/fixtures/sites.json
vantage-mvp/src/data/fixtures/dashboard-metrics.json
vantage-mvp/src/data/fixtures/moor-house/model-performance.json
vantage-mvp/src/data/fixtures/moor-house/physics-ml.json
vantage-mvp/src/data/fixtures/moor-house/uncertainty.json
vantage-mvp/src/data/fixtures/moor-house/observations.json
vantage-mvp/src/data/fixtures/moor-house/sar-config.json
vantage-mvp/src/data/fixtures/moor-house/data-quality.json
vantage-mvp/src/data/fixtures/moor-house/narrative.json
vantage-mvp/src/data/fixtures/moor-house/hydrology.json
vantage-mvp/src/data/fixtures/moor-house/vegetation.json
```

### vantage-mvp build (per VantageSpec — all phases)
```
(All files specified in VantageSpec 00-CONSTITUTION.md §8 project structure)
```
