# SPEC_PHASE5.md — Phase 5: Write-Up & Carbon13 Materials
# ECHO PoC — Vantage

**Prerequisite:** Phase 4 gate passed (both parts: `export_manifest.json` exists + VantageSpec Gate 5 passed)  
**Version:** 1.0 — 06 March 2026  
**Goal:** Produce a complete, honest written record of the PoC experiment; update the three Vantage documents that carry technical claims; and prepare all Carbon13 presentation materials. Phase 5 is complete when every claim in every Carbon13-facing document is backed by a measured number or explicitly flagged as a projection.

---

## P5. Overview

Phase 5 has four parts:

**Part A — PoC Results Document:** The canonical written record of the experiment. Methods, results, interpretation, diagnostics, and limitations. Stands alone as a technical document. Feeds everything else in this phase.

**Part B — Document updates:** Three Vantage documents contain technical claims that were written before real results existed. Each is updated in a controlled way — specific sections replaced, version incremented, changes logged.

**Part C — Carbon13 Materials:** Pitch deck update and co-founder brief. Both derived from the PoC results document. No claims in Carbon13 materials that are not in Part A or Part B.

**Part D — Gate 5:** Quality criteria for every deliverable. The gate is the final checkpoint before Carbon13 presentation. Passes or fails as a whole — no partial credit.

All content in Phase 5 is outcome-conditional. The spec provides parallel tracks for each outcome category. The export script already wrote `outcomeCategory` into `moor-house/narrative.json` — Phase 5 reads it from there.

---

## P5.1 PoC Results Document

**File:** `echo-poc/outputs/write-up/poc_results.md`  
**Format:** Markdown, rendered to PDF for distribution  
**Audience:** Carbon13 programme team, potential co-founders, PhD supervisors (if shared)  
**Length:** 2,500–4,000 words. Tight. No padding.

### Required sections

#### 1. Executive Summary (200 words max)

One paragraph per item:
- What the experiment tested
- How it was conducted (one sentence on methodology)
- What the result was — the outcome category, stated plainly
- What the result means for the venture

The executive summary is written last. It must not overstate the result. If outcome is Inconclusive, the summary says "inconclusive" — not "promising early evidence."

#### 2. Research Question and Hypotheses

Copied verbatim from `SPEC.md §2`. Do not paraphrase. The pre-registered question is the pre-registered question.

Null hypothesis stated explicitly: "The PINN offers no statistically significant RMSE improvement over the best-performing baseline at N=25 training samples."

#### 3. Study Site and Data

- Moor House description: location, ecosystem type, COSMOS-UK sensor details
- Data sources: Sentinel-1 (collection, orbit, date range), Sentinel-2 NDVI, ERA5-Land precipitation, SRTM terrain
- QC pipeline: attrition table from `gate_1_result.json` — exact numbers, not "approximately"
- Final dataset: N total, N training pool, N test set

**Attrition table format:**
```
Stage                          Observations    Removed    Retained
─────────────────────────────────────────────────────────────────
S1 overpasses acquired         168             —          168
After VWC QC flag exclusion    168             6          162
After frozen ground exclusion  162             12         150
After snow exclusion           150             4          146
After feature NaN exclusion    146             4          142
─────────────────────────────────────────────────────────────────
Final paired dataset           —               26         142
Train/test split (70/30)       —               —          99 / 43
```
Numbers populated from `gate_1_result.json` → `attrition_log`.

#### 4. Methods

Four subsections, each tight:

**4.1 Experimental design**
- 40 configurations: 4 training sizes × 10 repetitions
- Stratified subsampling (meteorological seasons: DJF/MAM/JJA/SON)
- Sealed chronological test set
- Pre-registration rationale: success criteria fixed before any model training

**4.2 Baseline models**
- Null model (seasonal climatological mean) — RMSE floor
- Random Forest: GridSearchCV, 5-fold CV, parameter grid
- Standard NN: 3-layer architecture, Adam, early stopping

**4.3 PINN architecture**
- Two-branch design: physics_net + correction_net
- WCM forward model: equations written out, parameter initialisations cited
- Dobson dielectric model: parameterisation for blanket bog, citation
- Composite loss: four terms with λ values from search
- Sigmoid reparameterisation for WCM parameter bounds

**4.4 Evaluation**
- Metrics: RMSE, R², mean bias on sealed test set
- Learning curves: median ± IQR across 10 reps per training size
- Statistical test: Wilcoxon signed-rank, both uncorrected (α=0.05) and Bonferroni-corrected (α=0.0125) p-values reported
- Identifiability diagnostics: residual ratio, parameter sensitivity, Dobson/Mironov comparison

#### 5. Results

**Structure is outcome-conditional.** All five tracks share the same section headings — the content differs.

**5.1 Baseline performance**

Always written the same way regardless of outcome:

```
Table: Baseline RMSE (cm³/cm³) by training size — median [IQR] across 10 reps

Model         │ N≈10 (10%) │ N≈25 (25%) │ N≈50 (50%) │ N≈99 (100%)
──────────────┼────────────┼────────────┼────────────┼─────────────
Null (floor)  │    0.XXX   │    0.XXX   │    0.XXX   │    0.XXX
RF            │ 0.XXX [IQR]│ 0.XXX [IQR]│ 0.XXX [IQR]│ 0.XXX [IQR]
NN            │ 0.XXX [IQR]│ 0.XXX [IQR]│ 0.XXX [IQR]│ 0.XXX [IQR]
```

Both baselines beat the null model at N=100%: state explicitly. If they do not: state that too.

**5.2 PINN performance and comparison**

Table same format as 5.1, with PINN row added.

Lead sentence states the outcome category plainly:

- **Strong:** "The PINN achieves a {pct}% RMSE reduction over the best baseline at N=25, with non-overlapping IQRs (Wilcoxon p={p_uncorrected}, Bonferroni-corrected p={p_bonferroni}). The null hypothesis is rejected."
- **Significant:** "The PINN achieves a {pct}% RMSE reduction over the best baseline at N=25, which is statistically significant under Bonferroni correction (p={p_bonferroni} < 0.0125). The null hypothesis is rejected."
- **Moderate:** "The PINN achieves a {pct}% RMSE reduction over the best baseline at N=25. This falls below the pre-registered Significant threshold. The result is directionally consistent at N=50 but does not meet the criterion for statistical significance under Bonferroni correction (p={p_bonferroni})."
- **Inconclusive:** "The PINN achieves a {pct}% RMSE reduction over the best baseline at N=25, below the pre-registered Moderate threshold of 10%. The result is inconclusive. The null hypothesis is not rejected."
- **Negative:** "The PINN does not outperform the best baseline at any training size. The null hypothesis is not rejected. The standard NN achieves lower RMSE than the PINN at {n_fractions_where_nn_wins} of 4 training sizes."

**5.3 Physics↔ML decomposition**

Always written regardless of outcome. Two items:

1. The decomposition values: physics estimate, ML correction, combined estimate at the most recent observation. Numbers from `gate_3_result.fixture_values_for_mvp`.

2. The residual ratio analysis: "At N=25, the median residual ratio (std(δ) / std(m_v_physics)) was {ratio}. A ratio {above/below} 1.0 indicates that the {ML correction / physics estimate} contributed more variance to the final prediction." — value from `diagnostics_residual_ratio.json`.

**5.4 Identifiability diagnostics**

Three paragraphs, one per diagnostic. Each paragraph reports the finding and its interpretation. No diagnostic is omitted regardless of what it shows.

Paragraph 1 — Residual ratio: "The residual ratio remained {below / above} 1.0 for {all / most / some} training sizes, indicating {the physics branch made meaningful contributions to the final estimate / the ML branch progressively dominated as training data increased}."

Paragraph 2 — Parameter sensitivity: "WCM parameters A and B were {well / poorly} constrained by the Moor House observations. The sensitivity of σ°_total to parameter A was {sensitivity_A:.2f} dB per unit change across the observed moisture range ({well-constrained / poorly-constrained} by the threshold of 0.1 dB). The calibrated values (A={A_final:.4f}, B={B_final:.3f}) {are consistent with / deviate from} the literature range for heathland/blanket bog vegetation."

Paragraph 3 — Dobson vs Mironov: "Running the identical PINN configuration with the Mironov (2009) dielectric model produced a median RMSE of {mironov_rmse:.3f} cm³/cm³ compared to {dobson_rmse:.3f} for Dobson (1985), a {relative_improvement:.1%} {improvement / degradation}. {Dobson was retained as the primary model. / The Mironov model was adopted as the primary model and the full 40-configuration experiment was re-run.}"

#### 6. Interpretation

**Outcome-conditional. Length: 300–500 words.**

| Outcome | Core argument |
|---------|--------------|
| **Strong** | Physics-informed ML generalises better from sparse data at this site and sensor combination. The WCM provides a structured prior that constrains the solution space — the improvement at N=25 but not N=100 (where the gap narrows) is consistent with the expected data-efficiency mechanism. The residual ratio confirms the physics branch is contributing. |
| **Significant** | Statistically reliable advantage at the critical data-volume threshold. The improvement is real but modest in absolute terms. The data-efficiency mechanism is supported but the effect size is smaller than the Strong scenario implies. The residual ratio provides secondary evidence on whether the advantage is architectural or incidental. |
| **Moderate** | Directional but not statistically robust at N=25 with 10 repetitions. Three explanations are considered: (1) the effect is real but underpowered at N=25; (2) the PINN architecture needs refinement for this site; (3) C-band SAR at this moisture range has limited physics structure for the WCM to exploit. The diagnostics help discriminate. |
| **Inconclusive** | The SAR-moisture signal is present (baselines beat null), but the physics structure of the PINN does not measurably improve retrieval at any training size. Most likely explanation is that the ML correction network has sufficient capacity to learn the residual without needing the physics scaffold at the available data volumes. This does not invalidate the platform — it informs the PhD programme's research questions. |
| **Negative** | The PINN underperforms standard ML. The most probable cause — indicated by the residual ratio — is that the ML branch is overriding the physics estimate rather than correcting it. The physics layer is not engaged. This is an architectural finding: the capacity asymmetry (physics_net 32→16, correction_net 64→32→16) was insufficient to prevent the correction net dominating. |

#### 7. Limitations

Six bullet points, always the same regardless of outcome:

1. **Single site:** All results are from Moor House. Generalisation to other peatland types (raised bog, fen, degraded agricultural peat) is untested.
2. **Single sensor combination:** Sentinel-1 C-band VV+VH. L-band SAR (ALOS-2, NISAR) has different penetration depth and vegetation interaction.
3. **WCM simplification:** The Oh soil scattering model uses a fixed roughness parameter (ks={KS_ROUGHNESS}). Actual surface roughness varies with season and disturbance.
4. **Data volume:** N=142 paired observations is small. The learning curves at N=10 have high variance. Results at 10% training size should be interpreted cautiously.
5. **Temporal coverage:** The test set covers the most recent ~15 months of the 2021–2024 period. Performance in earlier years (not in test set) is not evaluated.
6. **Dielectric model uncertainty:** Neither Dobson nor Mironov was developed for peat soils. The organic-soil parameterisation (zero clay fraction, low bulk density) is an approximation.

#### 8. Implications for the venture

**Outcome-conditional. Length: 200–300 words.**

This section is the bridge from science to strategy. It references the four-phase progression and speaks to Carbon13 audiences.

| Outcome | Core message |
|---------|-------------|
| **Strong / Significant** | "The PoC validates the core technical thesis: physics-informed ML achieves better soil moisture retrieval than standard ML under data-scarce conditions. This is the mechanism that makes the per-site economics work — a new site reaches useful accuracy with 25 observations rather than hundreds. The data flywheel compounds from there. Phase 2 (causal attribution) is the next validation target: the same architecture that estimates current state can, in principle, attribute observed changes to interventions." |
| **Moderate** | "The PoC shows a directional advantage consistent with the core thesis, though not yet at the pre-registered strength threshold. The monitoring platform is validated — continuous, uncertainty-quantified state estimation is operational. The data-efficiency claim requires qualification: 'physics-informed ML shows a directional accuracy advantage at limited data volumes' rather than 'demonstrated superiority.' The PhD programme will investigate whether a refined physics model or modified architecture recovers the stronger result." |
| **Inconclusive / Negative** | "The PoC validates the monitoring platform and the experimental methodology, but does not validate the specific data-efficiency claim for this site, sensor, and architecture combination. The venture thesis has two distinct components: (1) the platform capability — continuous, uncertainty-quantified environmental monitoring from satellite data — which is demonstrated; (2) the physics-informed advantage — better performance from sparse data — which requires further investigation. Carbon13 materials will lead with the platform capability and be transparent about the current status of the physics-advantage claim." |

---

## P5.2 Document updates

Each document update follows the same process:
1. Identify exactly which sections contain claims that depend on PoC results
2. Replace those sections with outcome-conditional content
3. Increment the document version and add a changelog entry
4. Do not modify sections unrelated to PoC results

### White Paper update

**Sections requiring update:**

| Section | Current claim | Replacement source |
|---------|--------------|-------------------|
| Technical validation | "physics-informed ML expected to outperform..." | Measured result from §5.2 |
| Data efficiency | "estimated N=25 advantage" | Actual outcome category + measured values |
| Model performance | Placeholder RMSE/R² | `gate_3_result.fixture_values_for_mvp` |
| Architecture description | WCM parameter initialisations | Calibrated A and B from diagnostics |

**Replacement rules:**
- Strong/Significant: replace "expected to" with "demonstrated to" + measured numbers
- Moderate: replace "expected to" with "shows a directional advantage" + measured numbers + qualification
- Inconclusive/Negative: replace with platform-capability framing; move physics-advantage claim to "research programme" framing

**Version increment:** e.g. `v1.2 → v1.3`

**Changelog entry format:**
```
## Changelog

### v1.3 — [date]
Updated §[N] Technical Validation with PoC results from ECHO Phase 3.
Outcome category: [category]. RMSE at N=25: [value]. See poc_results.md for full
experimental record.
```

### Yellow Paper update

The Yellow Paper carries the most technical claims. Three specific updates:

**Update 1 — WCM parameter values**

Replace initialisation values with calibrated values from `diagnostics_parameter_sensitivity.json`:

```
Before: A = 0.10 (literature initialisation, Attema & Ulaby 1978)
After:  A = {A_final:.4f} (site-calibrated from {training_pairs} Moor House observations)
        Literature initialisation: 0.10 (Attema & Ulaby 1978)
        Calibration shift: {A_final - 0.10:+.4f}
```

Same format for B.

**Update 2 — Dielectric model selection**

If Dobson retained: add one sentence confirming the empirical comparison and its outcome.
If Mironov adopted: replace the Dobson justification paragraph with the Mironov result and rationale.

**Update 3 — Architecture specification**

Replace "proposed architecture" language with "implemented architecture." Add:
- Actual physics_net capacity (32→16 hidden sizes confirmed)
- Correction_net capacity (64→32→16 confirmed)  
- Selected λ triple from `lambda_search_result.json`
- Stopped_at_epoch statistics (median across 100% configs)

**Version increment:** `v[N] → v[N+1]`

### Green Paper update

The Green Paper carries the PhD research programme. Phase 3 results directly inform three research questions.

**Update 1 — Research Question 1 status**

RQ1 is typically: "Does physics-informed ML improve SAR soil moisture retrieval under data-scarce conditions?"

Add a section: **"PoC Evidence (Moor House, 2021–2024)"** immediately after the RQ1 statement:

| Outcome | Status text |
|---------|------------|
| Strong/Significant | "The Moor House PoC provides supporting evidence for RQ1 at Sentinel-1 C-band on blanket bog peat. The PINN achieves {pct}% RMSE reduction at N=25 (p={p_bonferroni} under Bonferroni correction). This result warrants generalisation testing across ecosystem types and sensor configurations — the core of the PhD experimental programme." |
| Moderate | "The Moor House PoC provides directional but inconclusive evidence for RQ1. The effect is consistent but falls below the pre-registered significance threshold. The PhD programme will investigate whether a refined architecture or physics model recovers a stronger result, and whether the moderate advantage generalises." |
| Inconclusive/Negative | "The Moor House PoC does not provide evidence for RQ1 at this site and sensor configuration. This is a substantive finding that reframes the PhD research question: rather than assuming the physics advantage exists and studying its generalisation, the PhD programme must first characterise the conditions under which a physics advantage emerges. The identifiability diagnostics from the PoC (residual ratio, parameter sensitivity) provide the diagnostic framework." |

**Update 2 — Identifiability as a research theme**

Add a research question (or strengthen an existing one) based on the diagnostic findings:

"**RQn — Identifiability of physics structure in PINN architectures for SAR remote sensing.** Under what conditions does the physics branch of a dual-branch PINN contribute structural constraints rather than being overridden by the ML correction network? The residual ratio diagnostic developed in the Moor House PoC (std(δ)/std(m_v_physics)) provides a quantitative measure for this. Systematically varying architecture capacity ratios, physics model complexity, and data volume will characterise the identifiability boundary."

**Update 3 — Moor House as reference site**

Add a paragraph to the study sites section confirming Moor House as the PhD reference site and recording the baseline performance metrics that future experiments will need to exceed.

**Version increment:** `v[N] → v[N+1]`

---

## P5.3 Carbon13 materials

### Pitch deck update

**File:** `carbon13/pitch_deck_v[N+1].pptx` (or `.key` / `.pdf` as appropriate)

The pitch deck is not rebuilt from scratch. Specific slides are updated. Everything else is unchanged.

**Slides requiring update:**

| Slide | What changes | Source |
|-------|-------------|--------|
| Technical validation slide | Replace "PoC in progress" with result | `poc_results.md §5.2` |
| Core thesis slide | Replace projected improvement with measured | `gate_3_result.outcome` |
| Traction / evidence slide | Add "Moor House PoC complete" milestone | `export_manifest.json` |
| Technology slide | RMSE, R², training pairs | `gate_3_result.fixture_values_for_mvp` |
| Demo slide / screenshot | MVP screenshot at URL | VantageSpec Gate 5 deployed URL |

**Language rules per outcome:**

| Outcome | Headline claim |
|---------|---------------|
| Strong | "Validated: {pct}% accuracy improvement at 25 training samples vs standard ML" |
| Significant | "Statistically validated: physics-informed ML advantage at low data volume" |
| Moderate | "Directional advantage demonstrated: physics-informed ML at limited data volumes" |
| Inconclusive | "Continuous monitoring platform validated: {training_pairs} paired observations, RMSE {rmse} cm³/cm³" |
| Negative | "Monitoring platform validated: continuous, uncertainty-quantified peatland state estimation operational" |

**Rule:** Every number on every updated slide must appear in `poc_results.md`. No number from memory, from the original Yellow Paper projections, or from the VantageSpec placeholder fixtures.

### Co-founder brief update

**File:** `carbon13/cofounder_brief_v[N+1].md`

The co-founder brief is for Chief Scientist and Commercial CEO recruitment conversations. Update two sections:

**Section: "What's been built"**

Add:
```
**ECHO PoC (completed [date])**
- 142 paired SAR-moisture observations at Moor House (2021–2024)
- Three models trained and evaluated: null baseline, Random Forest, Standard NN, PINN
- Outcome: [category] — [one sentence summary from outcome_summary field in narrative.json]
- Physics↔ML decomposition chart: physics estimate {physics_estimate} cm³/cm³,
  ML correction {ml_correction} cm³/cm³, combined {final_estimate} cm³/cm³
- Live demo: [deployed URL]
- Full experimental record: [link to poc_results.md or summary]
```

**Section: "What's needed next"**

The capability gaps are constant (Chief Scientist: applied remote sensing science; Commercial CEO: environmental sector networks) but the evidence context changes:

| Outcome | Chief Scientist framing |
|---------|------------------------|
| Strong/Significant | "The PINN architecture is validated on blanket bog. The Chief Scientist role extends this to multiple ecosystem types, L-band sensor compatibility, and multi-site calibration transfer — the PhD programme provides the research structure." |
| Moderate/Inconclusive | "The PoC identifies an open architectural question: under what conditions does physics-informed ML engage its physics structure rather than being dominated by the ML residual? The Chief Scientist role leads the investigation and owns the answer." |
| Negative | "The PoC reveals that the current WCM-PINN architecture does not engage the physics structure at Moor House. The Chief Scientist role begins with diagnosing why — is this a C-band limitation, a peat-specific dielectric model problem, or an architecture capacity issue — and redesigning accordingly." |

---

## P5.4 Gate 5 criteria

**Run with:** `python echo-poc/poc/gates/gate_5.py [--confirm-deviations]`  
**Exit code:** 0 = pass, 1 = fail

| ID | Criterion | Verification | Auto-checkable |
|----|-----------|-------------|----------------|
| G5-01 | `poc_results.md` exists and is complete | All 8 sections present | Partial (section header check) |
| G5-02 | Results section states outcome category explicitly | String match against `gate_3_result.outcome.category` | Yes |
| G5-03 | Attrition table numbers match `gate_1_result.json` | Exact value comparison | Yes |
| G5-04 | RMSE/R² values in results match `gate_3_result` | Exact value comparison | Yes |
| G5-05 | All three diagnostics reported | Residual ratio, param sensitivity, Dobson/Mironov present | Partial |
| G5-06 | Limitations section present with ≥ 5 items | Count of bullet points | Yes |
| G5-07 | No numbers in `poc_results.md` that contradict `gate_3_result.json` | Cross-reference scan | Yes |
| G5-08 | White Paper version incremented | `v[N] → v[N+1]` in document | Partial |
| G5-09 | White Paper technical sections updated | Diff against previous version: §Technical Validation changed | Manual |
| G5-10 | Yellow Paper WCM parameters updated | Calibrated A and B match `diagnostics_parameter_sensitivity.json` | Yes |
| G5-11 | Yellow Paper architecture section updated | λ values match `lambda_search_result.json` | Yes |
| G5-12 | Green Paper PoC evidence section added | Section header present | Partial |
| G5-13 | Green Paper identifiability RQ present | Section header present | Partial |
| G5-14 | Pitch deck updated slides present | Manual review | Manual |
| G5-15 | Pitch deck numbers traceable to `poc_results.md` | Cross-reference check | Manual |
| G5-16 | Co-founder brief "What's been built" updated | Date + outcome in brief | Partial |
| G5-17 | Demo walkthrough rehearsed and timed | Manual sign-off | No |
| G5-18 | MVP deployed URL accessible | HTTP 200 | Yes |
| G5-19 | DEVIATIONS.md reviewed (final review) | `--confirm-deviations` flag | No |
| G5-20 | No `TODO` or `FIXME` in any Phase 5 deliverable | String scan | Yes |

### Gate 5 failure categories

**Hard failures** (gate does not pass until resolved): G5-02, G5-03, G5-04, G5-07, G5-10, G5-11, G5-18

**Soft failures** (gate passes with logged warning, must be resolved before Carbon13 presentation): G5-05, G5-06, G5-14, G5-15, G5-16

**Manual sign-offs** (checked by human, not script): G5-09, G5-13, G5-17, G5-19

### Automated checks (`poc/gates/gate_5.py`)

```python
def check_numbers_consistent(
    poc_results_path: Path,
    gate3_result_path: Path,
) -> list[str]:
    """
    Extract all floating-point numbers from poc_results.md that
    appear in contexts matching known metric patterns (RMSE, R², bias,
    reduction percentage, p-value).

    Cross-reference each against gate_3_result.json.

    Returns list of inconsistency descriptions. Empty list = pass.

    Tolerance: ±0.001 for absolute values, ±0.5% for percentage values.
    This allows for legitimate rounding in prose ("approximately 17%")
    while catching transcription errors.
    """

def check_attrition_table(
    poc_results_path: Path,
    gate1_result_path: Path,
) -> list[str]:
    """
    Parse the attrition table from poc_results.md.
    Compare each row's numbers against gate_1_result.json → attrition_log.
    Zero tolerance — these are exact counts, not rounded values.
    """

def check_wcm_parameters(
    yellow_paper_path: Path,
    diagnostics_sensitivity_path: Path,
) -> list[str]:
    """
    Extract WCM A and B values from Yellow Paper.
    Compare against diagnostics_parameter_sensitivity.json median values.
    Tolerance: ±0.0001 (4 decimal places).
    """
```

---

## P5.5 Deliverable summary

| # | Deliverable | File path | Gate criteria |
|---|------------|-----------|--------------|
| 5.1 | PoC Results Document | `echo-poc/outputs/write-up/poc_results.md` | G5-01 through G5-07 |
| 5.2 | White Paper v[N+1] | `documents/white-paper/white-paper-v[N+1].md` | G5-08, G5-09 |
| 5.3 | Yellow Paper v[N+1] | `documents/yellow-paper/yellow-paper-v[N+1].md` | G5-10, G5-11 |
| 5.4 | Green Paper v[N+1] | `documents/green-paper/green-paper-v[N+1].md` | G5-12, G5-13 |
| 5.5 | Pitch Deck v[N+1] | `carbon13/pitch-deck-v[N+1].[ext]` | G5-14, G5-15 |
| 5.6 | Co-Founder Brief v[N+1] | `carbon13/cofounder-brief-v[N+1].md` | G5-16 |
| 5.7 | Demo walkthrough rehearsed | (no file — manual sign-off) | G5-17 |
| 5.8 | Gate 5 result | `echo-poc/outputs/gates/gate_5_result.json` | All |

---

## P5.6 Writing standards

These apply to all Phase 5 documents.

**Precision over hedging.** "RMSE of 0.059 cm³/cm³ at N=25" is always better than "approximately 6% error." Use measured numbers. If a number is a projection or estimate rather than a measured value, say so.

**Outcome stated once, clearly.** The outcome category (Strong / Significant / Moderate / Inconclusive / Negative) appears in the executive summary of the PoC results document. It is not repeated in every paragraph. It does not need qualifying adjectives ("slightly inconclusive," "borderline significant").

**Diagnostics are not decoration.** The three identifiability diagnostics are reported because they characterise the trustworthiness of the result, not because they look thorough. If the residual ratio is > 1.0 at N=25, that matters — the results section says it matters and the interpretation section explains what it means.

**No retrospective hypothesis adjustment.** The pre-registered success criteria are in `SPEC.md §6.2`. They do not change because the result is disappointing. The results section applies the pre-registered criteria. The interpretation section may discuss what a different threshold would have shown — but only as additional context, never as a substitute for the pre-registered evaluation.

**Carbon13 materials lead with capability, not just outcome.** The pitch deck and co-founder brief should foreground what the platform does (continuous satellite monitoring, quantified uncertainty, operational MVP) before what the experiment found. The science is evidence for the platform's credibility — it is not the only evidence.

---

## P5.7 Phase 5 file manifest

```
echo-poc/
├── outputs/
│   ├── write-up/
│   │   └── poc_results.md
│   └── gates/
│       └── gate_5_result.json
└── poc/
    └── gates/
        └── gate_5.py

documents/
├── white-paper/
│   └── white-paper-v[N+1].md
├── yellow-paper/
│   └── yellow-paper-v[N+1].md
└── green-paper/
    └── green-paper-v[N+1].md

carbon13/
├── pitch-deck-v[N+1].[ext]
└── cofounder-brief-v[N+1].md
```

---

## P5.8 The complete ECHO spec suite

With Phase 5 complete, the full specification suite is:

| Document | Scope | Gate |
|----------|-------|------|
| `CLAUDE.md` | Engineering constitution — immutable laws | N/A |
| `SPEC.md` | Top-level specification — research question, models, evaluation | N/A (superseded by phase specs) |
| `DEVIATIONS.md` | Append-only log of spec deviations | Reviewed at every gate |
| `SPEC_PHASE1.md` | Data acquisition and alignment | Gate 1 |
| `SPEC_PHASE2.md` | Baseline models and evaluation harness | Gate 2 |
| `SPEC_PHASE3.md` | PINN, identifiability diagnostics, pre-registered evaluation | Gate 3 |
| `SPEC_PHASE4.md` | Fixture export pipeline + MVP dashboard (VantageSpec) | Gate 4 |
| `SPEC_PHASE5.md` | Write-up, document updates, Carbon13 materials | Gate 5 |

```
═══════════════════════════════════════════════════════
  ECHO PoC SPECIFICATION SUITE — COMPLETE
  All phases specced. All gates defined. Ready to build.
═══════════════════════════════════════════════════════
```
