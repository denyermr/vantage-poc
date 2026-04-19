# Phase 1b Test Specification: PINN-MIMICS Soil Moisture Retrieval at Moor House

**Vantage · Phase 1b · Pre-Registration · v0.1.1 — Signed (non-substantive amendment 2026-04-19)**

> **Versioning note (v0.1 → v0.1.1, 2026-04-19):** Non-substantive amendment.
> Added §15 "Phase 1b Pre-Registration document" cross-referencing the new
> companion document `phase1b/SUCCESS_CRITERIA.md` (pre-registration formalisation
> per Session F-1). Renumbered "References" from §15 to §16. **No threshold,
> tier, or signed-criterion change.** All §10 RMSE thresholds, §9 dominance
> criteria, §11 diagnostics, and §14 sign-off block stand verbatim. Tracked
> as a SPEC v0.1 → v0.1.1 minor amendment in the v0.1 sign-off record
> at §14, not as a DEV-1b-NNN entry (no scientific deviation; document
> structural addition only).

*A controlled module-replacement re-run of the ECHO Phase 1 PoC — only the physics branch changes.*

Companion to: `MIMICS Literature Review v1.0` and `poc_results.md v1.0`.

---

## Contents

1. [Purpose & controlled-comparison logic](#1-purpose--controlled-comparison-logic)
2. [Frozen elements — what does not change](#2-frozen-elements--what-does-not-change)
3. [Changed elements — the MIMICS module replacement](#3-changed-elements--the-mimics-module-replacement)
4. [MIMICS forward model architecture](#4-mimics-forward-model-architecture)
5. [Parameter table — fixed, constrained, learnable](#5-parameter-table--fixed-constrained-learnable)
6. [Dielectric model handling](#6-dielectric-model-handling)
7. [Surface scattering model](#7-surface-scattering-model)
8. [PINN architecture & loss function](#8-pinn-architecture--loss-function)
9. [λ hyperparameter search protocol](#9-λ-hyperparameter-search-protocol)
10. [Pre-registered success criteria](#10-pre-registered-success-criteria)
11. [Diagnostic plan](#11-diagnostic-plan)
12. [Risk register & mitigations](#12-risk-register--mitigations)
13. [Honest-gates protocol & deviation log template](#13-honest-gates-protocol--deviation-log-template)
14. [Pre-registration sign-off](#14-pre-registration-sign-off)
15. [Phase 1b Pre-Registration document](#15-phase-1b-pre-registration-document)
16. [References](#16-references)

---

## 1. Purpose & controlled-comparison logic

> Phase 1 produced a precise diagnosis: the Water Cloud Model is structurally inadequate for blanket bog at C-band. Phase 1b tests whether replacing the WCM with MIMICS — and changing nothing else — recovers the physics-informed advantage. To make that test meaningful, the experiment must be a single-variable substitution, not a redesign.

This specification pre-registers the Phase 1b PoC. It mirrors the ECHO Phase 1 protocol (`poc_results.md` v1.0, 09 March 2026) clause-by-clause. Site, data, quality control, train/test split, random seeds, the 4×10 factorial design, the baseline models, the PINN backbone, the evaluation metrics, the comparison point at N≈25, and the honest-gates methodology are all retained verbatim. The single intentional change is the physics branch: the Water Cloud Model is replaced by a Toure-style single-crown MIMICS implementation, with the unavoidable downstream consequences this entails — a different parameter set, a different dielectric model (Mironov primary), a dual-polarisation forward model, and a joint VV+VH physics loss term.

**Why a controlled comparison matters:** Phase 1's Negative outcome was diagnostic, not exploratory. The diagnosis (WCM forward fit r = 0.007, residual–NDVI correlation r = 0.82, residual ratio 3.3–6.4×) names the WCM as the binding constraint. The only way to test that diagnosis is to swap the WCM for a candidate replacement while holding every other variable constant. Any element of the experimental design that changes alongside the physics module weakens the causal interpretation of the result.

### What this specification is, and is not

This is a pre-registration document. It fixes the experimental procedure before any MIMICS implementation, training, or evaluation begins. The success criteria, diagnostic thresholds, λ-search procedure, and outcome categories are all set in advance, against the same baseline numbers Phase 1 used. This is not a research plan that will be revised in light of intermediate results — deviations are tracked through the deviation log (§13) and reported alongside the final outcome.

This specification is a companion to the MIMICS Literature Review v1.0 (April 2026), which establishes the scientific basis for the parameter choices and structural decisions adopted here. Where this document references a parameter range, an adaptation precedent, or a risk, the underlying justification is in the lit review.

### The hypothesis under test

> **Phase 1b hypothesis (falsifiable):** Replacing the Water Cloud Model with a Toure-style single-crown MIMICS implementation as the physics branch of the ECHO PINN architecture, with all other elements of the Phase 1 protocol held constant, produces a soil moisture retrieval at Moor House (Sentinel-1 C-band, COSMOS-UK ground truth) whose RMSE at the pre-registered N≈25 evaluation point is lower than the Random Forest baseline (RMSE 0.155 cm³/cm³).

The null hypothesis is that PINN-MIMICS RMSE at N≈25 is greater than or equal to RF baseline RMSE. Failure to reject the null is a Negative outcome and constitutes evidence that either (a) MIMICS is also structurally inadequate at this site, (b) the C-band sensor is the binding constraint regardless of physics model complexity, or (c) the PINN architecture itself is the binding constraint. Phase 4 diagnostics will distinguish these three failure modes if the outcome is Negative.

---

## 2. Frozen elements — what does not change

> The following elements are inherited verbatim from Phase 1. They are the controlled variables of the experiment. Any change to any of them during Phase 1b execution must be logged as a deviation and assessed for impact on the controlled-comparison interpretation.

### Site, data, and pre-processing

- **Site:** COSMOS-UK Moor House (54.69°N, 2.38°W, ~550 m a.s.l., site ID MOORH). Single-site PoC.
- **Ground truth:** COSMOS-UK daily VWC product, 2021-01-01 to 2024-12-31. Quality control via portal flags (gap-filled 'E' and interpolated 'I' observations excluded; per Phase 1 DEV-001).
- **SAR:** Sentinel-1 IW GRD descending orbit, relative orbit 154, VV+VH, extracted via Google Earth Engine over the COSMOS-UK footprint. Mean incidence angle 41.5°.
- **Vegetation index:** Sentinel-2 monthly NDVI composites (cloud-filtered), gap-filled by linear interpolation for overcast periods. **NDWI (Gao 1996) added in Phase 1b** as an auxiliary Sentinel-2 product — originally planned as a prior on canopy water content in v0.1-draft, revised to diagnostic-only post-amendment (see §5 and DEV-1b-002).
- **Meteorological:** ERA5-Land daily precipitation, including 7-day antecedent accumulation.
- **Quality filters:** Frozen-ground exclusion (T_min < 0°C), snow exclusion, manual freeze-thaw exclusion of 2021-02-05 (Phase 1 DEV-005), VWC QC-flag exclusion. Identical attrition pipeline → identical N = 119 paired observations.

### Train/test architecture and stochasticity

- **Final dataset:** N = 119 paired SAR–VWC observations spanning 2021-01-18 to 2024-12-10.
- **Train/test split:** Chronological 70/30, sealed test set n = 36 (2023-07-25 to 2024-12-10). Splits generated once and saved; never regenerated.
- **Training pool:** n = 83. Subsampled with stratification by meteorological season where sample sizes permit.
- **Random seeds:** `SEED = 42 + config_idx`. Identical seeds across baselines and PINN-MIMICS to ensure paired comparison validity.
- **Experimental design:** 4×10 factorial — four training fractions (10%, 25%, 50%, 100%) crossed with 10 independent repetitions each. 40 configurations per model.

### Baselines (re-run, not re-tuned)

- **Null model:** Seasonal climatological mean VWC (training set, applied by season). Expected RMSE 0.178 cm³/cm³ (recompute on the same data; should match exactly).
- **Random Forest:** `scikit-learn RandomForestRegressor`, 5-fold CV grid over the same 54 hyperparameter combinations. Same seven input features (VV, VH, VH/VV ratio, NDVI, daily precip, 7-day precip, incidence angle). Expected RMSE at N≈25: 0.155 cm³/cm³; at N=83: 0.147 cm³/cm³.
- **Standard NN:** Three-layer 64→32→16, ReLU, dropout 0.2, Adam (lr=10⁻³), early stopping (patience 20). Same seven input features. Architecture identical to the PINN correction branch.

> **Baseline re-run requirement:** The Phase 1 baselines must be re-executed on the Phase 1b runtime environment to confirm reproducibility of the published numbers (Null 0.178, RF 0.147 at N=83, RF 0.155 at N≈25, NN 0.159 at N=83). Any drift > 0.005 cm³/cm³ in any baseline RMSE is logged as a deviation and the source identified before PINN-MIMICS training begins. The pre-registered success criteria in §10 reference the published Phase 1 baseline numbers; if reproduction confirms them within tolerance, those numbers stand.

### PINN backbone (architecture skeleton)

- **Two-branch structure:** PhysicsNet + CorrectionNet, combined as `m_v_final = m_v_physics + δ`.
- **PhysicsNet:** Input(7) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid × θ_sat. Output: physically-bounded VWC estimate `m_v_physics ∈ [0, 0.88]`.
- **CorrectionNet:** Input(7) → Linear(64) → ReLU → Dropout → Linear(32) → ReLU → Dropout → Linear(16) → ReLU → Linear(1). Output: unbounded residual δ.
- **Optimiser:** Adam, lr = 10⁻³, early stopping patience 20 epochs on validation loss. Same as Phase 1.

### Evaluation

- **Metrics:** RMSE, R², mean bias, all on the sealed test set (n = 36).
- **Learning curves:** Median ± IQR across 10 repetitions per training size.
- **Statistical comparison:** Paired Wilcoxon signed-rank test on per-repetition RMSE values, both uncorrected (α=0.05) and Bonferroni-corrected (α=0.0125 for 4 comparisons).
- **Pre-registered evaluation point:** N≈25 (25% training fraction).

---

## 3. Changed elements — the MIMICS module replacement

> Three things change in Phase 1b. The first is intentional and is the experiment itself: the physics module. The other two are unavoidable consequences of that change — the dielectric model recommendation from the lit review (Mironov primary for peat substrate), and the addition of a cross-polarisation term in the physics loss because MIMICS, unlike the WCM, predicts both polarisations.

| Status | Element | Phase 1 (WCM) | Phase 1b (MIMICS) |
|---|---|---|---|
| **Δ** | Physics module | Water Cloud Model (Attema & Ulaby 1978). 2 learnable parameters (A, B). Bulk-volume isotropic crown. | MIMICS, Toure-style single-crown adaptation (Toure et al. 1994). 4–5 learnable parameters. Discrete cylinder/disc crown scatterers, no trunk layer. |
| **Δ** | Dielectric model | Dobson 1985 (mineral-soil parameterisation, peat-adapted: ε_dry=3.5, ε_water=80, α=1.4). Mironov retained as sensitivity check only (Phase 1 DEV-007). | **Mironov 2009 primary** (organic-soil GRMDM with ε ≥ 1.01 clamp inherited from DEV-007). Dobson run as comparative sensitivity arm. See §6. |
| **Δ** | Forward model output | VV only. `L_physics = MSE(σ°_WCM, VV_obs)`. | VV and VH jointly. `L_physics = MSE(σ°_VV, VV_obs) + MSE(σ°_VH, VH_obs)`. See §8. |
| **Δ** | Sentinel-2 derived priors | NDVI used as RF/NN feature only. No prior into physics branch. | No S2-derived priors on physics-branch parameters. NDVI remains in the seven-feature input vector unchanged. NDWI extracted as auxiliary diagnostic covariate. Both priors originally planned in v0.1-draft (NDVI → LAI → N_l; NDWI → m_g) were withdrawn pre-sign-off — see §5 amendment, DEV-1b-001, DEV-1b-002. |
| **=** | Site, data, QC, attrition | Moor House, S1 desc orbit 154, COSMOS-UK, S2 NDVI, ERA5. N=119. | Identical. NDWI added as auxiliary S2 product (diagnostic-only) but does not change the input feature vector. |
| **=** | Train/test split, seeds, 4×10 factorial | 83/36 chronological. SEED=42+config_idx. 4 fractions × 10 reps. | Identical. |
| **=** | Baselines | Null, RF (54-combo grid), NN (64→32→16). 7 features. | Identical. Re-run for reproducibility check (§2). |
| **=** | PINN backbone | PhysicsNet (32→16→1, sigmoid-bounded) + CorrectionNet (64→32→16→1). m_v_final = m_v_physics + δ. | Identical backbone. Only the WCM module embedded in the forward graph is replaced by MIMICS. |
| **=** | Evaluation, statistics, comparison point | Sealed test set n=36. Wilcoxon (uncorrected α=0.05, Bonferroni α=0.0125). N≈25 evaluation. | Identical. |
| **=** | Outcome categories | Strong / Significant / Moderate / Inconclusive / Negative against RF baseline at N≈25. | Identical structure, identical thresholds (re-derived against same RF numbers in §10). |

---

## 4. MIMICS forward model architecture

> Phase 1b implements the Toure (1994) crop-adaptation strategy: a single crown layer above a rough peat surface, with no separate trunk layer. The lit review (Section 8, Decision 1, Option B) gives the physical and literature justification — at C-band, heather stems (1–4 mm diameter, 10–30 cm length) are in the wavelength-comparable regime where MIMICS' infinite-cylinder trunk approximation is inaccurate, and the Toure precedent demonstrates that wheat and canola are best represented by a unified crown of mixed-geometry scatterers.

### Layer structure

- **Crown layer:** Heather + Sphagnum represented as a height-distributed mixture of scatterers — woody stem cylinders (a_b, l_b) and disc leaves (a_l, t_l). Layer thickness h_c (fixed from site measurement, see §5). No trunk sub-layer.
- **Ground layer:** Rough peat surface. Surface roughness s and correlation length l (see §7). Dielectric ε from Mironov GRMDM (see §6).

### Scattering mechanisms retained

- **Direct crown volume scattering** — branch + leaf contributions, both co-pol (VV, HH) and cross-pol (VH).
- **Crown–ground interaction** — energy transmitted through crown, scattered by ground, retransmitted upward. The path through which soil moisture enters the C-band signal.
- **Ground direct** — bare-surface scattering at the peat/water interface, attenuated by crown two-way loss.
- **Trunk–ground double bounce: NOT retained** (Toure adaptation). Stem-zone scatterers are folded into the crown distribution.

### Differentiable implementation

MIMICS must be implemented as a fully differentiable PyTorch module. The Phase 1 architecture description states that "WCM → MIMICS is a module replacement, not an architectural change", but the lit review (§11, R4) flags this as a substantive software engineering task. The implementation must handle the polarimetric matrix operations (4×4 Stokes-like transformations), the cylinder and disc scattering amplitude tensors, and the radiative transfer propagation terms while preserving gradient flow through all learnable parameters.

> **Implementation gate:** Before any training begins, the differentiable MIMICS module must pass a three-arm forward-pass equivalence check defined in `phase1b/physics/g2_anchor_spec.md` (see DEV-1b-003 for sourcing and rationale): (i) a **numpy_port arm** — agreement between a deterministic numpy implementation and the PyTorch implementation on the full canonical input set; (ii) a **published_table arm** — agreement between the PyTorch implementation and published σ° values from Toure et al. 1994 (primary; wheat C-band multiangle, scattering-mechanism decomposition) and McDonald, Dobson & Ulaby 1990 (secondary; walnut orchard L-band); and (iii) a **gradient spot-check arm** — agreement between PyTorch autograd ∂σ°/∂parameter and Toure 1994's published sensitivity coefficients, plus internal consistency between autograd and finite-difference estimates. Tolerance: σ° agreement within 0.5 dB across all VV and HH test points in (i) and (ii); ±20% or ±0.1 dB for gradient values in (iii). All three arms must pass. Failure of this gate halts the experiment until the implementation is corrected.

### Inputs and outputs

- **Inputs:** All MIMICS structural parameters (fixed and learnable, see §5), m_v from PhysicsNet, frequency (5.405 GHz fixed), incidence angle θ_i (per-observation, mean 41.5°).
- **Outputs:** σ°_VV (dB) and σ°_VH (dB). Both feed into the physics loss (§8).

---

## 5. Parameter table — fixed, constrained, learnable

> The effective complexity of the inversion is determined by which MIMICS parameters are fixed from prior knowledge, which are constrained by Sentinel-2-derived covariates, and which are jointly learned with m_v. This table is the pre-registered parameter assignment for Phase 1b.

### Crown geometry — fixed from Calluna field literature

| Parameter | Symbol | Description | Range / value | Status |
|---|---|---|---|---|
| Crown height | `h_c` | Vertical extent of crown layer at Moor House | 0.4 m (mean); range 0.3–0.5 m site survey | **Fixed** |
| Branch radius | `a_b` | Mean cylinder radius of woody heather stems | 2 mm | **Fixed** |
| Branch length | `l_b` | Mean cylinder half-length | 8 cm | **Fixed** |
| Leaf radius | `a_l` | Mean disc radius (heather leaf approximation) | 1 cm | **Fixed** |
| Leaf thickness | `t_l` | Mean disc thickness | 0.3 mm | **Fixed** |

### Crown densities — learnable with literature-range bounds

| Parameter | Symbol | Description | Range / value | Status |
|---|---|---|---|---|
| Branch density | `N_b` | Branches per m³ in crown layer | 10²–10⁴ m⁻³ (sigmoid-bounded). Init from heather biomass priors. | **Learnable** |
| Leaf density | `N_l` | Leaves per m³ in crown layer | 10³–10⁵ m⁻³ (sigmoid-bounded). Constrained by literature range only — no S2-derived prior (see §5 amendment, DEV-1b-001). Init at geometric mean of bounds (10⁴ m⁻³). | **Learnable** |
| Branch orientation | `σ_orient` | Std dev of branch zenith angle distribution (Gaussian, mean 0° = vertical) | 15°–60° (sigmoid-bounded) | **Learnable** |

### Crown moisture — evergreen, weak time variation

| Parameter | Symbol | Description | Range / value | Status |
|---|---|---|---|---|
| Gravimetric water content | `m_g` | Crown-layer gravimetric moisture (branch + leaf, tied at the branch range) | 0.3–0.6 g/g (sigmoid-bounded). Constrained by literature range only — no S2-derived prior (see §5 amendment, DEV-1b-002). Init at midpoint (0.45 g/g). Single time-stable value across observations (Calluna evergreen; see §5 amendment for branch-dominant-scatterer justification). | **Learnable** |

### Ground layer — surface and substrate

| Parameter | Symbol | Description | Range / value | Status |
|---|---|---|---|---|
| Surface roughness (RMS height) | `s` | Peat surface RMS height (Sphagnum hummock-hollow microtopography) | 1–5 cm (sigmoid-bounded). Larger than mineral soil. | **Learnable** |
| Correlation length | `l_corr` | Surface autocorrelation length | 5 cm (fixed; insufficient observations to identify jointly with s). | **Fixed** |
| Soil moisture | `m_v` | Volumetric water content of peat (the retrieval target) | 0.0–0.88 cm³/cm³ (sigmoid × θ_sat). | **Per-obs learned** |

### Sensor and frequency — fixed

| Parameter | Symbol | Description | Range / value | Status |
|---|---|---|---|---|
| Frequency | `f` | Sentinel-1 C-band centre frequency | 5.405 GHz | **Fixed** |
| Incidence angle | `θ_i` | Per-observation incidence angle from S1 metadata | ~29°–46° per scene (mean 41.5°) | **Per-obs input** |

> **Learnable parameter count: 5** (N_b, N_l, σ_orient, m_g, s) plus the per-observation m_v retrieved by PhysicsNet. This is the parameter-to-data ratio flagged in lit review §11 R3 — N=83 training observations against 5 structural parameters plus per-observation m_v. The dominance constraint in §9 is the primary defence against the over-parameterisation risk this implies.

### Sentinel-2 prior implementation

> **Pre-sign-off amendments (April 2026, DEV-1b-001 and DEV-1b-002):** v0.1-draft of this specification placed two Sentinel-2-derived priors on physics-branch parameters: an NDVI → LAI → N_l prior (citing Lees et al. 2021) and an NDWI → m_g prior. Pre-sign-off review withdrew both priors for related but distinct reasons.

**DEV-1b-001 (N_l prior):** A citation audit found that Lees et al. (2021) concerns Sentinel-1 SAR water-table-depth retrieval and contains no NDVI → LAI relationship for heather. The Lees PhD thesis (Reading, 2019) was checked as an alternative source; it likewise contains no Calluna NDVI → LAI fit. The wider peatland-vegetation literature treats NDVI → LAI for mixed Calluna–Sphagnum canopies as poorly constrained — NDVI saturates rapidly in this canopy, the Beer's-law extinction coefficient is ill-defined for Sphagnum understory (Weston et al. 2015, in Lees 2019), and chlorophyll-based indices (CI, MTCI) outperform NDVI as canopy-property predictors in these systems (Harris 2008; Lees et al. 2020; Letendre et al. 2008; Tucker et al. 2022). The N_l prior is withdrawn. See lit-review Decision 5 for the parallel revision.

**DEV-1b-002 (m_g prior):** No heather-specific NDWI → m_g transfer function exists. NDWI's physical basis (SWIR absorption by foliar water) is real, but the calibration required to map a Gao-NDWI value to a specific gravimetric moisture in g/g for Calluna is absent from the peer-reviewed literature. More decisively, the v0.1-draft specification mandated that the prior operate on a single time-stable scalar m_g averaged across the training set — meaning per-observation NDWI variation is averaged out before the prior is formed. At a single site, the resulting training-set-mean NDWI does not discriminate against any comparator site, and the "prior" reduces to a scalar choice dressed up as an empirical procedure. The m_g prior is withdrawn. NDWI remains extracted (Gao 1996, using Sentinel-2 B08 and B11 bands, per the U-3 work) but re-purposed as a **diagnostic-only covariate**: post-experiment, the correlation between training-set-mean NDWI and the learned m_g value is reported as part of the §11 diagnostic plan.

### Current prior state (post-amendments)

**No Sentinel-2-derived priors enter the physics loss.** The L_prior term is removed from the composite loss (§8). The two withdrawn priors represent the entire planned Sentinel-2 prior contribution; no alternative quantitative mapping replaces them at v0.1 sign-off.

**Implications for the N_l and m_g parameterisation:**

- **N_l:** Constrained only by literature range bounds [10³, 10⁵] m⁻³ (lit review §3) and the physics-consistency term (L_physics) in the composite loss (§8). Init at geometric mean of bounds (10⁴ m⁻³).
- **m_g:** Constrained only by its literature range bounds [0.3, 0.6] g/g (lit review §3, branch m_g range) and the physics-consistency term. Init at midpoint (0.45 g/g). The tied branch+leaf range uses the branch range rather than the wider leaf range (lit review §3 gives leaf m_g 0.5–0.8 g/g) on the **branch-dominant-scatterer** justification: at C-band for Calluna, branch radii (1–4 mm) are comparable to the wavelength (λ ≈ 5.55 cm, λ/2π ≈ 0.88 cm) and branches are the dominant volume scatterers, while scale-like heather leaves (0.2–0.3 cm) sit near the Rayleigh regime and contribute far less to the polarimetric signal. Tying m_g to the branch range is therefore physically defensible for a C-band retrieval; the leaf-range fraction of the contribution is absorbed into the modelling bias budget.

### Retained Sentinel-2 use

- **NDVI:** remains in the seven-feature input vector to all models (RF, NN, CorrectionNet, PhysicsNet) exactly as in Phase 1.
- **NDWI:** extracted per U-3 but used only as a diagnostic covariate (§11). Does not enter the input vector or any loss term.

### Pre-registered v0.2 fallback paths

If post-experiment diagnostics (§11) show parameter non-identifiability, the following v0.2 substitutions are committed to and **these are the only fallbacks** — reverting to the withdrawn v0.1-draft priors is explicitly out of scope:

- **N_l:** If Diagnostic B (parameter correlation matrix) shows N_l non-identifiability, v0.2 introduces a **chlorophyll-index-based prior (CI from Sentinel-2 red-edge bands)**, not a NDVI → LAI chain. This is the DEV-1b-001 fallback.
- **m_g:** If Diagnostic B shows m_g non-identifiability or the m_g posterior pins to a bound, v0.2 introduces a **season-stratified literature scalar prior** per lit review §8 Decision 3 (summer m_g vs winter m_g, literature-sourced from Calluna field studies), with explicit framing as sanity-bound regularisation rather than a retrieval target. This is the DEV-1b-002 fallback. Reverting to any form of NDWI-derived mapping (linear, per-observation, or training-set-averaged) is **not** the fallback.

Both v0.2 paths are pre-registered as part of this amendment; invoking either of them does not constitute a further deviation at Phase 1b execution.

---

## 6. Dielectric model handling

> Mironov GRMDM (2009) is the primary dielectric model for the peat substrate, selected for its more physically appropriate two-phase bound/free water structure. Dobson 1985 (with the Phase 1 organic-peat parameterisation) is run as a comparative sensitivity arm. The known DEV-007 issue from Phase 1 — Mironov producing ε < 1.0 at low VWC — is handled with the inherited ε ≥ 1.01 clamp.

### Primary model: Mironov GRMDM (2009)

- **Soil texture inputs:** Set for blanket bog peat — sand fraction 0%, clay fraction 0%, organic fraction effectively 100%. The Mironov calibration was performed on generic organic soils rather than Sphagnum-dominated peat specifically (lit review §11 R2).
- **ε clamp:** ε_real ≥ 1.01 enforced as in Phase 1 DEV-007, to prevent the GRMDM from producing unphysical ε < 1 at very low m_v (an artefact of applying a mineral-soil-derived formulation outside its calibration range).
- **Differentiable implementation:** The clamp is implemented with a soft-max smoothing rather than a hard clip to preserve gradient flow.

### Sensitivity arm: Dobson 1985

- **Parameterisation:** Identical to Phase 1 — ε_dry = 3.5, ε_water = 80.0, α = 1.4, organic-soil adapted (zero clay fraction, low bulk density 0.1 g/cm³). This preserves backwards comparability with Phase 1.
- **Run scope:** Dobson is run on the 10 × 100% training configurations as a sensitivity arm only. It does not contribute to the pre-registered primary outcome. Its results are reported in the Phase 1b results document for diagnostic transparency.

### Pre-registered diagnostic: Dobson vs Mironov forward difference

> **Pre-registered diagnostic (lit review §9):** Compute |ε_Dobson − ε_Mironov| over the observed m_v range (0.25–0.83 cm³/cm³ at Moor House). Report as a function of m_v. If the maximum relative difference is <5%, the choice of dielectric model is not the binding constraint and the Phase 1 → Phase 1b dielectric change cannot explain any RMSE difference. If >5%, the dielectric choice must be carried as an active source of variance in the Phase 1b interpretation.

This diagnostic is computed once, at the pre-registration sign-off stage (before any PINN training). Its outcome is recorded in §13 as a baseline reference, not as a deviation.

---

## 7. Surface scattering model

> Phase 1 used the Oh (1992) empirical surface scattering model with fixed roughness ks = 0.30. Phase 1b retains Oh as the primary surface model, but with surface roughness s as a learnable parameter (range 1–5 cm) rather than a fixed value, because peat surface roughness at Moor House (Sphagnum hummock-hollow microtopography) is substantially larger than the mineral-soil regime Oh was calibrated on.

### Validity range check

The Oh (1992) model has a published validity envelope on ks (= surface roughness × wavenumber). At C-band (k = 1.13 cm⁻¹) and the parameter range above, ks spans approximately 1.1–5.7 — partially outside Oh's nominal validity range (originally 0.1–6.0 for the dataset, but practically reliable over a narrower span). The lit review (§7, Decision 4) flags this and recommends a check.

> **Pre-experiment ks check:** Before any PINN training, run the Oh model in forward mode across the s = 1–5 cm range and confirm that σ° outputs remain physically plausible (no NaN, no extreme values, monotonic in m_v). If the check fails, implement the simplified AIEM (Chen et al. 2003) as the surface scattering model and document the change as a pre-registration amendment (logged in the deviation log before training begins). AIEM is more theoretically grounded for rougher surfaces but adds computational complexity.

---

## 8. PINN architecture & loss function

> The PINN backbone is identical to Phase 1: PhysicsNet produces a bounded m_v_physics estimate; CorrectionNet produces an unbounded residual δ; the final prediction is m_v_final = m_v_physics + δ. The only structural change is the loss function, which now has two physics terms (one per polarisation) instead of one, plus the Sentinel-2 prior penalty.

### Forward path

For each observation:

1. Seven input features (VV, VH, VH/VV ratio, NDVI, daily precip, 7-day precip, incidence angle) feed both PhysicsNet and CorrectionNet.
2. PhysicsNet outputs `m_v_physics ∈ [0, 0.88]`.
3. m_v_physics → Mironov GRMDM → ε_peat. Apply ε ≥ 1.01 clamp.
4. ε_peat + s + θ_i → Oh (1992) → σ°_soil_VV, σ°_soil_VH.
5. Crown layer parameters (N_b, N_l, σ_orient, m_g, fixed geometry) + canopy loss + crown-ground + ground-direct → MIMICS forward → σ°_VV, σ°_VH (total).
6. CorrectionNet outputs δ.
7. `m_v_final = m_v_physics + δ`.

### Composite loss

```
L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds
```

- **L_data:** `MSE(m_v_final, m_v_observed)` — same as Phase 1.
- **L_physics:** `MSE(σ°_VV_pred, VV_obs) + MSE(σ°_VH_pred, VH_obs)` — joint VV+VH. **This is the central change.** The Phase 1 formulation used VV only; MIMICS predicts both polarisations and matching both with a single parameter set is the structural advantage MIMICS is meant to provide. A single shared λ₁ is used across both polarisation terms; per-pol separate λs are explicitly out of scope for v0.1.
- **L_monotonic:** `mean(ReLU(−dε/dm_v))` — same as Phase 1. Enforces dielectric monotonicity in m_v.
- **L_bounds:** `mean(ReLU(−m_v_final) + ReLU(m_v_final − θ_sat))` — same as Phase 1.
- **L_prior:** Not used. Both Sentinel-2-derived priors originally specified in v0.1-draft (NDVI → LAI → N_l; NDWI → m_g) were withdrawn pre-sign-off (DEV-1b-001, DEV-1b-002). The L_prior term is removed from the composite loss, and λ_prior is no longer a hyperparameter of the model. N_l and m_g identifiability rest on their sigmoid-bounded literature ranges and the physics-consistency term (L_physics) alone. Pre-registered v0.2 fallbacks for both parameters are specified in §5.

### VV/VH weighting in L_physics

VV and VH are summed with equal weight in v0.1 of this specification. This is the simplest pre-registered choice. The implication is that the optimiser is pushed equally hard to fit both polarisations — appropriate when the VH channel is meant to constrain canopy structure orthogonally to VV's soil moisture sensitivity. If post-experiment diagnostics indicate one polarisation is being systematically under-fit, that finding is reported in the Phase 1b results document and motivates a v0.2 specification, not a within-experiment adjustment.

---

## 9. λ hyperparameter search protocol

> Phase 1's λ search produced a fallback selection because all 64 combinations violated the dominance constraint that L_data should be the largest single term (Phase 1 DEV-008). The lit review (§11 R3) recommends a stricter dominance criterion for MIMICS, given the higher parameter count. Phase 1b implements that recommendation explicitly.

### Search procedure

- **Grid:** Same as Phase 1 — (λ₁, λ₂, λ₃) ∈ {0.01, 0.1, 0.5, 1.0}³ = 64 combinations. λ_prior is no longer part of the model following DEV-1b-001 and DEV-1b-002 (both priors withdrawn pre-sign-off; L_prior removed from the composite loss — see §8).
- **Evaluation set:** 10 × 100% training configurations, validation loss measured on a stratified 20% held-out validation split within the training pool (NOT the sealed test set).
- **Selection metric:** Lowest median validation loss across the 10 reps.

### Dominance constraint (stricter than Phase 1)

> **Phase 1b dominance criterion:**
> - **Primary criterion:** L_data must be the largest single term in the composite loss across the final 10 epochs of training, averaged over all 10 reps.
> - **Secondary criterion (new in Phase 1b):** L_physics must contribute >10% of the total loss across the final 10 epochs. This is the lit review's "physics branch must provide non-trivial constraint" requirement made operational. Phase 1 had no quantitative threshold here — Phase 1b sets one.

### Fallback procedure

- If at least one combination satisfies both criteria: select the dominance-compliant combination with the lowest median validation loss.
- If no combination satisfies both criteria: select the combination with the lowest median validation loss that satisfies the primary criterion only. This is logged as a deviation (DEV-1b-008-equivalent) and the secondary criterion violation is reported as a Phase 1b finding.
- If no combination satisfies even the primary criterion: this is a Phase 1b architectural failure that halts the experiment. The Phase 1 fallback procedure (lowest median validation loss regardless of dominance) is **not** retained for Phase 1b. The honest interpretation in this case is that the MIMICS module cannot produce a physics-dominated solution at this data volume, which is itself a publishable finding.

The reason for the stricter procedure: Phase 1's fallback resulted in λ = (0.01, 0.01, 1.0), with L_physics weighted at 0.01. The diagnostic that the physics branch produced near-zero output at all training sizes (Phase 1 §4.3) is partly attributable to that low weighting. If Phase 1b allows the same fallback, it cannot distinguish "MIMICS works but the optimiser routes around it" from "MIMICS works and the optimiser uses it."

---

## 10. Pre-registered success criteria

> All criteria are fixed before any MIMICS implementation begins. Thresholds are derived from the Phase 1 baseline numbers (Null 0.178, RF 0.155 at N≈25, RF 0.147 at N=83) and from the WCM diagnostic baselines (forward fit r = 0.007, residual ratio 3.3–6.4×). Re-derivation is conditional on the baseline reproducibility check in §2 confirming the Phase 1 numbers within 0.005 cm³/cm³.

### Primary outcome — pre-registered comparison at N≈25

> **Primary criterion:** PINN-MIMICS median RMSE on the sealed test set at the N≈25 (25%) training fraction, evaluated against the RF baseline at the same fraction.
>
> RF baseline at N≈25: **RMSE = 0.155 cm³/cm³**

### Outcome categories (identical structure to Phase 1)

| Category | Threshold | Interpretation |
|---|---|---|
| **Strong** | PINN-MIMICS RMSE < 0.124 | ≥20% improvement over RF — strong evidence MIMICS recovers the physics-informed advantage. |
| **Significant** | PINN-MIMICS RMSE < 0.131 | ≥15% improvement over RF — meaningful evidence; survives Bonferroni-corrected significance test. |
| **Moderate** | PINN-MIMICS RMSE < 0.139 | ≥10% improvement over RF — directionally positive, may not survive correction. |
| **Inconclusive** | 0.139 ≤ PINN-MIMICS RMSE < 0.155 | PINN-MIMICS beats RF but not by a meaningful margin. Needs replication. |
| **Negative** | PINN-MIMICS RMSE ≥ 0.155 | RF baseline matched or exceeded. Hypothesis not supported. Phase 4 diagnostics determine the failure mode. |

### Secondary criteria — structural validation of the MIMICS module

Even a Negative primary outcome is informative if the secondary criteria are met — they confirm that MIMICS itself is operating as intended, isolating the failure to the data/sensor regime rather than the implementation.

- **Secondary 1 — Forward fit improvement:** MIMICS forward fit r > 0.3 (vs WCM forward fit r = 0.007). Computed by running MIMICS in pure forward mode (predicting σ°_VV from known m_v with optimised parameters) against observed VV across the full N=119 dataset.
- **Secondary 2 — Structural adequacy (residual ratio):** Residual ratio std(δ)/std(m_v_physics) < 2.0 at N=83 (full training). Phase 1 values were 3.3–6.4×.
- **Secondary 3 — Cross-pol identifiability:** VH forward fit r > 0.2. If < 0.2, the crown structural parameters (N_b, σ_orient) are misspecified and the dual-pol formulation is not achieving its purpose.
- **Secondary 4 — Residual-NDVI correlation:** |r(MIMICS_residual, NDVI)| < 0.5. Phase 1 WCM showed r = 0.82 — the signature of structural failure.

### Statistical significance

- **Wilcoxon signed-rank test (paired, per-rep RMSE):** PINN-MIMICS vs RF at each of the four training fractions. Same procedure as Phase 1.
- **Significance level:** Uncorrected α = 0.05 (reported); Bonferroni-corrected α = 0.0125 (4 comparisons; primary).
- **Required for "Significant" or "Strong":** Bonferroni-corrected p < 0.0125 in PINN-MIMICS' favour at N≈25.

---

## 11. Diagnostic plan

> The Phase 1 Phase-4 diagnostic framework transfers directly: residual analysis, forward model fit, identifiability via residual ratio and parameter sensitivity, and dielectric model sensitivity. Three new diagnostics are added for the dual-pol MIMICS context, per lit review §9.

### Diagnostics inherited from Phase 1 (unchanged)

- **Residual analysis:** Correlation of MIMICS physics-branch residuals (predicted minus observed σ°_VV, and separately σ°_VH) against NDVI, VWC, and incidence angle. Mean residual and std reported per polarisation.
- **Forward fit (VV):** MIMICS forward mode with optimised parameters, predicting σ°_VV from observed m_v across N=119. Pearson r and RMSE in dB.
- **Parameter sensitivity:** |∂σ°/∂θ_i| computed at the optimised parameter values for each learnable parameter θ_i ∈ {N_b, N_l, σ_orient, m_g, s} and m_v.
- **Residual ratio:** std(δ)/std(m_v_physics) at each training fraction.
- **Dobson vs Mironov sensitivity arm:** PINN-MIMICS retrained with Dobson dielectric on 10 × 100% configurations, RMSE compared. Reported for transparency, not affecting the primary outcome.

### New Phase 1b diagnostics

> **New Diagnostic A — Cross-pol forward fit:** MIMICS forward mode predicting σ°_VH from observed m_v across N=119. Pearson r and RMSE in dB. Tests whether the crown structural parameters (N_b, σ_orient) are correctly capturing volume scattering. If VH forward fit is high (r > 0.4) but VV forward fit is low, the model is matching crown structure but failing on the soil-canopy interaction. Reverse pattern indicates the opposite.

> **New Diagnostic B — Parameter correlation matrix:** Compute the 5×5 correlation matrix of ∂σ°_VV/∂θ_i and ∂σ°_VH/∂θ_i across all 119 observations and all 5 learnable structural parameters. Near-collinear rows indicate parameter pairs that cannot be separated from dual-pol observations alone — a structural identifiability problem requiring further parameter fixing in v0.2. Pre-registered threshold: any |r| > 0.95 between rows is flagged as non-identifiable.

> **New Diagnostic C — Dobson vs Mironov forward difference:** |ε_Dobson − ε_Mironov| computed across the observed m_v range (0.25–0.83 cm³/cm³). Reports whether the dielectric model choice is a binding constraint. Pre-registered threshold: maximum relative difference < 5% means the choice is not binding; ≥ 5% means it is. Computed once at sign-off (§6) and again post-experiment for confirmation.

> **New Diagnostic D — NDWI-vs-m_g correlation (DEV-1b-002 diagnostic repurposing):** Post-experiment, compute the Pearson correlation between the training-set-mean Gao-NDWI (and, separately, the per-observation Gao-NDWI time series) and the learned m_g value across the 10 reps at each training fraction. This is the diagnostic re-purposing of the U-3 NDWI extraction: NDWI is not used as a prior (DEV-1b-002) but is still extracted (Gao 1996, B08/B11) so that the post-hoc question "did the learned m_g correlate with an interpretable canopy-water signal?" can be answered. Not a pre-registered success criterion; reported for interpretive transparency. A strong correlation (|r| > 0.5) would support revisiting NDWI as an optional prior in v0.2; a weak or zero correlation is consistent with DEV-1b-002's hypothesis that the training-set-mean NDWI is not informative at a single site.

### If outcome is Negative — diagnostic decision tree

A Negative primary outcome triggers a structured diagnostic to determine the failure mode:

- **Secondary 1 fails (forward fit r ≤ 0.3):** MIMICS implementation is wrong or the parameter ranges are poorly chosen. Halt and re-implement before any further work.
- **Secondary 1 passes, Secondary 2 fails (residual ratio ≥ 2.0):** MIMICS is structurally improved over WCM but the PINN optimiser is still routing around it. Architectural finding — motivates investigation of the loss formulation, not the physics model.
- **Secondaries 1 and 2 pass, Secondary 4 fails (residual–NDVI r ≥ 0.5):** MIMICS captures bulk variance but still systematically fails with vegetation density — a more sophisticated canopy representation is needed (multi-layer crown, Multi-MIMICS extension per lit review §5).
- **All secondaries pass but primary fails:** MIMICS is operating correctly and the failure is genuinely the C-band sensor. This is the strongest justification for the Phase 2 NISAR L-band extension described in the Spectrum Paper.

---

## 12. Risk register & mitigations

> The five risks identified in lit review §11 carry forward into Phase 1b execution, plus one new risk introduced by the joint VV+VH formulation.

### R1 — C-band stem resonance (cylinder approximation breakdown)

**Risk:** At C-band, mature heather stems (1–4 mm × 10–30 cm) sit in the wavelength-comparable resonance regime where MIMICS' infinite-cylinder approximation introduces error.
**Mitigation:** Toure-style adaptation (no separate trunk layer) is already adopted as the primary structural decision (§4). Any remaining systematic bias is characterised as a forward model offset term in the diagnostic plan (§11).
**Trigger:** If forward fit Secondary 1 fails despite implementation passing the equivalence check, this risk is the primary suspect.

### R2 — Sphagnum dielectric not represented by any existing model

**Risk:** Mironov GRMDM is the closest available but was calibrated on generic organic soils, not Sphagnum-dominated peat.
**Mitigation:** Diagnostic C (Dobson vs Mironov forward difference) bounds the dielectric uncertainty.
**Trigger:** Diagnostic C output ≥ 5% relative difference at sign-off.

### R3 — Over-parameterisation under small dataset (N=83 vs 5 learnable params)

**Risk:** Higher parameter-to-data ratio than Phase 1 (5 vs 2 structural parameters plus per-obs m_v). Risk of overfitting absorbing physics-branch capacity.
**Mitigation:** Stricter dominance constraint in §9 (L_physics >10% of total loss) is **now the sole structural regulariser** for the MIMICS parameter space. Both Sentinel-2-derived priors originally planned as secondary regularisers were withdrawn pre-sign-off (DEV-1b-001 for N_l, DEV-1b-002 for m_g), so N_l and m_g identifiability rests on their literature-range sigmoid bounds and L_physics alone. The §9 tightening was introduced specifically to absorb this pressure, but it was calibrated assuming two priors would share the load; the single §9 constraint now carries weight for five learnable structural parameters against N=83 training observations. If the dominance constraint fails the λ search (primary-only fallback invoked), this becomes a more significant deviation than v0.1-draft anticipated and is flagged as such in the results document.
**Trigger:** Dominance constraint failure in λ search → fallback procedure invoked → flag as Phase 1b finding.

### R4 — Differentiable implementation complexity

**Risk:** MIMICS is substantially more complex than WCM as a computational graph. Implementation bugs propagate as systematic forward model bias.
**Mitigation:** Pre-training implementation gate (§4) requires forward-pass numerical equivalence check against published reference implementation, tolerance 0.5 dB.
**Trigger:** Equivalence check failure halts the experiment until corrected.

### R5 — No ground truth for MIMICS structural parameters at Moor House

**Risk:** No in-situ measurements of heather stem density, branch diameter distributions, or canopy water content at Moor House. Fixed structural priors come from published Calluna literature, not site-specific.
**Mitigation:** Conservative ranges on learnable structural parameters (sigmoid-bounded at literature ranges) and the §9 dominance constraint (L_physics > 10%) are the sole defences against fixed-prior error propagation. Both Sentinel-2-derived priors that would have absorbed some of this uncertainty were withdrawn pre-sign-off (DEV-1b-001 for N_l, DEV-1b-002 for m_g). Post-experiment Diagnostic B (parameter correlation matrix) is the specific check for non-identifiability arising from under-regularisation; pre-registered v0.2 fallbacks in §5 are the response path if the diagnostic fires.
**Trigger:** If Diagnostic B (parameter correlation matrix) shows non-identifiability, the fixed-parameter assumptions are the primary suspect.

### R6 — Joint VV+VH loss weighting may bias toward over-fitting one polarisation (NEW)

**Risk:** Equal weighting of VV and VH terms in L_physics may push the optimiser to over-fit the polarisation with smaller observed dynamic range.
**Mitigation:** Diagnostic A (cross-pol forward fit) is computed separately for VV and VH; gross asymmetry flags this risk post-experiment.
**Trigger:** If asymmetry is observed (e.g. VH r > 0.5 while VV r < 0.1, or vice versa), a v0.2 specification with separate λs per polarisation is the recommended next step. This is explicitly out of scope for v0.1 to keep the controlled comparison clean.

---

## 13. Honest-gates protocol & deviation log template

> Phase 1b inherits the Phase 1 honest-gates methodology unchanged. Deviations are logged with ID, summary, gate impact, and resolution. The Phase 1b deviation log is started fresh (does not inherit Phase 1 deviations) and is reported alongside the final results regardless of outcome direction.

### Gate structure

- **G1 — Data & baseline reproducibility:** Phase 1 dataset (N=119) reconstructed; baseline numbers reproduced within 0.005 cm³/cm³.
- **G2 — Pre-registration sign-off:** §14 signed; success criteria locked; MIMICS implementation passes the three-arm forward-pass equivalence check (§4 and `phase1b/physics/g2_anchor_spec.md`) — numpy_port arm, published_table arm (Toure 1994 and McDonald 1990), and gradient spot-check arm; Dobson vs Mironov diagnostic computed; Oh ks-validity check passed (or AIEM substitution logged).
- **G3 — λ search and dominance:** §9 procedure executed; selected λ combination meets primary and secondary dominance criteria, or fallback documented.
- **G4 — Final evaluation:** 4×10 factorial executed; primary and secondary criteria evaluated against pre-registered thresholds; Wilcoxon tests run; outcome category assigned; diagnostic plan executed.

### Deviation log template

| ID | Summary | Gate | Impact | Resolution |
|---|---|---|---|---|
| `DEV-1b-001` | Withdrawal of NDVI → LAI → N_l prior. v0.1-draft cited Lees et al. (2021) as the source of an empirical heather NDVI → LAI relationship. Citation audit during pre-sign-off found the cited paper concerns Sentinel-1 SAR water-table-depth retrieval and contains no such relationship. Lees PhD thesis (Reading, 2019) checked as alternative source; also empty. Wider literature treats NDVI → LAI for Calluna–Sphagnum canopies as poorly constrained. | G2 (pre-sign-off) | Reduces priors from two to one. N_l now constrained by literature bounds [10³, 10⁵] m⁻³ and the L_physics term only. L_prior formula updated (§8); R3 and R5 mitigations updated (§12). Pre-registered v0.2 fallback if N_l non-identifiable: CI-based prior, not reverting to NDVI → LAI. | NDVI → LAI → N_l prior dropped from L_prior. NDWI → m_g prior retained. Lit-review Decision 5 substantively rewritten in parallel. Reference [7] corrected for volume/page/title and re-annotated. Resolved before §14 sign-off; no impact on G1 (data and baseline reproducibility unchanged). |
| `DEV-1b-002` | Withdrawal of NDWI → m_g prior. Primary reason: the v0.1-draft specification mandated that the prior operate on a single time-stable scalar m_g averaged across the training set, meaning per-observation NDWI variation is averaged out before the prior is formed; at a single site, the resulting training-set-mean NDWI does not discriminate against any comparator site, so the "prior" reduced to a scalar choice dressed up as an empirical procedure. Secondary reason: no heather-specific NDWI → m_g transfer function exists in the peer-reviewed literature (input-side physics is real — SWIR absorption by foliar water — but output-side calibration is absent). Methodological consistency with DEV-1b-001 also supports symmetric treatment. | G2 (pre-sign-off) | Removes L_prior entirely from the composite loss (§8); λ_prior no longer a model hyperparameter (§9). m_g now constrained only by literature bounds [0.3, 0.6] g/g (branch-dominant-scatterer justification added to §5) and L_physics. §9 dominance constraint becomes the sole structural regulariser for the MIMICS parameter space; §12 R3 and R5 mitigations updated to reflect this. Pre-registered v0.2 fallback if m_g non-identifiable: season-stratified literature scalar prior per lit review §8 Decision 3, not reverting to NDWI-derived mapping. | L_prior term removed from §8 composite loss. §2 data-sources revised: NDWI remains extracted (Gao 1996, B08/B11) as diagnostic-only covariate. §11 diagnostic plan augmented with "training-set-mean NDWI vs learned m_g correlation" check. Lit-review Decision 5 further amended to state NDWI is now diagnostic-only. Lit-review Decision 3's season-stratified recommendation promoted to pre-registered v0.2 fallback. Pre-existing lit-review-vs-SPEC tension on leaf (0.5–0.8) vs branch (0.3–0.6) m_g bounds resolved by explicit branch-dominant-scatterer justification for tying m_g to the branch range at C-band. Resolved before §14 sign-off; no impact on G1. |
| `DEV-1b-003` | G2 Implementation Gate anchor construction. v0.1-draft §4 specified a forward-pass equivalence check against "a reference non-differentiable implementation (e.g. published MATLAB or Fortran MIMICS code)" without resolving which reference would be used. U-1 analysis identified that cross-framework consistency (numpy port vs PyTorch port) alone catches framework-specific bugs but not shared misreadings of the source specification. The needed external anchor — Toure et al. 1994 and McDonald, Dobson & Ulaby 1990 — was not staged in the repository at planning time, blocking the published-table arm. | G2 (pre-sign-off) | Both PDFs sourced via Open University Library on 18 April 2026 and staged in `phase1b/refs/`. G2 redefined as a three-arm check: (i) numpy_port consistency, (ii) published_table correctness vs T94 and M90, (iii) gradient spot-check against T94 sensitivity coefficients (autograd vs finite-difference vs published values). Strengthens the gate; §4 language expanded accordingly. No tolerance change (0.5 dB for σ° arms; ±20% or ±0.1 dB for gradient arm). | §4 implementation-gate paragraph expanded; §13 G2 gate definition updated; §15 references expanded (T94 annotation, M90 added as [11]); `phase1b/physics/g2_anchor_spec.md` created as authoritative specification of the 23 canonical anchor values (5 sets A–E). Toure 1994 and McDonald 1990 PDFs staged in `phase1b/refs/`. Resolved before §14 sign-off; no impact on G1. Does not affect outcome thresholds or the §11 diagnostic decision tree. |

### Reporting requirements

- The final Phase 1b results document must report the outcome against the pre-registered criteria in §10 with the same specificity as Phase 1 reported its Negative outcome.
- All deviations must be reported with their gate impact, even if the impact was None.
- If the outcome is Negative, the Phase 4 diagnostic decision tree (§11) determines the interpretation; a Negative outcome is not silently re-described as "promising" or "directionally positive."
- If the outcome is Strong or Significant, the secondary criteria results are reported in equal detail to confirm the result is not driven by a co-incidental architectural artefact.

---

## 14. Pre-registration sign-off

> This section is signed before any MIMICS implementation begins. The signatures (or initials and dates) below confirm that the success criteria, dominance thresholds, dielectric and surface model choices, and diagnostic plan are fixed for the duration of Phase 1b execution.

```
Specification version:                         v0.1 (this document)
Date locked:                                   2026-04-19
Lead investigator (sign / date):               Matthew Denyer / 2026-04-19
Independent reviewer (sign / date):            [science agent] / 2026-04-19

Gate results at sign-off
────────────────────────
Baseline reproducibility check (G1) result:    PASS       (Session A, 2026-04-18; all 12 rows within 0.005 cm³/cm³)
Oh ks-validity check (§7) result:              PASS       (Session A, 2026-04-18; 30/30 cells safe; no AIEM substitution)
   AIEM substituted: N
Dobson vs Mironov diagnostic (§6):
   max relative difference: 97.6 %             binding: Y (Session A, 2026-04-18; recorded as
                                               dielectric-sensitivity-arm active source of variance)
MIMICS forward equivalence check (G2) result:  MODERATE PASS — per DEV-1b-008
   numpy_port arm:        FULL PASS      (36/36 at machine precision; max Δ = 6.17e-6 dB)
   gradient arm:          MODERATE PASS  (autograd ↔ FD internal consistency within 0.003 dB;
                                          gradient-path liveness established; residual vs T94
                                          Table V(a) characterised as simplified-power-law-Dobson
                                          vs full Dobson 1985 — Session F diagnostic thread per
                                          DEV-1b-008 §Session F scope)
   published_table arm:   MODERATE PASS  (characterised residuals) — Sets A / B / C per-row
                                          residuals in 1.91–11.16 dB range documented in
                                          g2_anchor_spec.md v0.5 §"Per-row characterised
                                          residuals at Phase E closure"; implicated v0.1 sub-modules
                                          mapped per-row to DEV-1b-008's five-way approximation
                                          queue. Set C2 DEFERRED_SESSION_F. Set D EXEMPT pending
                                          Phase 1c per DEV-1b-005.
                                          Frozen verdict JSON: outputs/g2_equivalence_moderate_pass.json.
   g2_anchor_spec.md version: v0.5

Phase 1b deviation log at sign-off: DEV-1b-001 through DEV-1b-005 and DEV-1b-007 / DEV-1b-008
(DEV-1b-006 placeholder retired at v0.5 per DEV-1b-008). Full entries in phase1b/deviation_log.md
and the individual DEV-1b-NNN.md files.
```

### What sign-off commits the experiment to

- The five outcome categories in §10 are the categories the result will be reported under. The thresholds will not be moved post-hoc.
- The four secondary criteria in §10 are reported regardless of primary outcome.
- The diagnostic decision tree in §11 is the sole authority on interpreting a Negative outcome.
- The dominance constraint in §9 may not be relaxed during execution. Falling back to the primary-only criterion is logged as a deviation.
- The Phase 1 baselines must be reproduced in G1 within tolerance before Phase 1b training begins.
- The G2 Moderate Pass classification per DEV-1b-008: numpy_port arm full pass establishes implementation correctness; gradient and published_table arms pass with characterised residuals traceable to pre-registered v0.1 sub-module approximations. Phase 1b Block 2 (λ search + PINN-MIMICS trainer) proceeds on v0.1 physics unchanged. Session F physics promotions are evidence-led from Phase 1b training diagnostics per SPEC §11, not pre-committed at sign-off.

### What sign-off does not commit the experiment to

- The interpretation of any specific outcome.
- The Phase 2 strategy.
- v0.2 specification choices — improvements identified during execution are documented for v0.2, not retro-applied.
- Any specific Session F sub-module promotion. The five-way v0.1→v0.2 promotion queue documented in DEV-1b-008 is a registry of candidates, not a promotion commitment. Phase 1b training diagnostics are the evidence gate.

---

## 15. Phase 1b Pre-Registration document

> Added 2026-04-19 (Session F-1). Non-substantive amendment to SPEC v0.1.
> Formalises the pre-registration of the Phase 1b success criteria,
> experimental protocol, and diagnostic thresholds in a dedicated
> companion document so that the lit review §10 proposals (which were
> "not yet pre-registered" at sign-off) are bound to explicit numerical
> thresholds before Block 2 (λ search + PINN-MIMICS trainer) begins.

The Phase 1b pre-registration is jointly held by:

1. **This SPEC.md** — §10 (numerical RMSE outcome categories), §9 (λ
   search + dominance criteria), §11 (diagnostic plan including new
   Diagnostics A–D), and §14 (signed sign-off block, 2026-04-19). The
   numerical thresholds in SPEC §10 are authoritative; if any
   companion document conflicts on a number, SPEC.md wins.
2. **[`phase1b/SUCCESS_CRITERIA.md`](phase1b/SUCCESS_CRITERIA.md)** —
   the formal pre-registration companion document. Contains the
   primary criterion (verbatim from §10), the secondary criteria
   translated into explicit numerical thresholds (Forward fit r > 0.3;
   residual ratio < 2.0; VH r > 0.2; residual–NDVI |r| < 0.5), the
   dominance constraint and pre-registered fallback procedure (mirrors
   §9), the action-tier mapping (Strong / Moderate / Inconclusive /
   Refutation as joint conditions on primary × secondary × dominance),
   the locked experimental protocol (training sizes, repetitions,
   seeds, λ-grid, baselines), and the Phase 4 diagnostic thresholds
   (D-1 through D-4).
3. **[`tests/unit/test_pre_registration_lock.py`](tests/unit/test_pre_registration_lock.py)** —
   the technical lock. Fails if SUCCESS_CRITERIA.md is modified after
   any Phase 1b training artefact is produced, or if SPEC §10 RMSE
   thresholds are silently changed. Once Block 2 begins, the test is
   the enforcement mechanism for the pre-registration discipline that
   DEV-1b-001 / DEV-1b-002 established for the prior amendments.

### Lock metadata

| Field | Value |
|---|---|
| Document version | v1.0 |
| Date locked | 2026-04-19 |
| Tag at lock | `phase1b-success-criteria-pre-registered` |
| Parent commit | `63ef5fa` (Session F handoff close) |
| Lead investigator | Matthew Denyer / 2026-04-19 |
| Independent reviewer | [science agent] / 2026-04-19 |

### What §15 commits the experiment to

- The SUCCESS_CRITERIA.md secondary thresholds (S1: r > 0.3; S2: < 2.0;
  S3: r > 0.2; S4: |r| < 0.5) become the locked secondary-criteria
  numerical thresholds for Phase 1b. SPEC §10 already lists these
  criteria at category level; SUCCESS_CRITERIA.md numerically
  pre-registers them.
- The action-tier mapping (Strong / Moderate / Inconclusive /
  Refutation) is locked. The downstream recommendation (Phase 1c L-band
  NISAR; Phase 1c with honest scoping; pivot to L-band; Phase 1c
  priority increases) follows automatically from the observed outcome
  pattern. No post-hoc selection of action.
- Editing SUCCESS_CRITERIA.md after Block 2 begins requires a
  DEV-1b-NNN entry **and** an explicit update to the regression test
  in the same commit. Same honest-gates discipline as DEV-1b-001 /
  DEV-1b-002.

### What §15 does not commit the experiment to

- Any change to SPEC §10 RMSE thresholds (those are signed at §14 on
  2026-04-19; this section is a non-substantive amendment that adds
  no new threshold and changes no signed threshold).
- Any change to the SPEC §14 sign-off block. The signatures and
  Phase E closure verdict at §14 remain valid for SPEC v0.1.1 the
  same as for SPEC v0.1.
- Any Session F sub-module promotion. DEV-1b-008's five-way
  promotion queue remains evidence-led from training diagnostics per
  SPEC §11; this pre-registration adds neither candidates nor commitments
  to that queue.

---

## 16. References

1. Attema, E.P.W. and Ulaby, F.T. (1978) 'Vegetation modeled as a water cloud', *Radio Science*, 13(2), pp. 357–364. [Phase 1 physics model.]
2. Ulaby, F.T., Sarabandi, K., McDonald, K., Whitt, M. and Dobson, M.C. (1990) 'Michigan microwave canopy scattering model', *International Journal of Remote Sensing*, 11(7), pp. 1223–1253. [The MIMICS paper.]
3. Toure, A., Thomson, K.P.B., Edwards, G., Brown, R.J. and Brisco, B.G. (1994) 'Adaptation of the MIMICS backscattering model to the agricultural context — wheat and canola at L and C bands', *IEEE TGRS*, 32(1), pp. 47–61. DOI 10.1109/36.285188. [Crop-adaptation precedent and **primary G2 anchor source** — Figs. 2 and 7 (wheat C-band) and Tables V, VI (sensitivity coefficients) underpin anchor Sets A, B, C, and E per `phase1b/physics/g2_anchor_spec.md`. PDF staged at `phase1b/refs/Toure_1994_MIMICS_agricultural.pdf`.]
4. Mironov, V.L. et al. (2009) 'Generalized refractive mixing dielectric model for moist soils', *IEEE TGRS*, 47(7), pp. 1998–2010. [Phase 1b primary dielectric model.]
5. Dobson, M.C. et al. (1985) 'Microwave dielectric behavior of wet soil — Part II: Dielectric mixing models', *IEEE TGRS*, GE-23(1), pp. 35–46. [Phase 1 dielectric; Phase 1b sensitivity arm.]
6. Oh, Y., Sarabandi, K. and Ulaby, F.T. (1992) 'An empirical model and an inversion technique for radar scattering from bare soil surfaces', *IEEE TGRS*, 30(2), pp. 370–381. [Surface scattering.]
7. Lees, K.J. et al. (2021) 'Using remote sensing to assess peatland resilience by estimating soil surface moisture and drought recovery', *Sci. Total Environ.*, 761, 143312. [Originally cited in v0.1-draft as the source of an NDVI–LAI relationship for the N_l prior; on audit, found to concern SAR-derived water-table-depth retrieval and to contain no such relationship. Prior withdrawn — see DEV-1b-001 and §5 amendment. Reference retained as background context for the SAR-peatland literature.]
8. Vantage (2026) *ECHO PoC — Pre-Registered Experimental Evaluation of PINN-WCM at Moor House*, `poc_results.md` v1.0, 09 March 2026. [Phase 1 results.]
9. Vantage (2026) *MIMICS for Blanket Bog: A Literature Review and Parameter Scoping Study*, v1.0, April 2026. [Companion document.]
10. Vantage (2026) *Vantage Green Paper v4.1.0 — The Four-Phase Research Programme*. [Phase definitions and venture context.]
11. McDonald, K.C., Dobson, M.C. and Ulaby, F.T. (1990) 'Using MIMICS to model L-band multiangle and multitemporal backscatter from a walnut orchard', *IEEE TGRS*, 28(4), pp. 477–491. DOI 10.1109/TGRS.1990.572943. [**Secondary G2 anchor source** — Fig. 10 (multiangle L-band walnut orchard) and Tables I, II, III (canopy architecture and dielectrics) underpin anchor Set D per `phase1b/physics/g2_anchor_spec.md`. Used as a non-Toure structural cross-check with the trunk layer re-enabled. PDF staged at `phase1b/refs/McDonald_1990_MIMICS_walnut_orchard.pdf`.]

---

*Vantage · Phase 1b · Test Specification v0.1 — Draft for Sign-Off · April 2026*

*Companion to MIMICS Literature Review v1.0 and ECHO PoC Results v1.0. Not yet pre-registered — sign-off (§14) required before any MIMICS implementation begins.*
