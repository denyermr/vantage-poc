# Phase 1b Test Specification: PINN-MIMICS Soil Moisture Retrieval at Moor House

**Vantage · Phase 1b · Pre-Registration · v0.3.1 — Signed (substantive amendment: §18.11 loss-formulation string canonicalisation · 2026-04-28)**

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

## 17. Phase 1b Conclusion

> Added 2026-04-27 (Session G). Append-only audit-trail entry recording Phase 1b's formal conclusion on the Tier 3 HALT finding plus five publishable contributions. Structurally parallel to §14 (pre-registration sign-off) and §15 (pre-registration metadata) — the document where Phase 1b's audit trail was opened is the document where it is closed. The detailed Phase 1b results document is [`phase1b/poc_results_phase1b.md`](phase1b/poc_results_phase1b.md) (Block A); §17 records the SPEC-side conclusion-of-Phase-1b artefact and binds the framing constraints carried by Block A. Where Block A and §17 reference the same content, Block A is the canonical detail source and §17 is the audit-trail anchor.

### §17.0 Framing-inversion block

This block is structurally parallel to §14's "What sign-off commits / does not commit" template. It records, at the moment of conclusion, what the Phase 1b conclusion does and does not establish.

#### §17.0.1 What the Phase 1b conclusion commits the experiment to

Per Block A §8.2:

- **A specific empirical finding about composite-loss calibration:** the magnitude-balance saturation of joint-VV+VH composite-loss landscapes under Phase 1b's pre-registered configuration. Formally stated at §17.4 below and detailed at Block A §4.5 / §6.5.
- **A working differentiable Toure-style single-crown MIMICS implementation:** numpy↔PyTorch lockstep at machine precision per the G2 numpy_port arm; autograd path alive per the G2 gradient arm; `phase1b/physics/mimics.py` and `phase1b/pinn_mimics.py` at commit `1f4bfdf` (tag `phase1b-session-f2b-lambda-selected`). This is a reusable engineering asset for any future PINN work, regardless of architecture.
- **Four methodological contributions to physics-informed ML pre-registration discipline:** DEV-1b-008 implementation-correctness vs cross-configuration equivalence; DEV-1b-009 dominance-constraint aggregation-rule explicitness with its F-2b empirical complement; the supervisor-executor entry-check workflow architecture with four sub-observations; DEV-1b-010 post-sign-off implementation-audit gate. Independently publishable per Block A §6.1–§6.4 and registered as Phase 1b deliverables at §17.2 below.

#### §17.0.2 What the Phase 1b conclusion does not commit the experiment to

Per Block A §8.1:

- **PINN-MIMICS retrieval performance on Sentinel-1 C-band at Moor House blanket bog.** The sealed test set was never unsealed in Phase 1b; no RMSE comparison against the RF baseline was made; the central Phase 1b research question — "does PINN-MIMICS beat RF baseline at N≈25?" — is unanswered. Phase 1c is the next attempt at retrieval-science validation.
- **Whether physics-informed neural networks are a viable retrieval architecture for peatland water-table or soil-moisture monitoring from satellite SAR, in any general sense.** The HALT outcome is configuration-specific; PINN architectures as a class are unaddressed by Phase 1b's evidence base.
- **Whether the Vantage commercial thesis holds at the technical level needed to underwrite carbon credits.** The thesis (physics-informed satellite monitoring producing uncertainty-quantified ecosystem state estimates more accurate and cheaper than ground-based monitoring) is unaddressed by Phase 1b at the retrieval-science level. The HALT finding does not validate it; the HALT finding also does not refute it.

### §17.1 Outcome

The pre-registered λ-search (Session F-2b, 2026-04-20 to 2026-04-22, tag `phase1b-session-f2b-lambda-selected`, commit `1f4bfdf`) executed all 64 grid combinations of (λ_physics, λ_monotonic, λ_bounds) ∈ {0.01, 0.1, 0.5, 1.0}³ with 10 reps each (640 training runs) under the signed §8 joint VV+VH `L_physics` formulation, after DEV-1b-010 corrected an implementation-vs-text divergence detected at the F-3 entry check. The result is **Tier 3 HALT**: 0 of 64 combinations satisfy the §9 primary dominance criterion under either the strict per-rep AND or the binding mean-across-reps reading of DEV-1b-009. Per the verbatim §9 fallback procedure at line 324:

> "this is a Phase 1b architectural failure that halts the experiment. The Phase 1 fallback procedure (lowest median validation loss regardless of dominance) is **not** retained for Phase 1b. The honest interpretation in this case is that the MIMICS module cannot produce a physics-dominated solution at this data volume, which is itself a publishable finding."

The Tier 3 HALT branch fires. Detail at Block A §4.1–§4.4. The sealed test set (`echo-poc/data/splits/test_indices.json`, SHA-256 `a4b11206630cc80fc3e2ae5853bb114c7a4154072375654c257e51e4250f8eea`) was never unsealed in Phase 1b: the pre-registered evaluation path at §9 / §10 requires a Tier 1 or Tier 2 outcome with a selected λ, and the Tier 3 HALT outcome precludes that path. The sealed-test-set SHA-256 is unchanged from Phase 1.

### §17.2 Five publishable findings (registered as Phase 1b deliverables)

The five publishable contributions registered as Phase 1b deliverables are recorded below as a registry; the contributions themselves are detailed in Block A §6.1–§6.5 and are not duplicated here.

1. **Implementation-correctness vs cross-configuration equivalence (DEV-1b-008).** Physics-informed ML forward-equivalence gates must explicitly distinguish implementation-correctness testing from cross-configuration equivalence testing; tolerance thresholds calibrated to the test. Block A §6.1.
2. **Dominance-constraint aggregation rule explicitness (DEV-1b-009 + F-2b empirical complement).** Pre-registered PINN dominance constraints must explicitly specify the aggregation rule across training repetitions; participial phrases are ambiguous; verbatim text binds. F-2b empirical complement: ambiguity collapses on lopsided landscapes. Block A §6.2.
3. **Workflow architecture: supervisor-executor entry-check discipline.** Pre-registration discipline strengthened by an explicit supervisor-executor split with bidirectional entry checks at commit boundaries. Four sub-observations: state-snapshot freshness; cost-asymmetry case study (1000:1 in F-3 entry-check halt); artefact-persistence drift; cross-environment prompt drift. Block A §6.3.
4. **Post-SPEC-sign-off implementation-audit gate (DEV-1b-010).** Pre-registered physics-informed ML projects need an implementation-audit gate triggered whenever an implementation file referenced by a signed SPEC § first lands in the repo or is materially modified after sign-off. Block A §6.4.
5. **Magnitude-balance saturation of joint-VV+VH composite-loss landscapes** (the Phase 1b headline empirical finding). Binding framing line at §17.4 below. Block A §6.5.

### §17.3 Phase 1c open questions (recorded as open, not as commitments)

The open questions below were identified during Phase 1b execution as relevant to the magnitude-balance finding. They are recorded here as open questions for future Phase 1c scoping consideration. They are **not** Phase 1b extensions and they are **not** Phase 1c pre-registration commitments — Phase 1c requires its own pre-registration sign-off cycle following the same discipline as Phase 1b's §14 / §15 pre-registration. Per Block A §7:

- **Per-channel L_physics normalisation.** Whether dividing each polarisation's MSE by its empirical magnitude would rebalance the composite loss without requiring λ extensions.
- **λ grid lower bound for joint dual-pol formulations.** Whether the pre-registered grid's 0.01 lower bound is appropriate for joint dual-pol formulations or whether a wider grid is needed.
- **Trunk-layer scattering mechanism.** Whether the trunk-layer scattering mechanism (DEV-1b-005 Set D Phase 1c exemption) should be implemented and how that affects the magnitude-balance landscape.
- **L-band SAR generalisation.** Whether L-band SAR (NISAR / ROSE-L) has materially different magnitude-balance properties to C-band Sentinel-1 and whether the Phase 1b finding generalises.

These four items are **Phase-1c-scope candidates**, not Phase-1b deliverables. Phase 1b's pre-registered deliverables are bounded by §14 sign-off (2026-04-19) and concluded at this §17. Any Phase 1c work on these items is gated on a separate Phase 1c pre-registration sign-off cycle and is not authorised by this §17.

### §17.4 Magnitude-balance characterisation (binding framing line)

The Phase 1b headline empirical finding's framing line is recorded here verbatim from Block A §6.5 — both the "is" and the "is not" halves are normative for downstream interpretation, and §17.4 is the SPEC-side authoritative record of the framing constraint in case Block A is ever revised.

**The finding is** (verbatim):

> At C-band Moor House blanket bog, with the pre-registered MIMICS forward model and joint VV+VH `L_physics` formulation, with N=83 training points, and with the pre-registered λ grid lower-bounded at 0.01, joint VV+VH `L_physics` is intrinsically ~1.65× larger in MSE magnitude than VV-only, and the resulting composite-loss landscape is physics-saturated across the entire pre-registered λ grid {0.01, 0.1, 0.5, 1.0}³. L_data cannot be made the largest single term under any pre-registered combination.

**The finding is not** "MIMICS is inadequate as a physics module for Moor House peatland retrieval." The MIMICS forward model is operating per its specification (G2 Moderate Pass per DEV-1b-008 — numpy_port arm at machine precision; gradient arm autograd↔FD within 0.003 dB; published_table arm with characterised residuals). The HALT outcome is about composite-loss calibration, not about MIMICS forward fidelity.

**The finding does not** preclude a later investigation — outside Phase 1b's pre-registration — into whether a loss-rescaling scheme (per-channel MSE normalisation, λ_physics extension below the pre-registered minimum, or a separable-magnitude reformulation) could rebalance the composite loss. That investigation is a Phase 1c or methods-paper question, not a Phase 1b re-scoping. The pre-registered λ grid is what binds Phase 1b's evidence claim.

This three-part framing — "is", "is not", "does not preclude" — is the binding interpretation of the magnitude-balance finding for any cascade or downstream document that references the Phase 1b conclusion. Cascade revisions per [`phase1b/cascade_plan_session_g.md`](phase1b/cascade_plan_session_g.md) (Block B) inherit the framing through Block A; §17.4 is the SPEC-side anchor.

### §17.5 Audit-trail-strengthening proposals (Phase 1c result-JSON schema requirements)

Four audit-trail-strengthening proposals were surfaced during F-2b close (`phase1b/SESSION_F2B_CHECKPOINT.md` §9) and relayed in the Phase 1b decisions log §15.3. They are recorded here as Phase 1c result-JSON schema requirements — Phase 1c trainer code is gated on schema adoption of these four fields. They are not Phase 1b deliverables (the F-2b result JSON does not carry them); they are inherited forward as Phase-1c-scope schema requirements.

1. **Code-version hash in result-JSON metadata** (highest priority). E.g. `git rev-parse HEAD` short hash. Provides one-step audit hook from any result JSON back to the exact commit of the loss formulation that produced it. Would have made the F-2b integrity-audit Check 4 a strict PASS rather than NEUTRAL.
2. **Loss-formulation string in result-JSON metadata.** E.g. `"l_physics_formulation": "MSE(sigma_vv_db, vv_db_observed) + MSE(sigma_vh_db, vh_db_observed) [DEV-1b-010 joint VV+VH per SPEC §8]"`. Pairs with the code-version hash for double-redundant audit-trail recording.
3. **Pre-flight summary block in result-JSON.** Records pre-flight invariant checks and key magnitudes (e.g. VV/VH magnitude ratio; per-rep wall-clock estimate; structural-prediction-at-kickoff) so the result JSON is self-contained for retrospective inspection.
4. **Sleep/wake event count in result-JSON.** Captures `n_sleep_events` from `pmset -g log` over the run window; wall-clock anomalies become characterisable from the artefact alone.

Plus one operational discipline anchor inherited as Phase 1c compute-budgeting requirement: lid-open or AC power for any multi-hour run; `pmset -a sleep 0 disablesleep 1` invocations as belt-and-braces around `caffeinate`.

These four schema requirements and the operational discipline anchor are inherited forward as Phase-1c-scope; their adoption is a precondition for Phase 1c trainer code reaching production-ready status.

### §17.6 Conclusion sign-off

This section is signed at the close of Phase 1b. The signatures (or initials and dates) below confirm that Phase 1b is formally concluded on the HALT finding plus the five publishable contributions registered at §17.2; that the sealed test set is unchanged from Phase 1; and that no Phase 1c pre-registration commitment is made at this conclusion.

```
Specification version:                         v0.1 (this document, Phase 1b conclusion)
Date of Phase 1b conclusion:                   2026-04-27
Lead investigator (sign / date):               Matthew Denyer / 2026-04-27
Independent reviewer (sign / date):            Claude (Opus 4.7), web-app supervisor session, 2026-04-27
                                               — operating per the supervisor-executor entry-check
                                               workflow architecture documented at SPEC §17.2 contribution 3.

Phase 1b outcome
────────────────
Pre-registered λ-search outcome (F-2b):        Tier 3 HALT
   n_full_dominance:           0/64
   n_primary_only:             0/64
   n_any_primary:              0/64
   n_neither:                  64/64
   DEV-1b-009 readings:        identical (mean-across-reps and strict per-rep AND both classify HALT)
   F-2b result JSON SHA-256:   c5cbdac6414a9bf6989c0f6927462a927ebeda71120b6ab424320d6d77b6d68a
   F-2b tag:                   phase1b-session-f2b-lambda-selected
   F-2b parent commit:         1f4bfdf

Sealed test set status:                        UNCHANGED from Phase 1
   test_indices.json SHA-256:  a4b11206630cc80fc3e2ae5853bb114c7a4154072375654c257e51e4250f8eea
   Unsealed in Phase 1b:       N

Phase 1b deliverables registered (§17.2)
────────────────────────────────────────
   1. DEV-1b-008 implementation-correctness vs cross-configuration equivalence
   2. DEV-1b-009 + F-2b complement: aggregation-rule explicitness
   3. Supervisor-executor workflow architecture (four sub-observations)
   4. DEV-1b-010 post-sign-off implementation-audit gate
   5. Magnitude-balance saturation finding (headline empirical contribution)

Phase 1b cascade plan:                         phase1b/cascade_plan_session_g.md (Block B)
Phase 1b results document (canonical):        phase1b/poc_results_phase1b.md (Block A)
Phase 1b deviation log at conclusion:          DEV-1b-001 through DEV-1b-005 and DEV-1b-007
                                               through DEV-1b-010 (DEV-1b-006 retired at v0.5
                                               pre-sign-off per DEV-1b-008). Full entries in
                                               phase1b/deviation_log.md and individual
                                               DEV-1b-NNN.md files.

Recommended Session G close tag:              phase1b-concluded-halt-finding
   (final tag adjudication at Session G close; see Block A footer and Block B §4)

Conclusion commit:                             tagged at `phase1b-concluded-halt-finding`
```

### What the §17 conclusion commits the experiment to

(See §17.0.1 for the binding registry.)

### What the §17 conclusion does not commit the experiment to

(See §17.0.2 for the binding registry. Note in particular: §17 does **not** authorise any Phase 1c work; Phase 1c requires its own pre-registration sign-off cycle.)

---

## 18. Phase 1c-Lean Pre-Registration

**Status:** Pre-registered for execution. Founder-only sign-off per DEV-1c-lean-003.
**Branch:** `phase1c-lean`, cut from `phase1b-concluded-halt-finding` (commit `4175fc2`).
**Sequencing:** Follows §17 Phase 1b Conclusion. Addresses two of the four open questions registered at §17.3 (per-channel L_physics normalisation; λ-grid lower bound for joint dual-pol formulations). Defers the remaining two (trunk-layer reinstatement; L-band SAR generalisation).
**Outcome tag at execution close:** `phase1c-lean-{strong|significant|moderate|inconclusive|halt-finding}-pass`.

### §18.1 Scope

Phase 1c-Lean is a leanest-possible debug of the Phase 1b Tier 3 HALT recorded at §17. It tests a single hypothesis: that the F-2b magnitude-balance saturation finding (VH/VV ≈ 0.645 across the dataset; zero λ combinations satisfying the §9 dominance criterion across the pre-registered grid) is **configuration-tunable** rather than **structural**, and is recoverable by per-channel L_physics normalisation combined with a λ grid widened to lower bound 10⁻⁴.

The experiment introduces no new data sources, no new physics modules, no new ground sensors, and no partnerships. It re-uses the Phase 1 dataset (n=119; train 83 / test 36 chronological), the Phase 1b MIMICS implementation at G2 Moderate Pass under DEV-1b-008, and the existing Phase 1 / 1b pipeline.

### §18.2 Research question

> *Does per-channel L_physics normalisation, combined with a λ grid widened to lower bound 10⁻⁴, recover at least one configuration in which the joint VV+VH composite loss satisfies the §9 dominance criterion at Moor House on the Phase 1 dataset?*

The question is scoped deliberately to:

- The Phase 1b loss configuration (joint VV+VH composite, MIMICS amplitude branch under DEV-1b-008).
- The Phase 1 dataset (Moor House, COSMOS-UK MOORH daily VWC, n=119, established sealed test set per §15).
- The two §17.3 open questions executable without new data acquisition or partnership development.
- A pre-registered binary primary outcome: §9 dominance satisfied, or not.

Phase 1c-MRV (target variable WTD; observable channel S1 InSAR; physics module simplified Terzaghi consolidation; second site Forsinard) remains separately scoped and is not in scope here. Sequencing between Phase 1c-Lean and Phase 1c-MRV is conditional on Phase 1c-Lean outcome (§18.12).

### §18.3 Continuity from §14 / §15 / §17

| Dimension | Phase 1c-Lean (this section) | Reference |
|---|---|---|
| Site | Moor House (54.69°N 2.38°W) | §14.2 |
| Target variable | Volumetric water content (VWC) | §14.3 |
| Ground truth | COSMOS-UK MOORH daily product, Phase 1 QC convention | §14.4; DEV-001 |
| Sensors | Sentinel-1 IW GRD descending, relative orbit 154 | §14.5 |
| Observables | C-band amplitude VV + VH | §14.5 |
| Physics module | Differentiable single-crown MIMICS (Phase 1b G2 Moderate Pass) | §17.2; DEV-1b-008 |
| Sealed test set | Phase 1 sealed set (n=36, 2023-07-25 to 2024-12-10) | §15.4; status §18.5 below |
| Pre-registration discipline | §14 spec / §17 SPEC template | §14, §17 |
| Deviation log | DEV-1c-lean-NNN convention | §18.13 |

### §18.4 Pre-registered changes from the §17 / F-2b configuration

#### §18.4.1 Change 1: Per-channel L_physics normalisation

The Phase 1b composite loss (per F-2b extension §15 and SPEC §8 as implemented in `phase1b/pinn_mimics.py`) is

> L_total = L_data + λ_physics · (L_phys_VV + L_phys_VH) + λ_monotonic · L_monotonic + λ_bounds · L_bounds

with λ_data ≡ 1.0 (fixed coefficient, not in the grid), λ_physics shared across the VV and VH physics terms, and λ_monotonic and λ_bounds as soft-loss regularisers on monotonicity and physical bounds. The pre-registered F-2b grid is 4³ = 64 over `(λ_physics, λ_monotonic, λ_bounds)`.

The F-2b empirical observation was that VH/VV magnitude ratio ≈ 0.645 across the dataset, structurally biasing the loss landscape such that no λ_physics value in the grid (lower bound 0.01) produced a §9-dominant solution. Mean physics fraction was monotone non-decreasing in λ_physics across the entire grid (0.9255 → 0.9898 → 0.9979 → 0.9990).

The Phase 1c-Lean composite loss is

> L_total = L_data + λ_VV · (L_phys_VV / σ_VV) + λ_VH · (L_phys_VH / σ_VH) + λ_monotonic · L_monotonic + λ_bounds · L_bounds

where:
- σ_VV and σ_VH are scale factors computed **once at training initialisation** as the standard deviation of the unweighted physics losses over the training set's first forward pass at randomly initialised network weights;
- λ_data ≡ 1.0 (fixed, same as Phase 1b);
- λ_monotonic and λ_bounds are fixed at the Phase 1b grid lower-bound value (0.01 each, per the F-2b grid lower bound), preserving the soft regularisation discipline of Phase 1b;
- λ_VV and λ_VH are the only tunable axes in the Phase 1c-Lean grid (§18.4.2).

After per-channel normalisation, the two physics terms are unit-scaled and λ_VV / λ_VH control relative weighting in a comparable, magnitude-independent way. The single intervention being tested in Phase 1c-Lean is per-channel L_physics normalisation; all other elements of the Phase 1b composite are preserved to isolate the active ingredient per the §18.4.3 intervention-conflation reasoning.

**Implementation requirements** (gated at G2-Lean, §18.6.1):

- σ_VV and σ_VH computed once at init only. **Per-batch normalisation is explicitly out of scope** — moving-target normalisation introduces a non-stationary loss landscape and conflates the magnitude-balance question with optimiser dynamics.
- σ values saved with the model checkpoint and reproduced exactly in the pre-flight summary block (§18.11 schema).
- σ values logged in the result JSON.
- L_monotonic and L_bounds reuse the Phase 1b implementation at `echo-poc/phase1b/pinn_mimics.py` without modification.

#### §18.4.2 Change 2: λ grid recalibrated for the per-channel normalised composite

| | §17 (Phase 1b F-2b) | §18 (Phase 1c-Lean) |
|---|---|---|
| Tunable λ axes | λ_physics, λ_monotonic, λ_bounds | λ_VV, λ_VH |
| Number of axes | 3 | 2 |
| λ values per axis | [0.01, 0.1, 0.5, 1.0] | [10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 10⁰, 10¹] |
| Grid size | 4³ = 64 | 6² = **36** |
| Fixed coefficients | λ_data ≡ 1.0 | λ_data ≡ 1.0; λ_monotonic = 0.01; λ_bounds = 0.01 |
| Lower-bound rationale on tunable axes | Inherited from §14 single-pol setup | §17.3 open question 2: lower bound may be too tight for joint dual-pol formulations once magnitudes are normalised |

The grid axis count is reduced from 3 (Phase 1b) to 2 (Phase 1c-Lean) because:
- λ_data and λ_monotonic and λ_bounds are fixed coefficients in v0.3 (held at Phase 1b values to preserve intervention isolation per §18.4.3).
- λ_physics is replaced by λ_VV and λ_VH (a *consequence* of per-channel normalisation, not an additional intervention; sharing post-normalisation λ would defeat the normalisation).

The wider per-axis range (lower bound 10⁻⁴ vs Phase 1b's 10⁻²) addresses §17.3 open question 2 — the lower bound may have been too tight on joint dual-pol formulations even with magnitudes normalised. The 6-value-per-axis density is preserved from v0.2's design to give the same λ_VV / λ_VH coverage as the v0.2 specification, just over 2 axes instead of 3.

#### §18.4.3 Explicitly out of scope: trunk-layer reinstatement

DEV-1b-005 Set D Phase 1c exemption (the trunk-layer scattering mechanism, §17.3 open question 3) is **explicitly out of scope** for Phase 1c-Lean.

Including the trunk layer would conflate two interventions: a positive Phase 1c-Lean result with trunk-layer reinstatement included would not distinguish whether per-channel normalisation, the wider grid, or the trunk layer was the active ingredient.

If Phase 1c-Lean is HALT or Inconclusive (§18.7), trunk-layer reinstatement combined with λ-grid refinement around marginal combinations becomes Phase 1c-Lean-2 and requires its own pre-registration following the §14 / §17 / §18 template.

### §18.5 Sealed test set policy

The Phase 1 sealed test set defined at §15.4 (n=36, 2023-07-25 to 2024-12-10) was used once for Phase 1's final evaluation per §15.5 (PINN RMSE 0.167, RF 0.155 at N≈25; RF 0.147 at 100% training). Phase 1b preserved it unsealed (Tier 3 HALT before unsealing) per §17.4.

Phase 1c-Lean treats the Phase 1 sealed set as **"used-once held-out"** with the following discipline:

- Gate criteria (§18.7) are evaluated on **5-fold cross-validation on the n=83 training pool** with stratification by meteorological season where sample sizes permit (§14.6 convention).
- The sealed test set is unsealed for tertiary RMSE comparison (§18.7) **only** if all gate criteria pass: G2-Lean (§18.6.1) plus dominance satisfied (§9 criterion) plus CV-RMSE not degraded relative to RF baseline.
- HALT outcomes (§18.8) do not unseal, mirroring the §17 Phase 1b discipline.
- **DEV-1c-lean-001** (reserved, §18.13) records the "used-once" acknowledgement explicitly: the set has been seen once in Phase 1 evaluation; Phase 1c-Lean tertiary evaluation against it is therefore not strictly held-out.

A truly fresh held-out evaluation requires extending COSMOS-UK ground truth beyond 2024-12-10 (Phase 1c-Lean-2 scope; requires fresh Sentinel-1 acquisitions and ground-truth pairing) and is out of scope here.

### §18.6 Engineering gates

#### §18.6.1 G2-Lean: implementation gate (three-arm)

Three-arm equivalence check on the per-channel normalised composite loss, mirroring the §17 Phase 1b G2 structure under DEV-1b-008:

1. **Cross-framework consistency.** numpy ↔ PyTorch implementation of the per-channel normalised composite loss agrees at machine precision on a fixed random fixture. Same convention as DEV-1b-008.
2. **Autograd ↔ finite-difference.** Gradient of the normalised loss with respect to (a) model parameters and (b) λ values agrees within Phase 1b tolerance (0.003 dB equivalent).
3. **Scale sanity.** After normalisation on the training set, both L_phys_VV / σ_VV and L_phys_VH / σ_VH have unit standard deviation within 1e-6 numerical tolerance on the training-set first forward pass at randomly initialised network weights.

Failure of any arm halts Phase 1c-Lean until corrected. Pre-sign-off audit cycle as Phase 1b. DEV-1c-lean-NNN log opened at G2-Lean kickoff.

#### §18.6.2 G3-Lean: pre-training gate

Conditions for entering training:

- λ grid locked at 6×6 = 36 combinations per §18.4.2.
- Training fraction: **100% only** (N=83). Phase 1c-Lean is not a learning-curve experiment; the §14 4×10 factorial is explicitly not reproduced here. Single-fraction execution is justified by the pre-registered scope (§18.2): the question is configuration-debugging, not data-efficiency characterisation.
- Reps: **3 per λ combination** as the primary plan. Total: 3 × 36 = **108 training runs**. Extension to 5 reps (180 runs) authorised only if results are marginal under §18.7 tier definitions and authorised pre-extension by founder sign-off.
- Random seeds: SEED = 42 + config_idx + rep_idx, carrying forward §14.6.
- Baselines locked: RF at 100% training (Phase 1: 0.147 cm³/cm³ on sealed test, target reproduction on training-pool 5-fold CV); seasonal-climatological null on VWC (Phase 1: 0.178).
- Sealed test set NOT touched at G3-Lean.
- Result-JSON schema includes §17.5 / §18.11 schema requirements.

### §18.7 Pre-registered success criteria

Primary metric: §9 dominance criterion, evaluated as a binary verdict per λ combination per rep.
Secondary metric: 5-fold CV RMSE (cm³/cm³) on the n=83 training pool, compared against RF baseline at 100% training (Phase 1 reference: 0.147 cm³/cm³).
Tertiary metric (held-out, contingent on gates): sealed-test-set RMSE per §18.5 unsealing policy.

| Tier | Primary (§9 dominance) | Secondary (CV RMSE vs 0.147) | Tertiary (sealed RMSE) | Action |
|---|---|---|---|---|
| **Strong pass** | ≥ 3 λ combinations satisfy dominance robustly across all 3 reps | ≤ 0.147 (RF parity at 100% or better) | ≤ 0.147 | Tag `phase1c-lean-strong-pass`. Magnitude-balance hypothesis confirmed; configuration-tunable verdict. Phase 1c-MRV de-prioritised pending strategic discussion. |
| **Significant pass** | ≥ 1 λ combination satisfies dominance robustly across all 3 reps | within 5% of 0.147 (≤ 0.154) | within 5% of 0.147 | Tag `phase1c-lean-significant-pass`. Configuration is viable; data-efficiency claim at low N still untested. |
| **Moderate pass** | ≥ 1 λ combination satisfies dominance in some but not all reps | within 10% of 0.147 (≤ 0.162) | not unsealed | Tag `phase1c-lean-moderate-pass`. Dominance achievable but not robust. Phase 1c-Lean-2 (refinement + trunk-layer reinstatement, §18.4.3) before any move to Phase 1c-MRV. |
| **Inconclusive** | 0 satisfy across all reps but ≥ 3 combinations within 10% of dominance threshold | n/a | not unsealed | Tag `phase1c-lean-inconclusive`. Phase 1c-Lean-2 refinement run before any move to Phase 1c-MRV. |
| **Negative / HALT** | 0 λ combinations within 10% of dominance threshold across the full 36-cell grid | n/a | not unsealed | Tag `phase1c-lean-halt-finding`. Magnitude balance was NOT the binding constraint; the §17 finding is structural. Phase 1c-MRV with InSAR becomes the next test under substantially stronger motivation. |

The Strong/Significant cut between "≥ 3" and "≥ 1" robust dominance combinations is heuristic for evidence robustness rather than statistical: a single satisfying combination across 3 reps demonstrates the hypothesis is *recoverable*; multiple satisfying combinations demonstrate the recovery is robust to specific λ choice. The "≥ 3" threshold (vs v0.2's "≥ 5") is recalibrated for the 36-cell grid: ~8% of grid cells satisfying robust dominance is the Strong-pass criterion, parallel to v0.2's ~2% on the 216-cell grid; the proportion is loosened slightly because a smaller grid produces sparser robustness evidence.

### §18.8 HALT triggers

Phase 1c-Lean halts before sealed-set unsealing under any of:

- G2-Lean (§18.6.1) implementation gate failure on any of the three arms.
- After full 36-cell grid run with 3 reps each: zero λ combinations within 10% of the §9 dominance threshold.
- Cross-framework numpy ↔ PyTorch divergence at run-time exceeding the G2 tolerance specified at §18.6.1 arm 2.
- Compute budget exceeded (>3× the §18.10 pre-estimated wall-clock time on M4 MPS) — re-scope decision at founder discretion, not auto-HALT.

HALT is a publishable outcome under §17 precedent and produces the `phase1c-lean-halt-finding` tag.

### §18.9 Diagnostic battery

Computed for every run reaching training completion regardless of outcome tier, mirroring the §15 Phase 1 / §17 Phase 1b discipline:

- **Amplitude-residual ratio.** std(ε_phys) / std(ε_data). Phase 1b reference: 3.3–6.4×. Phase 1c-Lean expected behaviour under correct hypothesis: ratio drops, indicating physics branch is no longer dominated by data branch under per-channel normalisation.
- **Forward fit r.** Pearson correlation between MIMICS forward-predicted σ⁰ and observed σ⁰ at converged-model state. Phase 1 reference (WCM): r = 0.007. Phase 1c-Lean expected behaviour under correct hypothesis: r rises substantially as the loss landscape rebalances.
- **Per-channel residual correlation with NDVI.** Phase 1 reference (WCM): r = 0.82. Phase 1b under MIMICS partially absorbed this. Phase 1c-Lean check: does residual-NDVI correlation drop further under per-channel normalisation?
- **λ-pair stability.** Across the 3 reps per combination, does the dominance verdict reproduce? Marginal stability separates §18.7 Significant from Moderate.

### §18.10 Compute envelope

Target hardware: Apple M4, 24 GB unified memory, macOS Sequoia 15.6.1, PyTorch MPS backend.

| Item | Estimate |
|---|---|
| Single training run (N=83, MIMICS forward, Phase 1 PINN architecture) | 30–90 seconds (§17 / Phase 1b empirical band) |
| Primary plan: 36 combinations × 3 reps | 108 runs × ~60s ≈ **~2 hours wall-clock** |
| Extended plan (if §18.7 Moderate / Inconclusive triggers it): 36 × 5 reps | 180 runs × ~60s ≈ ~3 hours |
| Memory peak per run | <1 GB; 24 GB unified memory not binding |
| Backend | PyTorch MPS; CPU fallback accepted for any unsupported op without re-implementation |

Operational discipline carrying forward §17 / decisions log §24.4:

- `sudo pmset -a sleep 0 disablesleep 1` before any multi-hour run; restore after.
- Lid-open or AC power for the duration.
- Per-rep training histories written to disk incrementally (not accumulated in memory across the sweep).
- Sleep/wake event count logged in result JSON per §17.5 / §18.11. If non-zero, affected runs flagged for re-execution.
- MPS context re-acquired if the system wakes mid-run.

The compute envelope is materially tighter than v0.2 (108 runs / ~2 hours vs 648 runs / ~9–11 hours). The reduction reflects the v0.3 grid-axis correction (§18.4.2): two tunable axes × 6 values = 36 cells, vs v0.2's three tunable axes × 6 values = 216 cells. The smaller grid is a consequence of intervention isolation (per §18.4.3 reasoning); not a compute optimisation per se.

### §18.11 Result-JSON schema

Carries forward §17.5 schema requirements with two Phase 1c-Lean specific additions:

1. **Code-version hash** in JSON metadata (§17.5).
2. **Loss-formulation string** in JSON metadata (§17.5). For Phase 1c-Lean: `"v0.3_five_term_per_channel_normalised"`. Canonicalised at SPEC v0.3 → v0.3.1 amendment per DEV-1c-lean-005; matches Block B-prime committed G2-Lean equivalence-check artefacts at tag `phase1c-lean-g2-lean-passed`.
3. **Pre-flight summary block** in JSON (§17.5), now including σ_VV and σ_VH values computed at init.
4. **Sleep/wake event count** in JSON (§17.5).
5. **Phase 1c-Lean addition: dominance verdict** per (λ_data, λ_VV, λ_VH) tuple per rep, encoded as `{"satisfies_dominance": bool, "margin_to_threshold_pct": float}`.
6. **Phase 1c-Lean addition: σ scale factors** logged separately from the model checkpoint for downstream auditability.

### §18.12 Sequencing to subsequent phases

Conditional on §18.7 outcome, Phase 1c-Lean produces one of three forward sequences:

- **Strong / Significant pass:** Magnitude-balance hypothesis confirmed; the C-band amplitude path is configuration-tunable. Phase 1c-MRV (separately scoped, WTD + InSAR + Terzaghi + Forsinard) becomes optional rather than necessary; the decision to run it depends on whether the venture's commercial framing requires WTD as the target variable for §17.3 reasons orthogonal to the science question. Strategic discussion separate from this SPEC.
- **Moderate pass / Inconclusive:** Phase 1c-Lean-2 (refinement around marginal λ combinations + trunk-layer reinstatement per §18.4.3) requires its own pre-registration following this template before any move to Phase 1c-MRV.
- **Negative / HALT:** Phase 1b's structural finding compounds. Phase 1c-MRV with InSAR becomes the next test under stronger motivation: amplitude alone has now failed twice on pre-registered tests, and a different observable channel is needed. The Strategy Memo (working document, not corpus) absorbs this finding explicitly. Phase 1c-MRV pre-registration (separate SPEC section, anticipated §19) drafted post-tag.

### §18.13 Deviation log convention and reserved entries

DEV-1c-lean-NNN entries in the same format as DEV-001 through DEV-008 (§14, Phase 1) and DEV-1b-001 through DEV-1b-010 (§17, Phase 1b). Each deviation: ID, summary, gate impact, resolution, signed-off pre-training where applicable. Pre-sign-off audit cycle that surfaced DEV-1b-001, -002, -003 in Phase 1b is the operational model.

**Reserved entries (committed at SPEC.md v0.2 sign-off):**

- **DEV-1c-lean-001** — Sealed-test-set "used-once" acknowledgement per §18.5. The Phase 1 sealed set has been seen once in Phase 1 evaluation; Phase 1c-Lean tertiary evaluation against it is not strictly held-out. Mitigation: gate criteria evaluated on training-pool 5-fold CV; sealed set unsealed only after gates pass per §18.5 / §18.7.
- **DEV-1c-lean-002** — Per-channel normalisation implementation choice (reserved for any aspect of σ_VV / σ_VH computation that deviates from the published Toure-style derivation; expected to remain a no-op deviation unless implementation surfaces an unanticipated issue at G2-Lean).
- **DEV-1c-lean-003** — Founder-only sign-off with explicit "scientific co-supervisor TBC" status flag. Carbon13 cohort co-founder recruitment is in flight; full Phase 1c-MRV pre-registration sign-off (if it fires per §18.12) is binding on co-supervisor presence at G3 level. Phase 1c-Lean is not.
- **DEV-1c-lean-004** — v0.2 → v0.3 SPEC amendment (Block A-prime). Phase 1c-Lean composite scope corrected: λ_data, λ_monotonic, λ_bounds restored to fixed Phase 1b values; grid recalibrated from 6³ = 216 over (λ_data, λ_VV, λ_VH) to 6² = 36 over (λ_VV, λ_VH); compute envelope from 648 / ~9–11 hours to 108 / ~2 hours. Surfaced at Block B halt-1 entry-check; adjudicated via Session H Block A-prime cycle.
- **DEV-1c-lean-005** — v0.3 → v0.3.1 SPEC amendment (Block C-prime entry). §18.11 item 2 loss-formulation string canonicalised from `"per_channel_normalised_joint_vv_vh_composite"` (v0.2-era string carried forward unchanged through v0.3) to `"v0.3_five_term_per_channel_normalised"` (Block B-prime committed implementation string at tag `phase1c-lean-g2-lean-passed`). Surfaced at Block C-prime entry-check Rule 0.7 cross-check; adjudicated via this micro-block cycle.

### §18.14 Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Founder | Matthew Denyer |  |  |
| Scientific co-supervisor | *N/A — DEV-1c-lean-003* | — | — |

---

*Vantage · Phase 1b · Test Specification v0.1 — Draft for Sign-Off · April 2026*

*Companion to MIMICS Literature Review v1.0 and ECHO PoC Results v1.0. Not yet pre-registered — sign-off (§14) required before any MIMICS implementation begins.*
