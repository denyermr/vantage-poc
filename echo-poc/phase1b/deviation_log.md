# Phase 1b — Active Deviation Log

Append-only log of departures from [`SPEC.md`](../SPEC.md) during Phase 1b
execution. Entries follow the template in `SPEC.md` §13: ID, summary, gate
impact, resolution.

Phase 1 deviations (DEV-001 through DEV-008) live in the project-root
[`DEVIATIONS.md`](../../DEVIATIONS.md) and are **not** inherited here. This
log starts fresh for Phase 1b per `SPEC.md` §13: *"The Phase 1b deviation
log is started fresh (does not inherit Phase 1 deviations)"*.

---

## Entry template

```
### DEV-1b-NNN — YYYY-MM-DD

**Spec said:** [exact quote or section reference from SPEC.md]

**What was done:** [description of the actual implementation]

**Reason:** [why the deviation was necessary]

**Gate impact:** G1 / G2 / G3 / G4 — [none / reinterpreted / halts experiment]

**Resolution:** [description of action taken]

**Impact on pre-registered criteria:** [none / affects Gate N criterion X — explanation]
```

---

## Deviations

| ID | Date | Summary | Gate | Impact | Resolution |
|----|------|---------|------|--------|-----------|
| [`DEV-1b-001`](DEV-1b-001.md) | 2026-04-17 | Withdrawal of NDVI → LAI → N_l prior. v0.1-draft cited Lees et al. (2021) as the source of an empirical heather NDVI → LAI relationship; citation audit during pre-sign-off found the cited paper concerns Sentinel-1 SAR water-table-depth retrieval and contains no such relationship. Lees PhD thesis (Reading, 2019) checked as alternative; also empty. Wider literature treats NDVI → LAI for *Calluna–Sphagnum* canopies as poorly constrained. | G2 (pre-sign-off) | Reduced S2-derived priors from two to one *at the time of resolution*. N_l constrained by literature bounds [10³, 10⁵] m⁻³ and L_physics only. L_prior formula updated to a single m_g term; §12 R3 / R5 mitigations updated. Pre-registered v0.2 fallback on N_l non-identifiability is a CI-based prior, not reversion to NDVI → LAI. | NDVI → LAI → N_l prior dropped from L_prior. SPEC.md and lit review Decision 5 amended in parallel. Reference [7] corrected for volume / page / title. Resolved before §14 sign-off; no impact on G1. **Note:** the sibling deviation DEV-1b-002 (below) subsequently withdrew the retained NDWI → m_g prior as well; the final v0.1 state has no S2-derived priors in the composite loss. |
| [`DEV-1b-002`](DEV-1b-002.md) | 2026-04-18 | Withdrawal of NDWI → m_g prior. Primary reason: the v0.1-draft mandated that the prior operate on a single time-stable scalar m_g averaged across the training set, so per-observation NDWI variation is averaged out before the prior is formed; at a single site this reduces to a scalar choice dressed up as an empirical procedure. Secondary reason: no heather-specific NDWI → m_g transfer function exists in the peer-reviewed literature. Tertiary reason: methodological consistency with DEV-1b-001. | G2 (pre-sign-off) | Removes L_prior entirely from the composite loss (§8); λ_prior no longer a model hyperparameter; λ-search grid shrinks to `{0.01, 0.1, 0.5, 1.0}³` = 64 combinations only (§9). §9 dominance constraint becomes the sole structural regulariser; §12 R3 / R5 rewritten. m_g bound choice resolved via branch-dominant-scatterer justification (§5). Pre-registered v0.2 fallback on m_g non-identifiability is a season-stratified literature scalar prior per lit review §8 Decision 3, not reversion to any NDWI-derived mapping. | L_prior removed from §8 composite loss. NDWI remains extracted (Gao 1996, B08/B11) as diagnostic-only covariate. §11 augmented with Diagnostic D (post-experiment NDWI ↔ m_g correlation). Lit review Decision 3 promoted to v0.2 fallback; Decision 5 re-framed as diagnostic-only. Resolved before §14 sign-off; no impact on G1. |
| [`DEV-1b-003`](DEV-1b-003.md) | 2026-04-18 | G2 anchor-construction methodology. v0.1-draft §4 deferred the G2 reference source. Session D procured Toure 1994 and McDonald 1990 via Open University Library, staged at `phase1b/refs/`, and specified a three-arm G2 check: (i) numpy_port consistency, (ii) published_table vs T94 / M90, (iii) gradient spot-check vs T94 Table V(a). 22 anchor values refined from the PDFs in `anchor_reads_v1.json`. | G2 (pre-sign-off) | Strengthens G2 (adds published_table and gradient arms). No tolerance change. | Three-arm G2 harness at `phase1b/physics/equivalence_check.py`; `g2_anchor_spec.md` v0.2; first G2 run end-of-Session-D. |
| [`DEV-1b-004`](DEV-1b-004.md) | 2026-04-19 | G2 gradient-arm dielectric-configuration amendment for E.1 / E.2. First G2 run at end-of-Session-D returned `autograd = FD = 0 dB` vs T94 Table V(a) 1.21 / 1.16 dB. Diagnosis: the Moor House peat-Mironov configuration produces ε < 1.01 at T94 wheat reference m_v = 0.2 g/cm³, hitting the DEV-007 ε ≥ 1.01 clamp and nullifying the ∂ε/∂m_v path. The clamp is functioning as pre-registered. Finding: g2_anchor_spec.md v0.2 did not specify dielectric parameterisation for E.1 / E.2; T94 Table V(a) values are specific to T94's wheat-field Dobson configuration (T94 inherits Dobson 1985 from MIMICS [Ulaby 1990 ref 19], verified via text extraction 2026-04-19). | G2 (gradient spot-check arm). No impact on G1 / G3 / G4. | Amend `g2_anchor_spec.md` to v0.3 with an explicit dielectric-configuration block for E.1 / E.2 (T94 Dobson mineral-soil kwargs). Add `ground_epsilon_dobson_torch` (parameterised; Moor House peat defaults match frozen `phase1/physics/dielectric.DobsonDielectric`). Add `ground_dielectric_fn` kwarg to `mimics_toure_single_crown` and `mimics_toure_single_crown_breakdown_torch`; harness-only, production path uses `None` → Mironov default. Regression test in `tests/unit/test_mimics_torch.py::TestMoorHouseProductionPinning` pins the production call signature. Anchor values unchanged. Tolerance unchanged. Clamp unchanged. |
| [`DEV-1b-005`](DEV-1b-005.md) | 2026-04-19 | G2 Set D (M90 walnut orchard L-band) exemption pending Phase 1c trunk-layer build. First G2 run recorded all 4 Set D rows as status:UNIMPLEMENTED because `use_trunk_layer=True` raises `NotImplementedError` (Session D scope). Decision Memo 2 § "On P4" authorised exempting Set D rather than implementing the trunk-layer in Session E — the trunk-layer path (4 branch classes, cos⁶ zenith PDF, complex vegetation dielectric, L-band dual-dispersion UEL) is multi-session effort and is a Phase 1c critical-path prerequisite for NISAR L-band validation regardless of G2. | G2 (published_table arm). Set D rows return `status: EXEMPT` and do not count against arm pass/fail. No impact on other gates. | Amend `g2_anchor_spec.md` to v0.3 marking Set D as EXEMPT pending Phase 1c. Harness `_run_set_D` returns EXEMPT rather than UNIMPLEMENTED. Re-instatement criteria specified in DEV-1b-005 body (trunk-layer code path + complex vegetation dielectric + L-band canopy path + companion DEV-1c-00N entry). Anchor values (D.1 = −11.73, D.2 = −11.35, D.3 = −12.17, D.4 = −12.17 dB) retained verbatim. |
| [`DEV-1b-007`](DEV-1b-007.md) | 2026-04-19 | G2 published_table arm dielectric-configuration amendment for Sets A / B / C. Phase E-1 diagnostic probe via `mimics_toure_single_crown_breakdown_torch` (P3) revealed the Sets A / B / C failures are not canopy-extinction saturation (two-way extinction at Set A θ=30° is only 8.68 dB, far below the 30 dB criterion) but the same dielectric-configuration category error DEV-1b-004 addressed for the gradient arm: at T94 wheat m_v = 0.17 g/cm³ the peat-Mironov dielectric clamps to ε = 1.01 (DEV-007 floor), suppressing σ°_oh to −56 dB and crushing the ground-direct mechanism. T94 inherits Dobson 1985 mineral-soil from MIMICS [Ulaby 1990 ref 19] at §II.A (verified via text extraction 2026-04-19). | G2 (published_table arm). No impact on G1 / G3 / G4 / gradient arm. | Amend `g2_anchor_spec.md` to v0.4 with a shared dielectric-configuration block for Sets A / B / C: `ground_epsilon_dobson_torch(m_v, eps_dry=DOBSON_EPS_DRY_MINERAL=3.0, eps_water=80.0, alpha=DOBSON_ALPHA_MINERAL=0.65)`. Harness `_run_set_A_B` and `_run_set_C` pass the mineral-Dobson callable via `ground_dielectric_fn`. Companion proxy-choice correction: Set A HH proxy switched from VH (Session D default, cross-pol, systematically 5-8 dB below HH) to VV (co-pol, typically within ~1 dB of HH at C-band wheat); both `sigma_torch_vv_db` and `sigma_torch_vh_db` exposed in result JSON. Real HH as first-class channel is a Phase E-2 P5 deliverable. Anchor values unchanged; tolerances unchanged; DEV-007 clamp unchanged; Moor House production-path pinning unchanged (regression test passes). Phase E-1b re-run reduces peat-Mironov failure-mode dominant mechanism (mostly ground-direct-suppressed-by-clamp) to Dobson-mineral honest residual classification: Sets A/B residuals 1.9-8.0 dB with per-row implicated mechanisms fed to Phase E-2 per-row authorisation in `SESSION_E1B_CHECKPOINT.md`. Sibling to DEV-1b-004. |

Full entries: [`DEV-1b-001.md`](DEV-1b-001.md),
[`DEV-1b-002.md`](DEV-1b-002.md), [`DEV-1b-003.md`](DEV-1b-003.md),
[`DEV-1b-004.md`](DEV-1b-004.md), [`DEV-1b-005.md`](DEV-1b-005.md),
[`DEV-1b-007.md`](DEV-1b-007.md).
Companion document updates for
DEV-1b-002 (in addition to DEV-1b-001's): `SPEC.md` §2 (NDWI line
revised), §3 (Δ row rewritten), §5 (m_g row + prior subsection
rewritten; pre-registered v0.2 fallbacks for both parameters),
§8 (L_prior removed entirely), §9 (λ_prior removed from grid),
§11 (Diagnostic D added), §12 R3 and R5 (rewritten), §13 (row
added); `vantage-mimics-litreview-v1_0_2.html` §8 Decisions 3
(promoted to v0.2 fallback) and 5 (NDWI now diagnostic-only).
DEV-1b-004 and DEV-1b-005 amend `g2_anchor_spec.md` v0.2 → v0.3
(dielectric-configuration specificity for E.1 / E.2; Set D EXEMPT
pending Phase 1c); no SPEC.md amendments. DEV-1b-006 (forthcoming
in Phase E-2) will log the v0.2 physics promotion (UMF finite-cylinder
form factors, PyTorch mechanism decomposition already landed in
Phase E-1 per P3, HH channel exposure).

**Final v0.1 prior state:** No Sentinel-2-derived priors enter the
composite loss. `L_prior` is not a term; `λ_prior` is not a
hyperparameter. N_l and m_g identifiability rests on sigmoid-bounded
literature ranges and the `L_physics` term alone, with the §9
dominance constraint as the sole structural regulariser. Two v0.2
fallbacks are pre-registered in SPEC §5 and are the only permitted
substitutions if post-experiment diagnostics show non-identifiability.

---

*Log opened: 2026-04-17. Phase 1b repository re-layout (see
`ARCHITECTURE.md`) was performed as a prerequisite; it is engineering
housekeeping, not a scientific deviation, and is not logged here.*
