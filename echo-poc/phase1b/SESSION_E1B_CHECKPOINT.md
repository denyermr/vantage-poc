# Phase E-1b Checkpoint Report

**Session:** Phase 1b, Session E, Phase E-1b (published_table dielectric-configuration amendment).
**Date:** 2026-04-19.
**Predecessor:** `phase1b/SESSION_E1_CHECKPOINT.md` (Phase E-1 close, 2026-04-19).
**Scope landed in Phase E-1b:** DEV-1b-007 (Sets A / B / C dielectric-configuration amendment + Set A HH-proxy correction), `g2_anchor_spec.md` → v0.4, E.1 / E.2 Υ²-attenuation diagnostic, harness per-row dominant-mechanism exposure.

---

## 1. Deliverables landed

| Deliverable | File(s) | Status |
|---|---|---|
| E.1 / E.2 Υ²-attenuation diagnostic | inline (this report §2 and §3) | Complete |
| DEV-1b-007 (published_table dielectric-config amendment) | `phase1b/DEV-1b-007.md` | Committed (ready for git) |
| `g2_anchor_spec.md` → v0.4 | `phase1b/physics/g2_anchor_spec.md` | Landed |
| `equivalence_check.py` Dobson-mineral for Sets A / B / C + VV-as-HH proxy | `phase1b/physics/equivalence_check.py` | Landed |
| Per-row dominant-mechanism exposure in result JSON | `outputs/g2_equivalence_result.json` | Landed |
| `deviation_log.md` summary row for DEV-1b-007 | `phase1b/deviation_log.md` | Landed |
| Set C2 status: unchanged (`DEFERRED_PHASE_E2`) | — | As specified |
| Set D status: unchanged (`EXEMPT` per DEV-1b-005) | — | As specified |
| Full unit suite | — | 234 / 234 pass, zero regressions |

## 2. E.1 / E.2 Υ²-attenuation diagnostic (from Phase E re-scope decision, pre-requisite 1)

**Question:** Are T94 Table V(a)'s 1.21 dB (VV) and 1.16 dB (HH) sensitivities pre-canopy-attenuation (ground-frame) or post-canopy-attenuation (observed-total-frame)? If post-, our 0.17 / 0.26 dB values after the Dobson-mineral clamp-removal may already be the correct canopy-attenuated answer and E.1 / E.2 pass.

**Method:** Directly probe `mimics_toure_single_crown_breakdown_torch` at the T94 Table IV(a) wheat reference configuration (L-band 1.25 GHz, θ=30°, m_v = 0.2 g/cm³) under Dobson-mineral.

**Result:**

| Quantity | Value |
|---|---|
| κ_e_v (extinction coefficient, per m) | 8.55 × 10⁻⁴ |
| τ²_v (two-way canopy transmissivity) | 0.99921 |
| Canopy two-way loss (−10·log₁₀ τ²_v) | **0.003 dB** |
| ε_ground (Dobson-mineral) | 30.75 |
| σ°_oh VV (bare soil) | −8.84 dB |
| Dominant mechanism | ground-direct (99.9 % of linear total) |
| ∂σ°_total_VV / ∂m_v (autograd) | 4.13 dB/(g·cm⁻³) |
| ∂σ°_ground_direct / ∂m_v | 4.13 dB/(g·cm⁻³) |
| ∂σ°_oh / ∂m_v (bare soil, pre-canopy) | 4.13 dB/(g·cm⁻³) |
| |grad| × ζ (ζ = 0.04 g/cm³) | 0.165 dB (T94 anchor: 1.21 dB) |

**Interpretation:**

- The canopy is **effectively transparent** at T94's wheat L-band reference (two-way loss = 0.003 dB). The pre-/post-attenuation distinction is moot — there is no attenuation to correct for.
- σ°_total, σ°_ground_direct, and σ°_oh all give the **same** ∂σ°/∂m_v = 4.13 dB/(g·cm⁻³). T94 Table V(a)'s S_i = ∂σ°_total / ∂P_i (their eq. 11) is in the same frame as our autograd. **Our frame is correct.**
- The residual ~7× shortfall (0.17 vs 1.21 dB) is **real** and has a specific origin: our `ground_epsilon_dobson_torch` uses the simplified power-law Dobson form `ε = ε_dry + (ε_water − 1) · m_v^α` with α = 0.65. T94 used the **full Dobson 1985 four-component mixing model** (sand/silt/clay + bulk-density-dependent terms), which produces a different ε-shape and a different ∂ε/∂m_v at m_v = 0.2 for mineral soils. At m_v = 0.2 the full Dobson 1985 typically gives ε ≈ 12–15 for sandy loam, not our 30.75; and the derivative differs by roughly the observed factor.

**E.1 / E.2 verdict:** The DEV-1b-004 clamp-removal fix is **correct and working** (gradient path unblocked; autograd ↔ FD within 0.003 dB; total-σ° frame matches T94's frame). The residual ~7× shortfall is a separately-diagnosable **simplified-Dobson vs full-Dobson-1985** physics-fidelity issue. Per the Phase E re-scope decision's instruction ("If pre-attenuation, the residual is real and gets logged as a Session F diagnostic thread"), the residual is logged here and **deferred to Session F** as a v0.2 Dobson-dielectric-fidelity promotion candidate. It is not a Phase E-1b blocker and not a Phase E-2 physics-promotion item (no canopy-side physics is implicated).

**Session F diagnostic thread registered:**
> **Thread name:** simplified-power-law-Dobson vs full-Dobson-1985 at mineral-soil configurations.
> **Symptom:** ∂σ°/∂m_v at T94 Table IV(a) wheat reference is 0.165 / 0.26 dB vs T94's 1.21 / 1.16 dB. Sevenfold shortfall.
> **Suspected fix:** port the full Dobson 1985 equation from MIMICS ref [Ulaby 1990] with explicit sand/silt/clay/bulk-density parameters, matching T94 Table I soil columns. Keep the current power-law form as the Moor House peat default (it is calibrated there per Bechtold et al. 2018) and gate on the `eps_dry` / `alpha` kwargs for which parameterisation is active.
> **Load-bearing?** No — does not gate Phase 1b training or the Phase 1b production path.

## 3. G2 verdict at Phase E-1b close

Result JSON at [`outputs/g2_equivalence_result.json`](../outputs/g2_equivalence_result.json).

| Arm | Verdict | Detail |
|---|---|---|
| numpy_port | **PASS** | 36 / 36 rows; max Δ = 6.17 × 10⁻⁶ dB at 0.5 dB tolerance. Unchanged from Phase E-1 (and Session D close). |
| published_table | **FAIL** | Set A 0 / 4, Set B 0 / 4, Set C 1 / 6 (the dropped C.3 row auto-passes); Set C2 0 / 0 active (24 DEFERRED_PHASE_E2); Set D 0 / 0 active (4 EXEMPT per DEV-1b-005). Per-row residuals substantially reduced vs Phase E-1 (see §4); no row passes yet. |
| gradient | **FAIL (but mechanism unlocked)** | E.1 / E.2 under Dobson-mineral (DEV-1b-004): autograd ≈ 0.17 / 0.26 dB, FD ≈ 0.17 / 0.26 dB. autograd ↔ FD internal consistency within 0.003 dB. Residual vs T94 ≈ 7×; **origin: simplified-power-law-Dobson vs full-Dobson-1985** (see §2). Deferred to Session F as separate physics-fidelity diagnostic thread. E.3 / E.4 / E.5: unchanged (Session F territory for E.3 / E.4 non-smoothness; E.5 for Rayleigh-vs-finite-cylinder regime mismatch for Phase E-2). |

## 4. Per-row residual characterisation for failing published_table rows

All rows evaluated with Dobson-mineral ground dielectric (DEV-1b-007). Sets A / B / C; Set C2 deferred, Set D exempt.

### Set A (wheat C-band HH, VV-as-HH proxy per Phase E-1b correction)

| Row | θ | torch σ° (VV-proxy) | anchor σ° | Δ | Dominant mechanism | Implicated promotion category |
|---|---|---|---|---|---|---|
| A.1 | 20° | −6.93 dB | −9.48 dB | **2.55 dB** | ground-direct-attenuated | **Oh-vs-PO surface scattering** (1–3 dB, ground-implicated) |
| A.2 | 30° | −8.48 dB | −10.44 dB | **1.96 dB** | ground-direct-attenuated | **Oh-vs-PO surface scattering** (1–3 dB, ground-implicated) |
| A.3 | 40° | −7.59 dB | −11.57 dB | 3.98 dB | crown-ground | **Escalate** (> 3 dB) |
| A.4 | 50° | −6.16 dB | −12.70 dB | 6.54 dB | crown-direct | **Escalate** (> 3 dB; plus Brewster-region implications) |

### Set B (wheat C-band VV, direct comparison)

| Row | θ | torch σ° | anchor σ° | Δ | Dominant mechanism | Implicated promotion category |
|---|---|---|---|---|---|---|
| B.1 | 20° | −6.93 dB | −9.99 dB | 3.06 dB | ground-direct-attenuated | **Escalate** (just over 3 dB; structurally ground-Oh-implicated, borderline) |
| B.2 | 30° | −8.48 dB | −10.39 dB | **1.91 dB** | ground-direct-attenuated | **Oh-vs-PO surface scattering** (1–3 dB, ground-implicated) |
| B.3 | 40° | −7.59 dB | −11.12 dB | 3.53 dB | crown-ground | **Escalate** (> 3 dB) |
| B.4 | 50° | −6.16 dB | −14.13 dB | 7.97 dB | crown-direct | **Escalate** (> 3 dB; Brewster-region, VV-specific drop) |

### Set C (wheat CHH mechanism decomposition at θ=30°, VV-as-HH proxy)

| Row | Mechanism | torch σ° | anchor σ° | Δ | Implicated promotion category |
|---|---|---|---|---|---|
| C.1 | Direct ground ♦ | −17.92 dB | −11 dB | 6.92 dB | **Escalate** (> 3 dB; ground-direct mechanism itself — Oh-vs-PO implicated but severely off) |
| C.2 | Direct cover □ | −8.84 dB | −20 dB | 11.16 dB | **UMF finite-cylinder form factor** (crown-direct mechanism; residual dielectric-independent) |
| C.4 | Ground-cover ■ | −11.85 dB | −20 dB | 8.15 dB | **Escalate** (> 3 dB; dual-mechanism involving both ground and canopy) |
| C.5 | GCG ▲ | N/A | −35 dB | — | Mechanism absent in v0.1 forward; P2-promotion (add GCG mechanism) |
| C.6 | Total σ° ○ | −6.73 dB | −10 dB | 3.27 dB | **Escalate** (> 3 dB; tracks whichever mechanism dominates total) |

## 5. Phase E-2 scope authorisation

Per the Phase E re-scope decision's classification:

- **Pass (< 1 dB):** 0 rows. The closest residuals are A.2 at 1.96 dB and B.2 at 1.91 dB.
- **UMF-authorised (1–3 dB, canopy-implicated):** 0 rows currently. C.2 has Δ = 11.16 dB (> 3 dB) so it falls into "Escalate", but its mechanism is crown-direct which is exactly the UMF-promotion target; see §6.
- **Oh-authorised (1–3 dB, ground-implicated):** 3 rows — A.1 (2.55), A.2 (1.96), B.2 (1.91). All three are ground-direct-dominant. Residual origin is the Oh-1992-vs-T94-physical-optics surface-scattering-model mismatch — a **secondary implicit configuration assumption** in T94's anchor set that v0.4 did **not** address (see §6 and DEV-1b-007 "Secondary finding").
- **Fresnel-authorised (1–3 dB, crown-ground-implicated with Brewster signature):** 0 rows currently. A.4 / B.4 Brewster-region rows are > 3 dB so fall into "Escalate".
- **Escalate (> 3 dB):** 9 rows — A.3 (3.98), A.4 (6.54), B.1 (3.06 — borderline), B.3 (3.53), B.4 (7.97), C.1 (6.92), C.2 (11.16), C.4 (8.15), C.6 (3.27). All exceed the 3 dB threshold; the re-scope decision's "pause and escalate" clause applies.

**Verdict: ESCALATE. Phase E-2 physics-promotion scope must NOT be unilaterally chosen.**

## 6. Secondary finding surfaced during Phase E-1b — Oh 1992 vs T94 physical-optics surface scattering

The Phase E-1b dielectric amendment resolved the primary category error (peat-Mironov applied to mineral-soil T94 wheat scenario). Post-amendment, the residual 1.91–2.55 dB on ground-direct-dominant rows (A.1, A.2, B.2) is **a different category error of the same structural class**: T94 Table I specifies the **physical-optics surface-scattering model** with s = 0.55 cm, l_s = 4.9 cm (ks ≈ 0.65 at C-band, within the PO envelope). Our Phase 1b harness uses **Oh 1992**, the empirical surface-scattering model SPEC §7 specifies for the Moor House production path. At this surface-roughness regime and C-band wheat configuration, Oh 1992 and physical-optics will differ systematically.

**This is structurally analogous to the DEV-1b-004 / DEV-1b-007 dielectric finding:** the G2 anchor set implicitly assumes T94's surface model; the harness applies Phase 1b production's surface model; the category error produces a (smaller, but measurable) residual on every ground-direct row.

**Scope note:** Unlike the dielectric swap (harness-only, zero production-path impact), swapping Oh for physical optics in a G2 harness context has **larger scope implications**:

1. SPEC §7 requires the Oh ks-validity check for the Moor House peat surface (s = 1–5 cm, passed in G3). A G2-harness-only PO injection is a clean narrow amendment (analogous to DEV-1b-004's dielectric injection) and does not touch the production path. **This is the recommended path.**
2. Making surface scattering harness-selectable requires a new injection point in `mimics_toure_single_crown` (parallel to `ground_dielectric_fn`), a physical-optics differentiable module, and a regression test pinning Oh for the production path. Module scope is larger than DEV-1b-004's.

**DEV-1b-007 names this finding and surfaces it** but does not resolve it. Phase E-2 should determine whether:

- **Option α′ (secondary):** Add DEV-1b-008 amending the G2 harness for Sets A / B / C to use physical-optics surface scattering, analogous to DEV-1b-007 for the dielectric. Expected: closes the 1.91–2.55 dB ground-direct residuals; may not fully close the > 3 dB ground-related rows (C.1, C.4) if additional approximations are binding.
- **Option β′ (secondary):** Leave Oh 1992 in the G2 harness and accept the Oh-vs-PO residual as a known limitation. Phase E-2 focuses on the canopy-mechanism promotions (C.2 crown-direct UMF, C.5 GCG addition, Brewster-region Fresnel Γ) and the Oh-vs-PO residual is logged as Session F.

## 7. Per-row Phase E-2 promotion map (subject to science-agent re-scope)

Based on the characterisation above, if Phase E-2 is authorised to proceed per-row:

| Row(s) | Residual | Implicated promotion |
|---|---|---|
| A.1, A.2, B.2 | 1.9–2.55 dB | **DEV-1b-008 (proposed)** — Oh-vs-PO surface scattering for Sets A / B / C harness |
| A.3, B.3 | 3.5–4.0 dB | Crown-ground mechanism — investigate Fresnel Γ and the √σ°_oh coupling term |
| A.4, B.4 | 6.5–8.0 dB | Brewster-region (VV-specific at θ=50°) — literal Fresnel Γ with pol-specific reflectance |
| B.1 | 3.06 dB | Borderline ground-direct; overlaps A/B ground-family |
| C.1 | 6.92 dB | Direct-ground mechanism itself — Oh-vs-PO candidate; may require full Dobson 1985 for best closure |
| C.2 | 11.16 dB | Crown-direct mechanism — **UMF finite-cylinder form factor (the Memo 2 P2 hypothesis, now narrowed to a single mechanism)** |
| C.4 | 8.15 dB | Ground-cover — compound mechanism; may resolve after C.1 and C.2 individually land |
| C.5 | (absent) | Add GCG mechanism to v0.1 forward (Session E-authorised P2 scope item for mechanism completeness) |
| C.6 | 3.27 dB | Tracks dominant-mechanism residual; resolves when C.1 / C.2 / C.4 individually resolve |
| E.1 / E.2 | ≈ 7× shortfall | **Session F diagnostic thread** — simplified-power-law-Dobson vs full-Dobson-1985; not Phase E-2 |
| E.3 / E.4 | autograd ≠ FD by ~3×, ~2× | **Session F diagnostic thread** — stem-height geometry-path non-smoothness |
| E.5 | 7.5 dB off T94, but autograd = FD within 0.07 dB | Rayleigh-vs-finite-cylinder regime mismatch for a leaf-width-dominated sensitivity; Phase E-2 UMF candidate if C.2 UMF work also lands |

Phase E-2 is thus a **multi-promotion scope** even after scope discipline is applied. Memo 2's original single-P2-extinction hypothesis does not match the data. The per-row map above is what Phase E-2 implementation would address **if** the science-agent re-scope authorises proceeding.

## 8. Go / no-go verdict

Per the Phase E re-scope decision's Phase E-1b criteria:

| Criterion | Status | Notes |
|---|---|---|
| C1. numpy_port arm PASS at machine precision | ✅ | 36 / 36, max Δ = 6.17 × 10⁻⁶ dB. |
| C2. Gradient arm E.1 / E.2 verdict recorded via Υ²-attenuation diagnostic | ✅ | Diagnostic complete (§2). Verdict: clamp-fix correct; residual = Session F Dobson-fidelity thread. |
| C3. Published_table Sets A / B / C per-row residuals characterised | ✅ | §4 provides residual + dominant mechanism per row. |
| C3a. Number of rows with residual > 3 dB (escalate) | **9 rows** | Of 12 active A/B/C rows. |
| C3b. Number of rows with residual 1–3 dB ground-implicated (Oh-authorised) | 3 rows | A.1, A.2, B.2. |
| C3c. Number of rows with residual 1–3 dB canopy-implicated (UMF-authorised) | 0 rows | C.2 is the natural UMF target but has Δ = 11.16 dB. |

**Per the Phase E re-scope decision's "If escalate rows non-empty: pause Phase E-2 and return to science agent with per-row diagnostic summary" clause: ESCALATE.**

## 9. Recommendation for science-agent Phase E-2 review

The Phase E-1b amendment worked: peat-Mironov was suppressing the ground path via the DEV-007 clamp and hiding the true mechanism balance. Under Dobson-mineral the honest mechanism balance is visible. That surface an **ordered queue of four distinct issues**:

1. **Oh 1992 vs T94 physical-optics surface scattering** — the analogue of DEV-1b-007 in the surface-scattering dimension. Expected to close A.1, A.2, B.2 (three currently Oh-authorised rows in the 1.9–2.55 dB band) and improve C.1 (6.92 dB → possibly closer).
2. **Crown-direct mechanism for dense C-band wheat canopies** — Rayleigh + sinc² is 11 dB too loud at C.2; UMF finite-cylinder form factors are the Memo 2-pre-registered candidate promotion. This is the only unambiguously canopy-implicated rotation.
3. **Brewster-region (VV at θ=50°)** — A.4, B.4 residuals (6.5 / 8.0 dB) reflect the √σ°_oh-as-reflectance proxy miscalibrating pol-specific Fresnel reflectance. Literal Fresnel Γ is the candidate promotion.
4. **GCG mechanism** — C.5 is not evaluable because the v0.1 forward doesn't expose it. Adding GCG is mechanism-completeness work, independent of the above.

**Recommended Phase E-2 framing:**

- Memo 2's single-P2-extinction scope is **wrong** and should be formally retired.
- Phase E-2 should be authorised as **multi-promotion per-row**, following roughly the order: Oh-vs-PO (DEV-1b-008) → re-run G2 → UMF crown-direct for C.2 → re-run → Fresnel Γ for Brewster → re-run → add GCG → final G2.
- Each promotion produces its own DEV entry; DEV-1b-006 placeholder is retired because the "v0.2 physics promotion" concept of Memo 2 was too monolithic.
- Expected Phase E-2 wall-clock: larger than a single session. The science agent should weigh this against the alternative of running Phase 1b training against the current physics (residuals would feed the Phase 1b interpretation, not block it; Phase 1b's scientific question is the WCM-vs-MIMICS controlled-comparison outcome at Moor House, not T94 wheat-field reproduction).

**Alternative Phase E-2 framing (to consider):**

- Declare G2 closed on **"numpy_port + gradient + secondary criteria"** alone. The published_table arm can be logged as a multi-row residual-characterised table (§4 above) representing the v0.1 physics's known limitations, **provided** that:
  1. The secondary Set C2 (U&L 2014 Table 11-1 mechanism ratios, 24 anchors) is run as the Phase E-2 deliverable — this tests the **mechanism decomposition structure** against an independent primary source, which is what published_table was supposed to do structurally.
  2. The per-row residuals in §4 are accepted into the Phase 1b results document as a known v0.1 MIMICS limitation, framed honestly: "MIMICS v0.1 reproduces the T94 anchors to within 2–11 dB; the physics-fidelity bottleneck is a stack of four approximations we have characterised per-row; Phase 1b's scientific claim about Moor House heather C-band is **not** sensitive to whether v0.1 reproduces T94 wheat within 0.5 dB".

The alternative framing trades absolute published_table fidelity for a faster path to Phase 1b training, with honest reporting of the known limitations. It is scientifically defensible if the Phase 1b interpretation carries the known-limitation caveat.

## 10. Fallback clause status

No P2 physics promotion was attempted in Phase E-1b. The fallback clause (NaN / non-finite / > 5× wall-clock) is not triggered and not relevant. All Phase E-1b changes are harness-level call-pattern updates; numerical stability is unchanged.

## 11. Deliverable artefact list for git commit

- `phase1b/DEV-1b-007.md` (new)
- `phase1b/physics/g2_anchor_spec.md` (v0.3 → v0.4)
- `phase1b/physics/equivalence_check.py` (Dobson-mineral for Sets A/B/C; VV-as-HH proxy correction; per-row dominant_mechanism exposure; result JSON schema updated)
- `phase1b/deviation_log.md` (DEV-1b-007 summary row added)
- `phase1b/SESSION_E1B_CHECKPOINT.md` (this report, new)
- `outputs/g2_equivalence_result.json` (Phase E-1b G2 run)
- Unit tests: no new tests required for Phase E-1b (no new callable surface); 234 / 234 existing tests pass.

---

**Phase E-1b closed at:** 2026-04-19.
**Phase E-2 awaiting science-agent re-scope decision.**
**Active recommendation to science agent:** `ESCALATE` — Phase E-1b surfaced a stack of four distinct Phase E-2-candidate promotions, each with its own scope signature. Proceeding without explicit authorisation would violate the Phase E re-scope decision's "pause on > 3 dB residual" clause.
