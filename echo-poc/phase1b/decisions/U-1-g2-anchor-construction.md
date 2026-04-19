# U-1 — G2 anchor construction

**Status:** Resolved pre-sign-off (Option E executed with all three arms)
**Decision date:** 2026-04-18 (original); executed 2026-04-19 (Session D)
**Authoritative deviation entry:** [`../DEV-1b-003.md`](../DEV-1b-003.md)
**Authoritative anchor specification:** [`../physics/g2_anchor_spec.md`](../physics/g2_anchor_spec.md) (v0.2)
**Original discussion:** [`../physics/reference_mimics/README.md`](../physics/reference_mimics/README.md) — options analysis
**Related decisions:** [`U-3-ndwi-formulation.md`](U-3-ndwi-formulation.md), [`U-4-ndwi-mg-mapping.md`](U-4-ndwi-mg-mapping.md)

---

## Decision

**Option E — published tables + self-authored numpy Toure port, both required to agree with the differentiable PyTorch implementation for the G2 gate to pass — executed with a third arm added per DEV-1b-003.**

The final G2 gate construction is a **three-arm equivalence check**:

1. **numpy_port arm** — PyTorch vs numpy Toure reference across the 36 canonical numpy_port entries in
   [`reference_mimics/canonical_combinations.json`](../physics/reference_mimics/canonical_combinations.json). Tolerance 0.5 dB.
2. **published_table arm** — PyTorch vs 17 anchor values refined from Toure 1994 (Sets A/B/C; 13 rows) and McDonald 1990 (Set D; 4 rows), cited to page / figure / table / row in
   [`g2_anchor_spec.md`](../physics/g2_anchor_spec.md) v0.2 and
   [`refs/anchor_reads/anchor_reads_v1.json`](../refs/anchor_reads/anchor_reads_v1.json). Tolerance 0.5 dB (Set C row C.5 widened to ±2.0 dB).
3. **gradient spot-check arm** — PyTorch autograd ∂σ°/∂P and finite-difference ∂σ°/∂P vs T94 Table V(a) published sensitivity coefficients (Set E; 5 rows). Tolerance ±20 % or ±0.1 dB (whichever larger) for PyTorch-vs-T94; ±5 % or ±0.02 dB (whichever larger) for autograd-vs-FD.

All three arms must pass for G2 to pass. Fallback is not "relax the anchor"; it is "halt and fix the implementation" (SPEC §9, §13 honest-gates protocol).

---

## Options considered (summary)

The full options-and-rationale analysis is preserved in
[`../physics/reference_mimics/README.md`](../physics/reference_mimics/README.md). Summarised here:

| Option | Source | Adopted? | Reason |
|---|---|---|---|
| A | [py-mimics](https://github.com/katjensen/py-mimics) port | No | Provenance too weak; self-described work-in-progress, last activity 2018. |
| B | Self-authored numpy Toure port | Yes (as arm 1) | Independent code path from PyTorch module; extends coverage to Moor House operating envelope. |
| C | Published σ° tables | Yes (as arm 2) | Peer-reviewed independent authority; strongest evidence but narrow in coverage. |
| D | Request Fortran/MATLAB from Dobson lab / JPL | No | Time-to-response uncontrolled; parked as nice-to-have. |
| **E** | **B + C together** | **Yes** | **Chosen. Strengthens to three arms with gradient spot-check added per DEV-1b-003.** |

---

## Execution record (Session D, 2026-04-19)

### Sources staged

Both source papers procured via Open University Library access (IEEE Xplore) on 2026-04-18 and staged:

- [`../refs/Toure_1994_MIMICS_agricultural.pdf`](../refs/Toure_1994_MIMICS_agricultural.pdf) — DOI 10.1109/36.285188
- [`../refs/McDonald_1990_MIMICS_walnut_orchard.pdf`](../refs/McDonald_1990_MIMICS_walnut_orchard.pdf) — DOI 10.1109/TGRS.1990.572925

(DEV-1b-003 cites McDonald's DOI as `10.1109/TGRS.1990.572943`; the PDF metadata has `10.1109/TGRS.1990.572925`. Same paper; metadata is authoritative. Recorded in
[`g2_anchor_spec.md`](../physics/g2_anchor_spec.md) v0.2 and
[`anchor_reads_v1.json`](../refs/anchor_reads/anchor_reads_v1.json).)

### Anchor refinement

22 anchor values refined from v0.1 scaffolding estimates to v0.2 published values. Method by set:

- **Sets A, B (T94 Fig. 2 CHH/CVV, 8 rows):** 400 dpi render, hollow-square template-scoring, y-axis label-centre calibration. Human spot-check at A.1, B.2, B.3, B.4 confirmed methodology.
- **Set C (T94 Fig. 7 CHH, 5 rows; C.3 dropped):** pixel detection fell back to human transcription (continuation-prompt Step 3 fallback clause) because the v0.1 spec's marker-to-mechanism mapping conflicted with the T94 Fig. 7 caption and the middle-cluster markers overlap at the same dB.
- **Set D (M90 Fig. 10, 4 rows):** line-continuity detection on MIMICS curves (solid VV, dotted HH, dashed HV — the spec v0.1 description of "open-circle markers" was materially incorrect; markers in this figure are the MEASURED values).
- **Set E (T94 Tables IV(a), V(a), 5 rows):** `pdftotext -layout` extraction; all 5 values matched the v0.1 spec exactly.

Full provenance (calibration, pixel coordinates, template scores, second-best candidates, spot-check responses) is in
[`../refs/anchor_reads/anchor_reads_v1.json`](../refs/anchor_reads/anchor_reads_v1.json) and the annotated PNGs in the same directory.

### Implementation

- [`../physics/mimics.py`](../physics/mimics.py) — added `use_trunk_layer: bool = False` keyword-only flag. Default `False` (production). `True` currently raises `NotImplementedError` pointing at Session E (trunk layer structural extensions for M90 walnut orchard).
- [`../physics/equivalence_check.py`](../physics/equivalence_check.py) — three-arm harness. Loads anchors from `anchor_reads_v1.json`; writes result JSON to `outputs/g2_equivalence_result.json`; exits non-zero on any arm failure.
- [`../../Makefile`](../../Makefile) — `g2` target retargeted from `phase1b/implementation_gate/equivalence_check.py` to `phase1b/physics/equivalence_check.py`.
- [`../../tests/unit/test_equivalence_check.py`](../../tests/unit/test_equivalence_check.py) — 21 unit tests covering anchor-reads schema, canonical-combinations invariants (numpy_port count still 36), delta logic, end-to-end JSON schema, exit code, and `use_trunk_layer` flag contract. Full suite 223 pass / 0 fail (up from 202).

### First-run verdict

See [`../../outputs/g2_equivalence_result.json`](../../outputs/g2_equivalence_result.json) and the Session D session log entry for narrative.

**Overall: FAIL.** numpy_port arm PASS (36/36); published_table arm FAIL (1/18 — only the dropped C.3 row); gradient arm FAIL (0/5). All failures are real physics disagreements between the v0.1 PyTorch MIMICS (Rayleigh + sinc² form factor + real-only UEL + √σ°_oh coupling, tuned for Moor House heather) and the external T94/M90 anchors at the wheat/walnut scenarios those anchors describe. These are exactly the class of disagreements the G2 three-arm structure is designed to expose, per the honest-gates protocol (SPEC §9, §13). They do not constitute a G2-harness bug; they are the gate working as intended.

### Handoff to Session E

Session E picks up:
1. Diagnostic drill-down on the published_table arm per-row failures (mimics_toure_single_crown_breakdown decomposition by mechanism).
2. Implementation of the `use_trunk_layer=True` path for Set D.
3. Gradient-arm investigation: autograd returns 0 for m_v sensitivity — likely the Mironov ε < 1.01 clamp killing the gradient; needs a look.
4. Promotion of any v0.1 approximation found to be materially binding (full Ulaby–Moore–Fung finite-cylinder form factors, dual-dispersion UEL, literal Fresnel |Γ| in the coupling — called out in reference_toure.py's "Known limitations (v0.1)" block).
5. Re-run G2 until it passes.

§14 sign-off is gated on all three arms passing. Session E is where that happens.
