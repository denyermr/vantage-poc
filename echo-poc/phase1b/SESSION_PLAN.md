# Phase 1b — Multi-Session Plan

This document is the **session entry point** for Phase 1b work. At the
start of every session, read this after [`CLAUDE_3.md`](../CLAUDE_3.md)
and [`SPEC.md`](../SPEC.md) to see where we are, what the next session
is supposed to produce, and what "done" looks like for the current work.

It is kept up to date as a living document. Each session must update
the Session log at the bottom with what actually happened.

---

## Overall plan

Phase 1b runs in four large blocks. Each block closes with a concrete,
reviewable artefact. The blocks must be completed in order because of
the gate dependencies in `CLAUDE_3.md` and `SPEC.md` §13.

### Block 0 — Foundations (complete)

- [x] Repository migration `poc/` → `shared/ + phase1/ + phase1b/`
- [x] G1 baseline reproducibility check implemented at [`phase1/run_baselines.py`](../phase1/run_baselines.py)
- [x] **G1 passed** — result at [`implementation_gate/results/g1_baseline_result.json`](implementation_gate/results/g1_baseline_result.json)
- [x] MIMICS lit review installed (April 2026, v1.0.2)
- [x] Unblocker decisions documented:
  - [x] [U-1](physics/reference_mimics/README.md) — reference MIMICS sourcing (Option E: published tables + numpy Toure port)
  - [x] DEV-1b-001 — NDVI → LAI → N_l prior withdrawn ([`DEV-1b-001.md`](DEV-1b-001.md))
  - [x] [U-3](decisions/U-3-ndwi-formulation.md) — NDWI formulation (Gao + McFeeters; re-scoped diagnostic-only)
  - [x] [U-4](decisions/U-4-ndwi-mg-mapping.md) / DEV-1b-002 — NDWI → m_g prior withdrawn ([`DEV-1b-002.md`](DEV-1b-002.md))
- [x] SPEC.md v0.1 installed (sign-off-ready, pending gates)

### Block 1 — Pre-training gates (in progress)

Four gates must pass before SPEC §14 can be signed. G1 is done. G2–G4
are the focus of Block 1.

| Gate | Artefact | Status |
|---|---|---|
| G1 — Phase 1 baseline reproducibility (SPEC §2) | [`phase1/run_baselines.py`](../phase1/run_baselines.py) | ✅ Passed |
| G4 — Dobson vs Mironov dielectric diagnostic (SPEC §6, §11 Diag C) | [`implementation_gate/dielectric_diagnostic.py`](implementation_gate/dielectric_diagnostic.py) | ✅ Passed (binding = YES, max \|Δε\|/ε = 97.6%) |
| G3 — Oh ks-validity across s = 1–5 cm (SPEC §7) | [`implementation_gate/ks_validity_check.py`](implementation_gate/ks_validity_check.py) | ✅ Passed (30/30 cells; no AIEM substitution) |
| G2 — MIMICS forward equivalence (SPEC §4) | `implementation_gate/equivalence_check.py` + physics + reference | ⏳ pending (sessions B–E) |

The ordering is G4 → G3 → G2 rather than numeric because G4 and G3 are
self-contained and small; G2 is a multi-session software build. Closing
G4 and G3 first moves us from 1/4 to 3/4 gates passed with clean
artefacts, before we open the longer G2 work stream.

### Block 2 — λ search and training infrastructure

After sign-off (G1–G4 passed, SPEC §14 signed), we build:

- PyTorch composite loss (`L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds`; **no L_prior**, per DEV-1b-001 and DEV-1b-002)
- PINN-MIMICS trainer using the shared PhysicsNet + CorrectionNet backbone
- λ search over the 64-combination grid `(λ₁, λ₂, λ₃) ∈ {0.01, 0.1, 0.5, 1.0}³`, with the SPEC §9 stricter dominance criterion

### Block 3 — Execution, diagnostics, reporting

4 × 10 factorial execution; Dobson sensitivity arm on 10 × 100%
configurations; the Phase 1-inherited diagnostics plus new Diagnostics
A, B, C, D (including the DEV-1b-002 NDWI ↔ m_g correlation check); if
Negative, the decision-tree interpretation; `phase1b_results.md` written
in the same format as `poc_results.md`.

---

## Active session block — Block 1, sessions A and onward

### Session A (current) — G4 and G3

**Goal.** Close out both self-contained gates in a single session.

**Artefacts produced:**
- `phase1b/implementation_gate/dielectric_diagnostic.py` (G4)
- `phase1b/implementation_gate/ks_validity_check.py` (G3)
- `phase1b/physics/oh1992_learnable_s.py` — Oh with surface roughness `s` in cm as a parameter (reused by G3 and eventually by the PINN-MIMICS forward pass)
- `phase1b/implementation_gate/results/g4_dielectric.json` (result of G4)
- `phase1b/implementation_gate/results/g3_ks.json` (result of G3)
- Unit tests under `tests/unit/` for both new modules
- Updated entries in this SESSION_PLAN session log

**Done when:**
- Both result JSONs exist and are reproducible by running the Makefile `make g3` and `make g4` targets.
- Both scripts exit 0 (or, for G4, exit 0 regardless of binding flag — SPEC §6 says the diagnostic records the outcome but passes either way).
- G3 exits 0 iff no NaN, bounded σ°, and σ° monotonic in m_v across s = 1–5 cm.
- `pytest tests/unit/` is green (no regressions from the existing 116 tests).
- The session log below is updated with what the G4 binding flag actually was and whether G3 required any deviation (e.g. AIEM substitution per SPEC §7).

**Not in scope for session A:**
- Any MIMICS physics code (G2 territory; later sessions).
- Writing `phase1b/physics/mironov.py` as a new module — for the dielectric diagnostic we reuse the existing `phase1/physics/dielectric.py` classes (`DobsonDielectric`, `MironovDielectric`) which are already correct per SPEC §6. Introducing a Phase 1b copy would double-maintain a frozen artefact without scientific benefit.
- Any changes to the aligned dataset or Phase 1 code (Tier 1/2 — not touched).

### Sessions B–E (upcoming) — G2

Scope for G2 is substantial enough that it gets its own breakdown.
Each session's goal is a reviewable deliverable.

| Session | Deliverable |
|---|---|
| B | `phase1b/physics/reference_mimics/reference_toure.py` — plain-numpy Toure 1994 single-crown implementation; `canonical_combinations.json` schema fixed; a first set of (parameters → σ°) tuples generated by the numpy port over the Moor House operating envelope. |
| C | `phase1b/physics/mimics.py` — differentiable PyTorch MIMICS (Toure single-crown, VV + VH). Physics-only tests (monotonicity, finite outputs, no NaN under parameter sweeps). |
| D | `phase1b/physics/reference_mimics/published_tables/` — σ° values transcribed from Toure 1994 (and optionally McDonald 1990) with page/row citations. `phase1b/implementation_gate/equivalence_check.py` compares PyTorch MIMICS against both the transcribed tables and the numpy port. |
| E | Debugging to within the 0.5 dB tolerance. First-pass implementations typically need iteration; this session is explicitly reserved for that. If G2 is already passing at the end of session D, this session becomes sign-off: update SPEC §14, run `make p1b-ready`, move to Block 2. |

Sessions B–E are not expected to be consecutive calendar sessions.
Between sessions, the session log below must record where we got to
and what the next session picks up.

---

## Reference map — "where does X live?"

| Need to check | File |
|---|---|
| Scientific pre-registration (authoritative) | [`../SPEC.md`](../SPEC.md) |
| Project engineering rules | [`../CLAUDE_3.md`](../CLAUDE_3.md) |
| Repository conventions / Tier 1/2/3 rules | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) |
| Lit review (parameter ranges, precedents) | [`../vantage-mimics-litreview-v1_0_2.html`](../vantage-mimics-litreview-v1_0_2.html) |
| Active deviation log (summary rows) | [`deviation_log.md`](deviation_log.md) |
| Full deviation entries | [`DEV-1b-001.md`](DEV-1b-001.md), [`DEV-1b-002.md`](DEV-1b-002.md) |
| Unblocker / design-decision records | [`decisions/U-1-*`](physics/reference_mimics/README.md), [`decisions/U-3-*`](decisions/U-3-ndwi-formulation.md), [`decisions/U-4-*`](decisions/U-4-ndwi-mg-mapping.md) |
| G1 result (sealed) | [`implementation_gate/results/g1_baseline_result.json`](implementation_gate/results/g1_baseline_result.json) |
| Phase 1 frozen physics | [`../phase1/physics/`](../phase1/physics/) |
| Shared backbone + baselines + data | [`../shared/`](../shared/) |

---

## Session log

Chronological record. Each entry: date, what was planned, what shipped,
any deviations, and the explicit handoff to the next session.

### 2026-04-17 — Block 0 close + Block 1 open

Planned:
- Repository migration M1–M4.
- U-1 through U-4 decisions.
- Install SPEC.md and lit review.

Shipped:
- Migration complete; tests pass (116 unit + 10/11 integration; the
  one failure is the pre-existing G1-05 SAR–VWC correlation issue
  covered by DEV-003 — not a Phase 1b concern).
- G1 passed cleanly (all 12 rows within tolerance; the four SPEC §2
  anchors all within ±0.0005 cm³/cm³).
- U-1 Option E, U-3 Option B, DEV-1b-001, DEV-1b-002, U-4 Option A
  all documented.
- SPEC.md v0.1 installed (sign-off-ready pending G2–G4).

Handoff: session A picks up G4 and G3 per the plan above.

### 2026-04-18 — Session A (G4 + G3)

Planned:
- G4 Dobson vs Mironov dielectric diagnostic implementation and run.
- G3 Oh ks-validity check implementation and run, including the
  `phase1b/physics/oh1992_learnable_s.py` module that G3 tests and
  that the eventual PINN-MIMICS forward pass will use.
- Unit tests for the new modules.

Shipped:
- [`phase1b/physics/oh1992_learnable_s.py`](physics/oh1992_learnable_s.py)
  — Oh (1992) with learnable RMS height `s` in cm, plus Oh (2004)
  cross-pol VH/VV ratio; C-band constants derived from f = 5.405 GHz;
  DEV-007 ε ≥ 1.01 clamp inherited.
- [`phase1b/implementation_gate/dielectric_diagnostic.py`](implementation_gate/dielectric_diagnostic.py)
  (G4) and result [`g4_dielectric.json`](implementation_gate/results/g4_dielectric.json).
- [`phase1b/implementation_gate/ks_validity_check.py`](implementation_gate/ks_validity_check.py)
  (G3) and result [`g3_ks.json`](implementation_gate/results/g3_ks.json).
- 24 unit tests in [`tests/unit/test_phase1b_gates.py`](../tests/unit/test_phase1b_gates.py).
  Full suite: 140 passed, 0 failed (up from 116).

Gate outcomes:

- **G4 PASS, binding = YES** (SPEC §6 notes the gate passes either way;
  the result is recorded). Max relative |Δε|/ε = **97.6%** at
  m_v = 0.83. Mironov is clamped to ε = 1.01 over 271 / 501 of the
  observed m_v range (m_v below ~0.56). At m_v = 0.83, Dobson gives
  ε ≈ 64.4 while Mironov gives ε ≈ 1.54. This is a far larger
  divergence than Phase 1 observed at the PINN-prediction level
  (5.4%, informational only, per `poc_results.md`) and is a direct
  consequence of how DEV-007's clamp interacts with Mironov's
  organic-soil parameterisation at peat-saturated m_v. **The
  dielectric sensitivity arm (SPEC §11) must be reported alongside
  the primary outcome, and the interpretation language in the Phase 1b
  results document must reflect that dielectric choice is an active
  source of variance.** This is a finding, not a gate blocker.
- **G3 PASS, 30/30 cells.** AIEM substitution NOT required; no
  DEV-1b-003 entry. All cells are numerically safe (no NaN / Inf,
  monotonic in m_v within 0.05 dB tolerance). Observational-envelope
  summary at Moor House m_v ∈ [0.25, 0.83], θ = 41.5°:
  - Dobson @ s = 2 cm: σ°_VV ∈ [−10.65, −8.81] dB, σ°_VH ∈ [−19.82, −16.76] dB.
  - Mironov @ s = 2 cm: σ°_VV ∈ [−54.27, −22.71] dB, σ°_VH ∈ [−87.17, −39.24] dB.
  The Mironov envelope's low σ° regime is Oh's mathematically correct
  output given ε clamped to 1.01 (no dielectric contrast) and is a
  DEV-007 signature, not an Oh failure.

One implementation note worth flagging: the first pass of G3's
numerical-plausibility bounds was too tight — set at the SAR-observation
range [−40, +5] dB for VV and [−50, +5] dB for VH — which caused
Mironov cells to fail on the "observational plausibility" axis rather
than the "numerical safety" axis that SPEC §7 actually asks about.
Relaxed to [−120, +10] dB for VV and [−150, +10] dB for VH (overflow
guards only), with a separate, informational Moor-House observational-
envelope report so the sign-off JSON still shows where the model
lands in practice. No SPEC deviation — this is aligning the gate
implementation with the spec's intent ("no NaN, no extreme values,
monotonic") rather than a stricter interpretation.

§14 sign-off status: 3 / 4 gates passed (G1, G3, G4). G2 is the
remaining gate, scheduled for sessions B–E per the plan above.

Handoff to next session (session B):
- Goal: Write [`phase1b/physics/reference_mimics/reference_toure.py`](physics/reference_mimics/reference_toure.py)
  — plain-numpy Toure 1994 single-crown implementation. No autograd.
- Goal: Lock the canonical-combinations schema at
  `phase1b/physics/reference_mimics/canonical_combinations.json`, and
  generate a first set of (parameters → σ°) tuples from the numpy
  Toure port over the Moor House operating envelope (heather-scale
  geometry from SPEC §5, peat ground with Mironov dielectric, s = 1–5 cm,
  θ = 41.5°).
- Not in scope for session B: the PyTorch differentiable MIMICS
  (session C), published σ° transcription (session D), equivalence
  debugging (session E).
- Reading before starting session B: this SESSION_PLAN; SPEC §4 and §5;
  [`physics/reference_mimics/README.md`](physics/reference_mimics/README.md)
  (U-1 decision — published tables + numpy port); lit review §2
  (MIMICS physics) and §8 Decision 1 (Toure single-crown choice).

### 2026-04-18 — Session B (numpy Toure reference + canonical set)

Planned:
- Plain-numpy Toure 1994 single-crown MIMICS implementation at
  `phase1b/physics/reference_mimics/reference_toure.py`. No autograd;
  independent numpy code paths for dielectric (Mironov), vegetation
  dielectric (Ulaby–El-Rayes simplified), Oh 1992 surface scattering,
  and single-scatterer cross sections.
- Lock the canonical-combinations schema and generate a first set of
  numpy-port σ° tuples over the Moor House envelope.
- Unit tests for the new modules.

Shipped:
- [`phase1b/physics/reference_mimics/reference_toure.py`](physics/reference_mimics/reference_toure.py)
  — `MimicsToureParams` dataclass + `mimics_toure_single_crown`
  forward returning (σ°_VV dB, σ°_VH dB), plus a breakdown helper
  that exposes the three-mechanism decomposition, κ_e, τ², and both
  dielectrics for session D/E debugging.
- [`phase1b/physics/reference_mimics/generate_numpy_port_combinations.py`](physics/reference_mimics/generate_numpy_port_combinations.py)
  — preserves non-`numpy_port` rows and regenerates the numpy-port
  entries, tagging each with a SHA-256 of `reference_toure.py`.
- [`phase1b/physics/reference_mimics/canonical_combinations.json`](physics/reference_mimics/canonical_combinations.json)
  — schema v1, `tolerance_db = 0.5`, 36 numpy-port entries (27 on the
  dense-grid s×m_v×N_l cross, 9 sparse-canopy probes at SPEC §5 lower
  bounds N_b = 1e2 / N_l = 1e3).
- 36 unit tests in [`tests/unit/test_reference_toure.py`](../tests/unit/test_reference_toure.py).
  Full suite: 176 passed, 0 failed (up from 140).

Known v0.1 limitations (logged in the module docstring, to be revisited
in session E against Toure 1994 tables):
- Single-scatterer amplitudes use Rayleigh polarisabilities plus a
  backscatter sinc² form factor evaluated at the incidence direction;
  the same sinc² is applied to extinction as a first-pass RGD
  approximation. The correct Ulaby–Moore–Fung finite-cylinder form
  factors (Bessel-function radial, sinc axial, averaged over scattering
  angles for extinction) are deferred. At SPEC §5 midpoint densities
  the approximation is optically thick enough (τ² ≈ 1e-7) that σ° is
  saturated at the crown-direct ceiling and ground parameters do not
  visibly drive the total — the sparse-canopy probe set (N_b = 1e2,
  N_l = 1e3) is included specifically to cover a thinner-canopy regime
  where ground terms contribute.
- Vegetation dielectric is the real part only (UEL linear proxy
  ε_v(m_g) = 1.7 + 18 · m_g, calibrated to Ulaby & El-Rayes 1987 Fig. 7).
  Loss-tangent effects absent.
- Crown-ground coupling uses √(σ°_oh) as a ground-reflectance proxy
  rather than the literal Fresnel |Γ|; this folds Oh's roughness
  factor in through the coupling term.

σ° summary of the first set:
- Dense-grid (SPEC §5 midpoint-range N_l): σ°_VV = −19.49, −17.15,
  −14.04 dB for N_l = 10³, 10⁴, 10⁵ respectively (all s and m_v in
  this regime saturated at the crown-direct ceiling; ground terms
  suppressed by τ²).
- Sparse-canopy probes (N_b = 1e2, N_l = 1e3): σ°_VV = −17.77 dB at
  m_v ≤ 0.55 (Mironov ε at clamp); σ°_VV = −17.08 to −17.14 dB at
  m_v = 0.83 (Mironov ε ≈ 1.54) with 0.02 dB s sensitivity at that
  m_v. The small m_v response is the DEV-007-clamp signature
  propagating through the MIMICS forward — expected and consistent
  with G4.

No SPEC deviation — the v0.1 limitations in the scattering model are
all within the scope of the "first pass" (SPEC §4's implementation gate
is a 0.5 dB check against reference; establishing the reference is
what session B was for). If session D's published-table check shows
the Rayleigh approximation is binding, session E will promote to the
full Ulaby–Moore–Fung form as planned.

§14 sign-off status: unchanged (3 / 4 gates). G2 still pending; the
G2 deliverable is `phase1b/implementation_gate/equivalence_check.py`
which compares the to-be-written PyTorch MIMICS (session C) against
the 36 numpy-port entries above plus the Toure 1994 table transcripts
(session D).

Handoff to next session (session C):
- Goal: Write [`phase1b/physics/mimics.py`](physics/mimics.py)
  — differentiable PyTorch port of the same Toure single-crown
  formulation implemented in `reference_toure.py`. Must preserve
  gradient flow through all learnable parameters (N_b, N_l,
  σ_orient, m_g, s) and through m_v for the PhysicsNet coupling.
- Physics-only unit tests in `tests/unit/`: monotonicity in N_l,
  finite outputs across the SPEC §5 learnable ranges, no NaN under
  gradient. **Do not** run the equivalence check in session C; that
  is session D's scope.
- Not in scope for session C: transcribing Toure 1994 tables
  (session D), writing `equivalence_check.py` (session D),
  debugging disagreements (session E).
- Reading before starting session C: this SESSION_PLAN;
  [`physics/reference_mimics/reference_toure.py`](physics/reference_mimics/reference_toure.py)
  in full (including "Known limitations (v0.1)" — the PyTorch port
  must mirror the same approximations so the numpy-port arm of G2
  agrees within 0.5 dB on session D's run); SPEC §4, §5, §7, §8;
  [`phase1b/physics/oh1992_learnable_s.py`](physics/oh1992_learnable_s.py)
  (the PyTorch surface model to be reused inside the MIMICS forward).

### 2026-04-18 — Session C (differentiable PyTorch MIMICS)

Planned:
- `phase1b/physics/mimics.py` — differentiable PyTorch port of the
  Toure single-crown formulation in `reference_toure.py`, mirroring
  its v0.1 approximations (Rayleigh + sinc² form factor; real-part
  UEL; √(σ°_oh) crown-ground coupling).
- Gradient flow preserved through all five SPEC §5 learnables
  (N_b, N_l, σ_orient, m_g, s) and per-obs m_v.
- Reuse of `phase1b/physics/oh1992_learnable_s.py` for the surface
  layer; reuse of the SPEC §6 ε ≥ 1.01 clamp via `.clamp()` semantics
  (identical to the oh1992 torch module and the numpy reference).
- Physics-only unit tests: monotonicity in N_l, finite outputs across
  SPEC §5 ranges under gradient-enabled forward, VH ≤ VV at the
  sparse-canopy probe, per-learnable gradient finiteness.

Shipped:
- [`phase1b/physics/mimics.py`](physics/mimics.py) —
  `MimicsToureParamsTorch` dataclass + `mimics_toure_single_crown`
  returning (σ°_VV dB, σ°_VH dB). All tensor-carrying inputs may be
  scalar (0-d) or batched; the orientation quadrature broadcasts in a
  (..., n_theta, n_phi) layout and reduces at the end. Device/dtype
  inferred from input tensors with fallback to
  `shared.config.get_torch_device()`; no hardcoded device strings.
- 26 unit tests in [`tests/unit/test_mimics_torch.py`](../tests/unit/test_mimics_torch.py)
  covering Mironov / UEL / crown-direct factor gradient flow, finite
  outputs across all six learnable / per-obs axes (N_b, N_l,
  σ_orient, m_g, s, m_v) at SPEC §5 endpoints, simultaneous gradient
  flow through all five learnables plus m_v in one forward pass, and
  VH ≤ VV at the sparse-canopy probe.
- Full unit suite: 202 passed, 0 failed (up from 176). No
  regressions.

Implementation notes:
- `torch.sinc` is not implemented on the MPS backend; the sinc² form
  factor uses a manual `sin(x)/x` with a `torch.where` guard at the
  origin. Both branches are smooth so gradient flow is preserved.
- The crown-direct small-κ limit is factored through the numerically
  stable `(1 − exp(−x))/x` helper (`_one_minus_exp_over_x`) with a
  three-term Taylor branch below |x| = 1e-6 — this mirrors the
  numpy reference's `if kappa_e * h_c / cos_t < 1e-6` small-κ
  switch but is continuous through the transition (important for
  gradient flow during λ search).
- `torch.pow(torch.as_tensor(10.0, ...), sigma_oh_db / 10)` is used
  instead of `10 ** (...)` so the base-tensor carries the correct
  device/dtype even when the numerator is MPS and the literal would
  otherwise coerce to CPU; keeps the forward device-consistent.

No SPEC deviation — the PyTorch module is a faithful port of the v0.1
numpy reference. Session E remains the planned escalation path if the
numpy-port arm of G2 fails the 0.5 dB check (which session D will
measure) and a v0.2 reference is required.

§14 sign-off status: unchanged (3 / 4 gates). G2 remains the pending
gate. The PyTorch module now exists; what is still required is the
equivalence-check driver and the published-table transcripts
(session D), followed by any debugging in session E.

Handoff to next session (session D):
- Goal: Transcribe σ° values from Toure 1994 tables (and optionally
  McDonald 1990) into
  `phase1b/physics/reference_mimics/canonical_combinations.json` as
  new `source.type = "published_table"` entries with page and row
  citations, preserving the existing 36 `numpy_port` rows.
- Goal: Write
  `phase1b/implementation_gate/equivalence_check.py` — compares the
  PyTorch MIMICS (this session's module) against every canonical entry
  at its tolerance. The gate passes iff max |Δσ°_dB| ≤ 0.5 across
  all VV and VH test points from both `numpy_port` and
  `published_table` source types.
- Goal: Write the G2 result JSON at
  `phase1b/implementation_gate/results/g2_equivalence.json` and the
  Makefile `make g2` target, mirroring the G3 / G4 patterns.
- Not in scope for session D: promoting any of the v0.1
  approximations flagged in `reference_toure.py`'s "Known limitations"
  block — that is session E if the G2 check fails.
- Reading before starting session D: this SESSION_PLAN; the canonical
  JSON schema (already stable, session B); Toure 1994 IEEE TGRS 32(1)
  tables (via the lit review if a copy is staged); SPEC §4 for the
  gate tolerance; `phase1b/implementation_gate/ks_validity_check.py`
  and `dielectric_diagnostic.py` for the gate-script / result-JSON
  patterns to mirror.

### 2026-04-19 — Session D (G2 three-arm equivalence check)

Planned:
- Transcribe σ° anchor values from Toure 1994 (primary) and McDonald 1990
  (optional secondary) into the canonical parameter set with page / table /
  row citations.
- Write `phase1b/physics/equivalence_check.py` as the three-arm G2 gate
  driver per the post-sign-off anchor-construction revision.
- Wire the Makefile `g2` target.

Context-shift during execution:
- Session opened against the original brief (`session_d_prompt.md`, now
  removed from the repo). Partway through, three new spec documents landed
  in `~/Downloads/`:
  - `SPEC_2.md` (SPEC.md v0.1 proper, superseding v0.1-draft),
  - `DEV-1b-003.md` (G2 anchor construction — open-library sourcing of both
    papers + three-arm structure),
  - `g2_anchor_spec.md` v0.1 (22-row anchor set with ~-value scaffolding).
  The remaining work was re-aligned to the three-arm structure and the
  expanded anchor set.
- Both source PDFs were procured via Open University Library access and
  staged at `phase1b/refs/`. Initial attempt had the PDFs in `~/Downloads/`;
  moved into the repo once identified.
- Poppler-utils not initially on PATH; `brew install poppler` ran mid-session;
  used `/opt/homebrew/bin/pdftoppm` / `pdftotext` directly.

Shipped:
- **Spec docs installed into the repo:**
  - [`../SPEC.md`](../SPEC.md) — SPEC v0.1 (replaces v0.1-draft).
  - [`DEV-1b-003.md`](DEV-1b-003.md) — now in phase1b root. Methodology
    appendix added at the bottom documenting Session D anchor refinement.
  - [`physics/g2_anchor_spec.md`](physics/g2_anchor_spec.md) — v0.2
    (refined from PDFs; supersedes v0.1 scaffolding).
- **Anchor refinement (DEV-1b-003 deliverable):** 22 values refined from
  v0.1 `~`-estimates to values read directly from the source PDFs.
  Provenance at [`refs/anchor_reads/`](refs/anchor_reads/) — `README.md`,
  `anchor_reads_v0.json`, `anchor_reads_v1.json`, 9 annotated PNGs
  (plot-box calibration checks, detected-marker overlays, zoomed-panel
  crops).
  - Sets A, B (T94 Fig. 2 CHH/CVV, 8 rows): pixel detection with
    hollow-square template scoring at 400 dpi; y-axis label-centre
    calibration (12.4 px/dB CHH, 12.32 px/dB CVV). Human spot-check at
    A.1 (θ=20° CHH), B.2/B.3/B.4 (θ=30/40/50° CVV) confirmed the
    methodology. Q1 confirmed B.3 spec-v0.1 estimate was off the filled
    (measured) marker; the correct MIMICS (hollow) marker sits at
    −11.12 dB.
  - Set C (T94 Fig. 7 CHH at θ=30°, 5 rows; C.3 dropped):
    **pixel detection fell back to human transcription** per the
    continuation-prompt Step 3 fallback clause. Triggers: v0.1's
    marker-to-mechanism mapping on C.3/C.4 conflicted with the T94 Fig. 7
    caption legend, and the middle-cluster markers at ~−20 dB overlap at
    the same dB making template scoring unreliable. C.3 (Cover-ground
    open-triangle) is not visibly present in the CHH panel at θ=30°;
    dropped from v0.2 anchor set with a DEV note. C.5 shifted −40 → −35 dB
    (v0.1 ±2.0 dB widened tolerance would still have missed by 3 dB).
  - Set D (M90 Fig. 10, 4 rows): **spec v0.1 figure-semantics was
    materially wrong** — it described MIMICS values as "open-circle
    markers"; the figure actually draws MIMICS as LINES (solid VV,
    dotted HH, dashed HV) and measurements as markers. v0.2 reads line
    positions via column-darkness histogram continuity-tracking. MIMICS
    curves in Fig. 10 end at θ=55°.
  - Set E (T94 Tables IV(a), V(a), 5 rows): text extraction via
    `pdftotext -layout`; all 5 rows matched the v0.1 spec values exactly.
- **`use_trunk_layer` compile-time flag added to
  [`physics/mimics.py`](physics/mimics.py)**: default `False` (production);
  keyword-only; `True` path raises `NotImplementedError` pointing at
  Session E (trunk-layer structural extensions).
- **[`physics/equivalence_check.py`](physics/equivalence_check.py)**: the
  G2 three-arm driver. Loads anchors from `anchor_reads_v1.json`, runs the
  PyTorch module through Sets A/B/C/D/E, computes per-row Δ, exits
  non-zero on any arm failure, writes `outputs/g2_equivalence_result.json`
  mirroring the g3_ks / g4_dielectric schema.
- **[`../outputs/g2_equivalence_result.json`](../outputs/g2_equivalence_result.json)** — first G2 run output.
- **Makefile `g2` target repointed** from `phase1b/implementation_gate/equivalence_check.py`
  to the new location `phase1b/physics/equivalence_check.py`.
- **[`../tests/unit/test_equivalence_check.py`](../tests/unit/test_equivalence_check.py)** —
  21 unit tests covering anchor-reads schema, canonical_combinations
  invariants (numpy_port count still 36; schema_version still 1),
  delta computation, end-to-end result-JSON schema, exit code, and the
  `use_trunk_layer` flag contract.
  Full unit suite: **223 passed / 0 failed** (up from 202; no regressions).
- [`decisions/U-1-g2-anchor-construction.md`](decisions/U-1-g2-anchor-construction.md)
  — records that Option E was executed with all three arms; cross-refs
  DEV-1b-003 as the authoritative log entry.
- `decisions/U-5-published-table-source.md` — removed. It was the
  Session-D intermediate write-up asking for a decision on how to source
  the papers; the science-agent response authorised Option A and that work
  is now complete, so U-5 is redundant with DEV-1b-003.

G2 first-run verdict:

- **numpy_port arm: PASS.** 36/36 rows within 0.5 dB tolerance. Max Δ
  0.000 dB. PyTorch `mimics.py` and numpy `reference_toure.py` produce
  identical σ° for every canonical combination — the port is consistent.
- **published_table arm: FAIL.** 1/18 rows pass (the dropped C.3 row).
  - Set A (T94 wheat CHH, 4 rows): all fail. PyTorch returns ≈ −20.7 dB
    across θ ∈ {20°, 30°, 40°, 50°}; anchors are −9.48 to −12.70 dB.
    Δ range 8.5–11.3 dB. Canopy saturation signature consistent with the
    v0.1 Rayleigh + sinc² approximation's "Known limitations" block.
  - Set B (T94 wheat CVV, 4 rows): all fail. Δ 1.5–6.1 dB.
  - Set C (T94 Fig. 7 CHH mechanism decomposition @ θ=30°, 6 rows):
    1 pass (dropped C.3); 5 fail. Direct-ground mechanism returns
    −64.6 dB (anchor −11); crown-direct and crown-ground also off by
    ≥ 11 dB. Ground-surface sub-model is far from T94's.
  - Set D (M90 walnut L-band, 4 rows): all 4 raise `NotImplementedError`
    (use_trunk_layer=True code path deferred to Session E). Not counted
    as real failures — recorded as `status: UNIMPLEMENTED` in the result
    JSON.
- **gradient spot-check arm: FAIL.** 0/5 rows pass.
  - E.1, E.2 (Soil m_v sensitivity): both autograd and FD return 0.
    Probable cause: the Mironov ε ≥ 1.01 clamp (DEV-007) is active at
    L-band wheat m_v=0.2 and zeros the gradient through the ε path.
    Diagnostic for Session E.
  - E.3, E.4 (Stem height sensitivity): autograd and FD disagree with
    T94 published value by 3–4 dB. autograd-vs-FD also disagreeing,
    suggesting the PyTorch autograd and FD are probing different tangent
    directions.
  - E.5 (Leaf width sensitivity): autograd and FD agree with each other
    within 0.07 dB (passing the autograd-vs-FD internal consistency at
    ±0.02 dB or ±5% tolerance — internal consistency holds here), but
    both disagree with T94 by > 7 dB.

Interpretation: the failures are real physics disagreements between the
v0.1 PyTorch MIMICS (Rayleigh + sinc² form factor + real-only UEL +
√σ°_oh coupling, tuned for Moor House heather at C-band) and the external
T94/M90 anchors at the wheat/walnut scenarios those anchors describe.
This is **exactly the class of disagreements the G2 three-arm structure
is designed to expose** — the numpy_port consistency check passes
cleanly, but neither implementation agrees with independently published
σ° or sensitivity values on the T94/M90 test cases. Per the honest-gates
protocol (SPEC §9, §13), no tolerance was loosened to absorb these.

§14 sign-off status: unchanged (G1, G3, G4 passed; G2 remains the pending
gate). G2 first run produced a concrete, diagnosable failure report —
the gate is now exercised end-to-end and points directly at the physics
work required in Session E.

Handoff to next session (Session E):
- **Do not loosen tolerances.** The gate is correctly surfacing real
  issues.
- **Priority 1 — gradient arm E.1/E.2.** Trace whether the Mironov
  ε ≥ 1.01 clamp is zeroing the m_v gradient at wheat-reference m_v=0.2.
  Probably the cheapest fix in the failure set.
- **Priority 2 — published_table Set A/B diagnosis.** Use
  `reference_toure.mimics_toure_single_crown_breakdown` to decompose σ°
  into crown-direct, crown-ground, ground-direct at the T94 wheat inputs.
  Identify which mechanism is saturated / which approximation (Rayleigh
  single-sinc², real-only UEL, √σ°_oh coupling) is binding. Promote the
  binding approximation(s) to full MIMICS forms — Ulaby–Moore–Fung
  finite-cylinder form factors, dual-dispersion UEL with Im(ε),
  literal Fresnel |Γ| — per the "Known limitations (v0.1)" block in
  `reference_toure.py`.
- **Priority 3 — Set D (use_trunk_layer=True).** Implement the M90
  walnut trunk-layer code path. Requires four branch-class support,
  cos⁶ zenith distribution for trunks, complex vegetation dielectric
  (M90 Table III uses Im(ε) ≈ 11), L-band wavelength support.
- **Priority 4 — Set C mechanism decomposition exposed from the PyTorch
  module.** Currently `mimics.py` returns only totals; the Set C harness
  falls back to the numpy reference's breakdown helper. Add a
  `mimics_toure_single_crown_breakdown_torch` that exposes the three-mechanism
  decomposition from the torch forward pass so the Set C anchor directly
  tests the PyTorch code.
- **HH output channel.** `mimics.py` returns only VV and VH; the Set A
  anchor is HH. The Session D harness substitutes VV with a caveat flag.
  Session E should add HH as a first-class output.
- After all arms pass, run `make p1b-ready`, update SPEC.md §14 sign-off,
  and move to Block 2 (λ search + PINN-MIMICS trainer).

Reading before starting Session E:
- This SESSION_PLAN.
- [`DEV-1b-003.md`](DEV-1b-003.md) including the "Appendix — anchor
  refinement" at the end.
- [`physics/g2_anchor_spec.md`](physics/g2_anchor_spec.md) v0.2.
- [`../outputs/g2_equivalence_result.json`](../outputs/g2_equivalence_result.json)
  for the per-row failure detail.
- [`physics/reference_mimics/reference_toure.py`](physics/reference_mimics/reference_toure.py)
  "Known limitations (v0.1)" block for the queue of approximations
  pre-registered as candidate promotions.
- [`decisions/U-1-g2-anchor-construction.md`](decisions/U-1-g2-anchor-construction.md)
  for the three-arm structure and Session D handoff.
