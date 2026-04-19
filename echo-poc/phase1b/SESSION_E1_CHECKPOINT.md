# Phase E-1 Checkpoint Report

**Session:** Phase 1b, Session E, Phase E-1 (non-physics).
**Date:** 2026-04-19.
**Scope landed:** DEV-1b-004 (E.1 / E.2 dielectric amendment), DEV-1b-005 (Set D Phase 1c exemption), P3 (PyTorch mechanism decomposition), `g2_anchor_spec.md` → v0.3, Set C2 Table 11-1 anchor registration (harness deferred to Phase E-2), 11 new unit tests.

---

## 1. Non-physics deliverables landed

| Deliverable | File(s) | Status |
|---|---|---|
| `ground_epsilon_dobson_torch` (parameterised, Moor-House-peat defaults bit-identical to frozen `DobsonDielectric`) | `phase1b/physics/mimics.py` | Landed |
| `ground_dielectric_fn` kwarg on `mimics_toure_single_crown` and `mimics_toure_single_crown_breakdown_torch` | `phase1b/physics/mimics.py` | Landed |
| `mimics_toure_single_crown_breakdown_torch` (P3 — PyTorch mechanism decomposition) | `phase1b/physics/mimics.py` | Landed |
| DEV-1b-004 — gradient-arm dielectric-configuration amendment for E.1 / E.2 | `phase1b/DEV-1b-004.md` | Committed (draft; ready for git once repo baseline initialised — see §7) |
| DEV-1b-005 — Set D Phase 1c exemption | `phase1b/DEV-1b-005.md` | Committed (draft; ready for git once repo baseline initialised) |
| Deviation log summary rows for DEV-1b-003, DEV-1b-004, DEV-1b-005 | `phase1b/deviation_log.md` | Landed |
| `g2_anchor_spec.md` → v0.3 (E.1 / E.2 dielectric block; Set D EXEMPT; Set C2 Table 11-1 anchor registration) | `phase1b/physics/g2_anchor_spec.md` | Landed |
| `equivalence_check.py` updates: Dobson-mineral callable for E.1 / E.2; Set D returns `status: EXEMPT`; Set C2 returns `status: DEFERRED_PHASE_E2`; Set C now uses the PyTorch breakdown (P3) instead of the numpy reference; print report shows exempt / deferred counts separately from active pass / fail | `phase1b/physics/equivalence_check.py` | Landed |
| Phase E-1 unit tests (Dobson torch default parity with frozen; mineral kwargs unclamped at T94 wheat m_v = 0.2; clamp preserved; gradient non-zero; breakdown sum = total; breakdown keys match numpy reference; breakdown total matches numpy within 0.5 dB; Moor House production pinning; Set D NotImplementedError cross-references DEV-1b-005) | `tests/unit/test_mimics_torch.py` | 37 / 37 new-file tests pass; full suite **234 / 234** (up from 223; zero regressions) |

## 2. G2 verdict at Phase E-1 close

Result JSON at [`outputs/g2_equivalence_result.json`](../outputs/g2_equivalence_result.json).

| Arm | Verdict | Detail |
|---|---|---|
| numpy_port | **PASS** | 36 / 36 rows; max Δ = 6.17 × 10⁻⁶ dB at 0.5 dB tolerance. Unchanged from Session D end. Refactor (shared `_forward_internal`, optional `ground_dielectric_fn`, P3 breakdown) is bit-identical on the production path. |
| published_table | **FAIL** | Set A 0 / 4, Set B 0 / 4, Set C 1 / 6 (the dropped C.3 row auto-passes), Set C2 0 / 0 active (24 DEFERRED_PHASE_E2), Set D 0 / 0 active (4 EXEMPT per DEV-1b-005). No arm-level pass improvement yet — Phase E-1 is scope-disciplined to non-physics deliverables. |
| gradient | **FAIL** (but mechanism unlocked) | 0 / 5 rows pass. **E.1 / E.2 moved from `autograd = FD = 0.000 dB` (clamped) to `autograd ≈ 0.17 / 0.26 dB, FD ≈ 0.17 / 0.26 dB` under Dobson-mineral (DEV-1b-004 effective).** autograd ↔ FD internal consistency is now excellent (within 0.003 dB). Gradient flow through ε path confirmed alive. E.3 / E.4 / E.5 unchanged (expected — Session F territory for E.3 / E.4 non-smoothness; E.5 Rayleigh-vs-finite-cylinder regime mismatch for Phase E-2). |

### Set A detail (T94 wheat CHH multiangle)

| Row | θ | PyTorch σ° | Anchor σ° | Δ |
|---|---|---|---|---|
| A.1 | 20° | −20.79 dB | −9.48 dB | 11.31 dB |
| A.2 | 30° | −20.63 dB | −10.44 dB | 10.19 dB |
| A.3 | 40° | −20.70 dB | −11.57 dB | 9.13 dB |
| A.4 | 50° | −21.23 dB | −12.70 dB | 8.53 dB |

### Set C detail (T94 wheat mechanism decomposition @ θ = 30° CHH)

| Row | Mechanism | PyTorch σ° | Anchor σ° | Δ | Notes |
|---|---|---|---|---|---|
| C.1 | Direct ground ♦ | −64.63 dB | −11 dB | **53.63 dB** | Ground-direct crushed by peat-Mironov clamp (ε = 1.01) — see §4 |
| C.2 | Direct cover □ | −8.84 dB | −20 dB | 11.16 dB | Direct-crown ~11 dB higher than anchor |
| C.3 | (dropped v0.2) | — | — | — | auto-pass |
| C.4 | Ground-cover ■ | −35.21 dB | −20 dB | 15.21 dB | Similarly suppressed by clamp |
| C.5 | GCG ▲ | N/A | −35 dB | — | GCG mechanism absent in v0.1 forward |
| C.6 | Total σ° ○ | −8.83 dB | −10 dB | 1.17 dB | Total driven by direct-crown alone |

### Gradient detail (Set E)

| Row | Parameter | Dielectric | autograd | FD | T94 | autograd↔FD Δ | pass |
|---|---|---|---|---|---|---|---|
| E.1 | Soil m_v | **T94-Dobson-mineral** (DEV-1b-004) | 0.165 dB | 0.168 dB | 1.210 dB | 0.003 dB | ❌ |
| E.2 | Soil m_v | **T94-Dobson-mineral** (DEV-1b-004) | 0.260 dB | 0.265 dB | 1.160 dB | 0.005 dB | ❌ |
| E.3 | Stem height | Mironov (default) | 3.861 dB | 0.916 dB | 0.100 dB | 2.944 dB | ❌ |
| E.4 | Stem height | Mironov (default) | 1.249 dB | 0.380 dB | 0.050 dB | 0.869 dB | ❌ |
| E.5 | Leaf width | Mironov (default) | 7.446 dB | 7.512 dB | 0.180 dB | 0.066 dB | ❌ |

## 3. Mechanism decomposition evidence at Set A θ = 30°

Probed the PyTorch breakdown helper (P3) at T94 Set A θ = 30° CHH inputs under BOTH the peat-Mironov (current production default) and the Dobson-mineral (DEV-1b-004 configuration) ground dielectrics.

### 3a — Peat-Mironov (Session D baseline)

| Quantity | Value |
|---|---|
| ε_ground | 1.010 (CLAMPED — DEV-007 floor) |
| σ°_oh bare soil | −55.95 dB |
| τ²_v (two-way canopy extinction) | 0.135 → −8.68 dB |
| Direct-crown | −8.84 dB |
| Crown-ground | −35.21 dB |
| Ground-direct (attenuated) | −64.63 dB |
| Total VV | −8.83 dB |

### 3b — Dobson-mineral (DEV-1b-004 configuration applied as diagnostic to Set A)

| Quantity | Value |
|---|---|
| ε_ground | 27.97 (unclamped) |
| σ°_oh bare soil | −9.23 dB |
| τ²_v (two-way canopy extinction) | 0.135 → −8.68 dB **(unchanged — canopy path is independent of ground ε)** |
| Direct-crown | −8.84 dB **(unchanged)** |
| Crown-ground | −11.85 dB |
| Ground-direct (attenuated) | −17.92 dB |
| Total VV | −6.73 dB |

### Hypothesis check (Memo 2 Phase E-1 criterion 3)

| Check | Pass? | Detail |
|---|---|---|
| ground-direct ≤ (direct-crown − 40 dB) | ✅ | Under peat-Mironov: gap = 55.8 dB; under Dobson-mineral: gap = 9.1 dB. |
| Two-way crown extinction ≥ 30 dB | ❌ | Actual: 8.68 dB (~1 e-opacity). |

The two criterion-3 sub-checks **disagree** about which mechanism is binding: the large direct-crown-vs-ground-direct gap under peat-Mironov is produced by the **clamp**, not by canopy extinction. Once the dielectric is switched to Dobson-mineral (the T94-consistent path), the gap drops from 55.8 dB to 9.1 dB — meaning most of the gap that Memo 2 attributed to extinction saturation is actually dielectric-configuration-driven.

## 4. Go / no-go verdict

| Criterion | Status | Notes |
|---|---|---|
| C1. numpy_port arm PASS at machine precision on unchanged v0.1 physics | ✅ | 36 / 36; max Δ = 6.17 × 10⁻⁶ dB. The Phase E-1 refactor (shared `_forward_internal`, optional dielectric injection, P3 breakdown) did not alter the v0.1 Moor House production path. |
| C2. Gradient arm E.1 and E.2 PASS under Dobson-with-T94-texture path | ❌ (literal reading) / ✅ (intent) | Literal: autograd = FD ≈ 0.17 / 0.26 dB, T94 = 1.21 / 1.16 dB → ~7× shortfall. Intent: the Mironov clamp is no longer the binding issue (autograd and FD are no longer zero; autograd ↔ FD agree within 0.003 dB — excellent internal consistency; gradient flow through the ε path is alive). The residual ~7× shortfall is the **same root cause** that drives the Set A / B / C failures (see C3). |
| C3. PyTorch mechanism decomposition at Set A θ = 30° confirms **extinction-saturation** hypothesis | ❌ — **extinction-saturation is NOT confirmed** | Two-way canopy extinction is only 8.68 dB, far below the 30 dB criterion. The binding issue at Set A / B / C is NOT canopy extinction saturation. Instead, §3 shows that the dominant failure mode is the dielectric-configuration category error: at T94's wheat m_v = 0.17 under peat-Mironov, ε clamps to 1.01, bare-soil σ° = −55.95 dB, and the ground-direct path is crushed. Switching to Dobson-mineral alone (keeping all v0.1 physics otherwise unchanged) closes Set A.2 from Δ = 10.19 dB to Δ = 3.71 dB, and closes C.1 (direct-ground) from Δ = 53.63 dB to Δ = 6.92 dB (= \|−17.92 − (−11)\|). |

**Overall: NO-GO.** Criterion C3 has produced a **different diagnostic finding** than Memo 2 anticipated. Memo 2's Phase E-2 P2 plan is scoped around UMF finite-cylinder form factors for extinction. The first-run evidence says extinction is not the primary binding approximation — the peat-Mironov-applied-to-T94-wheat dielectric-configuration category error is.

This is the "evidence-before-action" checkpoint doing its job. Rather than proceed to P2-extinction on a hypothesis the data does not support, Phase E-2 scope should be re-decided with this finding in hand.

## 5. Proposed re-scope options for Phase E-2

Three viable scopes. All respect Rules 1–5 and the fallback clause. I recommend **Option α**.

### Option α — Broaden DEV-1b-004 to Sets A, B, C (published_table arm)

Rationale: the same dielectric-configuration category error that DEV-1b-004 addresses for E.1 / E.2 applies to every anchor row whose canonical inputs come from T94's wheat scenario — i.e. Sets A, B, and C. T94 uses Dobson 1985 soil dielectric throughout; our Session D harness inadvertently applied the Moor House peat-Mironov to these rows. Switching Sets A / B / C to Dobson-mineral in the harness (no physics change; same Session E pattern of making the configuration-of-test explicit per anchor) is expected to:

- Close Set A.2 from Δ = 10.2 dB → Δ ≈ 3.7 dB (1-row probe above).
- Close Set C.1 from Δ = 53.6 dB → Δ ≈ 7 dB.
- Improve Sets A.1, A.3, A.4 proportionally.

After the Dobson-mineral sweep, residual failures (likely 1–4 dB on the A / B / C rows) can be decomposed via the P3 breakdown (now exposed) to identify which remaining v0.1 approximation is binding. Decisions about subsequent promotions (UMF form factors for extinction, UMF for backscatter, dual-dispersion UEL, literal Fresnel Γ) are then evidence-driven at the per-row level rather than scope-level.

**Scope to ship as Phase E-2a:** amend DEV-1b-004 (or draft DEV-1b-004.1 / DEV-1b-006 broader) extending the dielectric-configuration amendment to Sets A, B, C; re-run G2; read per-row residuals from the P3 breakdown; draft a data-driven Phase E-2b plan for whichever approximation-promotions are actually implicated.

- **Pros:** Smallest scope that tests the simpler hypothesis first. No new physics. Matches Memo 2's "evidence-before-action" principle and Memo 1's "making configuration-of-test explicit strengthens pre-registration" framing. Closes a large fraction of the failures cheaply.
- **Cons:** Still leaves the published_table arm FAIL after Phase E-2a if the residual 1–4 dB per row are above 0.5 dB tolerance. Phase E-2b scope is data-driven but still TBD.
- **Rule check:** No tolerance change; no anchor re-read; Sets A / B / C anchor values unchanged; this is the same amendment pattern as DEV-1b-004, applied consistently to all T94-wheat-scenario rows.

### Option β — Proceed with Memo 2 Phase E-2 as written (P2 UMF form factors)

Rationale: Memo 2's locked sequence (P3 → P2-extinction → P2-backscatter → P5). The extinction-saturation hypothesis is not confirmed at Set A / B / C, but UMF form factors are a pre-registered candidate promotion from `reference_toure.py`'s "Known limitations (v0.1)" block and will move some σ° values regardless.

- **Pros:** No amendment to the Memo 2 locked sequence. Honors the memo as written.
- **Cons:** The data says this is not the binding issue at Set A / B / C. UMF form factors correct a real v0.1 approximation, but the ~10 dB Set A failures are not dominantly an extinction-saturation problem; the extinction is only 8.68 dB. Extinction-promotion may close the residual 3.7 dB Set A.2 gap after Option α has done the bulk work, but executing it first risks spending effort on a secondary-order issue.
- **Rule check:** Clean; Memo 2 is science-agent-approved.

### Option γ — Pause and surface the finding to science agent / user

Rationale: the finding is material. Decision Memo 2 chose Option 4 (diagnostic-led targeted promotion) on the premise that the diagnosis was already complete. Phase E-1 evidence now contradicts the diagnosis — the binding approximation is different from what Memo 2 assumed. A second science-agent review of Phase E-2 scope is warranted.

- **Pros:** Preserves full pre-registration integrity. The Phase E-2 scope decision is made with complete evidence.
- **Cons:** G2 close delayed by one decision cycle.
- **Rule check:** Matches the honest-gates protocol's "when in doubt, surface" posture.

## 6. What Phase E-1 proves

- The Mironov clamp is **no longer the binding gradient-path issue** at T94 wheat m_v = 0.2 under Dobson-mineral (DEV-1b-004 effective).
- autograd and FD **agree within 0.003 dB** at E.1 / E.2 — the PyTorch forward is correctly differentiable through the ε path.
- PyTorch **mechanism decomposition** is now exposed (P3) and produces mechanism-level σ° values that sum back to the total within machine precision; the same intermediate tensors form both the total and the decomposition, so numpy_port agreement implies mechanism agreement.
- The **Moor House production path is pinned** — 3 unit tests in `TestMoorHouseProductionPinning` guarantee that production uses `ground_dielectric_fn=None` → Mironov, and that the DEV-1b-004 harness override cannot accidentally leak into training or inference.
- **Set D is EXEMPT** per DEV-1b-005 and is now reported as such in the result JSON; it no longer artificially counts against the published_table arm.
- **Set C2** (Ulaby & Long 2014 Table 11-1 mechanism ratios) is **registered** in v0.3 of the anchor spec — 24 rows with dB values — but **harness evaluation is deferred to Phase E-2** (DEFERRED_PHASE_E2 status) alongside the finalised physics promotion decision.
- 234 / 234 unit tests pass, zero regressions, 11 new tests added in Phase E-1.

## 7. Open procedural item: git baseline

The repo shows `git log: No commits yet` at the working directory. Phase E-1 deliverables are on disk but not committed. Memo 2 § Phase E-1 ended with "tag `phase1b-session-e1`" — that tag cannot be created without an initial commit. Options for the user:

1. Initialise the git baseline now (`git add -A && git commit -m "Phase 1b baseline through Session D"`) so Phase E-1 work can be tagged.
2. Defer the baseline commit; Phase E-1 deliverables remain in the working tree and are reviewable via their file paths (the diff is inspectable via git-status-equivalent commands even without a commit).

The Phase E-1 artefact set is complete regardless — the question is only about git metadata.

## 8. Recommendation

**Option α** (broaden DEV-1b-004 to Sets A, B, C) is the scope-smallest and evidence-most-aligned Phase E-2a. If after α the residual failures exceed 0.5 dB, the next promotions are data-driven per-row via the P3 breakdown.

If the science agent prefers to honour Memo 2's original sequence verbatim, **Option β** is acceptable but will have lower impact per unit of engineering effort than Option α.

**Option γ** is the conservative path: pause, present this report, receive an updated Phase E-2 scope decision.

I will **pause here** and await the Phase E-2 go / no-go / re-scope decision before touching any physics files.

---

**Phase E-1 closed at:** 2026-04-19.
**Phase E-2 awaiting decision.**
