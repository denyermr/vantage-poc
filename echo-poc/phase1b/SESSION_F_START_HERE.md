# Session F — Start Here

**Purpose:** Single-page landing for the Session F kickoff. Read this first, then read the files in §"Required reading order" before touching any code.

---

## One-line state

Phase 1b Block 1 closed 2026-04-19 with SPEC §14 signed and G2 Moderate Pass per DEV-1b-008. Block 2 (λ search + PINN-MIMICS trainer) is Session F's scope.

## Absolute non-negotiables

1. **Do not touch the sealed test set.** `echo-poc/data/splits/test_indices.json` is read-only forever.
2. **Do not modify v0.1 physics** (`phase1b/physics/mimics.py`, `phase1b/physics/reference_mimics/reference_toure.py`, `phase1b/physics/oh1992_learnable_s.py`) without a new DEV entry and training-diagnostic evidence. The five-way DEV-1b-008 promotion queue is a registry of candidates, **not** a commitment.
3. **Do not touch Tier 1** (`phase1/`, `data/`, frozen `shared/` modules — see CLAUDE_3.md §Frozen-tier).
4. **Do not speculatively promote** any of DEV-1b-008's (a)–(e) approximations. Phase 1b training diagnostics (SPEC §11) are the evidence gate.
5. **Moor House production path pinning:** `mimics_toure_single_crown(params)` with `ground_dielectric_fn=None` → Mironov with DEV-007 clamp. Regression-tested in `tests/unit/test_mimics_torch.py::TestMoorHouseProductionPinning`.

## Verify environment at session start

Run these in order; all should pass.

```bash
cd /Users/matthewdenyer/Documents/Vantage/Science/PoC/vantage-poc/echo-poc

git log --oneline --decorate -1
# expected: 2331cd4 (HEAD -> main, tag: phase1b-session-e2-moderate-pass, origin/main) ...

.venv/bin/python -m pytest tests/unit/ -q | tail -3
# expected: 234 passed

.venv/bin/python phase1b/physics/equivalence_check.py | tail -15
# expected: numpy_port PASS 36/36; published_table FAIL (4 EXEMPT, 24 DEFERRED);
#           gradient 0/5 — overall exits non-zero by design per DEV-1b-008
#           honest-gates discipline. Verdict classification is in
#           outputs/g2_equivalence_moderate_pass.json > closure_classification.
```

If any of the three commands fail, **stop** and diagnose before doing anything else.

## Required reading order

Read all five in order before writing any code:

1. [`../CLAUDE_3.md`](../CLAUDE_3.md) — tier conventions, frozen-vs-active, gate protocol, deviation discipline.
2. [`../SPEC.md`](../SPEC.md) §8 (composite loss, L_physics dual-pol VV+VH), §9 (λ search + dominance), §10 (success criteria + outcome categories), §11 (diagnostic plan — Session F's evidence source), §12 (risk register), §14 (signed sign-off block).
3. [`SESSION_PLAN.md`](SESSION_PLAN.md) §"Session F handoff" at the bottom — the authoritative Session F scope.
4. [`DEV-1b-008.md`](DEV-1b-008.md) — Phase E closure. Section §"Session F scope" states the evidence-led-promotion rule.
5. [`physics/g2_anchor_spec.md`](physics/g2_anchor_spec.md) §"Per-row characterised residuals at Phase E closure" — reference table for when training diagnostics implicate a specific mechanism.

Phase E-1 / E-1b checkpoints are available as supporting reading if a Session F question touches the dielectric-configuration or mechanism-decomposition history:

- [`SESSION_E1_CHECKPOINT.md`](SESSION_E1_CHECKPOINT.md) — Phase E-1 report (DEV-1b-004 / DEV-1b-005 landing + mechanism-decomposition finding).
- [`SESSION_E1B_CHECKPOINT.md`](SESSION_E1B_CHECKPOINT.md) — Phase E-1b report (DEV-1b-007 landing + per-row residuals).

## Session F scope (one-page summary; canonical text in SESSION_PLAN.md)

**Build:**

1. PyTorch composite loss: `L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds` (no `L_prior` per DEV-1b-001 / DEV-1b-002). `L_physics = MSE(σ°_VV_pred, VV_obs) + MSE(σ°_VH_pred, VH_obs)` per SPEC §8; single shared λ₁.
2. PINN-MIMICS trainer using shared PhysicsNet + CorrectionNet backbone from Phase 1 with v0.1 MIMICS forward in the PhysicsNet graph.
3. λ search over 64-combination grid `(λ₁, λ₂, λ₃) ∈ {0.01, 0.1, 0.5, 1.0}³` with SPEC §9 stricter dominance criterion (primary: L_data largest single term; secondary: L_physics > 10% of total). Fallback and halt clauses per SPEC §9 — **do not** fall back to the Phase 1 procedure.
4. Seeds: `SEED = 42 + config_idx`, identical seeds across baselines (re-run at Session A) and PINN-MIMICS. Validation = stratified 20% held-out split within training pool. **Never the sealed test set.**
5. SPEC §11 Phase 4 diagnostic scaffolding — path ready to execute on 4×10 factorial outputs (Block 3).

**Ship:**

- Unit tests for composite loss and trainer; full suite stays at 234+ passing, zero regressions.
- Session F log entry appended to `SESSION_PLAN.md` at session close.
- DEV entry if the λ-search fallback is invoked, or if any v0.1 physics is modified.

**Do not:**

- Run the 4×10 factorial (Block 3).
- Promote v0.1 physics without training-diagnostic evidence.
- Audit the cross-document corpus (White / Yellow / Green / Pitch Deck) — user handles separately.

## Key artefacts referenced in Session F

| Artefact | Purpose |
|---|---|
| `echo-poc/phase1b/physics/mimics.py` | Differentiable v0.1 MIMICS forward; use in PhysicsNet graph. |
| `echo-poc/phase1b/physics/oh1992_learnable_s.py` | Oh 1992 surface scattering with learnable `s`. |
| `echo-poc/shared/pinn_backbone.py` | PhysicsNet + CorrectionNet backbone (frozen Tier 2; adapt into a Phase 1b trainer without modifying). |
| `echo-poc/phase1/pinn_trainer.py` | Reference trainer from Phase 1 (read-only Tier 1; use as structural reference for the Phase 1b trainer). |
| `echo-poc/data/splits/configs/config_*.json` | 40 pre-generated config JSON files — use these for the 4×10 factorial structure even though Session F only runs the λ-search subset (10 × 100% validation configs per SPEC §9). |
| `echo-poc/data/splits/test_indices.json` | **Read-only sealed test set.** Not accessed during training or λ search. |

## Deferred Session F items (unchanged from DEV-1b-008)

These are registered diagnostic threads that are NOT part of Session F's composite-loss-and-trainer scope but ARE evidence-gate candidates if Phase 1b training diagnostics implicate them:

- **(a) Simplified power-law Dobson vs full Dobson 1985** — evidence gate: if Dobson-vs-Mironov sensitivity arm (SPEC §11, already registered) shows dielectric choice is binding at retrieval.
- **(b) Oh 1992 vs physical-optics surface scattering** — evidence gate: if parameter correlation matrix (Diagnostic B) shows `s` non-identifiability structurally tied to the Oh empirical form.
- **(c) Rayleigh+sinc² vs UMF finite-cylinder form factor** — evidence gate: if residual-NDVI correlation (Secondary 4) is high AND the canopy-direct mechanism is the implicated failure mode.
- **(d) Real-only UEL vs dual-dispersion UEL** — evidence gate: if Diagnostic A cross-pol forward fit is asymmetric (VH r high, VV r low).
- **(e) √σ°_oh vs literal Fresnel Γ** — evidence gate: if crown-ground mechanism residual is dominant after (a)–(d) addressed, or if a θ=50° VV Brewster-region failure recurs at training time.

Session F should run training on v0.1 and let the diagnostics speak.

## Session F done-when

Same as SESSION_PLAN.md §"Session F done-when":

- Composite loss + PINN-MIMICS trainer implemented.
- λ search over 64-combination grid executed; selected combination meets SPEC §9 dominance or fallback documented in a new DEV entry.
- SPEC §11 diagnostic scaffolding ready (not executed — that's Block 3).
- Full unit suite at 234+ passing, zero regressions.
- Session log entry appended to `SESSION_PLAN.md`.

---

**Session F start:** TBD.
**Session F end:** When the checklist above is green and a Block 3 handoff is written into `SESSION_PLAN.md`.
