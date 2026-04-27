# Phase 1b Session G — Cascade Plan (scoping)

**Vantage Phase 1b · Session G · Cascade Plan**

*Version 1.0 — 27 April 2026 · Author: executor (CC), pending supervisor sign-off.*

This is a **scoping document**, not a deliverable. It specifies which sections in which Vantage corpus papers need what edits to reflect Phase 1b's HALT outcome and five publishable contributions. The corpus papers live web-app-side; the supervisor executes the actual paper edits in a separate session against the actual paper files. This document gives the supervisor a deterministic edit list to work from.

**Binding source-of-truth chain.** Every cascade edit specification in this document references a specific § in [phase1b/poc_results_phase1b.md](phase1b/poc_results_phase1b.md) (Block A) as its framing source. Block A is the canonical Phase 1b results document and the authoritative source for all cascade framing. Cascade edits derive from Block A; they do not invent new framing.

**Corpus baseline (per Session G entry-check (c)).** Yellow Paper v3.0.4; Green Paper v4.1.2; Pitch Deck v7.1; White Paper v11.3; Blue v1.2.2; Orange v1.0; Spectrum v1.1; Skills Coverage v1.1; Carbon Accounting Framework v1.1; MIMICS lit review v1.0.2. Edits specified against these versions. The Session-E `v3.0.5` / `v4.1.3` versions referenced in DEV-1b-008 §"Cross-document consistency follow-up" do not exist on disk per the artefact-persistence drift sub-observation in Block A §6.3.

---

## §1. Cascade overview

The Phase 1b cascade affects **two corpus papers substantively** (Yellow Paper v3.0.4; Green Paper v4.1.2), **one editorially** (Pitch Deck v7.1, Slide 9 only), and **seven not at all** (White Paper v11.3; Blue v1.2.2; Orange v1.0; Spectrum v1.1; Skills Coverage v1.1; Carbon Accounting Framework v1.1; MIMICS lit review v1.0.2). The substantive cascade represents a single consolidated jump per paper that incorporates two prior cascade-trigger events into one edit pass: (i) the deferred Session-E G2 Moderate Pass cascade implied by DEV-1b-008's cross-document consistency follow-up (Yellow / Green / White / Pitch Deck flagged at Phase E closure on 2026-04-19; cascade outputs lost to ephemeral storage and never persisted, hence Yellow remains at v3.0.4 and Green at v4.1.2 rather than the v3.0.5 / v4.1.3 referenced in earlier decisions-log entries), and (ii) the Phase 1b conclusion cascade triggered by the F-2b Tier 3 HALT outcome and the five publishable contributions per Block A. The "one consolidated jump per paper, not two" framing in the Session G kick-off prompt names this consolidation: the supervisor's execution-side cascade pass should fold both the G2 Moderate Pass methodology updates and the F-2b HALT conclusion updates into a single revision, rather than performing two sequential passes.

---

## §2. Per-paper cascade scope

### §2.1 Yellow Paper v3.0.4 (Technical Paper)

**Cascade verdict:** Substantive Edit.

**Rationale.** The Yellow Paper is the technical paper. Per DEV-1b-008 §"Cross-document consistency follow-up" it carries the Phase 1b methodology section that references G2 as a pre-registration arm; per DEV-1b-010 §"Cascade implications" it references Phase 1b F-2 results, with specific F-2 v1 numerics (median val loss 0.0353; physics fraction 0.435; selected λ=(0.01, 0.01, 0.10)) potentially appearing in technical-detail sections. Both cascade triggers land on this paper; both are folded into the Substantive Edit pass.

**Section-level edit specifications:** see §3.1 below for the deterministic edit list.

**Estimated effort:** High. Eight section-level edits spanning methodology, results, methodological-contributions, Phase 1c forward-look, limitations, and an F-2 numeric retraction sweep.

### §2.2 Green Paper v4.1.2 (Research Programme)

**Cascade verdict:** Substantive Edit.

**Rationale.** The Green Paper is the research-programme paper. Per DEV-1b-008 §"Cross-document consistency follow-up" it carries the four-phase research programme that references Phase 1b gates including G2; per DEV-1b-010 §"Cascade implications" it references Phase 1b F-2 results in the same potentially-numeric way as Yellow. Block A §9 references "Green Paper §8 publication strategy section" as the venue-scoping source — the venue options A/B/C in Block A §9 inherit there. Both prior cascade triggers and the venue-scoping inheritance land on this paper.

**Section-level edit specifications:** see §3.2 below.

**Estimated effort:** High. Seven section-level edits spanning research-programme overview, Phase 1b status, Phase 1c plan, publication strategy §8, commercial-thesis framing, programme conclusions, and an F-2 numeric retraction sweep.

### §2.3 Pitch Deck v7.1

**Cascade verdict:** Editorial Touch.

**Rationale.** Per Session G entry-check (c), Slide 9 carries the status framing "Phase 1b underway", which is now temporally stale post-F-2b. The remainder of the deck has no Phase 1b execution-detail references at slide-content granularity. The edit is a status update on a single slide; not Substantive because no methodology, no architecture, and no five-contribution framing belongs in pitch-deck depth.

**Section-level edit specifications:** Slide 9 status text — replace "Phase 1b underway" framing with the "Phase 1b concluded on the HALT finding plus five publishable contributions; Phase 1c separately scoped" framing per Block A §11. The single-slide framing should mirror Block A §11's calibrated tone — neither overclaiming the contribution as a thesis validation nor underclaiming it as a setback. If Slide 9 also carries a sub-bullet on "Phase 1b expected outcome" or similar, that sub-bullet should be replaced with the configuration-specific HALT framing per Block A §5.3.

**Estimated effort:** Low. One slide; one status update plus optional sub-bullet replacement.

### §2.4 White Paper v11.3

**Cascade verdict:** **Conditional.** Default **No Update Required**; upgrade to **Editorial Touch** if temporal-status language about Phase 1b is present in the White Paper.

**Verification step (supervisor, web-app side, approximately 2 minutes).** Read the White Paper v11.3 Phase 1b reference. Determine whether the reference is purely structural (e.g. "Phase 1b is part of the four-phase research programme; see Green Paper for detail" — no temporal language) or whether it contains temporal-status language ("Phase 1b is in progress", "Phase 1b will report at N≈25", "Phase 1b expected outcome", or similar).

**If purely structural:** verdict stands at No Update Required. The high-level reference can stand without substantive edit; detailed framing of the HALT outcome lives in Yellow + Green at depths matching their granularity. The DEV-1b-008 cross-doc list inheritance is acknowledged but does not force an update at this granularity.

**If temporal-status language present:** upgrade to Editorial Touch. One status-sync edit at Slide-9-equivalent framing per Block A §11 — replace the temporal-status language with "Phase 1b concluded on the HALT finding plus five publishable contributions; Phase 1c separately scoped" or equivalent calibrated framing. No deeper edit; no new sections introduced.

**Estimated effort:** None (default) or Low (if upgrade triggered). Not Medium or High under any reading.

### §2.5 Blue Paper v1.2.2

**Cascade verdict:** No Update Required.

**Rationale.** Per Session G entry-check (c), no Phase 1b execution references; no edits expected. Blue Paper sits outside the cascade scope.

**Estimated effort:** None.

### §2.6 Orange Paper v1.0

**Cascade verdict:** No Update Required.

**Rationale.** Per Session G entry-check (c), no Phase 1b execution references; no edits expected.

**Estimated effort:** None.

### §2.7 Spectrum Paper v1.1

**Cascade verdict:** No Update Required.

**Rationale.** Per Session G entry-check (c), no Phase 1b execution references; no edits expected. Block A §7 question 4 (L-band SAR generalisation) names the Spectrum Paper indirectly via "the strategic Vantage roadmap"; that reference is outbound from Phase 1b results to the Spectrum Paper, not a claim that Spectrum needs updating in light of Phase 1b. The L-band Phase 1c open question is recorded in Block A and the Green Paper Phase 1c plan section; the Spectrum Paper's L-band-roadmap framing does not require revision to accommodate the Phase 1b finding.

**Estimated effort:** None.

### §2.8 Skills Coverage Paper v1.1

**Cascade verdict:** No Update Required.

**Rationale.** Per Session G entry-check (c), no Phase 1b execution references; no edits expected.

**Estimated effort:** None.

### §2.9 Carbon Accounting Framework Paper v1.1

**Cascade verdict:** No Update Required.

**Rationale.** Per Session G entry-check (c), no Phase 1b execution references; no edits expected. Block A §10 limitation 7 (uncertainty quantification not addressed) is named within Block A and inherited by the Yellow / Green substantive cascades; it does not propagate to the Carbon Accounting Framework, which addresses commercial-mechanism design rather than retrieval-architecture choices.

**Estimated effort:** None.

### §2.10 MIMICS lit review v1.0.2

**Cascade verdict:** No Update Required.

**Rationale.** Per Session G entry-check (c), no Phase 1b execution references; no edits expected. The MIMICS lit review v1.0.2 was the input to Phase 1b's pre-registration (referenced as the source of the §10 secondary-criteria proposals at SUCCESS_CRITERIA.md §0); it is not a downstream document of the Phase 1b conclusion. The lit review's framing of the MIMICS literature is unchanged by the F-2b HALT outcome (which is, per Block A §6.5 binding framing, **not** "MIMICS is inadequate" but rather a composite-loss calibration finding).

**Estimated effort:** None.

---

### §2.x Cascade scope summary table

| Paper | Version | Verdict | Effort |
|---|---|---|---:|
| Yellow Paper | v3.0.4 | Substantive Edit | High |
| Green Paper | v4.1.2 | Substantive Edit | High |
| Pitch Deck | v7.1 | Editorial Touch (Slide 9) | Low |
| White Paper | v11.3 | Conditional: No Update Required default; Editorial Touch if temporal-status language present (supervisor verification at cascade execution time) | None / Low |
| Blue Paper | v1.2.2 | No Update Required | None |
| Orange Paper | v1.0 | No Update Required | None |
| Spectrum Paper | v1.1 | No Update Required | None |
| Skills Coverage | v1.1 | No Update Required | None |
| Carbon Accounting Framework | v1.1 | No Update Required | None |
| MIMICS lit review | v1.0.2 | No Update Required | None |

---

## §3. Section-level edit specifications (Yellow + Green)

This section gives the supervisor a deterministic edit list for the two Substantive cascades. Each row specifies: paper / section / paragraph reference; current framing (one-line); required revised framing (one-line, citing Block A source); Block A binding source (§ + topic); edit category. The supervisor authors the actual replacement prose at execution time per Block-B-specific anchor §2 (scope discipline).

**Edit categories** (per Session G kick-off prompt): structural / framing / numerical / status / venue.

### §3.0 Cascade-execution reference convention (general specification)

For any reference to Phase 1b results in any cascaded paper (Yellow Paper, Green Paper, Pitch Deck Slide 9, and conditionally White Paper per §2.4), the supervisor's actual paper edits should adopt the following stable-reference convention. This specification binds across both substantive cascade papers and is applied uniformly wherever Phase 1b results are referenced.

**Recommended reference convention — Option A: repo-path-plus-tag.** Cite Phase 1b results as `echo-poc/phase1b/poc_results_phase1b.md` at tag `phase1b-concluded-halt-finding` (subject to final tag-name adjudication at Session G close). Stable across time provided the tag persists; readers with repo access can resolve the reference deterministically.

**Alternative considered — Option B: preprint or DOI.** Defer until a preprint or DOI exists for the Phase 1b results document via the venue track adjudicated in Block A §9. Currently nonexistent; would require placeholder framing in the cascade revisions until a preprint or DOI lands.

**Block B recommendation: Option A.** Rationale: the cascade is happening now, before any preprint or DOI exists; placeholder framing in the corpus papers is worse than a stable repo-path-plus-tag reference because placeholder text drifts into staleness without an obvious update trigger. Option A's reference becomes the canonical pointer for all corpus references to Phase 1b's HALT outcome and five publishable contributions until a preprint or DOI supersedes it; at that point a single corpus revision pass updates each repo-path reference to the preprint / DOI in one consolidated cascade, and the persistence-handling discipline per §5 applies to that future cascade as it does to this one.

**Edit category:** structural (cascade-wide reference convention applied uniformly across all Phase 1b results references).

**Block A binding source:** Block A is itself the canonical reference target — no narrower § citation applies because the convention covers all references to Phase 1b results.

This general specification binds the per-paper edit specs in §3.1 and §3.2: any "current framing" that references Phase 1b results without a stable reference convention should be updated to use the repo-path-plus-tag convention; any "required revised framing" that introduces a new Phase 1b results reference should adopt the convention from the outset.

### §3.1 Yellow Paper v3.0.4 — section-level edit list

| # | Section / paragraph reference | Current framing (one-line) | Required revised framing (one-line) | Block A binding source | Edit category |
|---:|---|---|---|---|---|
| 1 | Phase 1b methodology section — G2 implementation gate paragraph | "G2 three-arm forward-equivalence check at ±0.5 dB tolerance" referenced as a single pre-registration arm | Replace with G2 Moderate Pass framing distinguishing implementation-correctness (numpy_port arm at machine precision) from cross-configuration equivalence (published_table arm with characterised residuals traced to five v0.1 sub-module simplifications) per DEV-1b-008 | Block A §6.1 | framing |
| 2 | Phase 1b methodology section — composite-loss paragraph | VV-only or unspecified `L_physics` framing | Specify joint VV+VH `L_physics = MSE(σ°_VV) + MSE(σ°_VH)` per signed SPEC §8; note DEV-1b-010 implementation-vs-text adjudication preserved F-2 v1 tag as superseded audit record | Block A §3.2, §3.4 | framing |
| 3 | Phase 1b methodology section — λ-search dominance constraint paragraph | Phase 1b dominance constraint and three-tier fallback procedure | Add the verbatim SPEC.md:324 honest-interpretation passage as the pre-authored Tier 3 HALT framing; cite the DEV-1b-009 dual-reading binding (mean-across-reps reading) | Block A §3.3, §4.4 | framing |
| 4 | Phase 1b results section — F-2b grid coverage and dominance distribution | Phase 1b retrieval-performance results, factorial-evaluation table, learning-curve framing (if present) | Replace with the F-2b grid coverage / dominance distribution / monotonicity / per-block summary per Block A §4.1–§4.2; explicitly note no factorial evaluation reached, no test-set RMSE reported | Block A §4.1, §4.2 | structural + numerical |
| 5 | Phase 1b results section — magnitude-balance characterisation (new sub-section) | Not present; the magnitude-balance finding is a new Phase 1b empirical contribution | Insert the §14.7 verbatim framing line (VH/VV ≈ 0.645; joint L_physics ~1.65× VV-only; physics_fraction ≥ 0.925 across grid) and both halves of the binding "is" / "is not" framing per Block A §4.5 + §6.5 | Block A §4.5, §6.5 | structural + framing |
| 6 | Phase 1b results section — five publishable contributions (new sub-section) | Not present (or present at lower granularity prior to Phase 1b's contribution accumulation) | Insert the five-contribution catalogue (DEV-1b-008 implementation-correctness vs cross-configuration equivalence; DEV-1b-009 aggregation-rule explicitness + F-2b empirical complement; supervisor-executor entry-check workflow architecture with four sub-observations; DEV-1b-010 post-sign-off implementation-audit gate; magnitude-balance saturation finding) per Block A §6.1–§6.5 | Block A §6 | structural |
| 7 | Phase 1b limitations / honest-gates section | Phase 1b limitations as scoped at sign-off | Update to the seven-item Phase 1b limitations list per Block A §10 (no retrieval-performance evidence; single-site/sensor scope; pre-registered λ grid scope; v0.1 sub-module simplifications; compute-cost operational details; corpus-side cascade outbound; UQ not addressed) | Block A §10 | framing |
| 8 | Phase 1b results section — F-2 v1 numeric retraction sweep | If F-2 v1 numerics appear (median val loss 0.0353; physics fraction 0.435; selected λ=(0.01, 0.01, 0.10); Tier 1 FULL_DOMINANCE classification) per DEV-1b-010 §"Cascade implications" | Retract; replace with F-2 v1's preserved-but-superseded audit-record status and F-2b's Tier 3 HALT outcome under the joint VV+VH formulation per Block A §3.4, §4.3 | Block A §3.4, §4.3 | numerical |
| 9 | Change-log / version-history section — v3.0.5 entry (new) | Existing change-log without the consolidated-cascade entry | Append: "v3.0.5 (2026-04-27) — consolidated cascade reflecting Session E G2 Moderate Pass per DEV-1b-008 plus F-2b HALT outcome and five publishable contributions per Block A. Previous v3.0.5 draft (Session E only) was authored 2026-04-19 but lost to ephemeral storage per the artefact-persistence drift sub-observation; this v3.0.5 supersedes that draft." | Block A §6.3 (sub-observation 3 — artefact-persistence drift) | structural |

The Yellow Paper edit pass also carries a forward-look update — the Phase 1c open questions per Block A §7 should appear in whatever section currently scopes the Phase 1c plan (forward-look, future-work, programme-roadmap, or equivalent). That edit overlaps with Green Paper §3.2 row 3 and is not double-listed here.

### §3.2 Green Paper v4.1.2 — section-level edit list

| # | Section / paragraph reference | Current framing (one-line) | Required revised framing (one-line) | Block A binding source | Edit category |
|---:|---|---|---|---|---|
| 1 | Four-phase research programme section — Phase 1b status paragraph | "Phase 1b in execution" / "Phase 1b underway" / similar status framing | Replace with "Phase 1b concluded on the HALT finding plus five publishable contributions; sealed test set never unsealed; Phase 1c separately scoped" per Block A §11 | Block A §11 | status |
| 2 | Phase 1b status section (or equivalent depth) — outcome framing | Phase 1b expected-outcome scoping (if present) | Replace with the configuration-specific HALT framing per Block A §5.3 ("HALT finding is configuration-specific, not architecture-killing") and §8 ("what Phase 1b validates / does not validate" verbatim binding framing) | Block A §5.3, §8 | framing |
| 3 | Phase 1c plan section (or equivalent forward-look) | Phase 1c plan as previously scoped | Promote the four open questions (per-channel L_physics normalisation; λ grid lower bound; trunk-layer mechanism; L-band SAR generalisation) to Phase 1c scope candidates with the explicit "Phase-1c-scope, not Phase-1b-extension" labelling per Block A §7 | Block A §7 | structural |
| 4 | Publication strategy §8 (per Block A §9 reference) | Existing publication strategy framing | Inherit the three venue options (A — methods venue; B — RSE/IEEE-domain paired contribution; C — two-paper split) per Block A §9 as candidate venues; retain the "options not decision" framing — final venue choice is supervisor adjudication outside this cascade | Block A §9 | venue |
| 5 | Programme overview / commercial-thesis section — Phase 1b position | Phase 1b's role in the commercial thesis | Replace with the calibrated framing per Block A §11 closing — "the Vantage thesis is unaddressed at the retrieval-science level by Phase 1b; the HALT finding does not validate, also does not refute"; explicitly preserve the "neither overclaiming nor underclaiming" tone | Block A §11 | framing |
| 6 | Methodological-contributions inheritance (new sub-section or augmentation) | Not present (or present at Phase-1-only granularity) | Insert reference to the five-contribution catalogue per Block A §6.1–§6.5; brief framing only — full per-contribution treatment lives in Yellow Paper per §3.1 row 6 | Block A §6 | structural |
| 7 | Programme-level F-2 v1 numeric retraction sweep | If F-2 v1 numerics appear (same trigger as Yellow Paper §3.1 row 8) per DEV-1b-010 §"Cascade implications" | Retract; replace with F-2b HALT outcome reference per Block A §3.4, §4.3 | Block A §3.4, §4.3 | numerical |
| 8 | Change-log / version-history section — v4.1.3 entry (new) | Existing change-log without the consolidated-cascade entry | Append: "v4.1.3 (2026-04-27) — consolidated cascade reflecting Session E G2 Moderate Pass per DEV-1b-008 plus F-2b HALT outcome and five publishable contributions per Block A. Previous v4.1.3 draft (Session E only) was authored 2026-04-19 but lost to ephemeral storage per the artefact-persistence drift sub-observation; this v4.1.3 supersedes that draft." | Block A §6.3 (sub-observation 3 — artefact-persistence drift) | structural |

Cross-paper note: the Yellow Paper's methodological-contributions section (§3.1 row 6) and the Green Paper's methodological-contributions inheritance (§3.2 row 6) are at different granularities. Yellow carries the full five-contribution treatment; Green inherits a brief reference. The supervisor's execution should preserve this asymmetry — Green Paper at programme-level granularity should not duplicate Yellow Paper at technical-detail granularity.

---

## §4. Version-numbering convention recommendation

**Two options:**

- **Sequential.** Yellow Paper v3.0.4 → v3.0.5; Green Paper v4.1.2 → v4.1.3; Pitch Deck v7.1 → v7.2. Continues the v3.0.x / v4.1.x / v7.x increment patterns. Treats each cascade as a calibrated update.
- **Jump.** Yellow Paper v3.0.4 → v3.1; Green Paper v4.1.2 → v4.2; Pitch Deck v7.1 → v8.0. Signals the consolidated cascade jump represents a substantive Phase 1b conclusion event with discoverable materiality in tag-history lookups.

**Block B recommendation: Sequential.**

**Reasoning.** Three considerations bear on the choice:

1. **The HALT outcome is configuration-specific, not architecture-killing.** Block A §5.3 and §11 explicitly frame the Phase 1b conclusion as a calibrated finding rather than a major-release event. Sequential numbering preserves that calibrated tone; Jump numbering risks framing the HALT as more dramatic than it is — semantically Jump numbering reads as "major release" which conflicts with Block A's "neither overclaiming nor underclaiming" closing instruction.

2. **The close tag is the discoverability mechanism.** The recommended Session G close tag `phase1b-concluded-halt-finding` is itself an outcome marker; tags are the right place for outcome-level discoverability in tag-history lookups. Version numbers and tags serve different purposes — version numbers track materiality of corpus-content-change at a calibrated granularity, tags mark phase-conclusion events.

3. **Sequential numbering matches the Phase 1 cascade precedent.** The Phase 1 Negative-outcome cascade did not use a Jump increment in the corpus papers (per Block A's reference convention to `outputs/write-up/poc_results.md` v1.0). Phase 1b's HALT outcome is structurally analogous — a pre-registered honest-gates conclusion, not a v0.x → v1.0 commercial milestone. Sequential numbering preserves the structural analogy.

**Counter-argument considered.** Jump numbering would make the consolidated cascade jump (incorporating both the deferred Session-E G2 Moderate Pass cascade and the F-2b HALT cascade) more visually evident in the corpus version history. The artefact-persistence-drift sub-observation in Block A §6.3 already records that the Session-E v3.0.5 / v4.1.3 increments were lost; a Jump increment to v3.1 / v4.2 would prevent any future inheriting-supervisor from confusing the consolidated cascade with another routine update. This is a real consideration; Block B nonetheless recommends Sequential because the Block A binding framing on calibrated tone outweighs the discoverability gain, and the close tag carries the discoverability load instead.

**Final adjudication at supervisor sign-off.** If supervisor adjudicates Jump, the supervisor's reasoning binds and Block B's recommendation is amended. The recommendation here is one supervisor input, not a binding decision.

---

## §5. Persistence handling reminder

Per the artefact-persistence drift sub-observation in Block A §6.3, the cascade execution outputs (the supervisor's actual paper edits at v3.0.5 / v4.1.3 / v7.2 — or v3.1 / v4.2 / v8.0 if Jump is adjudicated) require **deliberate persistence into the Vantage project knowledge after authoring**, not just authoring to ephemeral session storage. The Session-E cascade outputs were lost to ephemeral storage because no persistence step was invoked at authoring close; the inheriting Session F supervisor and the inheriting Session G supervisor both operated on stale corpus-state assumptions for three sessions before the artefact-persistence drift was surfaced at Session G entry check. That drift is now a Phase 1b methodological contribution (Block A §6.3 sub-observation 3); the cascade execution at Session G close is the immediate test of whether the contribution generalises to the supervisor's own discipline.

**Reminder to the supervisor at cascade execution close:**

1. After authoring the v3.0.5 / v4.1.3 / v7.2 (or Jump-equivalent) corpus paper revisions web-app-side, invoke an explicit persistence step uploading the revised papers into the Vantage project knowledge.
2. Verify persistence by listing the project-knowledge corpus state at the next inheriting session's entry check; the current versions should match the cascade outputs, not the v3.0.4 / v4.1.2 / v7.1 baseline.
3. If persistence fails or the project-knowledge state diverges from the cascade outputs, treat that as the same drift class as the Session-E cascade loss — surface as a Session-H entry-check halt-and-flag rather than silently re-authoring.

This persistence handling is the supervisor's web-app-side execution work and is not a deliverable of Block B. The reminder is recorded here so the cascade-plan document is self-contained.

---

## §6. Cascade dependencies

**No inter-paper cascade dependencies are identified.** Each substantive paper (Yellow Paper, Green Paper) and the editorial paper (Pitch Deck) inherits framing from Block A independently. The cascade execution sequence is therefore unconstrained by inter-paper dependencies; the supervisor can author the three cascade revisions in any order.

**Two intra-paper sequencing notes:**

1. **Yellow Paper §3.1 rows 6 and 8 are mutually informative.** Row 6 (insert five-contribution catalogue) and row 8 (F-2 v1 numeric retraction sweep) are both methodology-section edits; the supervisor may find it efficient to author them in adjacent passes, but neither blocks the other.

2. **Green Paper §3.2 row 1 (status update) and row 3 (Phase 1c plan) sit in different sections** but are framing-coherent — both convey that Phase 1b is concluded and Phase 1c is separately scoped. The supervisor should author them with mutual reference to ensure tone consistency, but neither blocks the other.

**One outbound dependency** worth noting: the Yellow Paper's Phase 1c forward-look (§3.1 implicit) and the Green Paper's Phase 1c plan section (§3.2 row 3) both depend on the Phase 1c-scope four open questions per Block A §7. Block A is the binding source for both; no separate Phase 1c-scoping document needs to land first.

---

## §7. Halts-and-flags surfaced during drafting

One borderline-call halt-and-flag and one note for supervisor adjudication.

### §7.1 White Paper verdict — resolution (formalised conditional)

The Session G kick-off prompt's verdict expectations ("Substantive verdicts are expected for Yellow and Green. The Editorial verdict is expected for Pitch Deck. The No Update verdicts are expected for the remaining six.") count to 9 of 10 corpus papers explicitly. The "remaining six" maps to Block B §2.5–§2.10 (Blue, Orange, Spectrum, Skills Coverage, Carbon Accounting Framework, MIMICS lit review) per the supervisor's (c) confirmation that those six papers carry no Phase 1b execution references. The White Paper verdict was therefore not explicitly fixed by the prompt's verdict expectations.

**Resolution (per supervisor adjudication, Block B sign-off, 2026-04-27):** the White Paper verdict is **conditional**. Default verdict: No Update Required. Upgrade to Editorial Touch if the White Paper's high-level Phase 1b reference contains temporal-status language ("Phase 1b is in progress", "Phase 1b will report at N≈25", "Phase 1b expected outcome", or similar). The supervisor (web-app side) verifies at cascade execution time per the verification step in §2.4.

This conditional structure pushes the verification to where it can happen — the supervisor has read access to White Paper v11.3 and can resolve the verdict in approximately 2 minutes; the executor (CC) does not have read access and cannot verify. The conditional structure is therefore not a halt-and-flag deferred for Block C — it is a **resolved verdict** whose final realisation depends on a supervisor-side observation that is cheap to make at the right time. The §2.4 subsection carries the full conditional spec with both branches (purely-structural → No Update; temporal-status → Editorial Touch one-line edit per Block A §11).

### §7.2 Cascade-trigger consolidation note (informational, not blocking)

The "one consolidated jump per paper, not two" framing in the Session G kick-off prompt is honoured. The cascade incorporates both the deferred Session-E G2 Moderate Pass cascade (per DEV-1b-008 §"Cross-document consistency follow-up") and the Phase 1b conclusion cascade (per Block A) into a single edit pass. No paper receives two sequential cascade revisions in this pass.

A second-order observation: if any future cascade trigger (Phase 1c authorisation, Phase 1c results, Phase 2 authorisation, etc.) lands before the Session G cascade is persisted into the project-knowledge corpus, the next cascade may inherit a third-order consolidation. The persistence reminder in §5 is the operational mitigation; the artefact-persistence discipline is the methodological mitigation.

This is informational, not a halt-and-flag.

---

*Vantage · Phase 1b · Session G · Cascade Plan · v1.0 · 2026-04-27 · Author: executor (CC), pending supervisor cross-check on cascade scope, version-numbering convention, and §7 halts-and-flags.*
