# Session G Close Notes — Phase 1b Adjudication-of-Outcome

**Vantage Phase 1b · Session G · Executor-side Close Notes**

*Version 1.0 — 27 April 2026 · Author: executor (CC), pending supervisor cross-check.*

This is the executor-side handoff document for Session G close. Its function is to make explicit what persists where after the Session G consolidated commit + tag + push, and to enumerate the supervisor-side artefacts that require separate persistence handling. It is the operational artefact that closes the cross-environment-prompt-drift sub-observation's mitigation loop ([Block A §6.3](../poc_results_phase1b.md) sub-observation 4).

This document is sibling to the existing decisions records at [`U-1-g2-anchor-construction.md`](U-1-g2-anchor-construction.md), [`U-3-ndwi-formulation.md`](U-3-ndwi-formulation.md), and [`U-4-ndwi-mg-mapping.md`](U-4-ndwi-mg-mapping.md) but distinct in role: the U-* documents record specification decisions; this document records session-close handoff state.

---

## §1. Session G summary

**Session G scope.** Session G is the four-block adjudication-of-outcome session pre-authored at the F-2b kickoff decision tree (`phase1b/SESSION_F2B_CHECKPOINT.md` §7) and authorised under the F-2b HALT branch fire. The session draws together the Phase 1b conclusion in writing without amending pre-registration, modifying code, or re-running experiments. The four blocks are: Block A — Phase 1b results document; Block B — cascade plan scoping; Block C — SPEC §17 Phase 1b Conclusion append; Block D — these close notes. The pre-registered evaluation path was honoured throughout: no sealed-test-set unsealing; no Phase 1c authorisation; no tolerance loosening.

**Block A–D deliverables in summary.** All four blocks locked at supervisor cross-check sign-off prior to this Block D draft.

| Block | File path | Status | Line count |
|---|---|---|---:|
| A | [`phase1b/poc_results_phase1b.md`](../poc_results_phase1b.md) | Locked at A1–A12 supervisor amendments | 421 |
| B | [`phase1b/cascade_plan_session_g.md`](../cascade_plan_session_g.md) | Locked at R1–R3 supervisor amendments | 260 |
| C | [`SPEC.md`](../../SPEC.md) §17 (append-only) | Locked at A1 supervisor amendment | 763 (full SPEC.md, post-append) |
| D | [`phase1b/decisions/session_g_close_notes.md`](session_g_close_notes.md) | This document; pending supervisor cross-check | (final count at draft close) |

**SPEC §17 amendment summary.** §17 Phase 1b Conclusion appended between §16 References (closes line 617 in pre-Block-C state) and the historical closing footer (now lines 759–761) per supervisor-confirmed option (b) structural placement. Pre-Block-C SPEC.md was 623 lines; post-Block-C is 763 lines (140 line additions). All §1–§16 content byte-identical to pre-Block-C state per Block C anchor §1 (append-only). The historical "Draft for Sign-Off · April 2026" footer is preserved verbatim and now sits between §17 and end-of-file as a deliberate audit-trail artefact (layered audit trail: pre-§14 baseline → §14 sign-off 2026-04-19 → §15 pre-registration metadata 2026-04-19 → §17 Phase 1b Conclusion 2026-04-27).

---

## §2. Per-deliverable persistence ledger

The ledger below enumerates every Session G artefact with its persistence path, persistence mechanism, persistence responsibility, and status at session close. It is the central deliverable of this document: the persistence ledger is the artefact-persistence-drift mitigation loop's closure mechanism.

| # | Artefact | Path | Persistence mechanism | Responsibility | Status at session close |
|---:|---|---|---|---|---|
| 1 | Block A — Phase 1b results document | [`phase1b/poc_results_phase1b.md`](../poc_results_phase1b.md) | Consolidated commit + tag + push at Session G close | CC | Locked, committing |
| 2 | Block B — Phase 1b cascade plan | [`phase1b/cascade_plan_session_g.md`](../cascade_plan_session_g.md) | Consolidated commit + tag + push at Session G close | CC | Locked, committing |
| 3 | Block C amendment — SPEC §17 Phase 1b Conclusion | [`SPEC.md`](../../SPEC.md) §17 (append-only insertion) | Consolidated commit + tag + push at Session G close | CC | Locked, committing (with A4 single-line amendment to §17.6 conclusion-commit reference applied at close — see §3 and §4 below) |
| 4 | Block D — Session G close notes | [`phase1b/decisions/session_g_close_notes.md`](session_g_close_notes.md) (this document) | Consolidated commit + tag + push at Session G close | CC | Locked, committing |
| 5 | Decisions log forward extension covering Session G | Web-app-side; not in this repo | Supervisor web-app-side action: author the forward extension covering Session G entry-check resolution, Block A–D adjudications, A1/A3/A4 amendments, R1–R3 amendments, and the §7 methodological-observation flag below; upload to Vantage project knowledge | Supervisor | Pending supervisor execution post-Session-G-close |
| 6 | Yellow Paper v3.0.4 → v3.0.5 cascade execution | Web-app-side; not in this repo | Supervisor web-app-side action: author the consolidated cascade revisions per [Block B §3.1](../cascade_plan_session_g.md) edit specs (9 section-level edits including row-9 change-log entry); upload v3.0.5 to project knowledge per [Block B §5](../cascade_plan_session_g.md) persistence reminder | Supervisor | Pending supervisor execution post-Session-G-close |
| 7 | Green Paper v4.1.2 → v4.1.3 cascade execution | Web-app-side; not in this repo | Supervisor web-app-side action: author the consolidated cascade revisions per [Block B §3.2](../cascade_plan_session_g.md) edit specs (8 section-level edits including row-8 change-log entry); upload v4.1.3 to project knowledge per Block B §5 | Supervisor | Pending supervisor execution post-Session-G-close |
| 8 | Pitch Deck v7.1 → v7.2 cascade execution (Slide 9 only) | Web-app-side; not in this repo | Supervisor web-app-side action: author the Slide 9 status-update edit per [Block B §2.3](../cascade_plan_session_g.md); upload v7.2 to project knowledge per Block B §5 | Supervisor | Pending supervisor execution post-Session-G-close |
| 9 | White Paper v11.3 verdict resolution + conditional execution | Web-app-side; not in this repo | Supervisor web-app-side action: ~2-minute verification per [Block B §2.4](../cascade_plan_session_g.md) verification step; if temporal-status language present, upgrade to Editorial Touch with one status-sync edit at Slide-9-equivalent framing per Block A §11; upload revised White Paper to project knowledge if upgrade triggered | Supervisor | Pending supervisor execution post-Session-G-close (conditional outcome) |

**Responsibility split:** 4 CC-side rows (rows 1–4) committing in the consolidated commit; 5 supervisor-side rows (rows 5–9) queued for post-Session-G-close web-app-side execution. The CC-side rows persist immediately at commit + tag + push; the supervisor-side rows persist via deliberate web-app-side authoring + project knowledge upload, per the artefact-persistence-drift discipline anchor inherited from Block A §6.3 sub-observation 3.

---

## §3. Session G consolidated commit specification

### §3.1 Files staged

The consolidated commit stages five paths:

1. [`phase1b/poc_results_phase1b.md`](../poc_results_phase1b.md) — Block A (new file).
2. [`phase1b/cascade_plan_session_g.md`](../cascade_plan_session_g.md) — Block B (new file).
3. [`SPEC.md`](../../SPEC.md) — Block C amendment (modified file; append-only +140 lines).
4. [`phase1b/decisions/session_g_close_notes.md`](session_g_close_notes.md) — Block D (this document, new file).
5. (After A4 single-line amendment to §17.6 conclusion-commit reference is applied — see §3.3 below.) `SPEC.md` will be re-staged with the amendment.

### §3.2 Commit message convention

Parallel to the F-2b close commit (`1f4bfdf`: "Phase 1b F-2b close: Tier 3 HALT under joint VV+VH (DEV-1b-010); 64/64 grid coverage; checkpoint + artefacts"). Proposed commit message:

```
Phase 1b Session G close: HALT-finding adjudication-of-outcome
(Block A results doc + Block B cascade plan + Block C SPEC §17
conclusion + Block D close notes)

Session G drew together the Phase 1b conclusion in writing per the
F-2b HALT-branch decision tree. Four blocks landed:

- Block A: phase1b/poc_results_phase1b.md (421 lines) — the canonical
  Phase 1b results document. Tier 3 HALT outcome; sealed test set
  unchanged; five publishable contributions (DEV-1b-008, -009 with
  F-2b complement, supervisor-executor workflow with four sub-
  observations, -010, magnitude-balance saturation finding).

- Block B: phase1b/cascade_plan_session_g.md (260 lines) — cascade
  plan scoping for Yellow + Green substantive, Pitch Deck Slide 9
  editorial, White Paper conditional, six other papers no update.
  Sequential version-numbering recommended; supervisor executes
  cascade web-app-side post-close.

- Block C: SPEC §17 Phase 1b Conclusion append (+140 lines; total
  SPEC.md 763 lines). Append-only; §1–§16 byte-identical to pre-
  Block-C state. Six subsections including framing-inversion (§17.0)
  parallel to §14 sign-off, magnitude-balance binding framing
  (§17.4) verbatim with three-part is/is not/does not preclude
  framing, and conclusion sign-off (§17.6).

- Block D: phase1b/decisions/session_g_close_notes.md — executor-
  side close notes with persistence ledger covering both CC-side
  and supervisor-side artefacts.

Sealed test set unchanged from Phase 1; SHA-256
a4b11206630cc80fc3e2ae5853bb114c7a4154072375654c257e51e4250f8eea
remains stable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### §3.3 Pre-commit verification steps

Before staging:

1. **A4 single-line amendment to SPEC.md §17.6 applied.** Replace `Conclusion commit:                             [to be set at Session G close consolidated commit]` with `Conclusion commit:                             tagged at \`phase1b-concluded-halt-finding\`` (per A4 supervisor-adjudicated option (ii) tag-based reference). Verify by `grep` after edit.
2. **A3 lead-investigator placeholder unchanged.** Verify `Matthew Denyer / 2026-04-27` stands as authorised at supervisor adjudication option (a). No amendment.
3. **A1 independent-reviewer attribution unchanged.** Verify `Claude (Opus 4.7), web-app supervisor session, 2026-04-27` stands per supervisor cross-check approval.
4. **Line counts re-verified.** Block A: 421. Block B: 260. SPEC.md: 763 (or 763 ± 0 after A4 — A4 swap is a same-line-count single-token amendment). Block D: final draft count.
5. **No unintended diffs in working tree.** `git diff --stat` and `git status` should show only the four CC-side deliverables + SPEC.md amendment as Session G changes; pre-existing items (`outputs/g2_equivalence_result.json` stale-timestamp diff; `phase1b/lambda_search/results_f2b/per_rep_histories/` untracked directory) are NOT staged in the consolidated commit and remain in their pre-existing state. See §halts-and-flags below for the per_rep_histories side-flag.

### §3.4 Tag specification

Tag: `phase1b-concluded-halt-finding` — outcome marker; semantically informative for tag-history lookups. Final tag selection at supervisor confirmation per [Block A footer](../poc_results_phase1b.md), [Block B §4](../cascade_plan_session_g.md), and [SPEC §17.6](../../SPEC.md). Tag points to the Session G consolidated commit. Tag carries an annotation referencing Block A as the canonical results document.

### §3.5 Push specification

Push tag and commit to `origin` per the external-tag-push audit-permanence rule referenced at F-2b close (`SESSION_F2B_CHECKPOINT.md` §10). Single `git push` followed by `git push origin phase1b-concluded-halt-finding` (or atomic `git push --follow-tags` per project convention). Push is the persistence boundary for CC-side artefacts: post-push, the four CC-side ledger rows are persistent in the origin remote and are no longer at risk of ephemeral-storage loss.

### §3.6 Authorisation note

Per Session G discipline anchor 7, no commits / tags / pushes during the draft phase. Session G consolidated commit + tag + push is authorised at the Session G close boundary, after Block D supervisor cross-check sign-off. The commit + tag + push sequence is the only Session-G-authorised git mutation; no intermediate commits.

### §3.7 Deliberately not staged at Session G consolidated commit

Two pre-existing working-tree items are explicitly **not** staged at the Session G consolidated commit. Recording the non-action here for audit-trail completeness — anyone reviewing the Session G commit will see these items in `git status` post-commit and benefit from knowing they were considered and intentionally excluded.

- **`outputs/g2_equivalence_result.json`** — stale G2 timestamp diff, pre-existing from before Session G (visible in the session-start `git status` snapshot per the Session G environment block). Deferred per supervisor instruction at Block B start. Not Session-G-introduced; not a Session-G-authorised stage.
- **`phase1b/lambda_search/results_f2b/per_rep_histories/`** — F-2b training-history output, untracked symmetric with F-2 v1 precedent (the F-2 v1 commit `5a2a994` also left its per_rep_histories untracked; the F-2b commit `1f4bfdf` inherited that convention). Adjudicated at §8.1 below — Option (iii) leave as-is, untracked. The `.gitignore` housekeeping question is deferred to Phase 1c schema-improvements work, not a Session G action.

The §3.7 enumeration converts the §3 commit specification from "what gets staged" to "what gets staged and what doesn't" — completing the working-tree accounting at session close.

---

## §4. A3 / A4 supervisor adjudications — both resolved

Both A-items raised during Block C cross-check are resolved at supervisor adjudication during this Session G chat. Recorded here for the audit trail.

- **A3 — Lead-investigator signature in SPEC §17.6 (resolved at option (a)).** Matthew authorised the placeholder `Matthew Denyer / 2026-04-27` to stand as the formal lead-investigator sign-off. No amendment required at Session G consolidated commit; placeholder is the sign-off.

- **A4 — Conclusion-commit reference in SPEC §17.6 (resolved at option (ii)).** Tag-based reference selected per CC's Block C recommendation. At Session G close, the placeholder `[to be set at Session G close consolidated commit]` is replaced with `tagged at \`phase1b-concluded-halt-finding\`` (or supervisor-adjudicated final tag name) before the consolidated commit is staged. Single-line amendment; no second-pass commit required. The amendment is part of the consolidated commit, not a separate commit.

**One operational action remains** for CC at session close: the A4 single-line SPEC.md §17.6 amendment per §3.3 step 1 above. This action does not block Block D lock; it is staged as part of the consolidated commit pre-commit verification.

No outstanding adjudications block the consolidated commit.

---

## §5. Web-app-side handoff (supervisor responsibility)

The five supervisor-side ledger rows (§2 rows 5–9) require deliberate post-Session-G-close action by the supervisor. Reminders below are organised per ledger row.

### §5.1 Decisions log forward extension covering Session G (ledger row 5)

The decisions log lives web-app-side per the Session G entry-check (a) bidirectional-architecture confirmation. The forward extension covering Session G should record, at minimum: the Session G entry-check resolution (four blocking divergences resolved per supervisor relay); Block A–D adjudications and amendments (A1–A12 on Block A; R1–R3 on Block B; A1 on Block C; A3 / A4 resolutions); the §7 methodological-observation flag below; the cascade-execution operational notes from Block B §5. Persistence mechanism: supervisor authoring web-app-side + explicit upload to Vantage project knowledge per the artefact-persistence-drift sub-observation. The §13.7 prescription from the F-2b log extension binds: persistence is explicit, not implicit; the upload step closes the persistence loop for the decisions log.

### §5.2 Yellow Paper v3.0.4 → v3.0.5 cascade execution (ledger row 6)

[Block B §3.1](../cascade_plan_session_g.md) carries the deterministic edit list — nine section-level edit specifications spanning methodology (rows 1–3), results (rows 4–5), methodological contributions (row 6), limitations (row 7), F-2 v1 numeric retraction sweep (row 8), and change-log entry (row 9). Each row cites a Block A § as binding source. Persistence mechanism: supervisor web-app-side authoring of v3.0.5 against the actual Yellow Paper file + project knowledge upload per Block B §5. Sequential version increment per Block B §4 supervisor-confirmed.

### §5.3 Green Paper v4.1.2 → v4.1.3 cascade execution (ledger row 7)

[Block B §3.2](../cascade_plan_session_g.md) carries the eight-row edit list spanning four-phase research programme (rows 1–2), Phase 1c plan (row 3), publication strategy §8 venue inheritance (row 4), commercial-thesis framing (row 5), methodological-contributions inheritance (row 6), F-2 v1 numeric retraction sweep (row 7), and change-log entry (row 8). Cross-paper note from Block B §3.2 binds: Green Paper inherits brief reference to the five-contribution catalogue at programme-level granularity; full per-contribution treatment lives in Yellow Paper. Persistence mechanism: same as Yellow.

### §5.4 Pitch Deck v7.1 → v7.2 cascade execution (ledger row 8)

[Block B §2.3](../cascade_plan_session_g.md) specifies a single-slide edit on Slide 9: replace "Phase 1b underway" framing with "Phase 1b concluded on the HALT finding plus five publishable contributions; Phase 1c separately scoped" per Block A §11 calibrated tone; if Slide 9 carries a sub-bullet on Phase 1b expected outcome, replace with the configuration-specific HALT framing per Block A §5.3. Estimated effort: Low. Persistence mechanism: supervisor web-app-side authoring + project knowledge upload.

### §5.5 White Paper v11.3 verdict resolution + conditional execution (ledger row 9)

[Block B §2.4](../cascade_plan_session_g.md) prescribes the ~2-minute verification step: read the White Paper Phase 1b reference, determine whether it is purely structural ("Phase 1b is part of the four-phase research programme; see Green Paper for detail" — no temporal language) or contains temporal-status language ("Phase 1b is in progress", "Phase 1b will report at N≈25", "Phase 1b expected outcome", or similar). Branch logic:

- **Purely structural:** verdict stands at No Update Required. No action.
- **Temporal-status language present:** verdict upgrades to Editorial Touch. One status-sync edit at Slide-9-equivalent framing per Block A §11. Upload revised White Paper to project knowledge per Block B §5.

The conditional structure pushes verification to where it can happen — supervisor has read access to White Paper v11.3 and can resolve the verdict in approximately 2 minutes. Block B §7.1 records this as a resolved verdict (not a halt-and-flag) per the Block B sign-off conditional adjudication.

---

## §6. Phase 1c readiness statement

Phase 1c open questions are recorded at [Block A §7](../poc_results_phase1b.md) and [SPEC §17.3](../../SPEC.md) as Phase-1c-scope candidates, **not** as Phase 1b extensions or Phase 1c pre-registration commitments. Phase 1c requires its own pre-registration sign-off cycle following the Phase 1b discipline (§14 sign-off + §15 pre-registration metadata + companion document tagged-at-lock + technical regression test). No Phase 1c work is authorised by Session G; Session G concludes Phase 1b on the HALT finding plus five publishable contributions and inherits four open questions forward as Phase-1c-scope candidates only. The audit-trail-strengthening proposals at SPEC §17.5 are similarly inherited as Phase 1c result-JSON schema requirements, not as Phase 1b deliverables.

---

## §7. Methodological-observation extension flag (for supervisor's decisions log forward extension)

This section flags one or more candidate methodological observations surfaced during Session G that may belong in the supervisor's decisions log forward extension as either (i) empirical complements to existing Block A §6.3 sub-observations or (ii) new sub-observations in their own right. Block D records the flag without pre-empting the adjudication; characterisation of the observations is decisions-log scope, not Block D scope.

### §7.1 SPEC.md path-prefix error (Session G entry check)

Pattern observed during Session G entry check: the inherited decisions log referenced `phase1b/SPEC.md` throughout (in the supervisor's relayed §§6.1–§17.3 source-of-truth chain), but the file actually lives at `SPEC.md` at the repo root. The pre-Block-C verification step ("read against the locked tag `phase1b-session-f2b-lambda-selected`") was the audit that surfaced the path-prefix error; the (b) grep-adjudication request was the executor-side halt-and-flag that bound the path correction.

This pattern is similar in shape to Block A §6.3 sub-observation 4 (cross-environment prompt drift) — a discipline gap surfaced through bidirectional verification at a session-boundary cross-check. It may be either:

- **An empirical complement to sub-observation 4.** The Session G pattern reinforces the existing observation with a second instance of cross-environment-induced reference drift (Session G Round 1 was the macOS-vs-web-app filesystem mounts; this is the path-prefix-inherited-from-decisions-log instance). Both are inherited-supervisor-side artefacts that diverge from the executor-side filesystem state. Treating as an empirical complement preserves a tight conceptual scope for the sub-observation.
- **A fifth sub-observation in its own right.** If the path-prefix-inherited-from-decisions-log pattern is structurally distinct from the cross-environment-mount-assumption pattern (the former is a within-supervisor session-to-session inheritance issue; the latter is a between-environments mount-mismatch issue), separate naming may be warranted. The decisions log forward extension is the appropriate venue to make this call.

### §7.2 Block D kick-off prompt truncation (mid-Session-G)

Pattern observed during Block D kickoff: the original Block D kick-off prompt was truncated mid-sentence at §7. The executor-side halt-and-flag surfaced the truncation; supervisor re-sent the complete prompt on confirmation; Block D drafting proceeded from the complete prompt. The mitigation was bidirectional — executor halted rather than filling the gap; supervisor re-sent rather than asking the executor to infer.

This pattern may be:

- **The same sub-observation as §7.1.** Both are executor-side halt-and-flag instances on incomplete or inherited supervisor artefacts (path-prefix error in §7.1; prompt truncation in §7.2). Conceptually unified as "executor-side halt-and-flag on inherited supervisor artefact whose completeness or correctness cannot be verified silently."
- **A distinct sub-observation in its own right.** Prompt-completeness verification as a discipline anchor in its own right — the prompt is the supervisor's authoring artefact; the executor's discipline is to verify completeness before draft begins, not to infer from truncated text. This pattern could generalise beyond Phase 1b (any future supervisor-executor session benefits from explicit prompt-completeness verification).

### §7.3 Supervisor adjudication required

The decisions log forward extension is the appropriate venue for supervisor adjudication on whether §7.1 and §7.2 are:

- Two empirical complements to existing Block A §6.3 sub-observation 4.
- One fifth sub-observation (combining §7.1 and §7.2 under a single new heading).
- Two distinct fifth and sixth sub-observations.
- Sub-observations to a different existing sub-observation (e.g. §6.3 sub-observation 1 — state-snapshot freshness — has structural resonance with both).

Block D records the flag for supervisor consideration only; Block D does not author the methodological-observation extension itself. Per Block D anchor §1 (operational-only scope), characterisation of these observations is decisions-log scope, not Block D scope.

---

## §8. Halts-and-flags surfaced during drafting

One side-flag surfaced during §3 drafting; one informational note on §7 acknowledged.

### §8.1 `phase1b/lambda_search/results_f2b/per_rep_histories/` directory — adjudicated (Option iii confirmed)

**Block-D §3 trigger fired** on the working-tree state during Block D drafting. The directory `phase1b/lambda_search/results_f2b/per_rep_histories/` is present in the working tree as untracked content but is neither one of the four Session G CC-side deliverables, nor the SPEC.md amendment, nor the previously-acknowledged stale `outputs/g2_equivalence_result.json` timestamp diff.

Origin: the directory contains per-rep training histories from the F-2b λ-search run (2026-04-20 to 2026-04-22) and was deliberately left untracked at F-2b close (commit `1f4bfdf`) — symmetric with the F-2 v1 commit (`5a2a994`) which also left its per_rep_histories untracked. The directory is pre-existing in the Session-G-start working tree and is not introduced by Session G work.

Three options surfaced for supervisor adjudication:
- **(i)** Stage in consolidated commit.
- **(ii)** Add to `.gitignore` to make its untracked status explicit.
- **(iii)** Leave as-is, untracked, for the Session G consolidated commit.

**Adjudication at Block D supervisor cross-check (2026-04-27): Option (iii) — leave as-is, untracked.** Three supervisor-recorded reasons:

1. **F-2 v1 precedent applies.** F-2 v1's per_rep_histories directory was also left untracked at F-2 v1 close (commit `5a2a994`); the F-2b commit (`1f4bfdf`) inherited that convention by leaving F-2b's per_rep_histories untracked. Adding F-2b's per_rep_histories at Session G close would break the F-2b commit's audit-trail symmetry retroactively (the F-2b commit's stated artefact list excluded it; adding it now would create a "what changed?" forensic question for future auditors that has no clean answer).

2. **Per-rep histories are debug-grade artefacts, not authoritative.** The authoritative F-2b numerical evidence is `lambda_search_f2_result.json` (which IS tracked at the F-2b commit). The per_rep_histories are intermediate per-rep training trajectories useful for debugging but not for tier classification or the magnitude-balance finding. The integrity audit's three-route triangulation (deviation log + test suite + empirical fingerprint) does not depend on per_rep_histories. Tracking them adds storage cost without strengthening the audit.

3. **Conflation hazard.** Staging F-2b artefacts in a Session G commit would conflate two separate sessions' deliverables. Session G is adjudication-of-outcome scope; F-2b is λ-search execution scope. The clean separation is preserved by leaving the directory as-is.

**`.gitignore` housekeeping question (Option ii) deferred to Phase 1c schema-improvements work.** Adding `phase1b/lambda_search/results_*/per_rep_histories/` (or equivalent glob) to `.gitignore` is a separate housekeeping decision that belongs with Phase 1c result-JSON schema requirements per [SPEC §17.5](../../SPEC.md) — specifically as a complement to schema requirement 1 (code-version hash in result-JSON metadata) and schema requirement 3 (pre-flight summary block). Not a Session G action.

The adjudication is recorded here as resolved rather than left as a halt-and-flag. The directory remains untracked at Session G consolidated commit per Option (iii).

### §8.2 §7 methodological-observation extension flag (informational, not blocking)

The §7 flag is by design a flag for the supervisor's decisions log forward extension; it is not a halt-and-flag in the discipline-anchor sense (it does not block Block D lock). Recording in halts-and-flags for visibility because the flag's content depth (two candidate observation patterns; multiple sub-observation classification options) approaches the boundary of Block D anchor §1 (operational-only scope; no introduction of new framing). Block D explicitly does not characterise the observations beyond identifying their candidate status — the characterisation is decisions-log scope. If supervisor reads §7 as exceeding operational-only scope, easy single-Edit pruning at sign-off.

---

*Vantage · Phase 1b · Session G · Executor-side Close Notes · v1.0 · 2026-04-27 · Author: executor (CC), pending supervisor cross-check.*
