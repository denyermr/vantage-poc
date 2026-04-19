# U-3 — NDWI formulation and extraction scope

**Status:** Resolved pre-sign-off (extraction retained; use re-scoped by DEV-1b-002)
**Decision date:** 2026-04-17
**Re-scoped:** 2026-04-18 — see superseding-scope note below.
**Scope (original):** Sentinel-2 NDWI extraction to support the
NDWI → m_g canopy moisture prior in SPEC.md v0.1-draft §5 and §8.
**Scope (post-DEV-1b-002):** Sentinel-2 NDWI extraction as a
diagnostic-only covariate feeding the post-experiment
Diagnostic D in SPEC.md v0.1 §11.

---

## Superseding scope (DEV-1b-002, 2026-04-18)

The NDWI → m_g prior that this document was originally written to
support has been withdrawn pre-sign-off (see
[`../DEV-1b-002.md`](../DEV-1b-002.md) and SPEC.md v0.1 §5). The
decision recorded below — extract both Gao and McFeeters, keep Gao as
primary — **still stands as the extraction decision**, but the use of
the Gao column has changed:

- `ndwi_gao` is no longer fed into a prior or any loss term.
- It is now the input to SPEC §11 **Diagnostic D** — the post-experiment
  Pearson correlation between training-set-mean Gao-NDWI and the
  learned m_g across the 10 reps at each training fraction.
- `ndwi_mcfeeters` retains its original diagnostic-only role for
  standing-water contamination checks.

The U-3 extraction work is *not* wasted — it now supports the
post-hoc question "did the learned m_g correlate with an interpretable
canopy-water signal?" which is the honest scientific use of the
extraction given that no heather-specific NDWI → m_g calibration
exists in the peer-reviewed literature (see DEV-1b-002 §Finding).

The rest of this document is preserved unchanged as the decision
record for the extraction scope and column naming. Any language below
that refers to Gao as "input to the m_g prior" should be read in
light of DEV-1b-002: the extraction decision holds; the downstream
consumer is now Diagnostic D, not L_prior.

---

## Decision

**Option B — Extract both Gao (primary) and McFeeters (diagnostic-only)
NDWI variants as separate columns.**

Column naming (binding):
- `ndwi_gao` — Gao (1996): `(B8 − B11) / (B8 + B11)`. Originally
  intended as input to the physics-branch m_g prior; post-DEV-1b-002
  this feeds SPEC §11 Diagnostic D only.
- `ndwi_mcfeeters` — McFeeters (1996): `(B3 − B8) / (B3 + B8)`.
  Diagnostic covariate. Used to flag standing-water contamination in
  monthly composites. Never fed into any model.

Everything else in the Phase 1 S2 pipeline is held constant: same COSMOS-UK
footprint, same monthly cadence, same SCL cloud mask, same qualityMosaic
compositing, same linear interpolation to SAR overpass dates.

Neither column is added to the seven-feature input vector for RF, NN,
or the CorrectionNet. SPEC.md v0.1 §2 and §5 are explicit on this —
NDWI is a diagnostic-only auxiliary product.

---

## Context

`SPEC.md` §5 specifies an NDWI → m_g Gaussian prior on the MIMICS crown
gravimetric water content. It does not specify which NDWI. Two widely
used indices share the name:

| Name | Formula | Physical meaning | Native purpose |
|---|---|---|---|
| McFeeters (1996) | (B3 − B8) / (B3 + B8) | Green minus NIR | Open-water detection |
| Gao (1996) | (B8 − B11) / (B8 + B11) | NIR minus SWIR | Canopy water content |

At Sentinel-2: B3 = green (560 nm), B4 = red (665 nm), B8 = NIR
(842 nm), B11 = SWIR-1 (1610 nm). B11 is native 20 m vs 10 m for the
visible/NIR bands — immaterial at Moor House footprint scale.

---

## Why Gao, not McFeeters, for the prior

The physics-branch prior maps NDWI to a gravimetric canopy moisture in
the range [0.3, 0.6] g/g. The target quantity is leaf and branch
*moisture content*, not *surface water presence*.

- **Gao** is calibrated by construction to track canopy water content.
  SWIR at 1610 nm is absorbed by foliar water; NIR at 842 nm is
  governed primarily by canopy structure (LAI, leaf orientation). Their
  ratio isolates the water absorption signal.
- **McFeeters** is calibrated to contrast open water against
  everything else. High over lakes/ponds, low and noisy over
  vegetation. It does not resolve differences in vegetation water
  content.

Using McFeeters for the m_g prior would put the wrong index on the
wrong quantity. SPEC.md §5's phrasing ("NDWI mapped to a gravimetric
moisture prior") is reaching for Gao semantically; the formulation
decision here makes that explicit.

---

## Why McFeeters is extracted anyway (as a diagnostic)

Moor House is blanket bog. Wet-period hollows and ponded Sphagnum
depressions do host standing or near-standing water at the scale of
individual S2 pixels. A monthly qualityMosaic composite is not immune
to this — a Gao value that moves because some pixels in the composite
contain water is not telling us about canopy moisture.

McFeeters is the standard, cheap, unambiguous test for open-water
signature. Extracting it in the same GEE task costs one extra
`normalizedDifference` call and one extra reducer. The output column
is used only for post-hoc diagnostics — for example, flagging any
month where McFeeters rises above a plausibility threshold so that
month's Gao value can be inspected for water contamination.

---

## Options considered

| Option | Description | Chosen | Reason |
|---|---|---|---|
| A | Gao only | No | Minimally sufficient, but no standing-water diagnostic; tiny extra cost of B is easily justified. |
| **B** | **Gao (primary) + McFeeters (diagnostic)** | **Yes** | **See rationale above.** |
| C | Gao + raw B8 and B11 means | No | Adds post-hoc decomposition of Gao into NIR vs SWIR contributions. Useful in principle but higher validation and column-management overhead; not justified until we see whether Gao itself holds up. Parked as a possible v0.2 extension if the m_g prior proves uninformative. |

---

## Relationship to DEV-1b-001

DEV-1b-001 withdrew the NDVI → LAI → N_l prior because the cited source
does not contain the claimed relationship, and the wider literature
treats the chain as poorly constrained for *Calluna–Sphagnum* canopies.
The reasoning that supported withdrawal there (prefer no prior to a
mis-calibrated prior) raises a natural question about the NDWI → m_g
prior too.

The answer is that the two priors are not symmetric:

- NDWI → m_g concerns *canopy water content*. Gao's NIR − SWIR formulation
  has a direct, well-established physical basis: SWIR is a foliar water
  absorption region. The prior's inputs are physically sensible even
  before we fit any transfer function.
- NDVI → LAI concerns *leaf area density*. NDVI saturates rapidly in
  dense canopies (LAI ≈ 2–5 is where Moor House lives) and is not
  cleanly separable from background / understory effects in mixed
  *Calluna–Sphagnum* systems. The relationship is empirical, not
  physical, and no heather-specific calibration exists.

So: the m_g prior is retained, the extraction lands now, and U-4 will
resolve the separate question of *how* we map `ndwi_gao` to a scalar
m_g value for the training set.

---

## Implementation

### New files

| Path | Purpose |
|---|---|
| `shared/data/gee/extract_ndwi.py` | GEE extractor, mirroring [`extract_sentinel2.py`](../../../shared/data/gee/extract_sentinel2.py). |
| `data/raw/gee/sentinel2_ndwi_raw.csv` | Extractor output. **Written by a human step** — Claude Code has no GEE access. |

### Modified files

| Path | Change |
|---|---|
| [`shared/data/ancillary.py`](../../../shared/data/ancillary.py) | Join the NDWI monthly composite to the S2 panel alongside NDVI. |
| [`shared/data/alignment.py`](../../../shared/data/alignment.py) | Attach `ndwi_gao` and `ndwi_mcfeeters` columns to the aligned dataset without altering the seven-feature input vector. |
| [`tests/unit/`](../../../tests/) | Add NDWI-specific tests: range in [-1, 1], row count equals NDVI, interpolation behaviour, N=119 preserved. |

### Human step

The actual GEE export requires network access and an authorised GEE
project — the same human-step model as Phase 1 steps H1.1–H1.5 in
[`PROGRESS.md`](../../../../PROGRESS.md). The sequence is:

1. Claude Code writes [`shared/data/gee/extract_ndwi.py`](../../../shared/data/gee/extract_ndwi.py)
   and the supporting alignment / test changes.
2. Human runs `python shared/data/gee/extract_ndwi.py` against the
   live GEE project; the export is pulled from the GEE drive into
   `data/raw/gee/sentinel2_ndwi_raw.csv`.
3. Human runs `python shared/data/ancillary.py` (or the equivalent
   alignment driver) to rebuild `data/processed/aligned_dataset.csv`
   with the NDWI columns attached.
4. G1 re-runs: `python phase1/run_baselines.py --confirm` — the
   baselines must still reproduce within 0.005 cm³/cm³ after the
   aligned dataset is rebuilt. A G1 regression after a Tier-2 change
   is a halt condition per CLAUDE.md.

### Governance

This is a Tier-2 change by the letter of [`CLAUDE_3.md`](../../CLAUDE_3.md)
(shared utilities are modified — alignment.py gets new columns). Per
the governance rules:

- Explicit justification: this document.
- Passing G1 re-run: required after the alignment rebuild.
- Deviation log note: **not** required — SPEC.md §2 pre-authorises the
  NDWI addition. This decision only locks the *formulation*, which the
  spec leaves open.

---

## Pre-registered diagnostics around the NDWI extraction

These do not affect the primary outcome and are reported in the Phase 1b
results document for transparency:

- **D1 — Distribution sanity:** `ndwi_gao` values at Moor House
  expected in roughly [-0.1, +0.4] across the NDVI-active months.
  Values outside this range on any observation are flagged.
- **D2 — Wet-pixel contamination check:** For any month where
  `ndwi_mcfeeters` exceeds a plausibility threshold (to be calibrated
  from the training set), the underlying composite is inspected before
  that month's `ndwi_gao` is used in the prior.
- **D3 — Seasonality cross-check:** Compare `ndwi_gao` seasonal cycle
  to ERA5-Land precipitation and to COSMOS VWC seasonality. Strong
  anti-correlation with wet-season VWC or flat seasonality would both
  warrant scrutiny before training.

---

## References

1. McFeeters, S.K. (1996) 'The use of the Normalized Difference Water
   Index (NDWI) in the delineation of open water features',
   *International Journal of Remote Sensing*, 17(7), pp. 1425–1432.
2. Gao, B.-C. (1996) 'NDWI — a normalized difference water index for
   remote sensing of vegetation liquid water from space',
   *Remote Sensing of Environment*, 58(3), pp. 257–266.
3. Phase 1b [`SPEC.md`](../../SPEC.md) §2 (NDWI as new S2 auxiliary),
   §5 (NDWI → m_g prior), §8 (L_prior single term).
4. [`DEV-1b-001.md`](../DEV-1b-001.md) — companion deviation that
   withdrew the N_l prior; context for the retention of the m_g
   prior here.
