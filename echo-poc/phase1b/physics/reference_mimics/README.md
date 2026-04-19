# Reference MIMICS implementation — sourcing decision

This directory holds the non-differentiable reference MIMICS implementation
used by the Phase 1b G2 Implementation Gate ([`SPEC.md`](../../../SPEC.md)
§4). The gate requires the differentiable PyTorch MIMICS module
([`phase1b/physics/mimics.py`](../mimics.py), to be written) to agree with
a reference implementation within **0.5 dB in σ°** across a set of
canonical parameter combinations spanning the Moor House operating range.

The purpose of the gate is to catch implementation bugs in the
differentiable port — autograd code that looks right but has a sign flip
or a missing factor produces plausible σ° and quietly corrupts the
experiment. The reference must be independent enough of the new PyTorch
code that the check is meaningful.

---

## Decision

**Option E — published tables + self-authored numpy Toure port, both
required to agree with the differentiable implementation for the gate to
pass.**

The gate is specified as a two-step check:

1. **Primary — published tables.** σ°_VV and σ°_VH values transcribed
   from peer-reviewed MIMICS papers at named parameter combinations. This
   directory will hold the transcribed tables as JSON/CSV with the source
   citation and the exact parameter inputs. Candidate sources (to be
   confirmed as the code is written):
   - **Toure et al. (1994)**, *IEEE TGRS* 32(1) — wheat and canola at C
     and L bands. Primary precedent for the single-crown adaptation we
     are using, and the natural source for C-band validation even though
     the canopy is crop, not heather.
   - **McDonald, Dobson & Ulaby (1990)**, *IEEE TGRS* — walnut orchard
     MIMICS validation at L and X bands. Secondary, useful for
     cross-band sanity but further from Moor House.
   - Any additional C-band MIMICS paper with tabulated outputs will be
     added here as a supplementary source if it tightens the parameter
     coverage.

2. **Secondary — numpy reference port.** A plain-numpy implementation of
   the Toure single-crown C-band formulation, written directly from the
   paper equations in [`reference_toure.py`](reference_toure.py) (to be
   written). This extends coverage to parameter combinations the
   published tables do not happen to tabulate — notably the Moor House
   operating envelope (heather-scale branches, peat ground, Sphagnum
   roughness s = 1–5 cm).

The G2 gate passes only if both checks pass independently. Any
disagreement between the PyTorch module and either reference is a halt
condition.

---

## Options considered

| Option | Source | Gate strength | Effort | Notes |
|--------|--------|---------------|--------|-------|
| A | Port [py-mimics](https://github.com/katjensen/py-mimics) | Weak | Low (if it works as-is) | Self-described "Still a Work in Progress — files were rescued from old laptop". 8 commits, last activity ~2018, no license. Not an independent authority. |
| B | Self-authored numpy port of Toure (1994) | Medium | Medium | Independent *code path* from our PyTorch module, but same author interprets the paper on both sides. Bugs originating from misreading the paper propagate to both. |
| C | Published σ° tables only | Strong on what they cover; silent elsewhere | Low | Peer-reviewed, independent authors, different code. Constrained to whatever parameter combinations the papers tabulate — typically wheat/canola/walnut, not heather/Sphagnum. |
| D | Request Fortran/MATLAB code from Dobson lab / JPL / Hamdan et al. | Strongest | High elapsed time, uncertain | Gold standard independence. Licensing and timing unknown; Fortran toolchain may be needed. |
| **E** | **C + B together** | **Strong + broad** | **Medium** | **Chosen.** |

Option D is not ruled out — if a request were to return timely, compliant
code, it would strengthen the gate further. It is parked as a parallel
nice-to-have and does not block progress.

---

## Rationale

The G2 gate exists to distinguish two failure modes that look identical
at the metric layer: *"MIMICS is genuinely inadequate for this site"*
(the scientific question Phase 1b is answering) versus *"our
differentiable MIMICS has an implementation bug"*. Phase 1's Negative
outcome only has interpretive power because its physics model was
implemented correctly; a Phase 1b Negative outcome with a bugged MIMICS
would be scientifically uninformative.

- **Option C (published tables) carries the independent-authority
  evidence.** When our PyTorch MIMICS reproduces Toure's Table N at the
  exact inputs Toure used, that is validation against a peer-reviewed
  source with no code ancestry in common with ours. This is the
  strongest form of evidence the gate can provide.
- **Option B (numpy port) extends coverage.** Toure's tables do not
  cover heather-scale scatterers over peat at Sphagnum roughness, and
  those are exactly the parameter combinations where Phase 1b will spend
  most of its time. A plain-numpy port lets us generate reference σ° at
  the Moor House operating envelope without waiting on a third party.
- **Both are required.** The weakness of B in isolation (we author both
  codes) is offset by C providing the independent-authority anchor.
  The narrowness of C in isolation (crop canopies, not heather) is
  offset by B extending to our operating range. A bug would have to
  produce the same σ° error in both our PyTorch code *and* our numpy
  port *and* the disagreement with published tables would still have to
  be under 0.5 dB — a much narrower attack surface than either check
  alone.
- **Option A was rejected** because py-mimics' stability and provenance
  are too weak to serve as a reference authority. If py-mimics itself
  has bugs, G2 would pass with both codes wrong the same way.
- **Option D was deferred** because the time-to-response is uncontrolled
  and the licensing is unknown. It strengthens the gate if it lands in
  time but does not belong on the critical path.

---

## Gate formulation

The G2 Implementation Gate
([`phase1b/implementation_gate/equivalence_check.py`](../../implementation_gate/equivalence_check.py),
to be written) will:

1. Load a set of canonical parameter combinations from
   `canonical_combinations.json` in this directory. The set will
   include:
   - Every (parameter combination, σ°) tuple transcribed from the
     published tables (primary check).
   - A grid of combinations covering the Moor House operating range —
     heather crown geometry from `SPEC.md` §5, peat ground at Mironov
     dielectric, Sphagnum roughness s ∈ {1, 2, 3, 4, 5} cm, incidence
     angle 41.5° — with reference σ° produced by the numpy Toure port
     (secondary check).
2. For each combination, compute σ°_VV and σ°_VH from the differentiable
   PyTorch MIMICS.
3. Require `max |σ°_diff|` ≤ **0.5 dB** (per `SPEC.md` §4) separately
   against the primary and secondary references.
4. Write `results/g2_equivalence.json` with the full diff table.

If the PyTorch MIMICS disagrees with either reference by more than
0.5 dB on any canonical combination, the gate halts and the disagreement
is investigated before any training begins. Disagreement with the
published tables is a stronger signal than disagreement with the numpy
port, because of the authority asymmetry noted above.

---

## What this directory will contain

Once the G2 work begins (task P1b.02 per the migration plan):

```
reference_mimics/
├── README.md                       (this file)
├── reference_toure.py              (numpy implementation of Toure 1994)
├── canonical_combinations.json     (input-parameter + reference-σ° tuples)
├── published_tables/
│   ├── toure_1994_table_N.csv      (transcribed with page/row citation)
│   └── mcdonald_1990_table_N.csv   (likewise)
└── LICENSE                          (for any third-party code we vendor)
```

The `published_tables/` files will each carry a header comment citing
the paper, DOI, table number, and the exact parameter inputs. Any
numeric values that had to be read off a figure rather than a table
will be flagged as lower-confidence in the canonical-combinations
manifest.

---

## References

- Toure, A., et al. (1994). *Adaptation of the MIMICS backscattering
  model to the agricultural context — wheat and canola at L and C
  bands.* IEEE TGRS, 32(1), pp. 47–61. — primary single-crown precedent.
- McDonald, K., Dobson, M.C., & Ulaby, F.T. (1990). *Using MIMICS to
  model L-band multiangle and multitemporal backscatter from a walnut
  orchard.* IEEE TGRS, 28(4), pp. 477–491. — canonical validation.
- Ulaby, F.T., et al. (1990). *Michigan Microwave Canopy Scattering
  Model.* Int. J. Remote Sensing, 11(7), pp. 1223–1253. — original
  MIMICS specification.
- [SPEC.md](../../../SPEC.md) §4 — G2 gate definition and tolerance.
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — directory conventions
  for `reference_mimics/`.
- [py-mimics](https://github.com/katjensen/py-mimics) — rejected
  option A; retained here as a citation so the decision is reviewable.
