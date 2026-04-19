# ECHO — Vantage PoC

Physics-informed machine learning for SAR-based peatland soil moisture retrieval.

**Research question:** Does embedding the Water Cloud Model into a neural network architecture improve soil moisture retrieval from Sentinel-1 SAR in data-scarce conditions, compared to standard ML baselines?

**Study site:** Moor House, North Pennines, England (COSMOS-UK station MOORH)  
**Sensor:** Sentinel-1 C-band SAR (VV + VH, IW GRD)  
**Ground truth:** COSMOS-UK CRNS volumetric water content, 2021–2024

---

## Repository structure

```
echo-vantage/
├── echo-poc/      ← Python science pipeline (this README)
└── vantage-mvp/   ← Next.js MVP dashboard (see vantage-mvp/README.md)
```

---

## Quickstart

### 1. Clone and set up Python environment

```bash
git clone https://github.com/[your-org]/echo-vantage.git
cd echo-vantage/echo-poc

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.11**. Uses PyTorch with Apple Silicon MPS acceleration automatically on M-series Macs (falls back to CPU on other hardware).

### 2. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

Follow the browser prompt. Your credentials are stored locally and are never committed to git.

Test access:
```bash
python poc/data/gee_sentinel1.py --dry-run
# Should print: "GEE authenticated. Moor House S1 scenes available: ~240"
```

### 3. Download raw data (human steps — requires external access)

These four steps cannot be automated. Complete them before running the pipeline.

**Step A — COSMOS-UK ground truth**
1. Go to https://cosmos.ceh.ac.uk/
2. Navigate to station MOORH → Download → Daily product
3. Select date range: 2021-01-01 to 2024-12-31
4. Save to: `data/raw/cosmos/COSMOS_UK_MOORH_1D_202101010000_202412310000.csv`

**Step B — Sentinel-1 SAR extractions (via GEE)**
```bash
python poc/data/gee_sentinel1.py
# Runs GEE job. May take 5–15 min. Downloads to:
# data/raw/sentinel1/sentinel1_extractions.csv
```

**Step C — Sentinel-2 NDVI composites (via GEE)**
```bash
python poc/data/gee_sentinel2.py
# Downloads to: data/raw/sentinel2/ndvi_composites.csv
```

**Step D — ERA5-Land precipitation**
```bash
python poc/data/era5_download.py
# Requires CDS API key (see §CDS Setup below)
# Downloads to: data/raw/precipitation/era5_land_moor_house.csv
```

**Terrain (one-time, already in repo for Moor House)**
```
data/raw/terrain/slope_deg.tif
data/raw/terrain/aspect_deg.tif
data/raw/terrain/twi.tif
```
These are pre-computed from EA 1m LiDAR and committed to git (small files, static).

### 4. Run the full pipeline

Once raw data is downloaded:

```bash
make poc
# Equivalent to: python poc/pipeline.py --from-raw
```

This runs all phases sequentially, stops at each gate, and exits if a gate fails. Total runtime on MacBook Air M-series: approximately 45–75 minutes.

To run a specific phase:
```bash
make phase-1    # Data acquisition and alignment only
make phase-2    # Baseline models (requires Phase 1 complete)
make phase-3    # PINN training (requires Phase 2 complete)
```

### 5. Check gate status

```bash
make gate-1     # Run Gate 1 check
make gate-2     # Run Gate 2 check
# etc.
```

Gate results are written to `outputs/gates/gate_N_result.json`.

### 6. Export fixtures to MVP

After Gate 3 passes:

```bash
make export-fixtures
# Reads gate_3_result.json, writes all fixture JSON files to
# ../vantage-mvp/src/data/fixtures/
```

Verify the export:
```bash
cd ../vantage-mvp && pnpm validate:data
```

---

## CDS API setup (ERA5-Land precipitation)

1. Register at https://cds.climate.copernicus.eu/
2. Go to your profile → API key
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR-UID:YOUR-API-KEY
   ```
4. Test: `python -c "import cdsapi; cdsapi.Client()"`

---

## Running tests

```bash
# Unit tests only (fast, no data required)
pytest tests/unit/ -v

# Integration tests (requires data/processed/ to exist)
pytest tests/integration/ -v --require-data

# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=poc --cov-report=term-missing
```

Coverage targets: ≥ 80% on `poc/data/`, `poc/models/`, `poc/evaluation/`.

---

## Reproducing results from scratch

All results are fully reproducible from raw data:

```bash
# Delete all processed data and outputs
make clean

# Re-download raw data (Steps A–D above)

# Regenerate everything
make poc

# Export fixtures
make export-fixtures
```

The sealed test set (`data/splits/test_indices.json`) is committed to git and never regenerated. This guarantees that results on the test set are always comparable across runs.

---

## Key files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Engineering constitution — read before any Claude Code session |
| `SPEC.md` | Top-level research specification |
| `SPEC_PHASE{1-5}.md` | Detailed phase-by-phase specs |
| `DEVIATIONS.md` | Append-only log of spec deviations |
| `PROGRESS.md` | Current build state — updated each session |
| `poc/config.py` | All constants, paths, seeds, hyperparameters |
| `data/splits/test_indices.json` | Sealed test set — never modify |
| `outputs/gates/gate_3_result.json` | Pre-registered outcome classification |

---

## Dependencies

All pinned in `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2.x | PINN training (MPS on Apple Silicon) |
| `scikit-learn` | 1.4.x | Random Forest |
| `earthengine-api` | 0.1.x | GEE data extraction |
| `scipy` | 1.12.x | Wilcoxon signed-rank test |
| `pandas` | 2.x | Data processing |
| `matplotlib` | 3.8.x | Figures |
| `pydantic` | 2.x | Fixture schema validation |
| `cdsapi` | 0.6.x | ERA5-Land download |

---

## For Claude Code sessions

At the start of every session, read:
1. `CLAUDE.md` — engineering constitution
2. `PROGRESS.md` — where we are
3. `SPEC_PHASE{N}.md` — current phase detail
4. `DEVIATIONS.md` — active deviations

Update `PROGRESS.md` at the end of every session.

---

## License

Private — Vantage / Matthew Douthwaite. Not for public distribution.
