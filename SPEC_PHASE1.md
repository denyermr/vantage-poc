# SPEC.md — Phase 1: Data Acquisition & Alignment
# ECHO PoC — Vantage

**Replaces:** §3 (Data Specification), §4.1 partial, §9 (Gate 1 Failure Protocol), §12 Phase 1 manifest  
**Version:** 1.1 — 06 March 2026

---

## P1. Overview

Phase 1 produces a single master analytical dataset (`data/processed/aligned_dataset.csv`) from four independent data sources. All sources are accessed via Google Earth Engine or the COSMOS-UK portal — there is no local geospatial processing in Phase 1. The phase is complete when the Gate 1 script exits with code 0.

**Sources:**

| # | Source | Data | Access method |
|---|--------|------|---------------|
| 1 | COSMOS-UK portal | Daily VWC, temperature, snow | CSV download (already done) |
| 2 | GEE `COPERNICUS/S1_GRD` | Sentinel-1 SAR backscatter | GEE → Google Drive → local |
| 3 | GEE `COPERNICUS/S2_SR_HARMONIZED` | NDVI | GEE → Google Drive → local |
| 4 | GEE `ECMWF/ERA5_LAND/DAILY_AGGR` | Precipitation | GEE → Google Drive → local |
| 5 | GEE `USGS/SRTMGL1_003` + `MERIT/Hydro/v1_0_1` | Slope, aspect, TWI | GEE → Google Drive → local |

**EA LiDAR note:** The Environment Agency 1 m LiDAR DTM offers higher-resolution terrain inputs than SRTM 30 m. It is **not used in Phase 1** — SRTM via GEE is the primary terrain source for the PoC. EA LiDAR download and processing instructions are documented in §P1.7 for future use as a named upgrade (ECHO-TERRAIN-v2). Do not implement it in Phase 1.

---

## P1.1 Repository structure (Phase 1)

```
echo-poc/
├── data/
│   ├── raw/
│   │   ├── cosmos/
│   │   │   └── COSMOS_UK_MOORH_1D_202101010000_202412310000.csv   ← already present
│   │   └── gee/
│   │       ├── sentinel1_raw.csv          ← downloaded from Google Drive
│   │       ├── sentinel2_ndvi_raw.csv     ← downloaded from Google Drive
│   │       ├── era5_precip_raw.csv        ← downloaded from Google Drive
│   │       └── terrain_static_raw.csv     ← downloaded from Google Drive
│   ├── processed/
│   │   ├── cosmos_daily_vwc.csv
│   │   ├── sentinel1_extractions.csv
│   │   ├── ancillary_features.csv
│   │   └── aligned_dataset.csv
│   └── splits/
│       └── test_indices.json              ← generated at end of Phase 1
├── poc/
│   ├── config.py
│   └── data/
│       ├── cosmos.py
│       ├── gee/
│       │   ├── __init__.py
│       │   ├── extract_sentinel1.py
│       │   ├── extract_sentinel2.py
│       │   ├── extract_era5.py
│       │   └── extract_terrain.py
│       ├── ancillary.py
│       └── alignment.py
├── poc/gates/
│   └── gate_1.py
└── outputs/
    ├── figures/
    │   ├── p1_cosmos_diagnostic.png
    │   ├── p1_sar_diagnostic.png
    │   ├── p1_ancillary_diagnostic.png
    │   └── p1_aligned_dataset_summary.png
    └── gates/
        └── gate_1_result.json
```

---

## P1.2 Configuration (`poc/config.py`)

All constants used in Phase 1 must be defined here. No magic numbers in any other file.

```python
# --- Site ---
SITE_NAME         = "Moor House"
SITE_ID           = "MOORH"
SITE_LAT          = 54.69          # degrees N
SITE_LON          = -2.38          # degrees E
SITE_RADIUS_M     = 200            # COSMOS-UK footprint radius
STUDY_START       = "2021-01-01"
STUDY_END         = "2024-12-31"

# --- GEE ---
GEE_PROJECT       = ""             # REQUIRED: user sets their GEE Cloud project ID
GEE_DRIVE_FOLDER  = "echo_poc_gee_exports"
S1_COLLECTION     = "COPERNICUS/S1_GRD"
S2_COLLECTION     = "COPERNICUS/S2_SR_HARMONIZED"
ERA5_COLLECTION   = "ECMWF/ERA5_LAND/DAILY_AGGR"
SRTM_COLLECTION   = "USGS/SRTMGL1_003"
MERIT_COLLECTION  = "MERIT/Hydro/v1_0_1"

# --- SAR ---
S1_PASS           = "DESCENDING"
S1_MODE           = "IW"
S1_MIN_PIXELS     = 10             # minimum valid pixels in footprint
VV_RANGE_MIN      = -20.0          # dB — hard lower bound for validation
VV_RANGE_MAX      = -5.0           # dB — hard upper bound for validation

# --- Physics constants (used in Phase 3, defined here) ---
# Saturated VWC for blanket bog peat (Bechtold et al., 2018, Table 1)
PEAT_THETA_SAT    = 0.88           # cm³/cm³
# Dielectric constant of dry peat (Bechtold et al., 2018)
EPSILON_DRY_PEAT  = 3.5
# Dielectric constant of free water at ~10°C
EPSILON_WATER     = 80.0
# Dobson mixing model exponent for organic soil
DOBSON_ALPHA      = 1.4

# --- Reproducibility ---
SEED              = 42

# --- Paths (resolve from project root) ---
from pathlib import Path
PROJECT_ROOT      = Path(__file__).parent.parent
DATA_RAW          = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED    = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS       = PROJECT_ROOT / "data" / "splits"
OUTPUTS_FIGURES   = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_METRICS   = PROJECT_ROOT / "outputs" / "metrics"
OUTPUTS_GATES     = PROJECT_ROOT / "outputs" / "gates"
```

**Rule:** `GEE_PROJECT` must be set before any GEE script can run. The script must check it is non-empty and raise `ValueError` with a clear message if not.

---

## P1.3 Source 1 — COSMOS-UK (already acquired)

**Status:** Complete. `data/raw/cosmos/COSMOS_UK_MOORH_1D_202101010000_202412310000.csv` is present.

**Processing script:** `poc/data/cosmos.py`  
**Function:** `load_cosmos(path: Path) -> pd.DataFrame`  
**Output:** `data/processed/cosmos_daily_vwc.csv`

### Input file structure

The COSMOS-UK daily CSV has a 5-row metadata header before column names:

```
Row 0: timestamp, <download_timestamp>
Row 1: site-id, MOORH
Row 2: site-name, Moor House
Row 3: date-installed, ...
Row 4: date-decommissioned, ...
Row 5: parameter-id, <col1>, <col2>, ...     ← column names
Row 6: parameter-name, ...                   ← skip
Row 7: parameter-units, ...                  ← skip
Row 8+: data rows
```

Parse with `pd.read_csv(path, skiprows=5, header=0)`, then drop rows 0 and 1 (parameter-name and parameter-units), then rename the index column from `parameter-id` to `date`.

### Columns to extract

| Raw column | Output column | Transformation |
|------------|---------------|----------------|
| `parameter-id` | `date` | `pd.to_datetime(...).dt.tz_localize(None)` — strip UTC tz for consistency |
| `cosmos_vwc` | `vwc_raw` | `pd.to_numeric / 100.0` — convert % → cm³/cm³ |
| `cosmos_vwc_flag` | `cosmos_vwc_flag` | string, preserve as-is |
| `ta_min` | `ta_min` | `pd.to_numeric` — °C |
| `snow` | `snow` | `pd.to_numeric` — mm |

### QC logic

```python
qc_pass = ~df['cosmos_vwc_flag'].isin(['E', 'I'])
df['vwc_qc'] = df['vwc_raw'].where(qc_pass)          # NaN where flag E or I
df['frozen_flag'] = (df['ta_min'] < 0).astype(int)    # 1 if ta_min < 0°C
df['snow_flag'] = (df['snow'] > 0).astype(float)      # 1 if snow > 0 mm
df['snow_flag'] = df['snow_flag'].fillna(0).astype(int)
```

### Output schema

| Column | Type | Units | Notes |
|--------|------|-------|-------|
| `date` | datetime | timezone-naive | Parsed from ISO 8601, UTC stripped |
| `vwc_raw` | float64 | cm³/cm³ | All rows including flagged |
| `vwc_qc` | float64 | cm³/cm³ | NaN where flag E or I |
| `cosmos_vwc_flag` | str | — | Original portal flag |
| `ta_min` | float64 | °C | Daily minimum temperature |
| `snow` | float64 | mm | Snow depth |
| `frozen_flag` | int64 | 0/1 | 1 if ta_min < 0 |
| `snow_flag` | int64 | 0/1 | 1 if snow > 0 |

### Validation (raise `ValueError` if any fail)

1. `vwc_qc` non-null values all in [0.10, 1.00] — if any outside range, stop
2. Date index is monotonically increasing with no duplicate dates
3. Date range includes 2021-01-01 and 2024-12-31
4. Seasonal check: mean `vwc_qc` in {Dec, Jan, Feb} > mean `vwc_qc` in {Jun, Jul, Aug}

### Known deviation

DEV-001 applies: daily product used instead of hourly. See `DEVIATIONS.md`.

---

## P1.4 Source 2 — Sentinel-1 SAR via GEE

**Script:** `poc/data/gee/extract_sentinel1.py`  
**Raw output:** `data/raw/gee/sentinel1_raw.csv` (downloaded from Google Drive)  
**Processed output:** `data/processed/sentinel1_extractions.csv`

### GEE extraction script specification

The script must be runnable as `python poc/data/gee/extract_sentinel1.py` and importable as a module. It must accept command-line arguments for site coordinates, radius, and optional orbit number.

**Initialisation:**
```python
import ee
ee.Initialize(project=config.GEE_PROJECT)
```
Raise `ValueError("GEE_PROJECT not set in config.py")` if `config.GEE_PROJECT` is empty.

**GEE pipeline:**

```
1. Load COPERNICUS/S1_GRD collection
2. Filter: date range STUDY_START to STUDY_END
3. Filter: instrumentMode == 'IW'
4. Filter: orbitProperties_pass == 'DESCENDING'
5. Filter: VV and VH transmitterReceiverPolarisation present
6. Filter: resolution_meters == 10
7. Define footprint: ee.Geometry.Point([SITE_LON, SITE_LAT]).buffer(SITE_RADIUS_M)
8. Filter: bounds intersect footprint
9. For each image, compute:
   - spatial mean VV (dB) within footprint  →  vv_db
   - spatial mean VH (dB) within footprint  →  vh_db
   - vhvv_db = vh_db - vv_db
   - mean local incidence angle (from 'angle' band)  →  incidence_angle_mean
   - pixel count (non-masked)  →  n_pixels
   - relative orbit number from image metadata  →  orbit_number
   - overpass date (image acquisition date, UTC)  →  date
10. If orbit_number not specified: count overpasses per relative orbit,
    select the most frequent. Log the selected orbit and counts.
11. Filter collection to selected orbit only
12. Convert to FeatureCollection
13. Export to Google Drive
```

**Export call:**
```python
task = ee.batch.Export.table.toDrive(
    collection=fc,
    description='echo_sentinel1_moorh',
    folder=config.GEE_DRIVE_FOLDER,
    fileNamePrefix='sentinel1_raw',
    fileFormat='CSV'
)
task.start()
print(f"Export task started: {task.id}")
print(f"Monitor at: https://code.earthengine.google.com/tasks")
print(f"Download to: data/raw/gee/sentinel1_raw.csv")
```

The script **does not wait** for the export to complete. It prints the task ID and exits. The user downloads the file manually from Google Drive.

**Post-download processing** (`poc/data/gee/extract_sentinel1.py`, function `process_raw(raw_path, output_path)`):

Clean the GEE CSV export (GEE adds a `.geo` column and sometimes a `system:index` column — drop both). Parse `date` from the `system:time_start` field (milliseconds since epoch: `pd.to_datetime(df['system:time_start'], unit='ms').dt.normalize()`). Round all float columns to 4 decimal places.

### Processed output schema

| Column | Type | Units | Notes |
|--------|------|-------|-------|
| `date` | datetime | timezone-naive | Overpass date, time normalised to midnight |
| `vv_db` | float64 | dB | Spatial mean VV backscatter |
| `vh_db` | float64 | dB | Spatial mean VH backscatter |
| `vhvv_db` | float64 | dB | VH − VV (cross-pol ratio in log space) |
| `orbit_number` | int64 | — | Relative orbit number (single value across all rows) |
| `n_pixels` | int64 | — | Number of valid pixels in footprint |
| `incidence_angle_mean` | float64 | degrees | Mean local incidence angle |

### Validation (raise `ValueError` if any fail)

1. All rows have the same `orbit_number`
2. `vv_db` values all in [−20, −5] dB
3. `n_pixels` > `S1_MIN_PIXELS` on every row
4. Annual overpass counts all in [45, 75] — warn (do not fail) if outside this range, log to console
5. No duplicate dates
6. Date range includes observations in each of the 4 years 2021–2024

---

## P1.5 Source 3 — Sentinel-2 NDVI via GEE

**Script:** `poc/data/gee/extract_sentinel2.py`  
**Raw output:** `data/raw/gee/sentinel2_ndvi_raw.csv`  
**Output column:** `ndvi` in `data/processed/ancillary_features.csv`

### GEE extraction script specification

**GEE pipeline:**

```
1. Load COPERNICUS/S2_SR_HARMONIZED collection
2. Filter: date range STUDY_START to STUDY_END
3. Filter: bounds intersect footprint (200m radius)
4. Cloud mask: use SCL (Scene Classification Layer) band
   - Keep pixels where SCL in {4=vegetation, 5=bare soil, 6=water, 7=unclassified}
   - Exclude: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus, 11=snow
5. Compute NDVI per pixel: (B8 - B4) / (B8 + B4)
6. For each calendar month, compute the maximum-NDVI composite:
   - Group images by year-month
   - Per pixel, take the image with highest NDVI (max-NDVI compositing)
   - Compute spatial mean NDVI over footprint for that composite
   - Record date as the 15th of the month (representative mid-month date)
7. Export monthly time series as FeatureCollection to Google Drive
```

**Export call:**
```python
task = ee.batch.Export.table.toDrive(
    collection=fc,
    description='echo_sentinel2_ndvi_moorh',
    folder=config.GEE_DRIVE_FOLDER,
    fileNamePrefix='sentinel2_ndvi_raw',
    fileFormat='CSV'
)
```

### Monthly raw output schema

| Column | Type | Notes |
|--------|------|-------|
| `year_month` | str | 'YYYY-MM' |
| `ndvi_mean` | float | Spatial mean NDVI over footprint, max-NDVI composite |
| `n_clear_pixels` | int | Clear pixels contributing to composite |
| `composite_date` | str | Set to 15th of month: 'YYYY-MM-15' |

### NDVI interpolation to SAR dates

Function `interpolate_ndvi(ndvi_monthly: pd.DataFrame, sar_dates: pd.DatetimeIndex) -> pd.Series`:

1. Convert `composite_date` to datetime
2. Set as index, sort ascending
3. Reindex to daily frequency using `asfreq('D')`
4. Linear interpolation: `interpolate(method='time', limit_direction='both')`
5. Extract values at SAR overpass dates

**Important:** If any SAR date falls outside the range of available monthly composites (e.g., due to persistent cloud cover), the interpolated NDVI is NaN. This is caught at the alignment stage (§P1.8) and the overpass is excluded.

**Validation:**
1. At least one valid composite per calendar quarter across all 4 years
2. `ndvi_mean` values all in [−0.2, 1.0]
3. After interpolation to SAR dates, NaN fraction < 10% — warn if exceeded

---

## P1.6 Source 4 — ERA5-Land precipitation via GEE

**Script:** `poc/data/gee/extract_era5.py`  
**Raw output:** `data/raw/gee/era5_precip_raw.csv`  
**Output columns:** `precip_mm`, `precip_7day_mm` in `data/processed/ancillary_features.csv`

### GEE extraction script specification

**Collection:** `ECMWF/ERA5_LAND/DAILY_AGGR`  
**Band:** `total_precipitation_sum`  
**Units in GEE:** metres/day — **multiply by 1000 to convert to mm/day**  
**Spatial resolution:** ~9 km native. Extract single pixel value nearest to site coordinates (point extraction, not spatial mean).

**GEE pipeline:**

```
1. Load ECMWF/ERA5_LAND/DAILY_AGGR
2. Filter: date range STUDY_START − 8 days to STUDY_END
   (8 days extra at start to allow 7-day antecedent index for first observations)
3. Select 'total_precipitation_sum' band
4. For each image:
   - Extract value at SITE_LON, SITE_LAT (point sample)
   - Convert to mm: value × 1000
   - Record date
5. Export full daily time series to Google Drive
```

**Export call:**
```python
task = ee.batch.Export.table.toDrive(
    collection=fc,
    description='echo_era5_precip_moorh',
    folder=config.GEE_DRIVE_FOLDER,
    fileNamePrefix='era5_precip_raw',
    fileFormat='CSV'
)
```

### Post-download processing

Function `process_era5(raw_path: Path, sar_dates: pd.DatetimeIndex) -> pd.DataFrame`:

1. Parse date from `system:time_start` (ms epoch)
2. Rename precip column to `precip_mm`, round to 3 decimal places
3. Compute 7-day antecedent index:
   ```python
   df['precip_7day_mm'] = df['precip_mm'].shift(1).rolling(7).sum()
   # shift(1) excludes current day; rolling(7) sums prior 7 days
   ```
4. Filter to SAR overpass dates only

### Output columns in ancillary_features.csv

| Column | Type | Units | Notes |
|--------|------|-------|-------|
| `precip_mm` | float64 | mm | ERA5-Land daily total, day of overpass |
| `precip_7day_mm` | float64 | mm | Sum of 7 days prior to overpass (not including overpass day) |

**Validation:**
1. `precip_mm` ≥ 0 on all rows
2. `precip_7day_mm` is non-null for all SAR dates (requires 7 days prior coverage)
3. Annual mean precip at Moor House expected ~1,800–2,500 mm — warn (do not fail) if outside this range

---

## P1.7 Source 5 — Terrain features via GEE (SRTM + MERIT Hydro)

**Script:** `poc/data/gee/extract_terrain.py`  
**Raw output:** `data/raw/gee/terrain_static_raw.csv`  
**Output columns:** `slope_deg`, `aspect_sin`, `aspect_cos`, `twi` in `data/processed/ancillary_features.csv`

These features are **static** — they are computed once for the COSMOS-UK footprint and are the same value on every overpass date.

### GEE pipeline

```
1. Load USGS/SRTMGL1_003 (30 m DEM)
2. Clip to footprint (200 m radius buffer)
3. Compute terrain products: ee.Terrain.products(dem)
   - 'slope': slope in degrees
   - 'aspect': aspect in degrees (0=N, 90=E, 180=S, 270=W)
4. Compute spatial means over footprint:
   - slope_deg   = mean of 'slope' band
   - aspect_deg  = mean of 'aspect' band (NOTE: circular mean — see below)
5. Compute circular encoding of aspect:
   - aspect_rad = aspect_deg × π / 180
   - aspect_sin = sin(aspect_rad)
   - aspect_cos = cos(aspect_rad)
6. Load MERIT/Hydro/v1_0_1
   - Band 'upa': upstream drainage area in km²
7. Extract spatial mean 'upa' over footprint  →  upa_km2
8. Compute TWI:
   - slope_rad = slope_deg × π / 180
   - tan_slope = max(tan(slope_rad), 0.001)   # floor to avoid log(inf)
   - upa_m2 = upa_km2 × 1e6
   - twi = ln(upa_m2 / tan_slope)             # natural log
9. Export single-row FeatureCollection to Google Drive
```

**Aspect circular mean note:** The simple arithmetic mean of aspect degrees is incorrect near 0°/360° (e.g., mean of 10° and 350° is not 180°). The circular mean is:
```
aspect_mean_rad = atan2(mean(sin(aspect_rad)), mean(cos(aspect_rad)))
aspect_mean_deg = aspect_mean_rad × 180 / π
```
Implement this in the GEE script using `ee.Reducer.mean()` on the sin and cos bands separately, then reconstruct.

**Export call:**
```python
task = ee.batch.Export.table.toDrive(
    collection=ee.FeatureCollection([terrain_feature]),
    description='echo_terrain_moorh',
    folder=config.GEE_DRIVE_FOLDER,
    fileNamePrefix='terrain_static_raw',
    fileFormat='CSV'
)
```

### Post-download processing

The raw CSV is a single row. Parse into a dict, validate, then broadcast to all SAR overpass dates:

```python
def load_terrain(raw_path: Path, sar_dates: pd.DatetimeIndex) -> pd.DataFrame:
    row = pd.read_csv(raw_path).iloc[0]
    terrain = {
        'slope_deg':   float(row['slope_deg']),
        'aspect_sin':  float(row['aspect_sin']),
        'aspect_cos':  float(row['aspect_cos']),
        'twi':         float(row['twi']),
    }
    # Validate
    assert 0 <= terrain['slope_deg'] <= 45, "slope_deg out of expected range"
    assert -1 <= terrain['aspect_sin'] <= 1
    assert -1 <= terrain['aspect_cos'] <= 1
    assert terrain['twi'] > 0
    # Broadcast to all dates
    return pd.DataFrame([terrain] * len(sar_dates), index=sar_dates)
```

### EA LiDAR — future upgrade (ECHO-TERRAIN-v2)

**Do not implement in Phase 1.** Instructions for future reference:

The Environment Agency National LiDAR Programme provides 1 m DTM tiles free of charge.

- **Portal:** https://environment.data.gov.uk/survey
- **Navigation:** Select "LIDAR Composite DTM" → 1m resolution → draw bounding box around Moor House (approximately NY74 grid square, ~54.6°N–54.8°N, ~2.2°W–2.6°W)
- **Format:** ASCII Grid (.asc) files, OSGB36 / BNG projection (EPSG:27700)
- **Processing:** Reproject to WGS84 (EPSG:4326), clip to 200 m footprint, compute slope/aspect/TWI using `pysheds` or `richdem`
- **TWI with pysheds:** `Grid.accumulation()` → `Grid.calculate_distance_to_ridge()` for upslope area; then TWI = ln(A / tan(β))
- **Trigger for upgrade:** if SRTM-derived features produce poor model performance and LiDAR resolution is suspected as a contributing factor

---

## P1.8 Data alignment and QC

**Script:** `poc/data/alignment.py`  
**Function:** `build_aligned_dataset(cosmos_path, s1_path, ancillary_path, output_path) -> pd.DataFrame`  
**Output:** `data/processed/aligned_dataset.csv`

### Assembly of ancillary_features.csv

Before alignment, a single ancillary features file is assembled:

```python
# poc/data/ancillary.py
def build_ancillary(s1_dates, ndvi_monthly_path, era5_raw_path, terrain_raw_path) -> pd.DataFrame:
    """
    Assembles ancillary_features.csv aligned to SAR overpass dates.
    All rows correspond 1:1 with Sentinel-1 overpass dates.
    """
```

| Column | Source | Notes |
|--------|--------|-------|
| `date` | S1 overpass dates | Join key |
| `ndvi` | S2 monthly composites | Linearly interpolated to S1 dates |
| `precip_mm` | ERA5-Land daily | Day-of-overpass value |
| `precip_7day_mm` | ERA5-Land rolling | Prior 7 days |
| `slope_deg` | SRTM terrain | Static — same on all rows |
| `aspect_sin` | SRTM terrain | Static |
| `aspect_cos` | SRTM terrain | Static |
| `twi` | MERIT Hydro | Static |

### Join logic

Primary join key: SAR overpass `date`. All joins are left joins from the S1 overpass date index.

```
aligned = s1_extractions
  LEFT JOIN cosmos_daily_vwc ON date        → adds vwc_qc, frozen_flag, snow_flag
  LEFT JOIN ancillary_features ON date      → adds all ancillary columns
```

### Exclusion criteria (applied in this order, attrition logged at each step)

| Step | Criterion | Reason |
|------|-----------|--------|
| 1 | `vwc_qc` is NaN | COSMOS-UK flagged E or I |
| 2 | `frozen_flag == 1` | Frozen soil invalidates dielectric model |
| 3 | `snow_flag == 1` | Snow confounds backscatter |
| 4 | Any feature column is NaN | Missing ancillary data |
| 5 | Not from the single selected orbit | Inconsistent look angle |

**Attrition must be logged** — print and save to `outputs/gates/gate_1_attrition.json`:

```json
{
  "s1_overpasses_raw": 247,
  "after_step1_vwc_flag": 235,
  "after_step2_frozen": 195,
  "after_step3_snow": 183,
  "after_step4_missing_ancillary": 180,
  "after_step5_orbit": 180,
  "final_paired_observations": 180
}
```

### Output schema (`aligned_dataset.csv`)

| Column | Type | Units | Source |
|--------|------|-------|--------|
| `date` | datetime | timezone-naive | S1 overpass date |
| `vwc` | float64 | cm³/cm³ | Target variable (= `vwc_qc`) |
| `vv_db` | float64 | dB | S1 |
| `vh_db` | float64 | dB | S1 |
| `vhvv_db` | float64 | dB | S1 (= `vh_db − vv_db`) |
| `ndvi` | float64 | dimensionless | S2 interpolated |
| `precip_mm` | float64 | mm | ERA5-Land |
| `precip_7day_mm` | float64 | mm | ERA5-Land rolling |
| `slope_deg` | float64 | degrees | SRTM static |
| `aspect_sin` | float64 | dimensionless | SRTM static |
| `aspect_cos` | float64 | dimensionless | SRTM static |
| `twi` | float64 | dimensionless | MERIT Hydro static |
| `incidence_angle_mean` | float64 | degrees | S1 |

**No NaN values permitted in the output file.** The alignment function must assert `df.isna().sum().sum() == 0` before writing to disk.

### Test/train split (generated at end of Phase 1)

Immediately after `aligned_dataset.csv` is written:

```python
n = len(df)
split_idx = int(np.floor(n * 0.70))
test_indices = df.index[split_idx:].tolist()   # integer positional indices
train_indices = df.index[:split_idx].tolist()

# Save
with open(DATA_SPLITS / 'test_indices.json', 'w') as f:
    json.dump({
        'split_idx': split_idx,
        'n_total': n,
        'n_train_pool': split_idx,
        'n_test': n - split_idx,
        'test_date_start': str(df['date'].iloc[split_idx]),
        'test_date_end': str(df['date'].iloc[-1]),
        'train_date_start': str(df['date'].iloc[0]),
        'train_date_end': str(df['date'].iloc[split_idx - 1]),
        'generated_at': datetime.utcnow().isoformat(),
    }, f, indent=2)
```

**The test set is sealed from this point.** It is never regenerated without human approval and a DEVIATIONS.md entry.

---

## P1.9 Diagnostic figures

Four figures must be produced by `poc/data/plots.py`. All figures use the Vantage dark theme (background `#0d1117`, surface `#111820`) and Outfit font.

### Figure 1: `p1_cosmos_diagnostic.png`

Three-panel figure:
- **Panel 1 (top):** Full 4-year VWC time series. Blue line = valid VWC. Orange scatter = frozen days. Light blue scatter = snow days. Red × = flagged E/I. X-axis: date. Y-axis: VWC (cm³/cm³).
- **Panel 2 (middle):** Monthly mean VWC ± 1 SD (non-frozen, non-snow days only). Green line with shaded band.
- **Panel 3 (bottom):** Day-of-year climatology (DOY 1–365). Purple line ± 1 SD. Horizontal dashed lines for winter mean and summer mean with labels.

### Figure 2: `p1_sar_diagnostic.png`

Three-panel figure:
- **Panel 1:** VV (dB) time series, scatter plot by date. Colour = orbit number (should be single colour). Y-axis range: [−20, −5].
- **Panel 2:** VH/VV ratio (dB) time series.
- **Panel 3:** Scatter plot of VV (dB) vs VWC (cm³/cm³) for aligned paired observations. Include Pearson r and p-value as text annotation. Colour points by season (DJF blue, MAM green, JJA amber, SON orange).

### Figure 3: `p1_ancillary_diagnostic.png`

Two-panel figure:
- **Panel 1:** NDVI time series at SAR overpass dates (after interpolation). Scatter plot. Annotate gaps where cloud prevented monthly composite.
- **Panel 2:** Precipitation (bar chart, daily mm). Overlay: 7-day antecedent index (line). X-axis: SAR overpass dates only.

### Figure 4: `p1_aligned_dataset_summary.png`

Two-panel figure:
- **Panel 1:** Attrition waterfall bar chart showing row counts at each QC step. Bars coloured green (retained) / red (excluded).
- **Panel 2:** Feature correlation heatmap of the final `aligned_dataset.csv` (Pearson r between all pairs). Include `vwc` as the first row/column. Use diverging colormap (red = negative, blue = positive, white = zero).

---

## P1.10 Gate 1 script

**Script:** `poc/gates/gate_1.py`  
**Run with:** `python poc/gates/gate_1.py` or `make gate-1`  
**Exit code:** 0 = pass, 1 = fail (any criterion not met)

### Gate 1 criteria

| ID | Criterion | Threshold | Source | Auto-checkable |
|----|-----------|-----------|--------|----------------|
| G1-01 | Paired observations | ≥ 100 | `aligned_dataset.csv` row count | Yes |
| G1-02 | VWC range (non-null) | All in [0.10, 1.00] cm³/cm³ | `vwc` column | Yes |
| G1-03 | VV backscatter range | All in [−20, −5] dB | `vv_db` column | Yes |
| G1-04 | Seasonal VWC signal | Winter mean > Summer mean | computed | Yes |
| G1-05 | Raw SAR–VWC correlation | Pearson r ≥ 0.30 | `vv_db` vs `vwc` | Yes |
| G1-06 | No NaN in aligned dataset | 0 missing values | all columns | Yes |
| G1-07 | Test indices file exists | file present + non-empty | `data/splits/test_indices.json` | Yes |
| G1-08 | DEVIATIONS.md reviewed | human sign-off | manual | No — requires `--confirm-deviations` flag |
| G1-09 | All diagnostic figures exist | 4 files present | `outputs/figures/` | Yes |
| G1-10 | GEE scripts importable | `import poc.data.gee.*` succeeds | all 4 gee scripts | Yes |

### Command-line interface

```
python poc/gates/gate_1.py [--confirm-deviations]

Options:
  --confirm-deviations    Provides human sign-off for G1-08.
                          Without this flag, G1-08 always fails.
```

### Terminal output format

```
═══════════════════════════════════════════════════════════════
 ECHO PoC — Gate 1: Data Acquisition & Alignment
 Run at: 2026-03-06T12:34:56
═══════════════════════════════════════════════════════════════

 Criterion              Threshold         Measured          Status
 ─────────────────────────────────────────────────────────────
 G1-01 Paired obs       ≥ 100             142               PASS
 G1-02 VWC range        [0.10, 1.00]      [0.234, 0.978]    PASS
 G1-03 VV range         [−20, −5] dB      [−16.8, −9.1]     PASS
 G1-04 Seasonal VWC     winter > summer   0.689 > 0.531     PASS
 G1-05 SAR–VWC corr     r ≥ 0.30          r = 0.61          PASS
 G1-06 No NaN           0                 0                 PASS
 G1-07 Test indices     file present      ✓                 PASS
 G1-08 Deviations       manual sign-off   NOT PROVIDED      FAIL
 G1-09 Figures          4 files           4 found           PASS
 G1-10 GEE importable   all pass          ✓                 PASS

 ─────────────────────────────────────────────────────────────
 Result: FAIL  (1 criterion not met)

 To pass G1-08: re-run with --confirm-deviations after reviewing
 DEVIATIONS.md and confirming all entries are acknowledged.
═══════════════════════════════════════════════════════════════
```

### JSON output (`outputs/gates/gate_1_result.json`)

Written on every run, regardless of pass/fail:

```json
{
  "gate": 1,
  "timestamp": "2026-03-06T12:34:56Z",
  "passed": false,
  "exit_code": 1,
  "criteria": {
    "G1-01": {"description": "Paired observations", "threshold": 100, "measured": 142, "passed": true},
    "G1-02": {"description": "VWC range", "threshold": [0.10, 1.00], "measured": [0.234, 0.978], "passed": true},
    "G1-03": {"description": "VV range", "threshold": [-20, -5], "measured": [-16.8, -9.1], "passed": true},
    "G1-04": {"description": "Seasonal signal", "threshold": "winter > summer", "measured": {"winter": 0.689, "summer": 0.531}, "passed": true},
    "G1-05": {"description": "SAR-VWC correlation", "threshold": 0.30, "measured": 0.61, "passed": true},
    "G1-06": {"description": "No NaN in aligned dataset", "threshold": 0, "measured": 0, "passed": true},
    "G1-07": {"description": "Test indices file", "threshold": "present", "measured": "present", "passed": true},
    "G1-08": {"description": "Deviations reviewed", "threshold": "manual", "measured": "not confirmed", "passed": false},
    "G1-09": {"description": "Diagnostic figures", "threshold": 4, "measured": 4, "passed": true},
    "G1-10": {"description": "GEE scripts importable", "threshold": "all pass", "measured": "all pass", "passed": true}
  },
  "attrition": {
    "s1_overpasses_raw": null,
    "final_paired_observations": 142
  },
  "warnings": []
}
```

---

## P1.11 Gate 1 failure protocol

If `final_paired_observations < 80`, **stop immediately**. Do not attempt options below until this is checked.

If `final_paired_observations` is in [80, 99], proceed with caution — statistical power is reduced. Document in DEVIATIONS.md.

If `final_paired_observations < 80`:

| Option | Action | Trigger condition |
|--------|--------|-------------------|
| **A — Extend period** | Re-run Task 1.1 with `STUDY_START = "2018-01-01"` | First choice — adds ~120 candidate observations |
| **B — Relax QC** | Allow ta_min in [−1°C, 0°C] instead of strict < 0 | Only if extension still insufficient |
| **C — Reduced power** | Proceed, acknowledge in all materials | Only if options A and B are exhausted |
| **D — Stop** | Document for Carbon13, reassess site | If r(VV, VWC) < 0.30 — signal too weak |

Any option taken requires a DEVIATIONS.md entry before proceeding.

If `G1-05` (SAR–VWC correlation) fails (r < 0.30), compute the scatter plot of VV vs VWC and examine by season. If the correlation is strong in winter but absent in summer (likely due to vegetation masking), document this as a known limitation and proceed. If no seasonal period shows r ≥ 0.30, escalate to Option D.

---

## P1.12 Phase 1 test requirements

**Coverage target:** ≥ 80% line coverage on `poc/data/` (excluding `poc/data/gee/` scripts — these are tested by integration tests only).

### Required unit tests (`tests/unit/test_data_p1.py`)

| Test | What it checks |
|------|---------------|
| `test_load_cosmos_converts_percent_to_fraction` | vwc_raw = input_percent / 100 |
| `test_load_cosmos_flags_exclusion_sets_nan` | vwc_qc is NaN when flag is 'E' or 'I' |
| `test_load_cosmos_frozen_flag` | frozen_flag=1 when ta_min=-0.1, frozen_flag=0 when ta_min=0.0 |
| `test_load_cosmos_snow_flag` | snow_flag=1 when snow=0.1, snow_flag=0 when snow=0.0 |
| `test_load_cosmos_validation_fails_on_wrong_range` | raises ValueError if vwc > 1.0 |
| `test_vhvv_computation` | vhvv_db = vh_db − vv_db (log-space ratio) |
| `test_antecedent_precip_excludes_current_day` | day 8 antecedent = sum of days 1–7, not day 8 |
| `test_aspect_circular_encoding` | aspect 0° → sin=0, cos=1; aspect 90° → sin=1, cos=0 |
| `test_twi_floor_prevents_log_infinity` | slope=0 does not raise ZeroDivisionError or return inf |
| `test_alignment_excludes_frozen` | rows with frozen_flag=1 absent from aligned output |
| `test_alignment_excludes_snow` | rows with snow_flag=1 absent from aligned output |
| `test_alignment_no_nan_in_output` | aligned_dataset has zero NaN values |
| `test_alignment_target_column_named_vwc` | output column is 'vwc' not 'vwc_qc' |
| `test_train_test_split_is_chronological` | all test dates > all training dates |
| `test_train_test_split_ratio` | test set is 28–32% of total (allows for rounding) |

### Required integration tests (`tests/integration/test_pipeline_p1.py`)

Require `data/processed/aligned_dataset.csv` to exist. Skip with `pytest.mark.skipif` if not present.

| Test | What it checks |
|------|---------------|
| `test_aligned_dataset_schema` | All columns present, correct dtypes |
| `test_aligned_dataset_no_nan` | Zero missing values |
| `test_aligned_dataset_date_range` | Dates span 2021–2024 |
| `test_vwc_range_plausible` | VWC in [0.10, 1.00] |
| `test_sar_vwc_correlation` | Pearson r(vv_db, vwc) ≥ 0.30 |

---

## P1.13 Phase 1 file manifest (gate requires all present)

```
data/raw/cosmos/COSMOS_UK_MOORH_1D_202101010000_202412310000.csv
data/raw/gee/sentinel1_raw.csv
data/raw/gee/sentinel2_ndvi_raw.csv
data/raw/gee/era5_precip_raw.csv
data/raw/gee/terrain_static_raw.csv
data/processed/cosmos_daily_vwc.csv
data/processed/sentinel1_extractions.csv
data/processed/ancillary_features.csv
data/processed/aligned_dataset.csv
data/splits/test_indices.json
outputs/figures/p1_cosmos_diagnostic.png
outputs/figures/p1_sar_diagnostic.png
outputs/figures/p1_ancillary_diagnostic.png
outputs/figures/p1_aligned_dataset_summary.png
outputs/gates/gate_1_attrition.json
outputs/gates/gate_1_result.json
poc/config.py
poc/data/cosmos.py
poc/data/ancillary.py
poc/data/alignment.py
poc/data/gee/__init__.py
poc/data/gee/extract_sentinel1.py
poc/data/gee/extract_sentinel2.py
poc/data/gee/extract_era5.py
poc/data/gee/extract_terrain.py
poc/gates/gate_1.py
DEVIATIONS.md
```
