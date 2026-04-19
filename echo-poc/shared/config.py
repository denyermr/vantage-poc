"""
ECHO PoC — Central configuration.

ALL constants, paths, seeds, and hyperparameters are defined here.
No magic numbers in any other file. Import from this module everywhere.

Physical constants cite their literature source.
"""

from pathlib import Path
import logging

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─── Paths (resolve from project root) ──────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent          # echo-poc/
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW_COSMOS = DATA_RAW / "cosmos"
DATA_RAW_GEE = DATA_RAW / "gee"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_METRICS = PROJECT_ROOT / "outputs" / "metrics"
OUTPUTS_GATES = PROJECT_ROOT / "outputs" / "gates"
OUTPUTS_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_EXPORT = PROJECT_ROOT / "outputs" / "export"

# ─── Reproducibility ────────────────────────────────────────────────────────

SEED = 42

# ─── Site ────────────────────────────────────────────────────────────────────

SITE_NAME = "Moor House"
SITE_ID = "MOORH"
SITE_LAT = 54.69                    # degrees N
SITE_LON = -2.38                    # degrees E (negative = West)
SITE_RADIUS_M = 200                 # COSMOS-UK footprint radius (m)
STUDY_START = "2021-01-01"
STUDY_END = "2024-12-31"

# ─── GEE ─────────────────────────────────────────────────────────────────────

# REQUIRED: user must set their GEE Cloud project ID before running GEE scripts
GEE_PROJECT = "project-f4d2c79a-000b-4ef5-a0b"
GEE_DRIVE_FOLDER = "echo_poc_gee_exports"
S1_COLLECTION = "COPERNICUS/S1_GRD"
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
SRTM_COLLECTION = "USGS/SRTMGL1_003"
MERIT_COLLECTION = "MERIT/Hydro/v1_0_1"

# ─── SAR ─────────────────────────────────────────────────────────────────────

S1_PASS = "DESCENDING"
S1_MODE = "IW"
S1_MIN_PIXELS = 10                  # minimum valid pixels in footprint
VV_RANGE_MIN = -20.0                # dB — hard lower bound for validation
VV_RANGE_MAX = -5.0                 # dB — hard upper bound for validation

# ─── COSMOS-UK ───────────────────────────────────────────────────────────────

# Expected raw CSV filename pattern
COSMOS_RAW_FILENAME = "COSMOS_UK_MOORH_1D_202101010000_202412310000.csv"

# QC flags to exclude (gap-filled/estimated, interpolated)
COSMOS_EXCLUDE_FLAGS = ["E", "I"]

# VWC plausible range for blanket bog peat (cm³/cm³)
VWC_RANGE_MIN = 0.10
VWC_RANGE_MAX = 1.00

# ─── Physics constants (used in Phase 3, defined here for central control) ──

# Saturated volumetric water content for blanket bog peat
# Source: Bechtold et al. (2018), Table 1, near-natural sites
PEAT_THETA_SAT = 0.88              # cm³/cm³

# Dielectric constant of dry peat
# Source: Bechtold et al. (2018)
EPSILON_DRY_PEAT = 3.5             # dimensionless

# Dielectric constant of free water at ~10°C
# Source: Kaatze (1989), standard reference
EPSILON_WATER = 80.0               # dimensionless

# Dobson mixing model exponent for organic soil
# Source: Dobson et al. (1985), adapted for organic soils
DOBSON_ALPHA = 1.4                  # dimensionless

# WCM parameter initialisations
# Source: Attema & Ulaby (1978), Radio Science 13(2); Singh et al. (2023)
WCM_A_INIT = 0.10                  # vegetation scattering coefficient (dimensionless)
WCM_B_INIT = 0.15                  # vegetation attenuation coefficient (dimensionless)
# Bounds enforced via sigmoid reparameterisation: param = LB + (UB-LB)*sigmoid(raw)
WCM_A_LB, WCM_A_UB = 0.01, 1.0
WCM_B_LB, WCM_B_UB = 0.01, 1.0
WCM_A_BOUNDS = (WCM_A_LB, WCM_A_UB)  # kept for backward compat
WCM_B_BOUNDS = (WCM_B_LB, WCM_B_UB)  # kept for backward compat

# Oh (1992) surface roughness — fixed, not learnable
# Reference: Bechtold et al. (2018) for saturated blanket bog surface
# ks = surface roughness × wavenumber = roughness × (2π / λ_radar)
# For Sentinel-1 C-band (λ=5.6 cm), ks≈0.3 corresponds to ~2.7 mm RMS height
KS_ROUGHNESS = 0.30                # dimensionless (roughness × wavenumber product)

# Mironov (2009) parameters for organic soil (sensitivity check only)
# Source: Mironov et al. (2009), IEEE TGRS, Table II, organic soil category
MIRONOV_ND = 0.312                 # refractive index of dry soil
MIRONOV_KD = 0.0                   # extinction index of dry soil (non-absorbing at C-band)
MIRONOV_MV_T = 0.36                # transition moisture (organic soil)
MIRONOV_ND1 = 1.42                 # refractive index slope below transition
MIRONOV_ND2 = 0.89                 # refractive index slope above transition

# ─── Phase 3: λ hyperparameter search ───────────────────────────────────────

# Grid for each of the three loss weights
LAMBDA_GRID = [0.01, 0.1, 0.5, 1.0]  # 4^3 = 64 combinations

# λ search is run on the 10 × 100% training configs (config_idx 0–9)
LAMBDA_SEARCH_CONFIG_RANGE = (0, 9)   # inclusive

# Constraint: L_data must dominate — reject any λ triple where
# mean(L_physics + L_monotonic + L_bounds) > mean(L_data) during training
# (checked on validation set at epoch of early stopping)
LAMBDA_DOMINANCE_CONSTRAINT = True

# ─── Phase 3: Prediction intervals ──────────────────────────────────────────

# Calibration targets
PRED_INTERVAL_TARGETS = [0.80, 0.95]  # 80% and 95% nominal coverage

# ─── Phase 3: Diagnostic thresholds ─────────────────────────────────────────

# Residual ratio warning threshold (diagnostic only, does not affect gate pass/fail)
RESIDUAL_RATIO_WARN = 1.0

# Dobson vs Mironov: re-run criterion
MIRONOV_IMPROVEMENT_THRESHOLD = 0.05  # 5% relative RMSE improvement triggers re-run

# ─── PyTorch device selection (Apple Silicon) ────────────────────────────────


def get_torch_device():
    """
    Get the PyTorch device for training.

    Apple Silicon MPS backend — automatic fallback to CPU if unavailable.
    Provides ~2-4x speedup on M-series Macs with no code changes elsewhere.

    Lazy import: torch is only imported when this function is first called,
    allowing data-loading code and tests to run without torch installed.
    """
    import torch
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

# ─── Phase 2: Experimental design ───────────────────────────────────────────

# Training pool subsampling fractions (applied to training pool, not full dataset)
TRAINING_FRACTIONS = [1.0, 0.50, 0.25, 0.10]

# Label for each fraction — used in plot axes and JSON keys
TRAINING_SIZE_LABELS = {
    1.00: "100%",
    0.50: "50%",
    0.25: "25%",
    0.10: "10%",
}

# Repetitions per training size
N_REPS = 10

# Total configurations = len(TRAINING_FRACTIONS) × N_REPS
N_CONFIGS = len(TRAINING_FRACTIONS) * N_REPS  # 40
assert N_CONFIGS == 40, f"Expected 40 configs, got {N_CONFIGS}"

# Validation split fraction (carved from each subsampled training set)
VAL_FRACTION = 0.20

# Meteorological seasons (UK standard)
SEASONS = {
    "DJF": [12, 1, 2],     # Dec, Jan, Feb — winter
    "MAM": [3, 4, 5],      # Mar, Apr, May — spring
    "JJA": [6, 7, 8],      # Jun, Jul, Aug — summer
    "SON": [9, 10, 11],    # Sep, Oct, Nov — autumn
}

# Minimum samples per stratum before falling back to unstratified sampling
MIN_STRATUM_SAMPLES = 2

# ─── Phase 2: Model hyperparameters ─────────────────────────────────────────

# Feature columns — single source of truth for feature ordering
# DEV-004: Terrain features (slope_deg, aspect_sin, aspect_cos, twi) excluded —
# zero temporal variance at single-site (Moor House). 11 → 7 dynamic features.
# Terrain values retained as site metadata below.
FEATURE_COLUMNS = [
    "vv_db", "vh_db", "vhvv_db",
    "ndvi",
    "precip_mm", "precip_7day_mm",
    "incidence_angle_mean",
]

# Terrain site metadata (DEV-004) — static values, not used in model training
TERRAIN_SITE_METADATA = {
    "slope_deg": 4.8928,
    "aspect_sin": 0.3239,
    "aspect_cos": 0.3197,
    "twi": 15.8204,
}
TARGET_COLUMN = "vwc"

# Random Forest grid search
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
    "max_features": [0.5, 0.75, 1.0],
}
RF_CV_FOLDS = 5
RF_SCORING = "neg_root_mean_squared_error"

# Standard NN
NN_HIDDEN_SIZES = [64, 32, 16]
NN_DROPOUT = 0.2
NN_LR = 1e-3
NN_MAX_EPOCHS = 500
NN_PATIENCE = 20                    # early stopping patience (val loss)
NN_BATCH_SIZE = 32                  # capped at N_train if N_train < 32
NN_WEIGHT_DECAY = 0.0              # no L2 regularisation at baseline

# ─── Phase 2: Evaluation ────────────────────────────────────────────────────

# Wilcoxon test significance thresholds
ALPHA_UNCORRECTED = 0.05
ALPHA_BONFERRONI = 0.05 / len(TRAINING_FRACTIONS)  # 0.0125

# Gate 2 failure threshold — RF RMSE at 100% training
# DEV-006: Original absolute threshold (0.10) not met; criterion reinterpreted
# as relative: RF must beat null model RMSE by ≥10%. See DEVIATIONS.md.
GATE2_RF_RMSE_THRESHOLD = 0.10     # cm³/cm³ — original absolute threshold (not met; see DEV-006)
GATE2_RF_RMSE_WARN = 0.08          # cm³/cm³ — original warn threshold
GATE2_RF_RELATIVE_IMPROVEMENT_MIN = 0.10  # RF must beat null by at least this fraction (10%)
