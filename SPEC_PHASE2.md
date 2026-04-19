# SPEC_PHASE2.md — Phase 2: Baseline Models & Evaluation Harness
# ECHO PoC — Vantage

**Prerequisite:** Phase 1 gate passed (`outputs/gates/gate_1_result.json` → `"passed": true`)  
**Version:** 1.0 — 06 March 2026  
**Goal:** Three baseline models (null model, Random Forest, standard NN) trained and evaluated across all 40 split configurations. Evaluation harness built and validated. Learning curve results in hand before the PINN enters Phase 3.

---

## P2. Overview

Phase 2 produces:

1. **40 saved train/val/test split configurations** — the sealed experimental design
2. **Three trained baseline models** per configuration — Baseline 0 (null), Baseline A (RF), Baseline B (NN)
3. **A complete metrics JSON** for every (model × config) combination
4. **The evaluation harness** — reusable by Phase 3 for the PINN without modification
5. **Two diagnostic figures** — learning curves and feature diagnostics

The Phase 3 PINN is trained on the **identical** splits produced here. The splits are the experimental contract between phases. They are never regenerated after this phase passes its gate.

---

## P2.1 Repository additions (Phase 2)

```
echo-poc/
├── data/
│   └── splits/
│       ├── test_indices.json              ← from Phase 1 (sealed)
│       ├── split_manifest.json            ← new: describes all 40 configurations
│       └── configs/
│           ├── config_000.json            ← train/val indices for size=100%, rep=0
│           ├── config_001.json            ← size=100%, rep=1
│           ...
│           └── config_039.json            ← size=10%, rep=9
├── poc/
│   ├── config.py                          ← additions only (see §P2.2)
│   └── models/
│       ├── __init__.py
│       ├── null_model.py                  ← Baseline 0
│       ├── random_forest.py               ← Baseline A
│       ├── standard_nn.py                 ← Baseline B
│       └── base.py                        ← abstract base class all models implement
│   └── evaluation/
│       ├── __init__.py
│       ├── harness.py                     ← metrics, statistical tests
│       ├── splits.py                      ← split generation and loading
│       └── plots.py                       ← learning curve figures
│   └── gates/
│       └── gate_2.py
├── outputs/
│   ├── models/
│   │   ├── baseline_0/                    ← null model artefacts
│   │   ├── baseline_a/                    ← RF artefacts per config
│   │   └── baseline_b/                    ← NN artefacts per config
│   ├── metrics/
│   │   ├── baseline_0_metrics.json        ← null model (single result, no configs)
│   │   ├── config_000_baseline_a.json
│   │   ├── config_000_baseline_b.json
│   │   ...
│   │   └── config_039_baseline_b.json
│   ├── figures/
│   │   ├── p2_learning_curves_baselines.png
│   │   └── p2_feature_diagnostics.png
│   └── gates/
│       └── gate_2_result.json
└── tests/
    ├── unit/
    │   └── test_models_p2.py
    └── integration/
        └── test_pipeline_p2.py
```

---

## P2.2 Config additions (`poc/config.py`)

Append to the existing `poc/config.py`. Do not modify existing constants.

```python
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
N_CONFIGS = 40  # assert at runtime

# Validation split fraction (carved from each subsampled training set)
# Random 80/20 split — seed = SEED + config_index for reproducibility
VAL_FRACTION = 0.20

# Meteorological seasons (UK standard)
# Used for stratified subsampling at each training size
SEASONS = {
    "DJF": [12, 1, 2],   # Dec, Jan, Feb — winter
    "MAM": [3, 4, 5],    # Mar, Apr, May — spring
    "JJA": [6, 7, 8],    # Jun, Jul, Aug — summer
    "SON": [9, 10, 11],  # Sep, Oct, Nov — autumn
}

# Minimum samples per stratum before falling back to unstratified sampling
# If any season has fewer than MIN_STRATUM_SAMPLES in the candidate pool,
# stratification is abandoned for that config and a warning is logged.
MIN_STRATUM_SAMPLES = 2

# ─── Phase 2: Model hyperparameters ─────────────────────────────────────────

# Random Forest grid search
RF_PARAM_GRID = {
    "n_estimators":    [100, 200],
    "max_depth":       [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
    "max_features":    [0.5, 0.75, 1.0],
}
RF_CV_FOLDS = 5
RF_SCORING   = "neg_root_mean_squared_error"  # GridSearchCV scoring

# Standard NN
NN_HIDDEN_SIZES  = [64, 32, 16]
NN_DROPOUT       = 0.2
NN_LR            = 1e-3
NN_MAX_EPOCHS    = 500
NN_PATIENCE      = 20           # early stopping patience (val loss)
NN_BATCH_SIZE    = 32           # capped at N_train if N_train < 32
NN_WEIGHT_DECAY  = 0.0          # no L2 regularisation at baseline

# ─── Phase 2: Evaluation ────────────────────────────────────────────────────

# Wilcoxon test: report both uncorrected p-value and Bonferroni threshold
# 4 comparisons (one per training size) → Bonferroni α = 0.05 / 4
ALPHA_UNCORRECTED  = 0.05
ALPHA_BONFERRONI   = 0.05 / len(TRAINING_FRACTIONS)   # = 0.0125

# Gate 2 failure threshold — RF RMSE at 100% training
GATE2_RF_RMSE_THRESHOLD = 0.10   # cm³/cm³ — hard fail above this
GATE2_RF_RMSE_WARN      = 0.08   # cm³/cm³ — warn above this, pass if ≤ 0.10
```

---

## P2.3 Abstract base model (`poc/models/base.py`)

All three baselines and the Phase 3 PINN must implement this interface. The evaluation harness only calls these methods — it has no model-specific logic.

```python
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """
    Abstract base for all ECHO PoC models.
    Enforces the interface the evaluation harness expects.
    """

    model_name: str  # must be set as class attribute

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X_train: Feature matrix, shape (N_train, n_features). Normalised.
            y_train: Target VWC, shape (N_train,). cm³/cm³.
            X_val:   Validation features, shape (N_val, n_features). Normalised.
            y_val:   Validation targets, shape (N_val,). cm³/cm³.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return point predictions.

        Args:
            X: Feature matrix, shape (N, n_features). Normalised.

        Returns:
            y_pred: Predicted VWC, shape (N,). cm³/cm³.
        """

    @abstractmethod
    def save(self, directory: Path) -> None:
        """
        Serialise model artefacts to directory.
        Must save config alongside weights so model can be reloaded
        without the original training script.
        """

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "BaseModel":
        """
        Restore model from saved artefacts.
        """
```

---

## P2.4 Split generation (`poc/evaluation/splits.py`)

This module generates and persists all 40 configurations. It is called **once** at the start of Phase 2 and never again.

### Season assignment

```python
def assign_season(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Assign meteorological season label to each date.

    Seasons (UK meteorological standard):
        DJF: Dec, Jan, Feb
        MAM: Mar, Apr, May
        JJA: Jun, Jul, Aug
        SON: Sep, Oct, Nov

    Returns:
        Array of season strings, same length as dates.
    """
    month_to_season = {}
    for season, months in config.SEASONS.items():
        for m in months:
            month_to_season[m] = season
    return np.array([month_to_season[d.month] for d in dates])
```

### Stratified subsampling

```python
def stratified_subsample(
    pool_indices: np.ndarray,
    dates: pd.DatetimeIndex,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a stratified random subsample from pool_indices.

    Stratification by meteorological season. Each season contributes
    proportionally to its representation in the pool.

    If any season has fewer than MIN_STRATUM_SAMPLES samples in the pool,
    falls back to unstratified random sampling and logs a warning.

    Args:
        pool_indices: Integer indices into aligned_dataset (training pool only).
        dates:        Full aligned_dataset date index (all N rows).
        fraction:     Target fraction of pool to retain (e.g. 0.25).
        rng:          NumPy Generator with fixed seed.

    Returns:
        Subsampled indices, length = floor(len(pool_indices) * fraction).
        Minimum 1 sample guaranteed.

    Raises:
        ValueError: If fraction not in (0, 1].
    """
```

**Stratification algorithm:**
```
n_target = max(1, floor(len(pool_indices) * fraction))
seasons = assign_season(dates[pool_indices])

for each season s in [DJF, MAM, JJA, SON]:
    n_season_pool = count of pool_indices in season s
    n_season_target = round(n_target * n_season_pool / len(pool_indices))
    n_season_target = max(1, n_season_target) if n_season_pool > 0 else 0

# If any season has < MIN_STRATUM_SAMPLES: fall back to unstratified
# Unstratified: rng.choice(pool_indices, size=n_target, replace=False)

# If stratified: sample n_season_target from each season, concatenate
# If total from stratification ≠ n_target: adjust largest stratum by ±1
```

### Configuration generation

```python
def generate_all_configs(
    aligned_dataset: pd.DataFrame,
    test_indices_path: Path,
    output_dir: Path,
) -> None:
    """
    Generate all 40 train/val split configurations and write to output_dir.

    Config index assignment (deterministic):
        config_idx = size_idx * N_REPS + rep_idx
        where size_idx ∈ {0,1,2,3} for fractions [1.0, 0.5, 0.25, 0.1]
        and   rep_idx  ∈ {0..9}

    So:
        configs 000–009: fraction=1.0, reps 0–9
        configs 010–019: fraction=0.5, reps 0–9
        configs 020–029: fraction=0.25, reps 0–9
        configs 030–039: fraction=0.10, reps 0–9

    Each config JSON contains:
        {
            "config_idx": 0,
            "fraction": 1.0,
            "fraction_label": "100%",
            "rep": 0,
            "seed_used": 42,               # SEED + config_idx
            "n_train": 74,                 # actual count after stratification
            "n_val": 18,
            "train_indices": [...],        # positional indices into aligned_dataset
            "val_indices": [...],          # positional indices into aligned_dataset
            "season_counts_train": {"DJF": 20, "MAM": 18, "JJA": 14, "SON": 22},
            "stratification_used": true,
            "generated_at": "2026-03-06T..."
        }

    Test indices are loaded from test_indices_path and are NOT stored in
    config files — they are always loaded separately from test_indices.json.
    """
```

**Validation split within each configuration:**
```
From the subsampled training set of size N_train:
    rng_val = np.random.default_rng(SEED + config_idx + 1000)  # distinct seed
    n_val = max(1, round(N_train * VAL_FRACTION))
    val_mask = rng_val.choice(N_train, size=n_val, replace=False)
    train_final = remaining after removing val_mask
    val_final = val_mask
```

**Important:** The validation set is carved from the **subsampled** training set, not the full training pool. This means the 10% configurations have a very small validation set (~2 samples). This is expected and intentional — the evaluation harness uses the **sealed test set** for all reported metrics. The validation set is only used for early stopping and CV, not for any reported result.

### Split manifest

After all 40 configs are generated, write `data/splits/split_manifest.json`:

```json
{
    "n_configs": 40,
    "n_test": 43,
    "training_pool_size": 99,
    "fractions": [1.0, 0.5, 0.25, 0.10],
    "n_reps": 10,
    "seed_base": 42,
    "stratification": "meteorological_DJF_MAM_JJA_SON",
    "val_fraction": 0.20,
    "generated_at": "2026-03-06T...",
    "configs_summary": [
        {"config_idx": 0, "fraction": 1.0, "rep": 0, "n_train": 74, "n_val": 18},
        ...
    ]
}
```

---

## P2.5 Baseline 0 — Null Model (`poc/models/null_model.py`)

### Definition

The null model predicts the **seasonal climatological mean** for each observation. It uses no SAR or ancillary features — only the month of year.

```
For each observation with date d:
    season(d) → one of {DJF, MAM, JJA, SON}
    prediction = mean(y_train[season == season(d)])
```

If a season has no training samples (possible at N=10), fall back to the global training mean.

### Purpose

- Establishes a **performance floor**: any ML model that cannot beat the null model RMSE adds no value beyond seasonal pattern recognition
- Provides the **observation uncertainty floor** for Phase 3 uncertainty quantification (§7.3 of main SPEC.md)
- Is evaluated once on the **full training pool** (not the 40 subsampled configs) — its RMSE does not vary with training size in a meaningful way

### Implementation requirements

```python
class NullModel(BaseModel):
    model_name = "null_baseline"

    def fit(self, X_train, y_train, X_val, y_val,
            train_dates: pd.DatetimeIndex) -> None:
        """
        X_train and X_val are accepted but ignored.
        train_dates required to assign seasons to training labels.
        """

    def predict(self, X: np.ndarray,
                pred_dates: pd.DatetimeIndex) -> np.ndarray:
        """
        X is ignored. Prediction based on pred_dates season only.
        """
```

**Note:** The null model's `fit` and `predict` signatures extend the base class with `dates` arguments. This is acceptable — the evaluation harness handles it via an `isinstance` check or a `uses_dates: bool` flag on the model class.

### Output file: `outputs/metrics/baseline_0_metrics.json`

```json
{
    "model": "null_baseline",
    "config_idx": "full_pool",
    "training_fraction": 1.0,
    "n_train": 99,
    "n_test": 43,
    "metrics": {
        "rmse": 0.112,
        "r_squared": 0.021,
        "mean_bias": 0.003
    },
    "seasonal_means_train": {
        "DJF": 0.721,
        "MAM": 0.643,
        "JJA": 0.512,
        "SON": 0.674
    },
    "note": "Performance floor. Any model with RMSE > this value adds no predictive value beyond seasonal climatology."
}
```

---

## P2.6 Baseline A — Random Forest (`poc/models/random_forest.py`)

### Training procedure

For each of the 40 configs:

```
1. Load config_{idx}.json → train_indices, val_indices
2. Load test_indices.json → test_indices
3. Extract X_train, y_train from aligned_dataset at train_indices
4. Extract X_val,   y_val   from aligned_dataset at val_indices
5. Extract X_test,  y_test  from aligned_dataset at test_indices

6. Fit normaliser on X_train ONLY:
       scaler = StandardScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_val_scaled   = scaler.transform(X_val)
       X_test_scaled  = scaler.transform(X_test)

7. GridSearchCV on (X_train_scaled, y_train):
       cv = KFold(n_splits=RF_CV_FOLDS, shuffle=True,
                  random_state=SEED + config_idx)
       grid = GridSearchCV(
           RandomForestRegressor(random_state=SEED + config_idx, n_jobs=-1),
           RF_PARAM_GRID,
           scoring=RF_SCORING,
           cv=cv,
           refit=True,   # refit on full X_train after search
       )
       grid.fit(X_train_scaled, y_train)

8. Best estimator = grid.best_estimator_
9. y_pred_test = best_estimator.predict(X_test_scaled)
10. Compute metrics on (y_pred_test, y_test)
11. Optionally compute feature importances (log, do not gate)
12. Save model artefacts to outputs/models/baseline_a/config_{idx}/
13. Write outputs/metrics/config_{idx}_baseline_a.json
```

**Note on GridSearchCV with small N:** At N_train=10 (10% config), 5-fold CV leaves only 2 samples per fold. This is statistically noisy but not an error — it's one of the documented data-scarcity conditions being tested. The grid search still runs; the best params may be poorly estimated. Log a warning for config_idx ≥ 30 (10% configs): `"WARNING: N_train={n} — CV reliability limited"`.

### Normaliser persistence

The `StandardScaler` fitted on the training set must be saved alongside the model:

```python
# In save():
joblib.dump(self.scaler, directory / "scaler.pkl")
joblib.dump(self.model,  directory / "model.pkl")
json.dump({"best_params": self.best_params_,
           "n_train": self.n_train_,
           "feature_names": FEATURE_COLUMNS,
           "config_idx": self.config_idx_},
          open(directory / "config.json", "w"))
```

The scaler is **always** saved with the model and **always** loaded with it. A model without its scaler is not a valid model.

### Feature columns

```python
# In poc/config.py (add to Phase 2 section):
FEATURE_COLUMNS = [
    "vv_db", "vh_db", "vhvv_db",
    "ndvi",
    "precip_mm", "precip_7day_mm",
    "slope_deg", "aspect_sin", "aspect_cos", "twi",
    "incidence_angle_mean",
]
TARGET_COLUMN = "vwc"
```

This list is the **single source of truth** for feature ordering. Both RF and NN extract features in this order. It is defined in `config.py` and imported everywhere — never repeated in model files.

### Feature importance (optional, not gated)

```python
if hasattr(self.model, "feature_importances_"):
    importances = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))
    json.dump(importances, open(directory / "feature_importances.json", "w"))
```

Log at INFO level. Included in the figure `p2_feature_diagnostics.png` (§P2.9).

---

## P2.7 Baseline B — Standard NN (`poc/models/standard_nn.py`)

### Architecture

```python
class StandardNN(nn.Module):
    """
    Three hidden layers: 64 → 32 → 16 → 1
    ReLU activations, Dropout(0.2) between hidden layers.
    Linear output — no activation on final layer (regression).
    """
    def __init__(self, n_features: int = 11,
                 hidden_sizes: list[int] = NN_HIDDEN_SIZES,
                 dropout: float = NN_DROPOUT):
        super().__init__()
        layers = []
        in_size = n_features
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

### Training procedure

For each of the 40 configs:

```
1. Load config, extract X_train, X_val, X_test (same as RF)
2. Fit StandardScaler on X_train
3. Set seeds:
       torch.manual_seed(SEED + config_idx)
       torch.cuda.manual_seed_all(SEED + config_idx)
       np.random.seed(SEED + config_idx)       # for any numpy ops in training
4. Instantiate StandardNN(n_features=11)
5. Optimiser: Adam(lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
6. Loss: nn.MSELoss()
7. batch_size = min(NN_BATCH_SIZE, len(X_train))
8. Training loop:
       best_val_loss = inf
       patience_counter = 0
       for epoch in range(NN_MAX_EPOCHS):
           # train step
           model.train()
           for batch in DataLoader(train_ds, batch_size, shuffle=True):
               optimizer.zero_grad()
               loss = criterion(model(X_batch), y_batch)
               loss.backward()
               optimizer.step()
           # validation step
           model.eval()
           with torch.no_grad():
               val_loss = criterion(model(X_val_t), y_val_t).item()
           if val_loss < best_val_loss - 1e-6:
               best_val_loss = val_loss
               patience_counter = 0
               save_checkpoint(model, "best_checkpoint.pt")
           else:
               patience_counter += 1
           if patience_counter >= NN_PATIENCE:
               break
       # load best checkpoint before evaluation
       model.load_state_dict(torch.load("best_checkpoint.pt"))
9. y_pred_test = model(X_test_t).detach().numpy()
10. Compute metrics, save artefacts, write JSON
```

**DataLoader shuffle seed:** The DataLoader shuffle uses a `torch.Generator` seeded with `SEED + config_idx + epoch` to ensure exact reproducibility across runs.

```python
g = torch.Generator()
g.manual_seed(SEED + config_idx + epoch)
loader = DataLoader(train_ds, batch_size=batch_size,
                    shuffle=True, generator=g)
```

### Saving artefacts

```
outputs/models/baseline_b/config_{idx}/
    model_weights.pt           ← best checkpoint state dict
    scaler.pkl                 ← fitted StandardScaler
    config.json                ← hyperparams, n_train, feature_names,
                                  training_history (loss per epoch),
                                  stopped_at_epoch, best_val_loss
```

`training_history` contains `{"epoch": [...], "train_loss": [...], "val_loss": [...]}` for the full training run. This is logged but not evaluated at the gate.

---

## P2.8 Metrics JSON schema

Every model output file follows this schema. Phase 3 extends it with `physics_diagnostics`. Defining it once here ensures consistency.

**File naming:** `outputs/metrics/config_{idx:03d}_{model_key}.json`

| `model_key` | Model |
|-------------|-------|
| `baseline_0` | Null model (one file total, `config_idx = "full_pool"`) |
| `baseline_a` | Random Forest |
| `baseline_b` | Standard NN |
| `pinn` | Phase 3 — extends this schema |

**Schema:**

```json
{
    "model":            "baseline_a",
    "config_idx":       0,
    "fraction":         1.0,
    "fraction_label":   "100%",
    "rep":              0,
    "seed_used":        42,
    "n_train":          74,
    "n_val":            18,
    "n_test":           43,
    "feature_columns":  ["vv_db", "vh_db", ...],
    "metrics": {
        "rmse":         0.051,
        "r_squared":    0.812,
        "mean_bias":    0.003
    },
    "training_metadata": {
        "best_params":          {"n_estimators": 200, "max_depth": null, ...},
        "stopped_at_epoch":     null,
        "best_val_loss":        null,
        "cv_warning":           false,
        "stratification_used":  true
    },
    "warnings":         [],
    "generated_at":     "2026-03-06T12:00:00Z"
}
```

**Rules:**
- `metrics.rmse` is always in cm³/cm³
- `metrics.r_squared` can be negative (valid for poor models — do not clamp)
- `warnings` is a list of strings — non-empty if any anomalies occurred during training
- `training_metadata` fields not applicable to a model are `null` (not omitted)
- The schema is validated by a Zod-equivalent schema in the gate script

---

## P2.9 Diagnostic figures

### Figure 1: `p2_learning_curves_baselines.png`

Two-panel figure using Vantage dark theme.

**Panel 1 — RMSE learning curves:**
- X-axis: Training set size label ["10%", "25%", "50%", "100%"] — categorical, evenly spaced
- Y-axis: RMSE (cm³/cm³)
- Three series:
  - Baseline A (RF): cyan `#38bdf8`, solid line connecting medians, IQR shaded band
  - Baseline B (NN): purple `#a78bfa`, solid line, IQR shaded band
  - Null model: amber `#fbbf24`, horizontal dashed line at its RMSE (single value)
- Each point: median RMSE across 10 repetitions
- IQR band: 25th–75th percentile across 10 repetitions
- Annotate the N=25 point with a vertical dashed line labelled "Critical threshold"
- Y-axis range: [0, max(null_rmse) × 1.1]
- Legend: bottom-right
- Title: "Baseline Learning Curves — RMSE vs Training Size"

**Panel 2 — R² learning curves:**
- Same structure as Panel 1, Y-axis: R²
- Y-axis range: [min(0, floor(min_r2)), 1.0]
- Title: "Baseline Learning Curves — R² vs Training Size"

**Required annotation on both panels:**
- Text box at N=25 showing: `"RF: {median_rmse:.3f}  NN: {median_rmse:.3f}"` for whichever panel

### Figure 2: `p2_feature_diagnostics.png`

Two-panel figure.

**Panel 1 — VV vs VWC scatter (coloured by season):**
- X: `vv_db` (dB), Y: `vwc` (cm³/cm³), from `aligned_dataset.csv`
- Colour: DJF=blue, MAM=green, JJA=amber, SON=orange
- Annotate Pearson r and p-value from `gate_1_result.json`
- Title: "SAR Backscatter vs Soil Moisture — Seasonal Context"

**Panel 2 — RF feature importances (full training pool only):**
- Horizontal bar chart, sorted descending
- Use median importances across the 10 × 100% configs (configs 000–009)
- Error bars: IQR across the 10 reps
- Title: "Random Forest Feature Importances (100% training, n=10 reps)"
- Note: only rendered if feature importance files exist. If not, Panel 2 shows a text card "Feature importances not computed".

---

## P2.10 Evaluation harness (`poc/evaluation/harness.py`)

The harness is called identically for all models (Baselines A, B, and Phase 3 PINN). It must not contain any model-specific logic.

### Core functions

```python
def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute RMSE, R², mean bias.

    Args:
        y_pred: Predicted VWC, shape (N,). cm³/cm³.
        y_true: Observed VWC, shape (N,). cm³/cm³.

    Returns:
        {"rmse": float, "r_squared": float, "mean_bias": float}

    Formulas:
        rmse       = sqrt(mean((y_pred - y_true)**2))
        r_squared  = 1 - sum((y_pred - y_true)**2) / sum((y_true - mean(y_true))**2)
        mean_bias  = mean(y_pred - y_true)
    """
```

```python
def aggregate_metrics_across_reps(
    metrics_list: list[dict],
) -> dict:
    """
    Aggregate metrics across N_REPS repetitions at a single training size.

    Returns:
        {
            "rmse_median": float,
            "rmse_q25": float,
            "rmse_q75": float,
            "rmse_mean": float,
            "rmse_std": float,
            "r_squared_median": float,
            "r_squared_q25": float,
            "r_squared_q75": float,
            "mean_bias_median": float,
            "n_reps": int,
        }
    """
```

```python
def wilcoxon_test(
    rmse_model_a: list[float],
    rmse_model_b: list[float],
) -> dict:
    """
    Paired Wilcoxon signed-rank test: H0 = distributions equal.

    Args:
        rmse_model_a: RMSE values for model A, length N_REPS.
        rmse_model_b: RMSE values for model B, length N_REPS.

    Returns:
        {
            "statistic": float,
            "p_value_uncorrected": float,
            "significant_uncorrected": bool,   # p < ALPHA_UNCORRECTED
            "significant_bonferroni": bool,    # p < ALPHA_BONFERRONI
            "alpha_uncorrected": 0.05,
            "alpha_bonferroni": 0.0125,
            "n_pairs": int,
        }

    Uses: scipy.stats.wilcoxon(alternative='two-sided', zero_method='wilcox')
    """
```

```python
def run_full_evaluation(
    model: BaseModel,
    aligned_dataset: pd.DataFrame,
    config_path: Path,
    test_indices_path: Path,
) -> dict:
    """
    Train one model on one configuration and return its metrics JSON.
    This is the single entry point for all model training in Phase 2 and 3.
    """
```

---

## P2.11 Gate 2 script (`poc/gates/gate_2.py`)

**Run with:** `python poc/gates/gate_2.py [--confirm-deviations]`  
**Exit code:** 0 = pass, 1 = fail

### Gate 2 criteria

| ID | Criterion | Threshold | Auto-checkable |
|----|-----------|-----------|----------------|
| G2-01 | All 40 config files exist and are valid JSON | 40 files, schema valid | Yes |
| G2-02 | Split manifest exists | file present | Yes |
| G2-03 | All train indices are chronologically before all test indices | no leakage | Yes |
| G2-04 | No overlap between train, val, and test sets in any config | 0 overlap | Yes |
| G2-05 | All 40 baseline_a metric files exist and are valid | 40 files | Yes |
| G2-06 | All 40 baseline_b metric files exist and are valid | 40 files | Yes |
| G2-07 | baseline_0 metrics file exists | 1 file | Yes |
| G2-08 | RF RMSE at 100% training (median across 10 reps) | ≤ 0.10 cm³/cm³ | Yes |
| G2-09 | Both baselines beat null model at 100% training | RMSE < null RMSE | Yes |
| G2-10 | Learning curve figures exist | 2 files | Yes |
| G2-11 | No NaN values in any metrics file | 0 NaN | Yes |
| G2-12 | DEVIATIONS.md reviewed | manual sign-off | No |
| G2-13 | `pytest tests/` passes with 0 failures | 0 failures | Yes |

### Gate 2 failure protocol

**If G2-08 fails** (RF RMSE > 0.10 at 100% training):

Do not proceed to Phase 3.

1. Compute Pearson r between `vv_db` and `vwc` in aligned_dataset.
2. If r < 0.30: the SAR–moisture signal is too weak. See §10 of main SPEC.md.
3. If r ≥ 0.30: investigate in this order:
   - Check feature importance — are any features dominant in an unexpected way?
   - Check mean bias — is the model systematically off? (normalisation issue?)
   - Check train vs val loss curves — is the model overfitting even at 100%?
   - Re-examine aligned_dataset QC — are any outliers present?
4. Document findings in DEVIATIONS.md before any remediation.

**If G2-09 fails** (a baseline does not beat the null model at 100% training):
This is a softer failure — the model may still add value at other training sizes. Log a warning, document in DEVIATIONS.md, and allow Phase 3 to proceed with explicit acknowledgment. **Do not fail the gate on G2-09 alone** — it is a warning criterion.

**Revised gate failure rule:** Gate 2 fails hard on G2-01 through G2-08, G2-11, G2-12, G2-13. G2-09 is a warning only.

### Terminal output format

```
═══════════════════════════════════════════════════════════════
 ECHO PoC — Gate 2: Baseline Models & Evaluation Harness
 Run at: 2026-03-06T14:22:11
═══════════════════════════════════════════════════════════════

 Criterion                    Threshold      Measured          Status
 ─────────────────────────────────────────────────────────────────────
 G2-01 Config files           40 valid       40                PASS
 G2-02 Split manifest         present        ✓                 PASS
 G2-03 Chronological splits   no leakage     verified          PASS
 G2-04 No set overlap         0 overlap      0                 PASS
 G2-05 RF metric files        40 valid       40                PASS
 G2-06 NN metric files        40 valid       40                PASS
 G2-07 Null model metrics     present        ✓                 PASS
 G2-08 RF RMSE @ 100%         ≤ 0.10         0.053             PASS
 G2-09 Baselines beat null    RMSE < 0.112   RF:0.053 NN:0.061 WARN
 G2-10 Figures exist          2 files        2                 PASS
 G2-11 No NaN in metrics      0              0                 PASS
 G2-12 Deviations reviewed    manual         ✓                 PASS
 G2-13 pytest passes          0 failures     0                 PASS

 ─────────────────────────────────────────────────────────────────────
 Baseline performance summary:

 Model         │ 10%          │ 25%          │ 50%          │ 100%
 ──────────────┼──────────────┼──────────────┼──────────────┼──────────────
 Null (floor)  │ 0.112        │ 0.112        │ 0.112        │ 0.112
 RF  (median)  │ 0.089 ±0.011 │ 0.071 ±0.008 │ 0.061 ±0.005 │ 0.053 ±0.003
 NN  (median)  │ 0.094 ±0.014 │ 0.075 ±0.010 │ 0.065 ±0.006 │ 0.061 ±0.004
 ──────────────┴──────────────┴──────────────┴──────────────┴──────────────
 Format: median RMSE (cm³/cm³) ± IQR/2

 Result: PASS

 PINN target at N=25 for Strong result (>20%): < 0.057 cm³/cm³
 PINN target at N=25 for Significant (15-20%): < 0.060–0.064 cm³/cm³
 PINN target at N=25 for Moderate (10-15%):    < 0.064–0.068 cm³/cm³
═══════════════════════════════════════════════════════════════
```

The gate script computes and prints the **PINN performance targets** derived from the actual baseline results. These replace the placeholder thresholds in the pre-registered criteria with exact values. They are written into `gate_2_result.json` and used verbatim by the Phase 3 gate script.

### Gate 2 JSON output (`outputs/gates/gate_2_result.json`)

```json
{
    "gate": 2,
    "timestamp": "2026-03-06T14:22:11Z",
    "passed": true,
    "exit_code": 0,
    "criteria": { ... },
    "baseline_summary": {
        "null_rmse": 0.112,
        "rf_rmse_by_fraction": {
            "10%":  {"median": 0.089, "q25": 0.083, "q75": 0.094},
            "25%":  {"median": 0.071, "q25": 0.067, "q75": 0.075},
            "50%":  {"median": 0.061, "q25": 0.058, "q75": 0.063},
            "100%": {"median": 0.053, "q25": 0.051, "q75": 0.054}
        },
        "nn_rmse_by_fraction": { ... }
    },
    "pinn_targets": {
        "best_baseline_at_n25": "rf",
        "best_baseline_rmse_at_n25": 0.071,
        "strong_threshold":       0.057,
        "significant_threshold":  0.060,
        "moderate_threshold":     0.064,
        "inconclusive_boundary":  0.064
    },
    "warnings": []
}
```

The `pinn_targets` block is the **live calibration of the pre-registered success criteria**. Phase 3 loads this file and uses these thresholds exactly.

---

## P2.12 Phase 2 test requirements

**Coverage target:** ≥ 80% line coverage on `poc/models/` and `poc/evaluation/`

### Required unit tests (`tests/unit/test_models_p2.py`)

| Test | What it checks |
|------|---------------|
| `test_season_assignment_djf` | Dec, Jan, Feb all return 'DJF' |
| `test_season_assignment_boundary` | Dec 1st = DJF, Mar 1st = MAM |
| `test_season_assignment_all_months` | All 12 months assigned exactly one season |
| `test_stratified_subsample_preserves_fraction` | Output length ≈ N × fraction (±1) |
| `test_stratified_subsample_all_seasons_represented` | All 4 seasons present in output when N large enough |
| `test_stratified_subsample_falls_back_below_min_stratum` | Warning logged when season < MIN_STRATUM_SAMPLES |
| `test_no_overlap_train_val_test` | Intersection of all three sets is empty |
| `test_train_indices_before_test_indices` | max(train_date) < min(test_date) |
| `test_compute_metrics_perfect_prediction` | RMSE=0, R²=1, bias=0 for y_pred=y_true |
| `test_compute_metrics_constant_prediction` | R² ≤ 0 when y_pred is constant (may be negative) |
| `test_compute_metrics_known_values` | Manually computed expected RMSE verified |
| `test_mean_bias_sign` | Positive bias when predictions systematically high |
| `test_null_model_predicts_seasonal_mean` | DJF prediction = mean of DJF training values |
| `test_null_model_fallback_missing_season` | Returns global mean when season absent from training |
| `test_null_model_beats_no_information` | RMSE < std(y_test) — better than zero-information |
| `test_rf_model_saves_and_loads` | Loaded RF produces identical predictions to original |
| `test_rf_scaler_saved_with_model` | `scaler.pkl` exists after save() |
| `test_rf_predicts_from_feature_matrix` | Output shape is (N,) for input (N, 11) |
| `test_nn_architecture_layer_count` | 3 hidden layers, confirmed by named modules |
| `test_nn_reproducible_with_seed` | Two trainings with same seed produce identical weights |
| `test_nn_early_stopping_triggers` | Stops before MAX_EPOCHS when val loss plateaus |
| `test_nn_saves_training_history` | `config.json` contains `training_history` key |
| `test_wilcoxon_identical_distributions` | p_value ≈ 1.0 for identical inputs |
| `test_wilcoxon_clearly_different` | p_value < 0.05 for sufficiently different inputs |
| `test_wilcoxon_bonferroni_flag_correct` | significant_bonferroni=False when 0.05 > p > 0.0125 |
| `test_metrics_json_schema_valid` | Pydantic/Zod-equivalent schema validates correct output |
| `test_metrics_json_rejects_nan` | Schema validation fails if rmse is NaN |
| `test_config_n_configs_assertion` | N_CONFIGS == len(TRAINING_FRACTIONS) × N_REPS |
| `test_feature_columns_length` | len(FEATURE_COLUMNS) == 11 |
| `test_vhvv_is_difference_not_ratio` | vhvv_db = vh_db - vv_db (subtraction in log space) |

### Required integration tests (`tests/integration/test_pipeline_p2.py`)

Require `data/processed/aligned_dataset.csv` and all split configs to exist.

| Test | What it checks |
|------|---------------|
| `test_all_40_configs_loadable` | All config JSON files parse without error |
| `test_no_data_leakage_across_configs` | Test set never appears in any train or val set |
| `test_baseline_a_runs_on_config_000` | RF trains and produces metrics without error |
| `test_baseline_b_runs_on_config_000` | NN trains and produces metrics without error |
| `test_null_model_runs_on_full_pool` | Null model trains and produces metrics |
| `test_rf_metrics_plausible_at_100pct` | RMSE in [0.01, 0.20] — not degenerate |

---

## P2.13 Phase 2 file manifest

All files must exist for Gate 2 to pass:

```
data/splits/split_manifest.json
data/splits/configs/config_000.json  ...  config_039.json   (40 files)
poc/models/__init__.py
poc/models/base.py
poc/models/null_model.py
poc/models/random_forest.py
poc/models/standard_nn.py
poc/evaluation/__init__.py
poc/evaluation/harness.py
poc/evaluation/splits.py
poc/evaluation/plots.py
poc/gates/gate_2.py
outputs/metrics/baseline_0_metrics.json
outputs/metrics/config_000_baseline_a.json  ...  config_039_baseline_a.json  (40 files)
outputs/metrics/config_000_baseline_b.json  ...  config_039_baseline_b.json  (40 files)
outputs/models/baseline_a/config_000/  ...  config_039/  (40 dirs, each with model.pkl + scaler.pkl + config.json)
outputs/models/baseline_b/config_000/  ...  config_039/  (40 dirs, each with model_weights.pt + scaler.pkl + config.json)
outputs/figures/p2_learning_curves_baselines.png
outputs/figures/p2_feature_diagnostics.png
outputs/gates/gate_2_result.json
```
