# SPEC_PHASE3.md — Phase 3: PINN & Evaluation
# ECHO PoC — Vantage

**Prerequisite:** Phase 2 gate passed (`outputs/gates/gate_2_result.json` → `"passed": true`)  
**Version:** 1.0 — 06 March 2026  
**Goal:** Train the physics-informed neural network across all 40 split configurations. Run three identifiability diagnostics. Evaluate the pre-registered success criteria using thresholds computed by the Phase 2 gate. Produce the complete comparative evaluation and all figures required for the Carbon13 materials.

---

## P3. Overview

Phase 3 produces:

1. **PINN model artefacts** for all 40 configurations — identical split configs as Phase 2
2. **λ hyperparameter selection** — searched once on 100% configs, fixed for all 40
3. **Three identifiability diagnostics** — residual monitoring, parameter sensitivity, Dobson vs Mironov
4. **Prediction intervals** — calibrated 80% and 95% intervals for the MVP fixture data
5. **Pre-registered outcome classification** — determined programmatically from Phase 2 thresholds
6. **Five diagnostic figures** — learning curves, physics↔ML decomposition, WCM convergence, sensitivity, Dobson/Mironov
7. **Gate 3 result** written to `outputs/gates/gate_3_result.json`

Phase 3 uses the **sealed test set** and the **identical 40 split configurations** from Phase 2. No new splits are generated. No test set exposure before final evaluation.

---

## P3.1 Repository additions (Phase 3)

```
echo-poc/
├── poc/
│   ├── models/
│   │   ├── pinn.py                        ← PINN architecture (this phase)
│   │   └── dielectric.py                  ← Dobson + Mironov as separate modules
│   └── evaluation/
│       ├── lambda_search.py               ← λ grid search on 100% configs
│       ├── diagnostics.py                 ← three identifiability diagnostics
│       ├── uncertainty.py                 ← prediction interval calibration
│       └── plots_p3.py                    ← Phase 3 figures
│   └── gates/
│       └── gate_3.py
├── outputs/
│   ├── models/
│   │   └── pinn/
│   │       ├── lambda_search_result.json  ← selected λ triple
│   │       └── config_000/ ... config_039/
│   ├── metrics/
│   │   ├── config_000_pinn.json ... config_039_pinn.json  (40 files)
│   │   ├── diagnostics_residual_ratio.json
│   │   ├── diagnostics_parameter_sensitivity.json
│   │   ├── diagnostics_dobson_vs_mironov.json
│   │   ├── uncertainty_calibration.json
│   │   └── gate_3_result.json
│   └── figures/
│       ├── p3_learning_curves_all_models.png
│       ├── p3_physics_ml_decomposition.png
│       ├── p3_wcm_parameter_convergence.png
│       ├── p3_parameter_sensitivity.png
│       └── p3_dobson_vs_mironov.png
└── tests/
    ├── unit/
    │   └── test_models_p3.py
    └── integration/
        └── test_pipeline_p3.py
```

---

## P3.2 Config additions (`poc/config.py`)

Append to existing `config.py`. Do not modify Phase 1 or Phase 2 constants.

```python
# ─── Phase 3: PINN physics constants ────────────────────────────────────────

# WCM learnable parameter initialisations (Attema & Ulaby 1978; Singh et al. 2023)
WCM_A_INIT = 0.10    # vegetation scattering coefficient (dimensionless)
WCM_B_INIT = 0.15    # vegetation attenuation coefficient (dimensionless)
# Bounds enforced via sigmoid reparameterisation: param = LB + (UB-LB)*sigmoid(raw)
WCM_A_LB, WCM_A_UB = 0.01, 1.0
WCM_B_LB, WCM_B_UB = 0.01, 1.0

# Sigmoid reparameterisation inverse: raw_init = logit((init - LB) / (UB - LB))
# Computed at runtime — do not hardcode

# Oh (1992) surface roughness — fixed, not learnable
# Reference: Bechtold et al. (2018) for saturated blanket bog surface
# ks = surface roughness × wavenumber = roughness × (2π / λ_radar)
# For Sentinel-1 C-band (λ=5.6 cm), ks≈0.3 corresponds to ~2.7 mm RMS height
KS_ROUGHNESS = 0.30  # dimensionless (roughness × wavenumber product)

# Dobson (1985) dielectric mixing model — defined in config Phase 1:
# PEAT_THETA_SAT, EPSILON_DRY_PEAT, EPSILON_WATER, DOBSON_ALPHA already present

# Mironov (2009) parameters for organic soil (sensitivity check only)
# Source: Mironov et al. (2009), IEEE TGRS, Table II, organic soil category
MIRONOV_ND   = 0.312   # refractive index of dry soil
MIRONOV_KD   = 0.0     # extinction index of dry soil (non-absorbing at C-band)
MIRONOV_MV_T = 0.36    # transition moisture (organic soil)
MIRONOV_ND1  = 1.42    # refractive index slope below transition
MIRONOV_ND2  = 0.89    # refractive index slope above transition

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
```

---

## P3.3 Dielectric models (`poc/models/dielectric.py`)

Two dielectric models as independent, differentiable PyTorch modules. Both share the same interface so the PINN can swap them with a single argument.

### Interface

```python
class DielectricModel(ABC):
    """
    Computes real part of soil dielectric constant from volumetric water content.
    Must be differentiable — implemented entirely with torch operations.
    """
    @abstractmethod
    def forward(self, m_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m_v: VWC, shape (...). Range [0, PEAT_THETA_SAT]. cm³/cm³.
        Returns:
            epsilon: Real dielectric constant, shape (...). Dimensionless.
        """
```

### Dobson (1985) — primary model

```python
class DobsonDielectric(DielectricModel):
    """
    Semi-empirical dielectric mixing model for organic soil at C-band.

    ε(m_v) = ε_dry + (ε_water − 1) · m_v^α

    Parameterised for blanket bog peat following Bechtold et al. (2018):
        ε_dry = EPSILON_DRY_PEAT = 3.5    (zero clay fraction, low bulk density)
        ε_water = EPSILON_WATER = 80.0
        α = DOBSON_ALPHA = 1.4            (organic soil empirical exponent)

    Reference: Dobson et al. (1985), IEEE TGRS 23(1), eq. 1
               Bechtold et al. (2018), Remote Sensing 10(2)
    """
    def forward(self, m_v: torch.Tensor) -> torch.Tensor:
        eps = EPSILON_DRY_PEAT + (EPSILON_WATER - 1.0) * m_v.pow(DOBSON_ALPHA)
        return eps
```

**Unit test requirement:** Verify `DobsonDielectric()(torch.tensor(0.0))` ≈ `EPSILON_DRY_PEAT` and `DobsonDielectric()(torch.tensor(0.5))` ≈ expected value computed manually from the equation.

### Mironov (2009) — sensitivity check only

```python
class MironovDielectric(DielectricModel):
    """
    Physical-statistical dielectric model.
    Used only for the Dobson vs Mironov sensitivity check (diagnostic 3).
    NOT used in the primary 40-config experiment.

    Piecewise model with transition moisture mv_t:
        n(m_v) = nd + nd1·m_v                     if m_v <= mv_t
        n(m_v) = nd + nd1·mv_t + nd2·(m_v - mv_t) if m_v > mv_t
        ε(m_v) ≈ n(m_v)^2   (real part, k≈0 at C-band for organic soil)

    Parameters for organic soil from Mironov et al. (2009), IEEE TGRS, Table II.

    Reference: Mironov et al. (2009), IEEE TGRS 57(7)
    """
    def forward(self, m_v: torch.Tensor) -> torch.Tensor:
        n_below = MIRONOV_ND + MIRONOV_ND1 * m_v
        n_above = (MIRONOV_ND + MIRONOV_ND1 * MIRONOV_MV_T
                   + MIRONOV_ND2 * (m_v - MIRONOV_MV_T))
        n = torch.where(m_v <= MIRONOV_MV_T, n_below, n_above)
        return n.pow(2)
```

---

## P3.4 WCM forward model (within `poc/models/pinn.py`)

The WCM forward model is a differentiable computation graph. It is called inside the PINN's `forward()` method and during the λ search.

### Oh (1992) soil backscatter

```python
def oh_soil_backscatter(
    epsilon: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    ks: float = KS_ROUGHNESS,
) -> torch.Tensor:
    """
    Simplified Oh (1992) model for soil surface backscatter at C-band VV.

    Computes Fresnel power reflectance at horizontal polarisation,
    then applies Oh roughness scaling.

    Fresnel reflectance (h-pol):
        Γ_h = |( cos θ - sqrt(ε - sin²θ) ) / ( cos θ + sqrt(ε - sin²θ) )|²

    Oh roughness correction (simplified, co-pol VV approximation):
        σ°_soil_linear = (ks^0.1 / 3) · (cos θ)^2.2 · Γ_h(ε, θ)

    Then convert to dB: σ°_soil_dB = 10 · log10(σ°_soil_linear + 1e-10)

    Args:
        epsilon:       Soil dielectric constant, shape (...). Dimensionless.
        theta_inc_rad: Incidence angle in radians, shape (...).
        ks:            Surface roughness parameter (fixed = KS_ROUGHNESS).

    Returns:
        sigma_soil_db: Soil backscatter in dB, shape (...).

    Reference: Oh et al. (1992), IEEE TGRS 30(2), simplified form
               Singh et al. (2023), Remote Sensing 15(4) — C-band application

    Numerical stability:
        - sqrt argument clamped to > 1e-6 to prevent NaN in autograd
        - log argument offset by 1e-10 to prevent log(0)
    """
    cos_theta = torch.cos(theta_inc_rad)
    sin_theta = torch.sin(theta_inc_rad)

    # Fresnel reflectance (h-pol)
    inner = (epsilon - sin_theta.pow(2)).clamp(min=1e-6)
    sqrt_inner = inner.sqrt()
    gamma_h = ((cos_theta - sqrt_inner) / (cos_theta + sqrt_inner + 1e-8)).pow(2)

    # Oh roughness scaling
    sigma_linear = (ks ** 0.1 / 3.0) * cos_theta.pow(2.2) * gamma_h
    sigma_db = 10.0 * torch.log10(sigma_linear + 1e-10)
    return sigma_db
```

### WCM vegetation terms

```python
def wcm_vegetation_terms(
    A: torch.Tensor,
    B: torch.Tensor,
    ndvi: torch.Tensor,
    theta_inc_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute WCM vegetation direct scattering and two-way transmissivity.

    Equations (Attema & Ulaby, 1978, Radio Science 13(2)):
        σ°_veg = A · NDVI · cos(θ)
        τ²     = exp(−2 · B · NDVI / cos(θ))

    Args:
        A:             Vegetation scattering coefficient (learnable, bounded).
        B:             Vegetation attenuation coefficient (learnable, bounded).
        ndvi:          Normalised Difference Vegetation Index, shape (...).
        theta_inc_rad: Incidence angle in radians, shape (...).

    Returns:
        sigma_veg:     Vegetation direct backscatter (linear, not dB), shape (...).
        tau_squared:   Two-way transmissivity factor [0, 1], shape (...).
    """
    cos_theta = torch.cos(theta_inc_rad)
    sigma_veg  = A * ndvi * cos_theta
    tau_squared = torch.exp(-2.0 * B * ndvi / (cos_theta + 1e-8))
    return sigma_veg, tau_squared
```

### WCM total backscatter

```python
def wcm_forward(
    m_v: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    ndvi: torch.Tensor,
    theta_inc_rad: torch.Tensor,
    dielectric_model: DielectricModel,
) -> torch.Tensor:
    """
    Full WCM forward pass: m_v → σ°_total_dB.

    σ°_total = σ°_veg + τ² · σ°_soil
    (all in linear before summing, then convert to dB)

    Args:
        m_v:              VWC estimate, shape (...). cm³/cm³.
        A, B:             WCM learnable parameters (already bounded via sigmoid).
        ndvi:             NDVI, shape (...).
        theta_inc_rad:    Incidence angle, shape (...). Radians.
        dielectric_model: DobsonDielectric or MironovDielectric.

    Returns:
        sigma_total_db: Predicted total backscatter, dB, shape (...).

    Note: σ°_veg is in linear units (Attema & Ulaby formulation).
          σ°_soil is converted from dB before adding. Both terms summed
          in linear, then result converted to dB.
    """
    epsilon = dielectric_model(m_v)
    sigma_soil_db = oh_soil_backscatter(epsilon, theta_inc_rad)
    sigma_soil_linear = 10.0 ** (sigma_soil_db / 10.0)

    sigma_veg, tau_sq = wcm_vegetation_terms(A, B, ndvi, theta_inc_rad)

    sigma_total_linear = sigma_veg + tau_sq * sigma_soil_linear
    sigma_total_db = 10.0 * torch.log10(sigma_total_linear.clamp(min=1e-10))
    return sigma_total_db
```

---

## P3.5 PINN architecture (`poc/models/pinn.py`)

### Overall architecture

Two jointly-trained sub-networks connected through the WCM forward model.

```
                    ┌─────────────────────────────────────────┐
                    │              PINN                         │
                    │                                           │
  X (11 features) ─┼─→ physics_net ──→ m_v_physics            │
                    │         │                                 │
                    │         └──→ WCM_forward(m_v_physics)    │
                    │                    │                      │
                    │                    └──→ σ°_wcm            │
                    │                                           │
  X (11 features) ─┼─→ correction_net ──→ δ                   │
                    │                                           │
                    │   m_v_final = m_v_physics + δ            │
                    └─────────────────────────────────────────┘

Training signals:
  L_data      = MSE(m_v_final,    m_v_observed)
  L_physics   = MSE(σ°_wcm,       σ°_observed_dB)
  L_monotonic = mean(ReLU(−∂ε/∂m_v_physics))
  L_bounds    = mean(ReLU(−m_v_final) + ReLU(m_v_final − PEAT_THETA_SAT))
```

### physics_net

```
Input (11 features)
  → Linear(11, 32) → ReLU
  → Linear(32, 16) → ReLU
  → Linear(16, 1)  → Sigmoid scaled to [0, PEAT_THETA_SAT]
  → m_v_physics

Output activation: sigmoid scaled to physical range
    m_v_physics = PEAT_THETA_SAT * torch.sigmoid(raw_output)
```

The physics_net is deliberately **smaller** than the correction_net (32→16 vs 64→32→16). This intentional capacity asymmetry biases the network towards relying on the physics for its primary estimate, with the larger correction_net handling residuals. If the correction_net dominates despite this, the residual ratio diagnostic will reveal it.

### correction_net

Identical to Baseline B:
```
Input (11 features)
  → Linear(11, 64) → ReLU → Dropout(0.2)
  → Linear(64, 32) → ReLU → Dropout(0.2)
  → Linear(32, 16) → ReLU
  → Linear(16, 1)  → δ (unbounded — can be positive or negative)
```

### WCM learnable parameters — sigmoid reparameterisation

```python
class PINN(nn.Module):
    def __init__(self, n_features: int = 11,
                 dielectric_model: DielectricModel = None):
        super().__init__()
        self.dielectric = dielectric_model or DobsonDielectric()

        self.physics_net    = PhysicsNet(n_features)
        self.correction_net = CorrectionNet(n_features)

        # WCM learnable parameters — stored as unconstrained raw values
        # Bounded at forward pass via sigmoid reparameterisation
        # Initialise raw values as logit of (init - LB) / (UB - LB)
        A_raw_init = math.log((WCM_A_INIT - WCM_A_LB) / (WCM_A_UB - WCM_A_INIT + 1e-8))
        B_raw_init = math.log((WCM_B_INIT - WCM_B_LB) / (WCM_B_UB - WCM_B_INIT + 1e-8))
        self.A_raw = nn.Parameter(torch.tensor(A_raw_init, dtype=torch.float32))
        self.B_raw = nn.Parameter(torch.tensor(B_raw_init, dtype=torch.float32))

    @property
    def A(self) -> torch.Tensor:
        """Bounded WCM A parameter. Always in [WCM_A_LB, WCM_A_UB]."""
        return WCM_A_LB + (WCM_A_UB - WCM_A_LB) * torch.sigmoid(self.A_raw)

    @property
    def B(self) -> torch.Tensor:
        """Bounded WCM B parameter. Always in [WCM_B_LB, WCM_B_UB]."""
        return WCM_B_LB + (WCM_B_UB - WCM_B_LB) * torch.sigmoid(self.B_raw)

    def forward(
        self,
        X: torch.Tensor,
        ndvi: torch.Tensor,
        theta_inc_rad: torch.Tensor,
        vv_db_observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict rather than a single tensor — all quantities needed
        for loss computation and diagnostics.

        Args:
            X:               Full feature matrix (N, 11), normalised.
            ndvi:            NDVI values (N,) — extracted from X for clarity.
            theta_inc_rad:   Incidence angles (N,) in radians.
            vv_db_observed:  Observed VV backscatter (N,) in dB.

        Returns dict with keys:
            m_v_physics:     Physics sub-network output (N,). cm³/cm³.
            delta_ml:        Correction sub-network output (N,). cm³/cm³.
            m_v_final:       m_v_physics + delta_ml (N,). cm³/cm³.
            sigma_wcm_db:    WCM forward pass on m_v_physics (N,). dB.
            epsilon:         Dielectric constant at m_v_physics (N,). For monotonic loss.
            A_current:       Current WCM A value (scalar).
            B_current:       Current WCM B value (scalar).
        """
        m_v_physics = self.physics_net(X)
        delta_ml    = self.correction_net(X)
        m_v_final   = m_v_physics + delta_ml

        sigma_wcm_db = wcm_forward(
            m_v_physics, self.A, self.B,
            ndvi, theta_inc_rad, self.dielectric
        )
        epsilon = self.dielectric(m_v_physics)

        return {
            "m_v_physics":  m_v_physics,
            "delta_ml":     delta_ml,
            "m_v_final":    m_v_final,
            "sigma_wcm_db": sigma_wcm_db,
            "epsilon":      epsilon,
            "A_current":    self.A,
            "B_current":    self.B,
        }
```

---

## P3.6 Composite loss function

```python
def compute_pinn_loss(
    outputs: dict[str, torch.Tensor],
    m_v_observed: torch.Tensor,
    vv_db_observed: torch.Tensor,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> dict[str, torch.Tensor]:
    """
    Compute composite PINN loss and all component terms.

    L = L_data + λ₁·L_physics + λ₂·L_monotonic + λ₃·L_bounds

    Args:
        outputs:         Dict from PINN.forward().
        m_v_observed:    Ground truth VWC (N,). cm³/cm³.
        vv_db_observed:  Observed VV backscatter (N,). dB.
        lambda1:         Weight for L_physics.
        lambda2:         Weight for L_monotonic.
        lambda3:         Weight for L_bounds.

    Returns dict with keys:
        total:      Scalar total loss.
        l_data:     MSE(m_v_final, m_v_observed). Scalar.
        l_physics:  MSE(sigma_wcm_db, vv_db_observed). Scalar.
        l_monotonic: Physics constraint penalty. Scalar.
        l_bounds:   Bounds constraint penalty. Scalar.

    Loss term formulas:
        L_data      = MSE(m_v_final, m_v_observed)
        L_physics   = MSE(sigma_wcm_db, vv_db_observed)

        L_monotonic: enforce dε/dm_v > 0 (always true for Dobson — but
                     log it as zero; for Mironov it can violate near mv_t)
            dm_v_probe = m_v_physics.detach() + small_perturbation (1e-3)
            eps_probe  = dielectric(dm_v_probe)
            eps_base   = outputs['epsilon'].detach()
            d_eps      = (eps_probe - eps_base) / 1e-3
            L_monotonic = mean(ReLU(-d_eps))

        L_bounds = mean(ReLU(-m_v_final) + ReLU(m_v_final - PEAT_THETA_SAT))

    Constraint check (LAMBDA_DOMINANCE_CONSTRAINT):
        After computing all terms, verify that L_data is the largest
        single term. Log a warning if not — do not raise an error.
    """
```

---

## P3.7 λ hyperparameter search (`poc/evaluation/lambda_search.py`)

### Procedure

```
1. Load configs 000–009 (100% training, 10 repetitions)
2. For each of 64 λ combinations in LAMBDA_GRID^3:
   For each of 10 configs (reps 0–9 at 100%):
       Train PINN on (X_train, X_val) with this λ triple
       Record val loss at early stopping epoch
       Check LAMBDA_DOMINANCE_CONSTRAINT — flag if violated
   Compute median val_loss across 10 reps for this λ triple
3. Select λ triple with lowest median val_loss among non-violating combinations
   If all combinations violate the dominance constraint: select lowest median
   val_loss regardless and log a DEVIATIONS.md-triggering warning
4. Write outputs/models/pinn/lambda_search_result.json
5. Log the top-5 λ triples by median val_loss (for transparency)
```

### Output: `lambda_search_result.json`

```json
{
    "selected": {
        "lambda1": 0.1,
        "lambda2": 0.01,
        "lambda3": 0.1
    },
    "median_val_loss": 0.0023,
    "dominance_constraint_satisfied": true,
    "search_configs_used": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "top5_candidates": [
        {"lambda1": 0.1,  "lambda2": 0.01, "lambda3": 0.1,  "median_val_loss": 0.0023},
        {"lambda1": 0.1,  "lambda2": 0.1,  "lambda3": 0.1,  "median_val_loss": 0.0025},
        {"lambda1": 0.01, "lambda2": 0.01, "lambda3": 0.1,  "median_val_loss": 0.0026},
        {"lambda1": 0.5,  "lambda2": 0.01, "lambda3": 0.1,  "median_val_loss": 0.0031},
        {"lambda1": 0.1,  "lambda2": 0.01, "lambda3": 0.01, "median_val_loss": 0.0033}
    ],
    "n_combinations_searched": 64,
    "n_violating_dominance": 3,
    "design_note": "λ searched once on 100% training configs, fixed for all 40 configs. See SPEC_PHASE3.md §P3.7.",
    "generated_at": "2026-03-06T..."
}
```

---

## P3.8 PINN training procedure

For each of the 40 configs, using the **fixed** λ triple from `lambda_search_result.json`:

```
1. Load config_{idx}.json → train_indices, val_indices
   Load test_indices.json → test_indices
   Load lambda_search_result.json → lambda1, lambda2, lambda3

2. Extract and normalise features (identical to Phase 2):
   Fit StandardScaler on X_train, apply to X_val and X_test

3. Extract physics inputs from normalised dataset:
   ndvi:          feature column index for 'ndvi' in FEATURE_COLUMNS
   theta_inc_rad: feature column index for 'incidence_angle_mean', converted to radians
   vv_db:         feature column index for 'vv_db' (not normalised — raw dB values)

   NOTE: vv_db for L_physics must be the RAW (unnormalised) dB values.
   The physics loss compares WCM output (dB) to observed VV (dB).
   Use the pre-normalisation vv_db values for L_physics only.
   The normalised feature matrix X is used for network inputs as normal.

4. Set seeds:
   torch.manual_seed(SEED + config_idx)
   torch.cuda.manual_seed_all(SEED + config_idx)

5. Instantiate PINN(n_features=11, dielectric_model=DobsonDielectric())

6. Optimiser: Adam(lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
   — same as Baseline B, applied to all parameters including A_raw and B_raw

7. Training loop (identical early stopping logic to Baseline B):
   For each epoch:
       model.train()
       For each batch:
           outputs = model(X_batch, ndvi_batch, theta_batch, vv_db_batch)
           loss_dict = compute_pinn_loss(outputs, y_batch, vv_db_batch, λ1, λ2, λ3)
           loss_dict['total'].backward()
           optimizer.step()

       model.eval()
       with torch.no_grad():
           val_outputs = model(X_val, ndvi_val, theta_val, vv_db_val)
           val_loss = compute_pinn_loss(val_outputs, y_val, vv_db_val, λ1, λ2, λ3)['total']

       Early stopping on val_loss (patience = NN_PATIENCE = 20)

       Log per epoch:
           epoch, train_loss, val_loss,
           A_current.item(), B_current.item(),
           residual_ratio = std(delta_ml) / (std(m_v_physics) + 1e-8)

8. Load best checkpoint, evaluate on TEST SET:
   test_outputs = model(X_test, ndvi_test, theta_test, vv_db_test)
   y_pred = test_outputs['m_v_final'].detach().numpy()
   Compute metrics via harness.compute_metrics(y_pred, y_test)

9. Compute final diagnostics (per config):
   - final_residual_ratio: std(delta_ml_test) / (std(m_v_physics_test) + 1e-8)
   - final_A: model.A.item()
   - final_B: model.B.item()

10. Save artefacts, write metrics JSON
```

### PINN metrics JSON (extends Phase 2 schema)

The Phase 2 metrics schema is extended with `physics_diagnostics`:

```json
{
    "model": "pinn",
    "config_idx": 0,
    "fraction": 1.0,
    "fraction_label": "100%",
    "rep": 0,
    "seed_used": 42,
    "n_train": 74,
    "n_val": 18,
    "n_test": 43,
    "feature_columns": ["vv_db", "vh_db", ...],
    "metrics": {
        "rmse": 0.041,
        "r_squared": 0.871,
        "mean_bias": 0.001
    },
    "training_metadata": {
        "lambda1": 0.1,
        "lambda2": 0.01,
        "lambda3": 0.1,
        "stopped_at_epoch": 187,
        "best_val_loss": 0.0019,
        "dominance_constraint_satisfied": true,
        "stratification_used": true
    },
    "physics_diagnostics": {
        "final_A": 0.091,
        "final_B": 0.163,
        "final_residual_ratio": 0.72,
        "residual_ratio_gt1_warning": false,
        "wcm_forward_rmse_vs_observed": 0.89
    },
    "warnings": [],
    "generated_at": "2026-03-06T..."
}
```

---

## P3.9 Identifiability diagnostics (`poc/evaluation/diagnostics.py`)

These three diagnostics are run **after** all 40 PINN configs are complete. They use saved artefacts — no retraining.

### Diagnostic 1 — Residual ratio monitoring

Aggregates the per-epoch residual ratio logs across all 40 configs.

**Output: `diagnostics_residual_ratio.json`**

```json
{
    "by_fraction": {
        "100%": {"median_final_ratio": 0.71, "q25": 0.61, "q75": 0.83, "n_gt1": 1},
        "50%":  {"median_final_ratio": 0.84, "q25": 0.70, "q75": 0.98, "n_gt1": 2},
        "25%":  {"median_final_ratio": 0.93, "q25": 0.78, "q75": 1.12, "n_gt1": 4},
        "10%":  {"median_final_ratio": 1.18, "q25": 0.95, "q75": 1.41, "n_gt1": 7}
    },
    "interpretation": "Ratio < 1.0 indicates physics branch dominates. Rising ratio at low N is expected and physically interpretable: physics provides more relative structure when data is scarce.",
    "warning_configs": [...]
}
```

**Interpretation rule** written into the gate script:
- Median ratio < 1.0 at N=100% and N=25%: physics is contributing meaningfully → no warning
- Median ratio > 1.0 at N=25%: log as diagnostic note, do not fail gate
- Median ratio > 1.5 at all training sizes: log as architectural concern in gate output

### Diagnostic 2 — WCM parameter sensitivity

Computed on the **full training pool model** (median of configs 000–009).

```python
def parameter_sensitivity(
    model: PINN,
    m_v_range: np.ndarray,            # linspace(0.1, PEAT_THETA_SAT, 100)
    ndvi_mean: float,                  # mean NDVI from training set
    theta_mean_rad: float,             # mean incidence angle from training set
    delta_param: float = 0.01,         # perturbation size
) -> dict:
    """
    Numerical sensitivity of σ°_total to perturbations in A, B, and ks.

    For each parameter p in {A, B}:
        ∂σ°_total/∂p ≈ (σ°_total(p + Δ) - σ°_total(p - Δ)) / (2Δ)
        Computed at each m_v in m_v_range, with other params at current values.
        Report: mean |∂σ°/∂p| across m_v_range.

    A parameter with mean |∂σ°/∂p| < 0.1 dB per unit change is "poorly constrained" —
    the forward model output is insensitive to that parameter across the observed
    moisture range, meaning its inferred value is unreliable.
    """
```

**Output: `diagnostics_parameter_sensitivity.json`**

```json
{
    "model_A_final_median": 0.091,
    "model_B_final_median": 0.163,
    "sensitivity_A_mean_abs_dSigma_dA": 1.84,
    "sensitivity_B_mean_abs_dSigma_dB": 0.93,
    "A_well_constrained": true,
    "B_well_constrained": true,
    "poorly_constrained_threshold": 0.1,
    "moisture_range_evaluated": [0.1, 0.88],
    "interpretation": "Both A and B produce meaningful variation in σ°_total across the observed moisture range. Parameter estimates are interpretable."
}
```

### Diagnostic 3 — Dobson vs Mironov sensitivity check

Train 10 additional PINN instances on configs 000–009 (100% training) using `MironovDielectric()` instead of `DobsonDielectric()`. All other settings identical.

```
PINN_mironov = PINN(dielectric_model=MironovDielectric())
Train on configs 000–009 using fixed λ triple
Evaluate on sealed test set
Compute metrics
```

**Comparison:**
```
rmse_dobson  = median RMSE across configs 000–009 (already computed)
rmse_mironov = median RMSE across 10 Mironov runs

relative_improvement = (rmse_dobson - rmse_mironov) / rmse_dobson
```

**Decision rule:**
- If `relative_improvement > MIRONOV_IMPROVEMENT_THRESHOLD (0.05)`:
  Re-run all 40 configs with Mironov as primary dielectric model.
  Add DEV entry to DEVIATIONS.md.
  Report both variants in results.
- Otherwise: retain Dobson. Report comparison as evidence for model selection.

**Output: `diagnostics_dobson_vs_mironov.json`**

```json
{
    "dobson_rmse_median": 0.051,
    "mironov_rmse_median": 0.053,
    "relative_improvement_mironov_over_dobson": -0.039,
    "mironov_wins": false,
    "rerun_triggered": false,
    "primary_dielectric_model": "Dobson (1985)",
    "conclusion": "Dobson retained. Mironov shows no meaningful improvement at C-band on blanket bog. Consistent with Park et al. (2019) finding of equivalent performance on organic soils.",
    "n_configs_compared": 10
}
```

---

## P3.10 Pre-registered outcome evaluation

Loaded directly from `gate_2_result.json` — the thresholds are those computed from actual baseline results, not the placeholder values in the original spec.

```python
def evaluate_outcome(
    gate2_result: dict,
    pinn_metrics_files: list[Path],
) -> dict:
    """
    Evaluate the pre-registered outcome category at N=25 critical threshold.

    Uses thresholds from gate_2_result['pinn_targets'].
    Evaluates PINN RMSE at fraction=0.25 (configs 020–029).

    Returns outcome category and supporting evidence.
    """
    targets = gate2_result['pinn_targets']
    best_baseline_key = targets['best_baseline_at_n25']  # 'rf' or 'nn'
    best_baseline_rmse = targets['best_baseline_rmse_at_n25']

    # Load PINN RMSE at 25% configs (020-029)
    pinn_rmse_n25 = [load_metric(f)['metrics']['rmse']
                     for f in pinn_metrics_files
                     if 'config_02' in f.name]

    pinn_median = np.median(pinn_rmse_n25)
    relative_reduction = (best_baseline_rmse - pinn_median) / best_baseline_rmse

    # Wilcoxon test vs best baseline at N=25
    baseline_rmse_n25 = load_baseline_rmse_at_fraction(best_baseline_key, 0.25)
    wilcoxon_result = wilcoxon_test(pinn_rmse_n25, baseline_rmse_n25)

    # IQR overlap check
    pinn_q25, pinn_q75 = np.percentile(pinn_rmse_n25, [25, 75])
    bl_q25, bl_q75 = np.percentile(baseline_rmse_n25, [25, 75])
    iqr_overlap = not (pinn_q75 < bl_q25 or bl_q75 < pinn_q25)

    # Category determination
    if relative_reduction > 0.20 and not iqr_overlap:
        category = "Strong"
    elif 0.15 <= relative_reduction <= 0.20 and wilcoxon_result['significant_bonferroni']:
        category = "Significant"
    elif 0.10 <= relative_reduction < 0.15:
        # Also check consistency at N=50
        pinn_n50 = median_pinn_rmse_at_fraction(0.50)
        bl_n50   = median_baseline_rmse_at_fraction(best_baseline_key, 0.50)
        consistent = (pinn_n50 < bl_n50)
        category = "Moderate" if consistent else "Inconclusive"
    elif relative_reduction < 0.10:
        category = "Inconclusive"
    else:
        category = "Negative"  # PINN worse than baseline
```

### Outcome categories and consequences

| Category | Criterion (at N=25) | Consequence |
|----------|--------------------|-|
| **Strong** | >20% RMSE reduction; non-overlapping IQR | Full confidence; real numbers in all Carbon13 materials |
| **Significant** | 15–20% reduction; Wilcoxon p < 0.0125 | Proceed confidently; state statistical significance |
| **Moderate** | 10–15% reduction; consistent direction at N=50 | Proceed with qualification: "directional advantage at low data volume" |
| **Inconclusive** | <10% at all sizes | Pivot: platform value, not physics-advantage claim |
| **Negative** | PINN worse than best baseline | Mandatory review of diagnostics before Phase 4; see §P3.11 |

**Wilcoxon reporting:** Both p-values reported in gate output and figures.
- `p_uncorrected`: tested against α=0.05
- `p_bonferroni`: tested against α=0.0125 (Bonferroni for 4 training-size comparisons)
- Gate and figures label both clearly. Neither is hidden.

---

## P3.11 Uncertainty quantification (`poc/evaluation/uncertainty.py`)

Produces the calibrated prediction intervals that populate the MVP fixture data.

### Ensemble uncertainty (epistemic)

```python
def compute_ensemble_uncertainty(
    pinn_metrics_files_at_fraction: list[Path],
    X_test: np.ndarray,
    model_dir: Path,
) -> np.ndarray:
    """
    Epistemic uncertainty: std of predictions across 10 repetitions at a training size.

    σ_epistemic(x_i) = std({ŷ_k(x_i) : k = 1..10})

    Computed on test set. Shape: (N_test,).
    """
```

### Calibration

```python
def calibrate_intervals(
    y_pred_ensemble: np.ndarray,   # shape (N_test, N_reps)
    y_test: np.ndarray,            # shape (N_test,)
    targets: list[float] = PRED_INTERVAL_TARGETS,
) -> dict:
    """
    Find scalar calibration factor k such that:
        coverage(y_pred_mean ± k · σ_epistemic) ≈ target

    Binary search over k ∈ [0.1, 10.0].
    Report empirical coverage on test set for each calibrated interval.

    Returns:
        {
            "0.80": {"k": 1.24, "empirical_coverage": 0.814},
            "0.95": {"k": 2.31, "empirical_coverage": 0.953}
        }
    """
```

**Output: `uncertainty_calibration.json`** — used directly by Phase 4 to populate MVP fixture values.

---

## P3.12 Diagnostic figures

### Figure 1: `p3_learning_curves_all_models.png`

Extends Phase 2 Figure 1 with PINN series added.

- All three models on same axes: RF (cyan), NN (purple), PINN (green `#B6FFCE`)
- Null model horizontal dashed line (amber)
- N=25 vertical dashed line labelled "Critical threshold"
- Outcome category annotation box at N=25: `"Result: {category}"`
- Both RMSE panel and R² panel

### Figure 2: `p3_physics_ml_decomposition.png`

The primary scientific figure. Shows what the MVP chart must faithfully reproduce.

- X-axis: Date (test set observations only, chronological)
- Y-axis: VWC (cm³/cm³)
- Three series:
  - Amber dashed: `m_v_physics` from the best-performing 100% config
  - Green solid: `m_v_final` from the same config
  - Cyan scatter: observed VWC (`y_test`)
- Shaded band around green line: ±1 σ_epistemic from 100% ensemble
- Annotation: `"Physics RMSE: {rmse_physics:.3f}  Fused RMSE: {rmse_fused:.3f}  Improvement: {pct:.1f}%"`
- Title: "Physics ↔ ML Decomposition — Test Set (Moor House)"

**Note:** `rmse_physics` is computed using `m_v_physics` as the prediction. This is the honest comparison. If the physics-only estimate is nearly as good as the fused estimate, the chart will show it.

### Figure 3: `p3_wcm_parameter_convergence.png`

Two-panel figure.

- **Panel 1:** A value at convergence across all 40 configs. Box plot grouped by training size. Horizontal dashed line at literature init (0.10) and at literature range for heathland vegetation (0.08–0.12, shaded).
- **Panel 2:** B value at convergence. Same structure. Literature range: 0.10–0.20.
- Annotate: "Well-constrained" or "Poorly constrained" per diagnostic 2 result.

### Figure 4: `p3_parameter_sensitivity.png`

- X-axis: VWC (cm³/cm³), range [0.1, PEAT_THETA_SAT]
- Y-axis: ∂σ°_total/∂param (dB per unit)
- Two series: sensitivity to A (cyan), sensitivity to B (purple)
- Horizontal dashed line at poorly-constrained threshold (0.1 dB)
- Title: "WCM Forward Model Sensitivity — Moor House"

### Figure 5: `p3_dobson_vs_mironov.png`

- Scatter plot: Dobson RMSE (x) vs Mironov RMSE (y) for each of the 10 comparison configs
- Diagonal line (y=x)
- Points above diagonal: Mironov worse; below: Mironov better
- Annotation: relative improvement value and decision outcome
- Title: "Dielectric Model Sensitivity — Dobson (1985) vs Mironov (2009)"

---

## P3.13 Negative result protocol

If the gate classifies the outcome as **Negative** (PINN worse than best baseline at all training sizes):

1. **Do not fail the gate.** The experiment was conducted correctly. The gate passes regardless of outcome category.
2. Examine diagnostic 1: is `median_final_residual_ratio > 1.5` at 100% training? If yes, the ML branch dominated — the physics structure was not engaged.
3. Examine diagnostic 2: are A and B poorly constrained (sensitivity < 0.1)? If yes, the WCM is not sufficiently sensitive to m_v at this site and sensor combination.
4. Examine the Dobson/Mironov comparison: did Mironov trigger a re-run? If re-run also produced Negative, the dielectric model choice is not the issue.
5. **Do not cherry-pick configurations, modify λ post-hoc, or re-run with altered success criteria.** The result is the result.
6. Document findings in DEVIATIONS.md entry DEV-00N.
7. Proceed to Phase 4. The MVP chart will show the actual decomposition — if m_v_final ≈ m_v_physics, the chart still demonstrates the physics engine running and producing estimates. The Carbon13 narrative pivots to platform value: continuous monitoring, uncertainty quantification, site-agnostic architecture.
8. The gate output `narrative_recommendation` field provides the pivot language:

```json
"narrative_recommendation": "Negative result: physics-ML fusion did not improve over standard ML at this site and data volume. Recommended narrative: the platform demonstrates (1) continuous automated state estimation with quantified uncertainty — a capability no competitor offers at site scale; (2) a rigorous experimental methodology that produces honest results; (3) an architecture ready to incorporate improved physics models as they emerge from the PhD programme."
```

---

## P3.14 Gate 3 script (`poc/gates/gate_3.py`)

**Run with:** `python poc/gates/gate_3.py [--confirm-deviations]`  
**Exit code:** 0 = pass, 1 = fail

### Gate 3 criteria

| ID | Criterion | Threshold | Auto-checkable |
|----|-----------|-----------|----------------|
| G3-01 | λ search result exists and valid | file present + schema valid | Yes |
| G3-02 | All 40 PINN metric files exist | 40 files | Yes |
| G3-03 | All 40 PINN metrics schema-valid | 0 errors | Yes |
| G3-04 | No NaN in any PINN metrics | 0 NaN | Yes |
| G3-05 | PINN physics forward pass unit test | VV output in [−20, −5] dB | Yes |
| G3-06 | All composite loss components non-negative | min(L_data, L_physics, L_monotonic, L_bounds) ≥ 0 | Yes |
| G3-07 | WCM A parameter in valid range | A ∈ [0.01, 1.0] for all 40 configs | Yes |
| G3-08 | WCM B parameter in valid range | B ∈ [0.01, 1.0] for all 40 configs | Yes |
| G3-09 | All three diagnostics files exist | 3 files | Yes |
| G3-10 | Dobson/Mironov decision made | file present + rerun_triggered resolved | Yes |
| G3-11 | Outcome category evaluated and recorded | gate_3_result.json contains category | Yes |
| G3-12 | Statistical tests completed | Wilcoxon results for all 4 fractions | Yes |
| G3-13 | Uncertainty calibration complete | uncertainty_calibration.json present | Yes |
| G3-14 | All 5 figures exist | 5 files | Yes |
| G3-15 | pytest passes with 0 failures | 0 failures | Yes |
| G3-16 | DEVIATIONS.md reviewed | manual sign-off | No |

**Gate 3 passes regardless of outcome category.** The gate validates experimental integrity, not hypothesis confirmation.

### Gate 3 JSON output (`outputs/gates/gate_3_result.json`)

```json
{
    "gate": 3,
    "timestamp": "...",
    "passed": true,
    "exit_code": 0,
    "outcome": {
        "category": "Significant",
        "pinn_rmse_n25_median": 0.059,
        "best_baseline_rmse_n25": 0.071,
        "relative_reduction": 0.169,
        "wilcoxon_p_uncorrected": 0.008,
        "wilcoxon_p_bonferroni_threshold": 0.0125,
        "significant_bonferroni": true,
        "iqr_overlap": true
    },
    "diagnostics_summary": {
        "residual_ratio_n25_median": 0.93,
        "residual_ratio_architectural_concern": false,
        "A_well_constrained": true,
        "B_well_constrained": true,
        "mironov_rerun_triggered": false,
        "primary_dielectric": "Dobson (1985)"
    },
    "pinn_vs_baselines": {
        "100%": {"pinn": 0.041, "rf": 0.053, "nn": 0.061, "reduction_vs_best": 0.226},
        "50%":  {"pinn": 0.052, "rf": 0.061, "nn": 0.065, "reduction_vs_best": 0.148},
        "25%":  {"pinn": 0.059, "rf": 0.071, "nn": 0.075, "reduction_vs_best": 0.169},
        "10%":  {"pinn": 0.078, "rf": 0.089, "nn": 0.094, "reduction_vs_best": 0.124}
    },
    "fixture_values_for_mvp": {
        "vwc_current": 0.72,
        "rmse": 0.041,
        "r_squared": 0.871,
        "training_pairs": 142,
        "physics_estimate": 0.68,
        "ml_correction": 0.04,
        "uncertainty_k_80pct": 1.24,
        "uncertainty_k_95pct": 2.31
    },
    "narrative_recommendation": "Significant result: proceed with physics-advantage claim, qualified with 'statistically significant advantage at 25-sample data volume'.",
    "criteria": { ... },
    "warnings": []
}
```

The `fixture_values_for_mvp` block is the **direct input to Phase 4**. Claude Code building the MVP reads this file and uses these values to populate `src/data/fixtures/moor-house/`. No manual transcription.

---

## P3.15 Phase 3 test requirements

**Coverage target:** ≥ 80% line coverage on `poc/models/pinn.py`, `poc/models/dielectric.py`, `poc/evaluation/diagnostics.py`, `poc/evaluation/uncertainty.py`

### Required unit tests (`tests/unit/test_models_p3.py`)

| Test | What it checks |
|------|----------------|
| `test_dobson_at_zero_moisture` | ε(0) = EPSILON_DRY_PEAT |
| `test_dobson_increases_with_moisture` | ε(0.3) > ε(0.1) — monotonically increasing |
| `test_dobson_at_saturation` | ε(PEAT_THETA_SAT) < EPSILON_WATER (bound check) |
| `test_mironov_piecewise_boundary` | Continuous at m_v = MIRONOV_MV_T (no step) |
| `test_mironov_increases_with_moisture` | ε(0.4) > ε(0.2) |
| `test_oh_backscatter_range` | σ°_soil_dB in [−30, 0] for ε ∈ [3, 80], θ ∈ [25°, 50°] |
| `test_oh_backscatter_increases_with_epsilon` | Higher ε → higher backscatter |
| `test_oh_backscatter_no_nan_no_inf` | No NaN/inf for any physically valid input |
| `test_wcm_vegetation_tau_bounded` | τ² ∈ [0, 1] for all NDVI ∈ [0, 1] |
| `test_wcm_total_range` | σ°_total_dB in [−20, −5] for typical peatland inputs |
| `test_wcm_known_values` | Manual calculation cross-check at A=0.1, B=0.15, NDVI=0.4, θ=38°, m_v=0.6 |
| `test_sigmoid_bounds_A` | PINN.A always in [WCM_A_LB, WCM_A_UB] for any A_raw |
| `test_sigmoid_bounds_B` | PINN.B always in [WCM_B_LB, WCM_B_UB] for any B_raw |
| `test_sigmoid_init_close_to_literature` | A at init ≈ WCM_A_INIT ± 0.01 |
| `test_pinn_forward_returns_all_keys` | Dict contains all 7 required keys |
| `test_pinn_m_v_final_is_sum` | m_v_final = m_v_physics + delta_ml (exactly) |
| `test_pinn_physics_net_smaller_than_correction` | physics_net param count < correction_net param count |
| `test_loss_l_data_nonnegative` | L_data ≥ 0 always |
| `test_loss_l_physics_nonnegative` | L_physics ≥ 0 always |
| `test_loss_l_monotonic_zero_for_dobson` | L_monotonic ≈ 0 for Dobson (always increasing) |
| `test_loss_l_bounds_zero_within_range` | L_bounds = 0 when all m_v in [0, PEAT_THETA_SAT] |
| `test_loss_l_bounds_positive_outside_range` | L_bounds > 0 when m_v < 0 or m_v > PEAT_THETA_SAT |
| `test_vv_db_used_unnormalised_in_l_physics` | L_physics computed on raw dB, not normalised values |
| `test_pinn_reproducible_with_seed` | Two runs with same seed produce identical weights |
| `test_residual_ratio_computation` | std(δ) / std(m_v_physics) computed correctly |
| `test_calibration_achieves_target_coverage` | Calibrated 80% interval achieves 75–85% empirical coverage on test data |
| `test_calibration_k_positive` | Calibration factor k > 0 |
| `test_outcome_strong_criteria` | >20% reduction + no IQR overlap → "Strong" |
| `test_outcome_negative_criteria` | PINN worse than baseline → "Negative" |
| `test_wilcoxon_both_thresholds_reported` | Result dict contains both p_uncorrected and significant_bonferroni |

---

## P3.16 Phase 3 file manifest

```
poc/models/pinn.py
poc/models/dielectric.py
poc/evaluation/lambda_search.py
poc/evaluation/diagnostics.py
poc/evaluation/uncertainty.py
poc/evaluation/plots_p3.py
poc/gates/gate_3.py
outputs/models/pinn/lambda_search_result.json
outputs/models/pinn/config_000/ ... config_039/   (40 dirs)
outputs/metrics/config_000_pinn.json ... config_039_pinn.json  (40 files)
outputs/metrics/diagnostics_residual_ratio.json
outputs/metrics/diagnostics_parameter_sensitivity.json
outputs/metrics/diagnostics_dobson_vs_mironov.json
outputs/metrics/uncertainty_calibration.json
outputs/gates/gate_3_result.json
outputs/figures/p3_learning_curves_all_models.png
outputs/figures/p3_physics_ml_decomposition.png
outputs/figures/p3_wcm_parameter_convergence.png
outputs/figures/p3_parameter_sensitivity.png
outputs/figures/p3_dobson_vs_mironov.png
```
