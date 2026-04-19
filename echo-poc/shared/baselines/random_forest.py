"""
Baseline A — Random Forest with GridSearchCV.

For each of the 40 configs: load split, fit StandardScaler on train only,
run GridSearchCV, evaluate on sealed test set, save artefacts.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from shared import config
from shared.baselines.base import BaseModel

logger = logging.getLogger(__name__)


class RFModel(BaseModel):
    """
    Random Forest regressor with GridSearchCV hyperparameter tuning.

    Wraps sklearn RandomForestRegressor with standardised feature scaling.
    The scaler is always saved alongside the model.
    """

    model_name = "baseline_a"

    def __init__(self, config_idx: int = 0):
        self.config_idx_ = config_idx
        self.scaler_: StandardScaler | None = None
        self.model_: RandomForestRegressor | None = None
        self.best_params_: dict | None = None
        self.n_train_: int = 0
        self.cv_warning_: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> None:
        """
        Fit StandardScaler on X_train, then GridSearchCV RF.

        X_val is not used for RF (CV handles validation internally).

        Args:
            X_train: Feature matrix, shape (N_train, n_features).
            y_train: Target VWC, shape (N_train,). cm³/cm³.
            X_val: Validation features (accepted but not used by RF).
            y_val: Validation targets (accepted but not used by RF).
        """
        self.n_train_ = len(X_train)
        seed = config.SEED + self.config_idx_

        # Warn for very small training sets
        if self.n_train_ <= 10:
            self.cv_warning_ = True
            logger.warning(
                "N_train=%d — CV reliability limited (config_idx=%d)",
                self.n_train_, self.config_idx_,
            )

        # Fit scaler on training data only
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)

        # GridSearchCV
        cv = KFold(
            n_splits=min(config.RF_CV_FOLDS, self.n_train_),
            shuffle=True,
            random_state=seed,
        )

        grid = GridSearchCV(
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            config.RF_PARAM_GRID,
            scoring=config.RF_SCORING,
            cv=cv,
            refit=True,
        )
        grid.fit(X_train_scaled, y_train)

        self.model_ = grid.best_estimator_
        self.best_params_ = grid.best_params_

        logger.info(
            "RF config_%03d: best_params=%s, best_cv_score=%.4f",
            self.config_idx_, self.best_params_, grid.best_score_,
        )

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict VWC using fitted RF model.

        Args:
            X: Feature matrix, shape (N, n_features). Raw (unscaled).

        Returns:
            y_pred: Predicted VWC, shape (N,). cm³/cm³.

        Raises:
            ValueError: If model not fitted.
        """
        if self.model_ is None or self.scaler_ is None:
            raise ValueError("RFModel not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def get_feature_importances(self) -> dict[str, float] | None:
        """Return feature importances as a dict, or None if not available."""
        if self.model_ is not None and hasattr(self.model_, "feature_importances_"):
            return dict(zip(config.FEATURE_COLUMNS, self.model_.feature_importances_))
        return None

    def save(self, directory: Path) -> None:
        """
        Save RF model, scaler, and config to directory.

        Saves:
            model.pkl — fitted RandomForestRegressor
            scaler.pkl — fitted StandardScaler
            config.json — best params, metadata
            feature_importances.json — if available
        """
        if self.model_ is None or self.scaler_ is None:
            raise ValueError("Cannot save unfitted model")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, directory / "model.pkl")
        joblib.dump(self.scaler_, directory / "scaler.pkl")

        meta = {
            "best_params": self.best_params_,
            "n_train": self.n_train_,
            "feature_names": config.FEATURE_COLUMNS,
            "config_idx": self.config_idx_,
            "cv_warning": self.cv_warning_,
        }
        with open(directory / "config.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        importances = self.get_feature_importances()
        if importances is not None:
            with open(directory / "feature_importances.json", "w") as f:
                json.dump(importances, f, indent=2)

        logger.info("RFModel saved to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "RFModel":
        """Load RF model from saved artefacts."""
        model_path = directory / "model.pkl"
        scaler_path = directory / "scaler.pkl"
        config_path = directory / "config.json"

        for p in [model_path, scaler_path, config_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing artefact: {p}")

        with open(config_path) as f:
            meta = json.load(f)

        instance = cls(config_idx=meta.get("config_idx", 0))
        instance.model_ = joblib.load(model_path)
        instance.scaler_ = joblib.load(scaler_path)
        instance.best_params_ = meta.get("best_params")
        instance.n_train_ = meta.get("n_train", 0)
        instance.cv_warning_ = meta.get("cv_warning", False)

        return instance
