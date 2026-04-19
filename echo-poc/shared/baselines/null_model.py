"""
Baseline 0 — Seasonal climatological null model.

Predicts the seasonal mean VWC from training data for each observation.
Uses no SAR or ancillary features — only the month of year.

Purpose:
  - Establishes performance floor
  - Any ML model that cannot beat this adds no value beyond seasonal pattern recognition
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from shared import config
from shared.splits import assign_season
from shared.baselines.base import BaseModel

logger = logging.getLogger(__name__)


class NullModel(BaseModel):
    """
    Seasonal climatological baseline.

    fit(): learns seasonal mean VWC from training data.
    predict(): returns the seasonal mean for each observation's date.
    """

    model_name = "null_baseline"
    uses_dates = True

    def __init__(self):
        self.seasonal_means_: dict[str, float] | None = None
        self.global_mean_: float | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            train_dates: pd.DatetimeIndex = None, **kwargs) -> None:
        """
        Learn seasonal mean VWC from training data.

        X_train and X_val are accepted but ignored.
        train_dates required to assign seasons to training labels.

        Args:
            X_train: Ignored.
            y_train: Training VWC values, shape (N_train,). cm³/cm³.
            X_val: Ignored.
            y_val: Ignored.
            train_dates: DatetimeIndex for training observations.

        Raises:
            ValueError: If train_dates is None or length mismatch.
        """
        if train_dates is None:
            raise ValueError("NullModel.fit() requires train_dates argument")
        if len(train_dates) != len(y_train):
            raise ValueError(
                f"train_dates length {len(train_dates)} != y_train length {len(y_train)}"
            )

        self.global_mean_ = float(np.mean(y_train))
        seasons = assign_season(train_dates)

        self.seasonal_means_ = {}
        for s in ["DJF", "MAM", "JJA", "SON"]:
            mask = seasons == s
            if mask.sum() > 0:
                self.seasonal_means_[s] = float(np.mean(y_train[mask]))
            else:
                logger.warning(
                    "Season %s has no training samples; using global mean %.4f",
                    s, self.global_mean_,
                )
                self.seasonal_means_[s] = self.global_mean_

        logger.info(
            "NullModel fit: global_mean=%.4f, seasonal_means=%s",
            self.global_mean_,
            {k: f"{v:.4f}" for k, v in self.seasonal_means_.items()},
        )

    def predict(self, X: np.ndarray,
                pred_dates: pd.DatetimeIndex = None, **kwargs) -> np.ndarray:
        """
        Predict seasonal mean VWC for each observation.

        X is ignored. Prediction based on pred_dates season only.

        Args:
            X: Ignored.
            pred_dates: DatetimeIndex for prediction observations.

        Returns:
            y_pred: Predicted VWC, shape (N,). cm³/cm³.

        Raises:
            ValueError: If model not fitted or pred_dates is None.
        """
        if self.seasonal_means_ is None:
            raise ValueError("NullModel not fitted. Call fit() first.")
        if pred_dates is None:
            raise ValueError("NullModel.predict() requires pred_dates argument")

        seasons = assign_season(pred_dates)
        predictions = np.array([
            self.seasonal_means_.get(s, self.global_mean_)
            for s in seasons
        ])
        return predictions

    def save(self, directory: Path) -> None:
        """Save seasonal means and global mean to JSON."""
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "global_mean": self.global_mean_,
            "seasonal_means": self.seasonal_means_,
        }
        with open(directory / "null_model.json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info("NullModel saved to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "NullModel":
        """Load NullModel from saved JSON."""
        path = directory / "null_model.json"
        if not path.exists():
            raise FileNotFoundError(f"NullModel artefact not found: {path}")
        with open(path) as f:
            data = json.load(f)
        model = cls()
        model.global_mean_ = data["global_mean"]
        model.seasonal_means_ = data["seasonal_means"]
        return model
