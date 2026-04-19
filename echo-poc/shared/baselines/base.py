"""
Abstract base model interface for all ECHO PoC models.

All baselines (null, RF, NN) and the Phase 3 PINN implement this interface.
The evaluation harness only calls these methods — it has no model-specific logic.
"""

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

    # Set True on models that need date arguments (e.g. NullModel)
    uses_dates: bool = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> None:
        """
        Train the model.

        Args:
            X_train: Feature matrix, shape (N_train, n_features). Normalised.
            y_train: Target VWC, shape (N_train,). cm³/cm³.
            X_val:   Validation features, shape (N_val, n_features). Normalised.
            y_val:   Validation targets, shape (N_val,). cm³/cm³.
            **kwargs: Model-specific args (e.g. train_dates for NullModel).
        """

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return point predictions.

        Args:
            X: Feature matrix, shape (N, n_features). Normalised.
            **kwargs: Model-specific args (e.g. pred_dates for NullModel).

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
