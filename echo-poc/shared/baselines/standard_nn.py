"""
Baseline B — Standard Neural Network with early stopping.

Architecture: input → 64 → 32 → 16 → 1
ReLU activations, Dropout(0.2) between hidden layers.
Linear output (regression). Adam optimiser. MSE loss.
Early stopping on validation loss with patience=20.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from shared import config
from shared.baselines.base import BaseModel

logger = logging.getLogger(__name__)


class StandardNNModule(nn.Module):
    """
    Three hidden layers: 64 → 32 → 16 → 1.
    ReLU activations, Dropout(0.2) between hidden layers.
    Linear output — no activation on final layer (regression).
    """

    def __init__(
        self,
        n_features: int = len(config.FEATURE_COLUMNS),
        hidden_sizes: list[int] = None,
        dropout: float = config.NN_DROPOUT,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = config.NN_HIDDEN_SIZES

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


class NNModel(BaseModel):
    """
    Standard NN baseline with early stopping.

    Wraps StandardNNModule with StandardScaler, training loop,
    and checkpoint management.
    """

    model_name = "baseline_b"

    def __init__(self, config_idx: int = 0):
        self.config_idx_ = config_idx
        self.scaler_: StandardScaler | None = None
        self.model_: StandardNNModule | None = None
        self.device_ = config.get_torch_device()
        self.training_history_: dict | None = None
        self.stopped_at_epoch_: int | None = None
        self.best_val_loss_: float | None = None
        self.n_train_: int = 0
        self._best_state_dict: dict | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> None:
        """
        Train NN with early stopping on validation loss.

        Args:
            X_train: Feature matrix, shape (N_train, n_features).
            y_train: Target VWC, shape (N_train,). cm³/cm³.
            X_val: Validation features, shape (N_val, n_features).
            y_val: Validation targets, shape (N_val,). cm³/cm³.
        """
        self.n_train_ = len(X_train)
        seed = config.SEED + self.config_idx_

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        if self.device_.type == "mps":
            torch.mps.manual_seed(seed)
        elif self.device_.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Fit scaler on training data only
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_val_scaled = self.scaler_.transform(X_val)

        # Convert to tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device_)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device_)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device_)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device_)

        # Model
        n_features = X_train.shape[1]
        self.model_ = StandardNNModule(n_features=n_features).to(self.device_)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=config.NN_LR,
            weight_decay=config.NN_WEIGHT_DECAY,
        )
        criterion = nn.MSELoss()

        batch_size = min(config.NN_BATCH_SIZE, len(X_train))
        train_ds = TensorDataset(X_train_t, y_train_t)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        epoch_train_losses = []
        epoch_val_losses = []

        for epoch in range(config.NN_MAX_EPOCHS):
            # Train step
            self.model_.train()
            g = torch.Generator()
            g.manual_seed(seed + epoch)
            loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, generator=g
            )

            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.model_(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            epoch_train_losses.append(epoch_loss / max(n_batches, 1))

            # Validation step
            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            epoch_val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state_dict = {
                    k: v.clone().cpu() for k, v in self.model_.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= config.NN_PATIENCE:
                logger.info(
                    "NN config_%03d: early stopping at epoch %d (val_loss=%.6f)",
                    self.config_idx_, epoch + 1, best_val_loss,
                )
                break

        # Load best checkpoint
        if self._best_state_dict is not None:
            self.model_.load_state_dict(self._best_state_dict)
            self.model_.to(self.device_)

        self.stopped_at_epoch_ = epoch + 1
        self.best_val_loss_ = best_val_loss
        self.training_history_ = {
            "epoch": list(range(1, len(epoch_train_losses) + 1)),
            "train_loss": epoch_train_losses,
            "val_loss": epoch_val_losses,
        }

        logger.info(
            "NN config_%03d: stopped_at_epoch=%d, best_val_loss=%.6f",
            self.config_idx_, self.stopped_at_epoch_, self.best_val_loss_,
        )

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict VWC using fitted NN model.

        Args:
            X: Feature matrix, shape (N, n_features). Raw (unscaled).

        Returns:
            y_pred: Predicted VWC, shape (N,). cm³/cm³.

        Raises:
            ValueError: If model not fitted.
        """
        if self.model_ is None or self.scaler_ is None:
            raise ValueError("NNModel not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            y_pred = self.model_(X_t).cpu().numpy()

        return y_pred

    def save(self, directory: Path) -> None:
        """
        Save NN model weights, scaler, and config to directory.

        Saves:
            model_weights.pt — best checkpoint state dict
            scaler.pkl — fitted StandardScaler
            config.json — hyperparams, training history, metadata
        """
        if self.model_ is None or self.scaler_ is None:
            raise ValueError("Cannot save unfitted model")

        directory.mkdir(parents=True, exist_ok=True)

        # Save best state dict (already on CPU)
        state_dict = self._best_state_dict or {
            k: v.cpu() for k, v in self.model_.state_dict().items()
        }
        torch.save(state_dict, directory / "model_weights.pt")
        joblib.dump(self.scaler_, directory / "scaler.pkl")

        meta = {
            "n_features": len(config.FEATURE_COLUMNS),
            "hidden_sizes": config.NN_HIDDEN_SIZES,
            "dropout": config.NN_DROPOUT,
            "lr": config.NN_LR,
            "n_train": self.n_train_,
            "config_idx": self.config_idx_,
            "stopped_at_epoch": self.stopped_at_epoch_,
            "best_val_loss": self.best_val_loss_,
            "feature_names": config.FEATURE_COLUMNS,
            "training_history": self.training_history_,
        }
        with open(directory / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("NNModel saved to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "NNModel":
        """Load NN model from saved artefacts."""
        weights_path = directory / "model_weights.pt"
        scaler_path = directory / "scaler.pkl"
        config_path = directory / "config.json"

        for p in [weights_path, scaler_path, config_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing artefact: {p}")

        with open(config_path) as f:
            meta = json.load(f)

        instance = cls(config_idx=meta.get("config_idx", 0))
        instance.scaler_ = joblib.load(scaler_path)
        instance.n_train_ = meta.get("n_train", 0)
        instance.stopped_at_epoch_ = meta.get("stopped_at_epoch")
        instance.best_val_loss_ = meta.get("best_val_loss")
        instance.training_history_ = meta.get("training_history")

        n_features = meta.get("n_features", len(config.FEATURE_COLUMNS))
        hidden_sizes = meta.get("hidden_sizes", config.NN_HIDDEN_SIZES)
        dropout = meta.get("dropout", config.NN_DROPOUT)

        instance.model_ = StandardNNModule(
            n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        instance.model_.load_state_dict(state_dict)
        instance.model_.to(instance.device_)

        return instance
