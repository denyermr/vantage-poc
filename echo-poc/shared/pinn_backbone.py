"""
PINN backbone — PhysicsNet and CorrectionNet sub-networks.

These classes define the architectural skeleton shared between Phase 1
(WCM-PINN) and Phase 1b (MIMICS-PINN) per `SPEC.md` §2. The physics forward
model (WCM, MIMICS, …) is wired in by the phase-specific code that composes
these two modules with a physics branch.

Moved here from the original `poc/models/pinn.py` as part of the
Phase 1b repository re-layout (see `ARCHITECTURE.md`). The sub-networks
themselves are unchanged.

References:
    SPEC.md §2 — PINN backbone (frozen across Phase 1 and Phase 1b)
    SPEC_PHASE3.md §P3.5 — original Phase 1 specification
"""

import torch
import torch.nn as nn

from shared.config import FEATURE_COLUMNS, NN_DROPOUT, PEAT_THETA_SAT


# ─── PhysicsNet sub-network ─────────────────────────────────────────────────


class PhysicsNet(nn.Module):
    """
    Physics sub-network: maps input features to VWC estimate.

    Architecture (deliberately smaller than CorrectionNet):
        Input(n_features) → Linear(32) → ReLU → Linear(16) → ReLU
        → Linear(1) → Sigmoid scaled to [0, PEAT_THETA_SAT]

    The output is bounded to the physically plausible VWC range via
    sigmoid activation scaled by the saturated water content of peat.

    Reference:
        SPEC_PHASE3.md §P3.5 — physics_net specification
    """

    def __init__(self, n_features: int = len(FEATURE_COLUMNS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: X → m_v_physics.

        Args:
            X: Feature matrix, shape (N, n_features). Normalised.

        Returns:
            m_v_physics: VWC estimate, shape (N,). Range [0, PEAT_THETA_SAT].
        """
        raw = self.net(X).squeeze(-1)
        return PEAT_THETA_SAT * torch.sigmoid(raw)


# ─── CorrectionNet sub-network ──────────────────────────────────────────────


class CorrectionNet(nn.Module):
    """
    ML correction sub-network: learns residuals between physics estimate and observed VWC.

    Architecture (identical to Baseline B — StandardNNModule):
        Input(n_features) → Linear(64) → ReLU → Dropout(0.2)
        → Linear(32) → ReLU → Dropout(0.2)
        → Linear(16) → ReLU
        → Linear(1) → δ (unbounded — can be positive or negative)

    The output is unbounded, allowing the correction to shift the physics
    estimate in either direction.

    Reference:
        SPEC_PHASE3.md §P3.5 — correction_net specification
    """

    def __init__(self, n_features: int = len(FEATURE_COLUMNS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(NN_DROPOUT),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(NN_DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: X → δ (correction term).

        Args:
            X: Feature matrix, shape (N, n_features). Normalised.

        Returns:
            delta_ml: Correction term, shape (N,). Unbounded.
        """
        return self.net(X).squeeze(-1)
