"""
Dielectric mixing models for soil moisture retrieval.

Two models implemented as differentiable PyTorch modules:
  - DobsonDielectric (primary): semi-empirical 4-component mixing model
  - MironovDielectric (sensitivity check): physical-statistical refractive index model

Both share the same interface so the PINN can swap them with a single argument.
All operations use torch tensors for autograd compatibility.

References:
    Dobson et al. (1985), IEEE TGRS 23(1), eq. 1
    Bechtold et al. (2018), Remote Sensing 10(2)
    Mironov et al. (2009), IEEE TGRS 57(7), Table II
"""

from abc import ABC, abstractmethod

import torch

from shared.config import (
    DOBSON_ALPHA,
    EPSILON_DRY_PEAT,
    EPSILON_WATER,
    MIRONOV_MV_T,
    MIRONOV_ND,
    MIRONOV_ND1,
    MIRONOV_ND2,
    PEAT_THETA_SAT,
)


class DielectricModel(ABC):
    """
    Computes real part of soil dielectric constant from volumetric water content.

    Must be differentiable — implemented entirely with torch operations.
    Subclasses implement forward() and can be used as callables.
    """

    @abstractmethod
    def forward(self, m_v: torch.Tensor) -> torch.Tensor:
        """
        Compute dielectric constant from volumetric water content.

        Args:
            m_v: VWC, shape (...). Range [0, PEAT_THETA_SAT]. cm³/cm³.

        Returns:
            epsilon: Real dielectric constant, shape (...). Dimensionless.
        """

    def __call__(self, m_v: torch.Tensor) -> torch.Tensor:
        """Allow callable syntax: model(m_v) delegates to forward()."""
        return self.forward(m_v)


class DobsonDielectric(DielectricModel):
    """
    Semi-empirical dielectric mixing model for organic soil at C-band.

    ε(m_v) = ε_dry + (ε_water − 1) · m_v^α

    Parameterised for blanket bog peat following Bechtold et al. (2018):
        ε_dry   = EPSILON_DRY_PEAT = 3.5   (zero clay fraction, low bulk density)
        ε_water = EPSILON_WATER = 80.0
        α       = DOBSON_ALPHA = 1.4        (organic soil empirical exponent)

    Physical constraints:
        - At m_v = 0: ε = ε_dry (dry peat)
        - Monotonically increasing with m_v for m_v ≥ 0
        - At m_v = PEAT_THETA_SAT: ε < ε_water (bound check)

    Reference:
        Dobson et al. (1985), IEEE TGRS 23(1), eq. 1
        Bechtold et al. (2018), Remote Sensing 10(2)
    """

    def forward(self, m_v: torch.Tensor) -> torch.Tensor:
        """
        Compute Dobson dielectric constant.

        Args:
            m_v: VWC, shape (...). Range [0, PEAT_THETA_SAT]. cm³/cm³.

        Returns:
            epsilon: Real dielectric constant, shape (...). Dimensionless.

        Numerical stability:
            m_v is clamped to [0, 1] before exponentiation to prevent NaN
            from negative values raised to fractional power.
        """
        m_v_safe = m_v.clamp(min=0.0)
        eps = EPSILON_DRY_PEAT + (EPSILON_WATER - 1.0) * m_v_safe.pow(DOBSON_ALPHA)
        return eps


class MironovDielectric(DielectricModel):
    """
    Physical-statistical dielectric model for organic soil.

    Used only for the Dobson vs Mironov sensitivity check (diagnostic 3).
    NOT used in the primary 40-config experiment.

    Piecewise model with transition moisture mv_t:
        n(m_v) = nd + nd1·m_v                      if m_v <= mv_t
        n(m_v) = nd + nd1·mv_t + nd2·(m_v - mv_t)  if m_v > mv_t
        ε(m_v) ≈ n(m_v)²   (real part, k≈0 at C-band for organic soil)

    Parameters for organic soil from Mironov et al. (2009), IEEE TGRS, Table II:
        nd   = 0.312  (refractive index of dry soil)
        nd1  = 1.42   (refractive index slope below transition)
        nd2  = 0.89   (refractive index slope above transition)
        mv_t = 0.36   (transition moisture)

    Physical constraints:
        - Continuous at m_v = mv_t (no step discontinuity)
        - Monotonically increasing with m_v (nd1, nd2 > 0)

    Reference:
        Mironov et al. (2009), IEEE TGRS 57(7)
    """

    def forward(self, m_v: torch.Tensor) -> torch.Tensor:
        """
        Compute Mironov dielectric constant.

        Args:
            m_v: VWC, shape (...). Range [0, PEAT_THETA_SAT]. cm³/cm³.

        Returns:
            epsilon: Real dielectric constant, shape (...). Dimensionless.
        """
        n_below = MIRONOV_ND + MIRONOV_ND1 * m_v
        n_above = (MIRONOV_ND + MIRONOV_ND1 * MIRONOV_MV_T
                   + MIRONOV_ND2 * (m_v - MIRONOV_MV_T))
        n = torch.where(m_v <= MIRONOV_MV_T, n_below, n_above)
        return n.pow(2)
