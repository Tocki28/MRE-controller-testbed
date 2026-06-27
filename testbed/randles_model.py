"""Shared Randles circuit model for EIS fitting.

4-parameter model: Rs, Rct, Cdl, sigma (Warburg coefficient).
Z(ω) = Rs + (Rct + Z_W) || Z_dl
where Z_W = sigma*(1-j)/sqrt(omega), Z_dl = 1/(jω·Cdl)

MOE anode baseline values from published literature (molten oxide electrolyte).
"""
from __future__ import annotations

import numpy as np


def randles_impedance(
    omega: np.ndarray, Rs: float, Rct: float, Cdl: float, sigma: float
) -> np.ndarray:
    """Full Randles model with Warburg diffusion element."""
    j = 1j
    Z_W = sigma * (1.0 - j) / np.sqrt(omega)
    Z_dl = 1.0 / (j * omega * Cdl)
    Z_ct_W = Rct + Z_W
    return Rs + (Z_ct_W * Z_dl) / (Z_ct_W + Z_dl)


# MOE anode baseline constants (fresh electrode, molten oxide electrolyte)
MOE_RS_BASELINE = 1.2       # Ω  solution resistance
MOE_RCT_NEW = 8.0           # Ω  charge-transfer resistance, fresh anode
MOE_RCT_END = 50.0          # Ω  Rct at end-of-life (6× nominal — conservative estimate)
MOE_CDL_BASELINE = 2.5e-4   # F  double-layer capacitance
MOE_SIGMA_BASELINE = 3.5    # Ω·s^-0.5  Warburg coefficient


def moe_frequencies() -> np.ndarray:
    """Standard MOE EIS frequency sweep: 0.01 Hz → 10 kHz (6 decades, 50 points)."""
    return np.logspace(-2, 4, 50)
