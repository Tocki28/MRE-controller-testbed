"""EIS-based health inference + OES composition mock + state estimator.

EIS sweep: every 60 sim-seconds generate a synthetic Randles impedance
spectrum, add noise, fit by least-squares to extract R_ct, normalise
to a health proxy.

OES: trivially returns composition from plant state + tiny noise.

State estimator: low-pass (α=0.3) filter on raw health measurement.
"""

from __future__ import annotations

import time

import numpy as np
import structlog

from testbed.interfaces import InferenceModule
from testbed.plant import PlantState

log = structlog.get_logger(__name__)

EIS_INTERVAL_S = 60.0   # sim-seconds between sweeps
ALPHA = 0.3              # EMA smoothing


def _randles_impedance(
    omega: np.ndarray,
    Rs: float,
    Rct: float,
    Cdl: float,
) -> np.ndarray:
    """Forward Randles model: Z(ω) = Rs + Rct / (1 + jω·Rct·Cdl)"""
    j = 1j
    return Rs + Rct / (1.0 + j * omega * Rct * Cdl)


def _fit_randles(
    omega: np.ndarray, Z_meas: np.ndarray
) -> tuple[float, float, float]:
    """Least-squares fit of Randles model to measured (noisy) impedance.

    Returns (Rs, Rct, Cdl) estimates.
    Uses a simple grid search over a coarse parameter space - sufficient
    for the v1 simulation without requiring scipy.
    """
    best_err = float("inf")
    best = (0.01, 1.0, 1e-4)
    for Rs in np.linspace(0.005, 0.1, 8):
        for Rct in np.linspace(0.1, 10.0, 20):
            for Cdl in np.logspace(-5, -2, 8):
                Z_fit = _randles_impedance(omega, Rs, Rct, Cdl)
                err = float(np.sum(np.abs(Z_meas - Z_fit) ** 2))
                if err < best_err:
                    best_err = err
                    best = (Rs, Rct, Cdl)
    return best


class MOEInference(InferenceModule):
    """Composed inference module: EIS + OES + EMA state estimator."""

    RCT_NEW = 0.5    # Ω - nominal Rct for a fresh electrode
    RCT_END = 5.0    # Ω - Rct at end-of-life

    def __init__(self) -> None:
        self._last_eis_time: float = -999.0
        self._health_est: float = 1.0
        self._health_history: list[float] = []
        self._rng = np.random.default_rng(seed=7)

        # Store last EIS data for dashboard
        self._eis_freq: list[float] = []
        self._eis_Z_re: list[float] = []
        self._eis_Z_im: list[float] = []
        self._predicted_ttf: float = 999.0

    # ------------------------------------------------------------------

    def update(self, state: PlantState) -> dict:
        needs_sweep = (state.uptime_s - self._last_eis_time) >= EIS_INTERVAL_S

        if needs_sweep:
            self._run_eis_sweep(state)
            self._last_eis_time = state.uptime_s

        # OES: composition + small noise
        comp_est = {
            k: max(0.0, v + self._rng.normal(0, 0.003))
            for k, v in state.composition.items()
        }
        total = sum(comp_est.values())
        comp_est = {k: v / total for k, v in comp_est.items()}

        # Time-to-failure estimate from degradation rate
        self._health_history.append(self._health_est)
        if len(self._health_history) > 300:
            self._health_history.pop(0)

        if len(self._health_history) >= 10:
            recent = np.array(self._health_history[-10:])
            dh_per_s = float(np.mean(np.diff(recent)))  # typically negative
            if dh_per_s < -1e-9:
                self._predicted_ttf = float(
                    np.clip(self._health_est / (-dh_per_s) / 3600.0, 0.0, 999.0)
                )

        return {
            "electrode_health_est": self._health_est,
            "composition_est": comp_est,
            "predicted_ttf_hrs": self._predicted_ttf,
            "eis_freq": self._eis_freq,
            "eis_Z_re": self._eis_Z_re,
            "eis_Z_im": self._eis_Z_im,
        }

    # ------------------------------------------------------------------

    def _run_eis_sweep(self, state: PlantState) -> None:
        """Generate synthetic EIS data, fit Randles, update health estimate."""
        freq = np.logspace(-2, 4, 50)   # 0.01 Hz → 10 kHz
        omega = 2.0 * np.pi * freq

        # True parameters: Rct scales with degradation (health ↓ → Rct ↑)
        true_Rct = self.RCT_NEW + (1.0 - state.electrode_health) * (
            self.RCT_END - self.RCT_NEW
        )
        Rs_true = 0.015
        Cdl_true = 1e-4

        Z_true = _randles_impedance(omega, Rs_true, true_Rct, Cdl_true)
        noise = self._rng.normal(0, 0.03, size=len(omega)) + 1j * self._rng.normal(
            0, 0.03, size=len(omega)
        )
        Z_meas = Z_true + noise

        # Fit to extract Rct
        _, Rct_fit, _ = _fit_randles(omega, Z_meas)

        # Normalise Rct → health proxy (inverted and clamped)
        raw = 1.0 - (Rct_fit - self.RCT_NEW) / (self.RCT_END - self.RCT_NEW)
        raw = float(np.clip(raw, 0.0, 1.0))

        # Low-pass filter
        self._health_est = ALPHA * raw + (1.0 - ALPHA) * self._health_est

        log.info(
            "eis_sweep_complete",
            Rct_fit=round(Rct_fit, 3),
            health_est=round(self._health_est, 3),
        )

        # Store for dashboard Nyquist plot
        self._eis_freq = freq.tolist()
        self._eis_Z_re = Z_meas.real.tolist()
        self._eis_Z_im = (-Z_meas.imag).tolist()   # Nyquist convention: -Im vs Re
