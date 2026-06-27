"""EIS-based health inference + OES composition mock + state estimator.

EIS sweep: every 60 sim-seconds acquire an EIS spectrum via EISDataSource,
fit by least-squares (scipy TRF) to extract R_ct, normalise to a health proxy.

OES: trivially returns composition from plant state + tiny noise.

State estimator: low-pass (α=0.3) filter on raw health measurement.
"""

from __future__ import annotations

import numpy as np
import structlog
from scipy.optimize import least_squares

from testbed.interfaces import EISDataSource, InferenceModule
from testbed.plant import PlantState
from testbed.randles_model import (
    MOE_CDL_BASELINE,
    MOE_RCT_END,
    MOE_RCT_NEW,
    MOE_RS_BASELINE,
    MOE_SIGMA_BASELINE,
    moe_frequencies,
    randles_impedance,
)

log = structlog.get_logger(__name__)

EIS_INTERVAL_S = 60.0   # sim-seconds between sweeps
ALPHA = 0.3              # EMA smoothing


# ---------------------------------------------------------------------------
# EIS data sources
# ---------------------------------------------------------------------------

class SyntheticEISSource(EISDataSource):
    """Generates synthetic Randles impedance from plant state (in-process)."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(seed=42)

    def acquire_spectrum(self, state: PlantState) -> tuple[np.ndarray, np.ndarray]:
        """Return (omega [rad/s], Z_complex [Ω]) for one synthetic sweep."""
        freq = moe_frequencies()
        omega = 2.0 * np.pi * freq

        Rs_true = MOE_RS_BASELINE
        Rct_true = MOE_RCT_NEW + (1.0 - state.electrode_health) * (
            MOE_RCT_END - MOE_RCT_NEW
        )
        Cdl_true = MOE_CDL_BASELINE
        sigma_true = MOE_SIGMA_BASELINE

        Z_true = randles_impedance(omega, Rs_true, Rct_true, Cdl_true, sigma_true)
        noise = (
            self._rng.normal(0, 0.03, size=len(omega))
            + 1j * self._rng.normal(0, 0.03, size=len(omega))
        )
        Z_meas = Z_true + noise
        return omega, Z_meas


class RodeostatEISSource(EISDataSource):
    """Hardware EIS source via Rodeostat potentiostat (requires M1.2)."""

    def acquire_spectrum(self, state: PlantState) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "RodeostatEISSource not yet implemented — requires M1.2 hardware"
        )


# ---------------------------------------------------------------------------
# Randles fitting
# ---------------------------------------------------------------------------

def fit_randles_scipy(
    omega: np.ndarray, Z_meas: np.ndarray
) -> tuple[float, float, float, float]:
    """Fit 4-parameter Randles model to measured impedance via scipy TRF.

    Returns (Rs, Rct, Cdl, sigma).
    """
    x0 = [MOE_RS_BASELINE, MOE_RCT_NEW * 1.5, MOE_CDL_BASELINE, MOE_SIGMA_BASELINE]
    lower = [0.01, 0.1, 1e-6, 0.01]
    upper = [20.0, 200.0, 1e-2, 50.0]

    def residuals(x: list[float]) -> np.ndarray:
        Z_fit = randles_impedance(omega, x[0], x[1], x[2], x[3])
        diff = Z_fit - Z_meas
        return np.concatenate([diff.real, diff.imag])

    result = least_squares(residuals, x0, bounds=(lower, upper), method="trf")
    Rs, Rct, Cdl, sigma = result.x
    return float(Rs), float(Rct), float(Cdl), float(sigma)


# ---------------------------------------------------------------------------
# Inference module
# ---------------------------------------------------------------------------

class MOEInference(InferenceModule):
    """Composed inference module: EIS + OES + EMA state estimator."""

    RCT_NEW = MOE_RCT_NEW    # Ω - nominal Rct for a fresh electrode
    RCT_END = MOE_RCT_END    # Ω - Rct at end-of-life

    def __init__(self, eis_source: EISDataSource | None = None) -> None:
        self._eis_source: EISDataSource = (
            eis_source if eis_source is not None else SyntheticEISSource()
        )
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

    def reset_for_new_batch(self) -> None:
        """Reset health estimate so dashboard shows 1.0 for the fresh electrode."""
        self._health_est = 1.0
        self._health_history = []
        self._last_eis_time = -999.0
        self._predicted_ttf = 999.0

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
        """Acquire EIS spectrum, fit Randles, update health estimate."""
        omega, Z_meas = self._eis_source.acquire_spectrum(state)
        freq = omega / (2.0 * np.pi)

        _, Rct_fit, _, _ = fit_randles_scipy(omega, Z_meas)

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
