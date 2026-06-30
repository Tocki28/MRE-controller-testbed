"""EIS analysis utilities — pure module, no Dash imports.

These constants and functions target the room-temp iron electrolysis testbed
(Fe/Ca/Mg/Al aqueous solution, ~298 K). They are NOT the MOE/MRE simulation
constants, which live in testbed/randles_model.py (Rs~1.2 Ω, T~1600 °C).
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# ── Randles fit constants (room-temp testbed, NOT MOE simulation) ────────────
_TESTBED_INITIAL_GUESS = [100.0, 50.0, 5.0, 1e-4]   # Rs, Rct, sigma, Cdl
_TESTBED_BOUNDS = (
    [10.0,   0.1,  0.01, 1e-7],    # lower: Rs, Rct, sigma, Cdl
    [1000.0, 500.0, 200.0, 1e-2],  # upper
)
_R_GAS = 8.314        # J / (mol·K)
_F_FARADAY = 96485.0  # C / mol


# ── Sweep parser ─────────────────────────────────────────────────────────────

class SweepParser:
    """Parse UART CSV lines into complete EIS sweeps.

    Line format: frequency_hz,real_ohm,imag_ohm
    Sweep sentinel: any line starting with '# DBG'
    """

    def __init__(self) -> None:
        self._buf: list[tuple[float, float, float]] = []

    def parse_line(self, line: str) -> Optional[tuple[float, float, float]]:
        """Return (freq_hz, re_ohm, im_ohm) on valid CSV line, None otherwise."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        try:
            parts = line.split(',')
            f, re, im = float(parts[0]), float(parts[1]), float(parts[2])
            self._buf.append((f, re, im))
            return (f, re, im)
        except (ValueError, IndexError):
            return None  # malformed line — skip silently

    def is_complete_after(self, line: str) -> bool:
        """True if this line is the sweep-end sentinel (# DBG)."""
        return line.strip().startswith('# DBG')

    def get_sweep(self) -> Optional[dict]:
        """Return completed sweep dict and clear internal buffer.

        Returns None if the buffer is empty (no data points collected).
        """
        if not self._buf:
            return None
        freqs = [p[0] for p in self._buf]
        res   = [p[1] for p in self._buf]
        ims   = [p[2] for p in self._buf]
        self._buf = []
        return {'freq': freqs, 'Z_re': res, 'Z_im': ims}


# ── Thread-safe EIS buffer ───────────────────────────────────────────────────

class EISBuffer:
    """Thread-safe store for EIS sweeps consumed by the Dash callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest: Optional[dict] = None
        self.previous: Optional[dict] = None
        self.history: deque = deque(maxlen=100)
        self.sweep_timestamp: Optional[datetime] = None
        self.sweep_count: int = 0
        self.status: str = 'NO_DATA'
        self.error_msg: str = ''

    def add_sweep(self, sweep: dict) -> None:
        with self._lock:
            self.previous = self.latest
            self.latest = sweep
            self.history.append(sweep)
            self.sweep_timestamp = datetime.now(timezone.utc)
            self.sweep_count += 1
            self.status = 'LIVE'

    def set_error(self, msg: str) -> None:
        with self._lock:
            self.status = 'ERROR'
            self.error_msg = msg

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def get_snapshot(self) -> dict:
        """Return a thread-safe copy of all state."""
        with self._lock:
            return {
                'latest': self.latest,
                'previous': self.previous,
                'history': list(self.history),
                'sweep_timestamp': self.sweep_timestamp,
                'sweep_count': self.sweep_count,
                'status': self.status,
                'error_msg': self.error_msg,
            }


# ── Analytical parameter extraction ──────────────────────────────────────────

def extract_analytical_params(
    freq: list[float],
    Z_re: list[float],
    Z_im: list[float],
) -> Optional[dict]:
    """Extract analytical EIS parameters without curve fitting.

    Args:
        freq:  Frequency points in Hz (ascending order expected).
        Z_re:  Real part of impedance in Ω.
        Z_im:  Imaginary part of impedance in Ω (signed; negative = inductive,
               positive = capacitive for conventional EIS sign convention).

    Returns:
        Dict with keys: Rs, Z_abs_1Hz, phase_100Hz, f_peak, tau_ms.
        Returns None if fewer than 3 frequency points provided.
    """
    if len(freq) < 3:
        return None

    freq_arr = np.asarray(freq, dtype=float)
    re_arr   = np.asarray(Z_re, dtype=float)
    im_arr   = np.asarray(Z_im, dtype=float)

    # Rs: Re(Z) at highest measured frequency (high-f real intercept)
    Rs = float(re_arr[np.argmax(freq_arr)])

    # |Z| at nearest point to 1 Hz
    idx_1hz = int(np.argmin(np.abs(freq_arr - 1.0)))
    Z_abs_1Hz = float(np.hypot(re_arr[idx_1hz], im_arr[idx_1hz]))

    # Phase at nearest point to 100 Hz (degrees)
    idx_100hz = int(np.argmin(np.abs(freq_arr - 100.0)))
    phase_100Hz = float(np.degrees(np.arctan2(im_arr[idx_100hz], re_arr[idx_100hz])))

    # f_peak: frequency where -Im(Z) is maximum (semicircle apex)
    neg_im = -im_arr
    peak_idx = int(np.argmax(neg_im))
    if neg_im[peak_idx] < 0.01 * float(np.max(np.abs(re_arr))):
        # Imaginary component negligible — pure resistive data, no time constant
        f_peak = None
        tau_ms = None
    else:
        f_peak = float(freq_arr[peak_idx])
        tau_ms = float(1.0 / (2.0 * np.pi * f_peak) * 1000.0)

    return {
        'Rs': Rs,
        'Z_abs_1Hz': Z_abs_1Hz,
        'phase_100Hz': phase_100Hz,
        'f_peak': f_peak,
        'tau_ms': tau_ms,
    }


# ── Randles circuit fit via impedance.py ─────────────────────────────────────

_FIT_FAIL: dict = {
    'fit_ok': False,
    'residual': float('inf'),
    'Rs_fit': None,
    'Rct': None,
    'Cdl_uF': None,
    'sigma': None,
    'i0_proxy_mA_cm2': None,
}


def fit_randles(
    freq: list[float],
    Z_re: list[float],
    Z_im: list[float],
    n_electrons: int = 2,
    T_K: float = 298.0,
    electrode_area_cm2: float = 1.0,
) -> dict:
    """Fit Randles circuit R0-p(R1-W1,C1) using impedance.py.

    Circuit topology: Rs + (Rct + Warburg) ∥ Cdl
    Parameter order in impedance.py: [R0=Rs, R1=Rct, W1=sigma, C1=Cdl]

    Returns a dict with fit_ok=False and all params=None on any failure.
    Residual thresholds: <5% green, <15% yellow, otherwise red.

    NOTE: T_K and electrode_area_cm2 are for the i0 proxy calculation only.
    At 298 K this is valid for room-temp aqueous electrolysis. Do not use
    the returned i0_proxy_mA_cm2 for high-temperature MOE (T~1873 K).
    """
    from impedance.models.circuits import CustomCircuit  # lazy import — no Dash dep

    if len(freq) < 3:
        return dict(_FIT_FAIL)

    freq_arr = np.asarray(freq, dtype=float)
    Z_complex = np.asarray(Z_re, dtype=float) + 1j * np.asarray(Z_im, dtype=float)

    try:
        circuit = CustomCircuit(
            'R0-p(R1-W1,C1)',
            initial_guess=_TESTBED_INITIAL_GUESS,
        )
        circuit.fit(freq_arr, Z_complex, bounds=_TESTBED_BOUNDS)
        Rs_fit, Rct, sigma, Cdl = circuit.parameters_

        Z_fit = circuit.predict(freq_arr)
        residual = float(np.mean(np.abs(Z_fit - Z_complex) / np.abs(Z_complex)))

        if residual > 0.5:  # >50% relative error → nonsense fit
            return dict(_FIT_FAIL)

        if float(Rct) <= 0:
            return dict(_FIT_FAIL)

        i0_proxy = (
            (_R_GAS * T_K) / (n_electrons * _F_FARADAY * float(Rct) * electrode_area_cm2)
        ) * 1000.0  # mA/cm²

        return {
            'fit_ok': True,
            'residual': residual,
            'Rs_fit': float(Rs_fit),
            'Rct': float(Rct),
            'Cdl_uF': float(Cdl) * 1e6,
            'sigma': float(sigma),
            'i0_proxy_mA_cm2': i0_proxy,
        }
    except Exception:
        return dict(_FIT_FAIL)
