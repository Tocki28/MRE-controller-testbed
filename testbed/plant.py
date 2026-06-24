"""MOE cell plant simulator. Thread-safe via a lock on every state mutation."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field

import numpy as np

# Faraday constant (C/mol)
_FARADAY = 96_485.0

# Internal bath depletion model - tracks how oxide fractions evolve as
# different oxides are preferentially reduced.  These phases are physics-
# internal only: the cathode output is always an alloy, not sequential
# pure metals.  Externally we surface O₂ production (the anode output),
# which IS pure and mission-critical.
DEPLETION_THRESHOLDS = {
    "Fe": 0.04,
    "Si": 0.06,
    "Al_Ti": 0.04,
}

_DEPLETION_RATE = {
    "Fe":  0.00055,
    "Si":  0.00070,
    "Al":  0.00045,
    "Ti":  0.00030,
}


@dataclass
class PlantState:
    T_bulk: float = 1580.0           # °C  bulk melt temperature
    I_cell: float = 100.0            # A   electrolysis current
    V_cell: float = 3.2              # V   terminal voltage
    electrode_health: float = 1.0   # 0–1, 1.0 = new
    heater_power: float = 5000.0    # W
    composition: dict = field(
        default_factory=lambda: {
            "Fe": 0.20, "Si": 0.30, "Al": 0.22, "Ti": 0.15, "Other": 0.13
        }
    )
    faradaic_efficiency: float = 0.80
    uptime_s: float = 0.0
    fault_active: str | None = None   # e.g. "anode_effect"

    # O₂ production - the primary mission output (anode side).
    # Computed from Faraday's law: 2O²⁻ → O₂ + 4e⁻
    O2_produced_mol: float = 0.0     # mol, cumulative

    # Internal bath oxide phase tracker - drives realistic composition drift.
    # Not surfaced in the dashboard; does not represent pure metal extraction.
    _bath_phase: str = field(default="Fe", repr=False)

    # EIS data for the last sweep (stored here so the UI can read it)
    eis_freq: list = field(default_factory=list)
    eis_Z_re: list = field(default_factory=list)
    eis_Z_im: list = field(default_factory=list)


class PlantSimulator:
    """Low-fidelity Euler-integrated MOE cell simulator.

    Thread-safe: all reads/writes go through ``_lock``.
    """

    # Nominal cell parameters
    R_BASE: float = 0.030   # Ω - nominal resistance at full health

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = PlantState()
        self._rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, dt: float, setpoints: dict) -> PlantState:
        """Advance simulation by *dt* seconds. Returns a snapshot copy."""
        with self._lock:
            s = self._state
            rng = self._rng

            # Apply setpoints (clamped)
            s.heater_power = float(np.clip(setpoints.get("heater_power", s.heater_power), 0, 10_000))
            I_sp = float(np.clip(setpoints.get("I_cell_setpoint", s.I_cell), 10, 200))

            # During fault the current controller fights the fault injector
            if s.fault_active == "anode_effect":
                # Recovery ramp: drive current toward zero quickly
                s.I_cell = max(10.0, s.I_cell - 20.0 * dt)
            else:
                # Slew-rate-limited current tracking (20 A/s)
                delta = float(np.clip(I_sp - s.I_cell, -20 * dt, 20 * dt))
                s.I_cell = float(np.clip(s.I_cell + delta, 10, 200))

            # Compute cell resistance from health
            R_cell = self.R_BASE / max(s.electrode_health, 0.05)

            # Temperature dynamics: heater in, joule heating, radiative loss, noise
            dT = (
                s.heater_power * 0.002
                + s.I_cell ** 2 * R_cell * 0.001
                - 0.001 * (s.T_bulk - 20.0)
            ) * dt + rng.normal(0, 0.3)
            s.T_bulk = float(np.clip(s.T_bulk + dT, 800.0, 2200.0))

            # Electrode health degrades faster at higher current
            d_health = -0.0002 * (s.I_cell / 100.0) * dt
            s.electrode_health = float(np.clip(s.electrode_health + d_health, 0.01, 1.0))

            # Terminal voltage: V = I * R + noise
            V_nominal = s.I_cell * R_cell + rng.normal(0, 0.05)
            if s.fault_active == "anode_effect":
                # Voltage spike during anode effect
                multiplier = getattr(self, "_fault_multiplier", 3.5)
                s.V_cell = float(V_nominal * multiplier + rng.normal(0, 0.3))
            else:
                s.V_cell = float(np.clip(V_nominal, 0.5, 20.0))

            # Faradaic efficiency drifts with health + noise
            s.faradaic_efficiency = float(
                np.clip(0.8 * s.electrode_health + 0.1 * rng.random(), 0.1, 0.98)
            )

            # O₂ production at anode - Faraday's law: 2O²⁻ → O₂ + 4e⁻
            # n(O₂) = I · η · dt / (4 · F)
            dO2 = s.I_cell * s.faradaic_efficiency * dt / (4.0 * _FARADAY)
            s.O2_produced_mol += dO2

            # Bath oxide composition drift - oxides are preferentially reduced
            # in order of decomposition potential (Fe₂O₃ first, then SiO₂, etc.).
            # The cathode output is always a mixed alloy; this tracks what's
            # left in the bath, not what's been purely extracted.
            comp = dict(s.composition)
            phase = s._bath_phase
            I_norm = s.I_cell / 150.0

            if phase == "Fe":
                rate = _DEPLETION_RATE["Fe"] * I_norm * s.faradaic_efficiency
                comp["Fe"] = max(0.005, comp["Fe"] - rate * dt + rng.normal(0, 0.0002) * dt)
                if comp["Fe"] < DEPLETION_THRESHOLDS["Fe"]:
                    s._bath_phase = "Si"
            elif phase == "Si":
                rate = _DEPLETION_RATE["Si"] * I_norm * s.faradaic_efficiency
                comp["Si"] = max(0.005, comp["Si"] - rate * dt + rng.normal(0, 0.0002) * dt)
                if comp["Si"] < DEPLETION_THRESHOLDS["Si"]:
                    s._bath_phase = "Al_Ti"
            elif phase == "Al_Ti":
                rate_al = _DEPLETION_RATE["Al"] * I_norm * s.faradaic_efficiency
                rate_ti = _DEPLETION_RATE["Ti"] * I_norm * s.faradaic_efficiency
                comp["Al"] = max(0.005, comp["Al"] - rate_al * dt + rng.normal(0, 0.0001) * dt)
                comp["Ti"] = max(0.005, comp["Ti"] - rate_ti * dt + rng.normal(0, 0.0001) * dt)
                if comp["Al"] < DEPLETION_THRESHOLDS["Al_Ti"] and comp["Ti"] < DEPLETION_THRESHOLDS["Al_Ti"]:
                    s._bath_phase = "complete"
            else:
                for k in comp:
                    comp[k] = max(0.005, comp[k] + rng.normal(0, 0.0002) * dt)

            for k in ("Other",):
                comp[k] = max(0.005, comp[k] + rng.normal(0, 0.0001) * dt)
            total = sum(comp.values())
            s.composition = {k: v / total for k, v in comp.items()}

            s.uptime_s += dt
            return copy.deepcopy(s)

    def get_state(self) -> PlantState:
        """Return a snapshot of the current state (no step)."""
        with self._lock:
            return copy.deepcopy(self._state)

    def apply_fault(self, fault_name: str, multiplier: float = 3.5) -> None:
        """Externally apply a fault (called by FaultInjector)."""
        with self._lock:
            self._state.fault_active = fault_name
            self._fault_multiplier = multiplier

    def clear_fault(self) -> None:
        with self._lock:
            self._state.fault_active = None
            self._fault_multiplier = 1.0

    def set_eis_data(self, freq: list, Z_re: list, Z_im: list) -> None:
        with self._lock:
            self._state.eis_freq = freq
            self._state.eis_Z_re = Z_re
            self._state.eis_Z_im = Z_im
