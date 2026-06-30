"""Heater PID + adaptive current controller.

HeaterPID: T_bulk → heater_power
AdaptiveCurrent: electrode_health_est → I_cell_setpoint
CompositeController: composes both.
"""

from __future__ import annotations

import numpy as np
import structlog

from testbed.interfaces import ControlModule
from testbed.plant import PlantState

log = structlog.get_logger(__name__)

T_SETPOINT = 1580.0   # °C
BASE_CURRENT = 150.0  # A nominal target current (fallback)

# Current setpoint per extraction phase.  Higher decomposition potential
# → higher current → higher cell voltage.  Health scaling is applied on top.
_PHASE_CURRENT: dict[str, float] = {
    "Fe":      80.0,   # Fe₂O₃  ~1.3 V decomposition
    "Si":     120.0,   # SiO₂   ~2.8 V
    "Al_Ti":  160.0,   # Al₂O₃/TiO₂  ~4.5 V
    "complete": 40.0,  # wind-down — bath exhausted
}


class HeaterPID:
    """Discrete-time PID on T_bulk → heater_power.

    Anti-windup: integral is clamped to keep output in range.
    """

    KP = 200.0
    KI = 5.0
    KD = 10.0
    OUT_MIN = 0.0
    OUT_MAX = 10_000.0

    def __init__(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, T_bulk: float, dt: float) -> float:
        error = T_SETPOINT - T_bulk
        self._integral = float(
            np.clip(self._integral + error * dt, -self.OUT_MAX, self.OUT_MAX)
        )
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        output = self.KP * error + self.KI * self._integral + self.KD * derivative
        return float(np.clip(output, self.OUT_MIN, self.OUT_MAX))


class TemperaturePID:
    """Discrete-time PID on T_bulk → heater_power with configurable setpoint.

    Identical gains to HeaterPID but accepts the setpoint at construction time,
    making it reusable across different target temperatures.

    Anti-windup: integral is clamped to [-OUT_MAX, OUT_MAX] before multiplication.
    """

    KP = 200.0
    KI = 5.0
    KD = 10.0
    OUT_MIN = 0.0
    OUT_MAX = 10_000.0

    def __init__(self, T_setpoint: float) -> None:
        self._T_setpoint = T_setpoint
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, T_bulk: float, dt: float) -> float:
        """Return heater_power in [OUT_MIN, OUT_MAX].

        Anti-windup: integral only accumulates when the pre-integration output
        would not be saturated, preventing integrator windup during large steps.
        The integral is also clamped to [-OUT_MAX, OUT_MAX] as a hard backstop.
        """
        error = self._T_setpoint - T_bulk
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error

        # Tentative output without integrating yet
        tentative = self.KP * error + self.KI * self._integral + self.KD * derivative
        # Only integrate when not saturated (conditional anti-windup)
        if self.OUT_MIN < tentative < self.OUT_MAX:
            self._integral = float(
                np.clip(self._integral + error * dt, -self.OUT_MAX, self.OUT_MAX)
            )

        output = self.KP * error + self.KI * self._integral + self.KD * derivative
        return float(np.clip(output, self.OUT_MIN, self.OUT_MAX))

    def reset(self) -> None:
        """Zero integral accumulator and previous error."""
        self._integral = 0.0
        self._prev_error = 0.0


class AdaptiveCurrent:
    """Scale I_cell setpoint down as electrode health falls.

    I_cell_sp = PHASE_BASE[phase] * (0.8 * health + 0.2)
    """

    I_MIN = 10.0
    I_MAX = 200.0

    def compute(self, electrode_health_est: float, bath_phase: str = "Fe") -> float:
        base = _PHASE_CURRENT.get(bath_phase, BASE_CURRENT)
        sp = base * (0.8 * electrode_health_est + 0.2)
        return float(np.clip(sp, self.I_MIN, self.I_MAX))


class CompositeController(ControlModule):
    """Runs HeaterPID and AdaptiveCurrent; gates outputs by operating mode."""

    def __init__(self) -> None:
        self._pid = HeaterPID()
        self._adaptive = AdaptiveCurrent()
        self._dt = 1.0  # nominal step

    def compute_setpoints(
        self, state: PlantState, inferred: dict, mode: str
    ) -> dict:
        health_est = float(inferred.get("electrode_health_est", state.electrode_health))

        if mode == "IDLE":
            return {"heater_power": 0.0, "I_cell_setpoint": 10.0}

        if mode == "HEATING":
            # Ramp heater hard; no electrolysis current until nominal
            hp = self._pid.compute(state.T_bulk, self._dt)
            return {"heater_power": hp, "I_cell_setpoint": 10.0}

        if mode == "RUN_NOMINAL":
            hp = self._pid.compute(state.T_bulk, self._dt)
            i_sp = self._adaptive.compute(health_est, state.bath_phase)
            return {"heater_power": hp, "I_cell_setpoint": i_sp}

        if mode == "FAULT_RECOVERY":
            # Kill current during recovery; keep heater ticking over
            hp = self._pid.compute(state.T_bulk, self._dt)
            return {"heater_power": hp * 0.5, "I_cell_setpoint": 10.0}

        if mode == "ELECTRODE_DEGRADING":
            health = inferred.get("electrode_health_est", state.electrode_health)
            I_sp = float(np.clip(self._adaptive.compute(health, state.bath_phase) * 0.5, 10, 200))
            hp = self._pid.compute(state.T_bulk, self._dt)
            return {"heater_power": hp, "I_cell_setpoint": I_sp}

        if mode == "ELECTRODE_SWAP":
            # Electrode being physically replaced: heater holds temperature, no current.
            hp = self._pid.compute(state.T_bulk, self._dt)
            return {"heater_power": hp, "I_cell_setpoint": 10.0}

        if mode == "BATH_DEPLETED":
            hp = self._pid.compute(state.T_bulk, self._dt)
            return {"heater_power": hp, "I_cell_setpoint": 10.0}

        if mode == "DRAINING":
            hp = self._pid.compute(state.T_bulk, self._dt)
            return {"heater_power": hp, "I_cell_setpoint": 10.0}

        if mode == "CLEANOUT":
            return {"heater_power": 0.0, "I_cell_setpoint": 10.0}

        # SAFE_SHUTDOWN or unknown: everything off
        return {"heater_power": 0.0, "I_cell_setpoint": 10.0}
