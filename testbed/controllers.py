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
BASE_CURRENT = 150.0  # A nominal target current


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


class AdaptiveCurrent:
    """Scale I_cell setpoint down as electrode health falls.

    I_cell_sp = PHASE_BASE * (0.8 * health + 0.2)
    Phase-aware: each extraction target has a different base current reflecting
    the decomposition voltage needed to reduce that oxide.
    """

    I_MIN = 10.0
    I_MAX = 200.0

    def compute(self, electrode_health_est: float) -> float:
        sp = BASE_CURRENT * (0.8 * electrode_health_est + 0.2)
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
            i_sp = self._adaptive.compute(health_est)
            return {"heater_power": hp, "I_cell_setpoint": i_sp}

        if mode == "FAULT_RECOVERY":
            # Kill current during recovery; keep heater ticking over
            hp = self._pid.compute(state.T_bulk, self._dt)
            return {"heater_power": hp * 0.5, "I_cell_setpoint": 10.0}

        # SAFE_SHUTDOWN or unknown: everything off
        return {"heater_power": 0.0, "I_cell_setpoint": 10.0}
