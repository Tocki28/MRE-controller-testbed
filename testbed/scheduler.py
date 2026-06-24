"""Background simulation loop.

SimLoop runs in a daemon thread, advancing the plant simulation every
real-second (or sub-second wall-clock tick mapped to sim dt=1 s).
The Streamlit frontend reads state via get_snapshot() which is protected
by a threading.Lock.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import structlog

from testbed.controllers import CompositeController
from testbed.faults import AnodeEffectDetector, FaultInjector
from testbed.inference import MOEInference
from testbed.logging_setup import configure_logging
from testbed.mode_manager import ModeManager
from testbed.plant import PlantSimulator, PlantState

log = structlog.get_logger(__name__)

# How many seconds to hold in FAULT_RECOVERY before returning to nominal
RECOVERY_HOLD_S = 10.0

# Wall-clock sleep between ticks (controls dashboard refresh rate)
TICK_WALL_S = 0.2   # 5 ticks/s wall-clock = 1 sim-s per tick at 5x speed


class SimLoop:
    """Orchestrates plant, inference, control, fault detection in a background thread."""

    def __init__(self) -> None:
        configure_logging()

        self.plant = PlantSimulator()
        self.inference = MOEInference()
        self.controller = CompositeController()
        self.fault_detector = AnodeEffectDetector()
        self.mode_manager = ModeManager()
        self.fault_injector = FaultInjector(self.plant)

        self._history: deque[PlantState] = deque(maxlen=300)
        self._snapshot: dict[str, Any] = {}
        self._lock = threading.Lock()

        self._fault_recovery_start: float | None = None
        self._started = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._loop, daemon=True, name="SimLoop")
        self._thread = t
        t.start()
        log.info("sim_loop_started")

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        # Auto-start: IDLE → HEATING
        self.mode_manager.safe_trigger("start")
        log.info("mode_auto_start", mode=self.mode_manager.mode)

        setpoints: dict = {"heater_power": 5000.0, "I_cell_setpoint": 10.0}

        while True:
            tick_start = time.monotonic()
            mode = self.mode_manager.mode

            # --- 1. Plant step (sim dt = 1 s per tick) ---------------
            state = self.plant.step(dt=1.0, setpoints=setpoints)
            self._history.append(state)
            history_list = list(self._history)

            # --- 2. Auto-transition HEATING → RUN_NOMINAL ------------
            if mode == "HEATING" and state.T_bulk > 1560.0:
                self.mode_manager.safe_trigger("target_reached")
                log.info("target_temperature_reached", T_bulk=round(state.T_bulk, 1))
                mode = self.mode_manager.mode

            # --- 3. Inference ----------------------------------------
            inferred = self.inference.update(state)

            # --- 4. Fault detection -----------------------------------
            if mode == "RUN_NOMINAL":
                fault = self.fault_detector.detect(state, history_list)
                if fault:
                    self.mode_manager.safe_trigger("fault_detected")
                    self._fault_recovery_start = state.uptime_s
                    log.error("fault_detected", fault=fault, V_cell=round(state.V_cell, 2))
                    mode = self.mode_manager.mode

            # --- 5. Recovery timer -----------------------------------
            if mode == "FAULT_RECOVERY" and self._fault_recovery_start is not None:
                elapsed = state.uptime_s - self._fault_recovery_start
                if elapsed >= RECOVERY_HOLD_S:
                    self.fault_injector.clear()
                    self.mode_manager.safe_trigger("recovery_complete")
                    self._fault_recovery_start = None
                    log.info("recovery_complete", uptime_s=round(state.uptime_s, 1))
                    mode = self.mode_manager.mode

            # --- 6. Control setpoints --------------------------------
            setpoints = self.controller.compute_setpoints(state, inferred, mode)

            # --- 7. Store snapshot -----------------------------------
            with self._lock:
                self._snapshot = {
                    "state": state,
                    "inferred": inferred,
                    "mode": mode,
                    "history": history_list[-60:],
                }

            # --- Wall-clock pacing -----------------------------------
            elapsed_wall = time.monotonic() - tick_start
            sleep_time = max(0.0, TICK_WALL_S - elapsed_wall)
            time.sleep(sleep_time)
