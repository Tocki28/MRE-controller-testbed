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
from testbed.faults import AnodeEffectDetector, ElectrodeDegradationDetector, FaultInjector
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
        self.electrode_degradation_detector = ElectrodeDegradationDetector()
        self.mode_manager = ModeManager()
        self.fault_injector = FaultInjector(self.plant)

        self._history: deque[PlantState] = deque(maxlen=300)
        self._snapshot: dict[str, Any] = {}
        self._lock = threading.Lock()

        self._fault_recovery_start: float | None = None
        self._drain_start: float | None = None
        self._cleanout_start: float | None = None
        self._bath_depleted_start: float | None = None
        self._recovery_outcomes: deque[bool] = deque(maxlen=5)
        self._pre_fault_mode: str = "RUN_NOMINAL"
        self._last_bath_phase: str = "Fe"
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

            # --- 3a. Electrode degradation ---------------------------
            if mode == "RUN_NOMINAL" and inferred.get("electrode_health_est", 1.0) < 0.6:
                self.mode_manager.safe_trigger("electrode_degrading")
                mode = self.mode_manager.mode

            # --- 3b. Bath phase transition logging -------------------
            if state.bath_phase != self._last_bath_phase:
                log.info(
                    "recipe_phase_transition",
                    from_phase=self._last_bath_phase,
                    to_phase=state.bath_phase,
                    uptime_s=round(state.uptime_s, 1),
                )
                self._last_bath_phase = state.bath_phase

            # --- 3b. Bath depletion ----------------------------------
            if mode in ("RUN_NOMINAL", "ELECTRODE_DEGRADING") and state.bath_phase == "complete":
                self.mode_manager.safe_trigger("bath_depleted")
                mode = self.mode_manager.mode

            # --- 3c. Critical degradation forces drain ---------------
            if mode == "ELECTRODE_DEGRADING" and inferred.get("electrode_health_est", 1.0) < 0.3:
                self.mode_manager.safe_trigger("drain_command")
                mode = self.mode_manager.mode

            # --- 4. Fault detection -----------------------------------
            if mode in ("RUN_NOMINAL", "ELECTRODE_DEGRADING", "BATH_DEPLETED"):
                fault = self.fault_detector.detect(state, history_list, inferred)
                if fault is None and mode == "ELECTRODE_DEGRADING":
                    fault = self.electrode_degradation_detector.detect(state, history_list, inferred)
                if fault:
                    self._pre_fault_mode = mode
                    self.mode_manager.safe_trigger("fault_detected")
                    self._fault_recovery_start = state.uptime_s
                    log.error("fault_detected", fault=fault, mode=mode, V_cell=round(state.V_cell, 2))
                    mode = self.mode_manager.mode

            # --- 5. Recovery timer -----------------------------------
            if mode == "FAULT_RECOVERY" and self._fault_recovery_start is not None:
                elapsed = state.uptime_s - self._fault_recovery_start
                if elapsed >= RECOVERY_HOLD_S:
                    self.fault_injector.clear()
                    self._fault_recovery_start = None
                    failed_in_window = sum(self._recovery_outcomes)
                    if failed_in_window >= 3:
                        # Too many recent failures — declare unrecoverable
                        self._recovery_outcomes.append(True)
                        self.mode_manager.safe_trigger("unrecoverable")
                        log.error("unrecoverable_fault", uptime_s=round(state.uptime_s, 1), recent_failures=failed_in_window)
                    else:
                        # Recovery succeeds
                        self._recovery_outcomes.append(False)
                        if self._pre_fault_mode == "ELECTRODE_DEGRADING":
                            self.mode_manager.safe_trigger("recovery_complete_degraded")
                        else:
                            self.mode_manager.safe_trigger("recovery_complete")
                        log.info("recovery_complete", uptime_s=round(state.uptime_s, 1), pre_fault_mode=self._pre_fault_mode)
                        self._pre_fault_mode = "RUN_NOMINAL"
                    mode = self.mode_manager.mode

            # --- 5a. Draining timed exit (60 sim-seconds) ------------
            if mode == "DRAINING":
                if self._drain_start is None:
                    self._drain_start = state.uptime_s
                elif state.uptime_s - self._drain_start >= 60.0:
                    self._drain_start = None
                    self.mode_manager.safe_trigger("drain_complete")
                    mode = self.mode_manager.mode
            else:
                self._drain_start = None

            # --- 5b. Cleanout timed exit (30 sim-seconds) ------------
            if mode == "CLEANOUT":
                if self._cleanout_start is None:
                    self._cleanout_start = state.uptime_s
                elif state.uptime_s - self._cleanout_start >= 30.0:
                    self._cleanout_start = None
                    self.plant.reset_for_new_batch()
                    self.inference.reset_for_new_batch()
                    self.mode_manager.safe_trigger("cleanout_complete")  # → IDLE
                    self.mode_manager.safe_trigger("start")              # IDLE → HEATING
                    log.info("new_batch_started", uptime_s=round(state.uptime_s, 1))
                    mode = self.mode_manager.mode
            else:
                self._cleanout_start = None

            # --- 5c. Bath depleted auto-drain (30 sim-seconds hold) --
            # No feedstock replenishment in simulation: drain after hold.
            if mode == "BATH_DEPLETED":
                if self._bath_depleted_start is None:
                    self._bath_depleted_start = state.uptime_s
                elif state.uptime_s - self._bath_depleted_start >= 30.0:
                    self._bath_depleted_start = None
                    self.mode_manager.safe_trigger("drain_command")
                    log.info("bath_depleted_auto_drain", uptime_s=round(state.uptime_s, 1))
                    mode = self.mode_manager.mode
            else:
                self._bath_depleted_start = None

            # --- 6. Control setpoints --------------------------------
            setpoints = self.controller.compute_setpoints(state, inferred, mode)

            # --- 7. Store snapshot -----------------------------------
            with self._lock:
                self._snapshot = {
                    "state": state,
                    "inferred": inferred,
                    "mode": mode,
                    "history": history_list[-60:],
                    "bath_phase": state.bath_phase,
                }

            # --- Wall-clock pacing -----------------------------------
            elapsed_wall = time.monotonic() - tick_start
            sleep_time = max(0.0, TICK_WALL_S - elapsed_wall)
            time.sleep(sleep_time)
