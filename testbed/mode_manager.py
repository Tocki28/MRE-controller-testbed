"""Global operational state machine for the MOE controller.

States (v2 — full batch cycle, 10 states):
  IDLE → HEATING → RUN_NOMINAL → ELECTRODE_DEGRADING → DRAINING → CLEANOUT → IDLE
  ELECTRODE_DEGRADING → ELECTRODE_SWAP → RUN_NOMINAL   (industrial hot-swap path)
  RUN_NOMINAL → BATH_DEPLETED → DRAINING
  RUN_NOMINAL / ELECTRODE_DEGRADING / BATH_DEPLETED → FAULT_RECOVERY → RUN_NOMINAL
  FAULT_RECOVERY → ELECTRODE_DEGRADING              (recovery_complete_degraded)
  FAULT_RECOVERY → SAFE_SHUTDOWN                    (unrecoverable)
  SAFE_SHUTDOWN → IDLE
  HEATING → SAFE_SHUTDOWN                              (abort on timeout / thermocouple dropout)
"""

from __future__ import annotations

import threading
import time

import structlog
from transitions import Machine

log = structlog.get_logger(__name__)


class ModeManager:
    """Wraps a ``transitions`` state machine. Thread-safe property access."""

    STATES = [
        "IDLE", "HEATING", "RUN_NOMINAL",
        "ELECTRODE_DEGRADING", "ELECTRODE_SWAP", "BATH_DEPLETED",
        "DRAINING", "CLEANOUT",
        "FAULT_RECOVERY", "SAFE_SHUTDOWN",
    ]

    TRANSITIONS = [
        # Normal startup
        {"trigger": "start",               "source": "IDLE",                "dest": "HEATING"},
        {"trigger": "target_reached",      "source": "HEATING",             "dest": "RUN_NOMINAL"},
        {"trigger": "abort",               "source": "HEATING",             "dest": "SAFE_SHUTDOWN"},
        # Electrode degradation track
        {"trigger": "electrode_degrading", "source": "RUN_NOMINAL",         "dest": "ELECTRODE_DEGRADING"},
        # Hot-swap path (industrial only — requires hot-swap hardware)
        {"trigger": "hot_swap_command",    "source": "ELECTRODE_DEGRADING", "dest": "ELECTRODE_SWAP"},
        {"trigger": "swap_complete",       "source": "ELECTRODE_SWAP",      "dest": "RUN_NOMINAL"},
        # Bath depletion
        {"trigger": "bath_depleted",       "source": "RUN_NOMINAL",         "dest": "BATH_DEPLETED"},
        {"trigger": "bath_depleted",       "source": "ELECTRODE_DEGRADING", "dest": "BATH_DEPLETED"},
        # Drain from any productive state
        {"trigger": "drain_command",       "source": "RUN_NOMINAL",         "dest": "DRAINING"},
        {"trigger": "drain_command",       "source": "ELECTRODE_DEGRADING", "dest": "DRAINING"},
        {"trigger": "drain_command",       "source": "ELECTRODE_SWAP",      "dest": "DRAINING"},
        {"trigger": "drain_command",       "source": "BATH_DEPLETED",       "dest": "DRAINING"},
        # End of batch
        {"trigger": "drain_complete",      "source": "DRAINING",            "dest": "CLEANOUT"},
        {"trigger": "cleanout_complete",   "source": "CLEANOUT",            "dest": "IDLE"},
        # Fault handling — detectable from any electrically active state
        {"trigger": "fault_detected",      "source": "RUN_NOMINAL",         "dest": "FAULT_RECOVERY"},
        {"trigger": "fault_detected",      "source": "ELECTRODE_DEGRADING", "dest": "FAULT_RECOVERY"},
        {"trigger": "fault_detected",      "source": "BATH_DEPLETED",       "dest": "FAULT_RECOVERY"},
        {"trigger": "recovery_complete",          "source": "FAULT_RECOVERY",      "dest": "RUN_NOMINAL"},
        {"trigger": "recovery_complete_degraded", "source": "FAULT_RECOVERY",      "dest": "ELECTRODE_DEGRADING"},
        {"trigger": "unrecoverable",             "source": "FAULT_RECOVERY",      "dest": "SAFE_SHUTDOWN"},
        # Shutdown paths
        {"trigger": "shutdown",            "source": "RUN_NOMINAL",         "dest": "SAFE_SHUTDOWN"},
        {"trigger": "shutdown_complete",   "source": "SAFE_SHUTDOWN",       "dest": "IDLE"},
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._machine = Machine(
            model=self,
            states=self.STATES,
            transitions=self.TRANSITIONS,
            initial="IDLE",
            ignore_invalid_triggers=True,
            after_state_change="_on_state_change",
        )

    def _on_state_change(self) -> None:
        log.info("mode_transition", new_mode=self.state, ts=time.time())

    @property
    def mode(self) -> str:
        with self._lock:
            return str(self.state)

    def safe_trigger(self, trigger: str) -> None:
        """Call a trigger from any thread."""
        with self._lock:
            trigger_fn = getattr(self, trigger, None)
            if trigger_fn is not None:
                trigger_fn()
