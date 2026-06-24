"""Global operational state machine for the MOE controller.

States (v1 subset):
  IDLE → HEATING → RUN_NOMINAL ⇄ FAULT_RECOVERY → SAFE_SHUTDOWN → IDLE
"""

from __future__ import annotations

import threading
import time

import structlog
from transitions import Machine

log = structlog.get_logger(__name__)


class ModeManager:
    """Wraps a ``transitions`` state machine. Thread-safe property access."""

    STATES = ["IDLE", "HEATING", "RUN_NOMINAL", "FAULT_RECOVERY", "SAFE_SHUTDOWN"]

    TRANSITIONS = [
        {"trigger": "start",             "source": "IDLE",           "dest": "HEATING"},
        {"trigger": "target_reached",    "source": "HEATING",        "dest": "RUN_NOMINAL"},
        {"trigger": "fault_detected",    "source": "RUN_NOMINAL",    "dest": "FAULT_RECOVERY"},
        {"trigger": "recovery_complete", "source": "FAULT_RECOVERY", "dest": "RUN_NOMINAL"},
        {"trigger": "shutdown",          "source": "RUN_NOMINAL",    "dest": "SAFE_SHUTDOWN"},
        {"trigger": "shutdown_complete", "source": "SAFE_SHUTDOWN",  "dest": "IDLE"},
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._machine = Machine(
            model=self,
            states=self.STATES,
            transitions=self.TRANSITIONS,
            initial="IDLE",
            ignore_invalid_triggers=True,  # silently drop triggers not valid in this state
            after_state_change="_on_state_change",
        )

    # transitions sets self.state automatically
    def _on_state_change(self) -> None:
        log.info("mode_transition", new_mode=self.state, ts=time.time())

    @property
    def mode(self) -> str:
        with self._lock:
            return str(self.state)

    # The trigger methods (start, target_reached, etc.) are injected by
    # Machine into *self* - we just need to call them safely.
    def safe_trigger(self, trigger: str) -> None:
        """Call a trigger from any thread."""
        with self._lock:
            trigger_fn = getattr(self, trigger, None)
            if trigger_fn is not None:
                trigger_fn()
