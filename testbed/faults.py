"""Fault injection and anode-effect detector.

FaultInjector: exposes inject()/clear() to the Streamlit UI.
AnodeEffectDetector: stateless detection on V_cell history.
ElectrodeDegradationDetector: EIS-inferred health-based detection.
"""

from __future__ import annotations

import threading

import structlog

from testbed.interfaces import FaultDetector
from testbed.plant import PlantSimulator, PlantState

log = structlog.get_logger(__name__)

# V_cell threshold that triggers anode-effect detection
ANODE_EFFECT_V_THRESH = 8.0   # V (roughly 3× nominal 2.5–3.5 V range)

# Severity 1–7 → V_cell multiplier 1.5–5.0
_SEV_TO_MULT = {i: 1.5 + (i - 1) * (5.0 - 1.5) / 6.0 for i in range(1, 8)}


class AnodeEffectDetector(FaultDetector):
    """Detect if V_cell > threshold for 2+ consecutive readings."""

    def detect(
        self, state: PlantState, history: list[PlantState], inferred: dict | None = None
    ) -> str | None:
        # Need at least 2 readings
        if len(history) < 2:
            return None
        last_two = history[-2:]
        if all(s.V_cell > ANODE_EFFECT_V_THRESH for s in last_two):
            return "anode_effect"
        return None


class ElectrodeDegradationDetector(FaultDetector):
    """Detects electrode degradation from EIS-inferred health estimate.

    Returns 'electrode_degradation' when electrode_health_est drops below
    threshold for CONSECUTIVE_REQUIRED readings. Uses EIS-inferred health
    (not ground-truth plant state) to validate the EIS inference pipeline.

    Requires inferred dict from MOEInference.update() passed as third arg.
    Returns None if inferred is not provided.
    """

    HEALTH_THRESHOLD = 0.6
    CONSECUTIVE_REQUIRED = 3

    def __init__(self) -> None:
        self._consecutive_count: int = 0

    def detect(
        self,
        state: PlantState,
        history: list[PlantState],
        inferred: dict | None = None,
    ) -> str | None:
        if inferred is None:
            return None
        health_est = float(inferred.get("electrode_health_est", 1.0))
        if health_est < self.HEALTH_THRESHOLD:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        if self._consecutive_count >= self.CONSECUTIVE_REQUIRED:
            return "electrode_degradation"
        return None


class FaultInjector:
    """Called from the Streamlit UI to inject or clear faults.

    Holds a reference to the PlantSimulator so it can apply fault dynamics.
    """

    def __init__(self, plant: PlantSimulator) -> None:
        self._plant = plant
        self._active: str | None = None
        self._lock = threading.Lock()

    def inject(self, fault_name: str, severity: int = 4) -> None:
        multiplier = _SEV_TO_MULT.get(int(severity), 3.5)
        with self._lock:
            self._active = fault_name
            self._plant.apply_fault(fault_name, multiplier=multiplier)
        log.warning(
            "fault_injected",
            fault=fault_name,
            severity=severity,
            multiplier=round(multiplier, 2),
        )

    def clear(self) -> None:
        with self._lock:
            self._active = None
            self._plant.clear_fault()
        log.info("fault_cleared")

    @property
    def active_fault(self) -> str | None:
        with self._lock:
            return self._active
