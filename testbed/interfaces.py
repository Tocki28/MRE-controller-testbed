"""Abstract base classes for plug-in subsystems.

Every concrete algorithm (EIS inference, OES mock, PID, fault detector)
implements one of these interfaces so the harness can swap them out
without touching the rest of the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from testbed.plant import PlantState


class InferenceModule(ABC):
    @abstractmethod
    def update(self, state: PlantState) -> dict:
        """Return inferred quantities:
        - electrode_health_est  (0–1)
        - composition_est       (dict, same keys as PlantState.composition)
        - predicted_ttf_hrs     (float, hours to failure)
        - eis_Z_re, eis_Z_im    (lists of floats for Nyquist plot)
        - eis_freq              (list of floats)
        """


class ControlModule(ABC):
    @abstractmethod
    def compute_setpoints(
        self, state: PlantState, inferred: dict, mode: str
    ) -> dict:
        """Return setpoints dict with keys:
        - heater_power      (W, 0–10 000)
        - I_cell_setpoint   (A, 10–200)
        """


class FaultDetector(ABC):
    @abstractmethod
    def detect(
        self, state: PlantState, history: list[PlantState]
    ) -> str | None:
        """Return a fault name string if a fault is detected, else None."""
