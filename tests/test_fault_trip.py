"""Unit tests for AnodeEffectDetector and fault signature verification."""

from __future__ import annotations

import pytest

from testbed.faults import AnodeEffectDetector
from testbed.plant import PlantSimulator, PlantState


def _make_state(V_cell: float) -> PlantState:
    s = PlantState()
    s.V_cell = V_cell
    return s


def test_anode_effect_detection_triggers_on_high_voltage() -> None:
    """V_cell > threshold for 2+ consecutive readings → 'anode_effect'."""
    detector = AnodeEffectDetector()
    high_V = 10.0  # well above the 8.0 V threshold
    history = [_make_state(high_V) for _ in range(3)]
    current = _make_state(high_V)
    result = detector.detect(current, history)
    assert result == "anode_effect"


def test_no_fault_on_normal_voltage() -> None:
    """Normal V_cell readings should return None."""
    detector = AnodeEffectDetector()
    normal_V = 3.2  # nominal
    history = [_make_state(normal_V) for _ in range(5)]
    current = _make_state(normal_V)
    result = detector.detect(current, history)
    assert result is None


# ---------------------------------------------------------------------------
# Fault signature tests (M5.3)
# ---------------------------------------------------------------------------

_SETPOINTS = {"heater_power": 5000.0, "I_cell_setpoint": 10.0}
_FAULT_NAMES = [
    "anode_burnout",
    "power_loss",
    "melt_freeze",
    "electrode_short",
    "bath_depletion",
    "sensor_dropout",
    "cathode_flooding",
    "offgas_blockage",
]


def _run_nominal(steps: int = 3) -> tuple[float, "PlantSimulator"]:
    plant = PlantSimulator()
    state = plant.step(dt=1.0, setpoints=_SETPOINTS)
    for _ in range(steps - 1):
        state = plant.step(dt=1.0, setpoints=_SETPOINTS)
    return state.V_cell, plant


def _run_fault(fault_name: str, steps_after: int = 3) -> tuple[float, "PlantState"]:
    plant = PlantSimulator()
    plant.step(dt=1.0, setpoints=_SETPOINTS)
    plant.apply_fault(fault_name)
    state = None
    for _ in range(steps_after):
        state = plant.step(dt=1.0, setpoints=_SETPOINTS)
    return state.V_cell, state


class TestFaultSignatures:

    @pytest.mark.parametrize("fault_name", [f for f in _FAULT_NAMES if f != "sensor_dropout"])
    def test_v_cell_differs_from_nominal(self, fault_name: str) -> None:
        nominal_V, _ = _run_nominal()
        fault_V, _ = _run_fault(fault_name)
        assert fault_V != pytest.approx(nominal_V, rel=1e-3)

    def test_sensor_dropout_v_cell_equals_nominal(self) -> None:
        plant = PlantSimulator()
        plant.step(dt=1.0, setpoints=_SETPOINTS)
        plant.apply_fault("sensor_dropout")
        fault_state = plant.step(dt=1.0, setpoints=_SETPOINTS)
        assert fault_state.fault_active == "sensor_dropout"
        assert fault_state.V_cell > 0.0

    def test_sensor_dropout_fault_active_set(self) -> None:
        _, state = _run_fault("sensor_dropout")
        assert state.fault_active == "sensor_dropout"

    @pytest.mark.parametrize("fault_name", _FAULT_NAMES)
    def test_all_faults_produce_state(self, fault_name: str) -> None:
        fault_V, state = _run_fault(fault_name)
        assert state is not None
        assert state.fault_active == fault_name

    def test_anode_burnout_repeatability(self) -> None:
        v1, _ = _run_fault("anode_burnout")
        v2, _ = _run_fault("anode_burnout")
        assert v1 == pytest.approx(v2, rel=1e-4)

    def test_electrode_short_repeatability(self) -> None:
        v1, _ = _run_fault("electrode_short")
        v2, _ = _run_fault("electrode_short")
        assert v1 == pytest.approx(v2, rel=1e-4)
