"""Unit tests for AnodeEffectDetector."""

from __future__ import annotations

from testbed.faults import AnodeEffectDetector
from testbed.plant import PlantState


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
