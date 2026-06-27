"""Unit tests for O₂ production physics and internal bath oxide depletion."""

from __future__ import annotations

import pytest

from testbed.plant import DEPLETION_THRESHOLDS, PlantSimulator

NOMINAL_SETPOINTS = {"heater_power": 5000.0, "I_cell_setpoint": 150.0}

_FARADAY = 96_485.0


def test_o2_production_increases_with_current() -> None:
    """O₂ produced must increase each step when current flows."""
    sim = PlantSimulator()
    state = sim.step(1.0, NOMINAL_SETPOINTS)
    assert state.O2_produced_mol > 0.0


def test_o2_follows_faradays_law() -> None:
    """After 100 steps at 150 A / 80% efficiency, O₂ should match Faraday's law."""
    sim = PlantSimulator()
    # Override faradaic efficiency to a fixed value for a clean check
    with sim._lock:
        sim._state.faradaic_efficiency = 0.80
    for _ in range(100):
        state = sim.step(1.0, {"heater_power": 5000.0, "I_cell_setpoint": 150.0})
    # Expected: ∑ I·η·dt / (4F)  ≈ 150 * 0.80 * 100 / (4 * 96485)
    # η actually drifts so just check it's in the right order of magnitude
    expected_approx = 150 * 0.80 * 100 / (4 * _FARADAY)
    assert state.O2_produced_mol == pytest.approx(expected_approx, rel=0.3)


def test_fe_oxide_depletes_with_current() -> None:
    """Running at 150 A reduces Fe fraction in the bath over time."""
    sim = PlantSimulator()
    initial_fe = sim.get_state().composition["Fe"]
    for _ in range(100):
        state = sim.step(1.0, NOMINAL_SETPOINTS)
    assert state.composition["Fe"] < initial_fe


def test_bath_phase_transitions_fe_to_si() -> None:
    """Internal bath phase transitions Fe → Si when Fe oxide drops below threshold."""
    sim = PlantSimulator()
    with sim._lock:
        sim._state.composition["Fe"] = DEPLETION_THRESHOLDS["Fe"] - 0.01
    state = sim.step(1.0, NOMINAL_SETPOINTS)
    assert state.bath_phase == "Si"


def test_bath_phase_transitions_si_to_al_ti() -> None:
    """Internal bath phase transitions Si → Al_Ti when Si oxide drops below threshold."""
    sim = PlantSimulator()
    sim._bath_phase = "Si"
    with sim._lock:
        sim._state.composition["Si"] = DEPLETION_THRESHOLDS["Si"] - 0.01
    state = sim.step(1.0, NOMINAL_SETPOINTS)
    assert state.bath_phase == "Al_Ti"


def test_bath_phase_transitions_al_ti_to_complete() -> None:
    """Internal bath phase completes when both Al and Ti oxides are depleted."""
    sim = PlantSimulator()
    thresh = DEPLETION_THRESHOLDS["Al_Ti"]
    sim._bath_phase = "Al_Ti"
    with sim._lock:
        sim._state.composition["Al"] = thresh - 0.01
        sim._state.composition["Ti"] = thresh - 0.01
    state = sim.step(1.0, NOMINAL_SETPOINTS)
    assert state.bath_phase == "complete"


def test_composition_sums_to_one() -> None:
    """Composition normalises to 1.0 after any depletion step."""
    sim = PlantSimulator()
    for _ in range(50):
        state = sim.step(1.0, NOMINAL_SETPOINTS)
    assert abs(sum(state.composition.values()) - 1.0) < 1e-6
