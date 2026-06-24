"""Unit tests for PlantSimulator dynamics."""

from __future__ import annotations

import pytest

from testbed.plant import PlantSimulator


NOMINAL_SETPOINTS = {"heater_power": 5000.0, "I_cell_setpoint": 100.0}


def test_plant_step_advances_time() -> None:
    """A single step of dt=1 should set uptime_s to 1.0."""
    sim = PlantSimulator()
    state = sim.step(1.0, NOMINAL_SETPOINTS)
    assert state.uptime_s == pytest.approx(1.0)


def test_temperature_responds_to_heater() -> None:
    """High heater power should raise T_bulk over 10 steps.

    We start at 20°C where loss = 0.8*(20-20) = 0 and heater input = 500 W-eq/s,
    so net dT is clearly positive regardless of noise.
    """
    sim = PlantSimulator()
    sim._state.T_bulk = 20.0   # room temp - zero radiative loss
    setpoints = {"heater_power": 10_000.0, "I_cell_setpoint": 10.0}
    initial_T = sim._state.T_bulk
    for _ in range(10):
        state = sim.step(1.0, setpoints)
    assert state.T_bulk > initial_T


def test_health_degrades_with_current() -> None:
    """Running at I_cell=150 A should degrade electrode_health over 100 steps."""
    sim = PlantSimulator()
    initial_health = sim._state.electrode_health
    setpoints = {"heater_power": 5000.0, "I_cell_setpoint": 150.0}
    for _ in range(100):
        state = sim.step(1.0, setpoints)
    # Health must have decreased
    assert state.electrode_health < initial_health
