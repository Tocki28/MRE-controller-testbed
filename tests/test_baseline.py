"""Regression test: open-loop baseline must be reproducible within 1%.

M4.2 — Baseline open-loop simulation benchmarked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from testbed.plant import PlantSimulator

# Path to the committed baseline artefact
_BASELINE_PATH = Path(__file__).parent.parent / "baselines" / "open_loop_baseline.json"

# Tolerance: 1 % relative
_TOL = 0.01


def _run_simulation(dt: float, n_steps: int, setpoints: dict) -> dict:
    """Run the simulator and return the same metric dict as generate_baseline.py."""
    sim = PlantSimulator()

    total_energy_J = 0.0
    phase_transitions = 0
    prev_phase: str | None = None
    final_state = None

    for _ in range(n_steps):
        state = sim.step(dt, setpoints)
        total_energy_J += state.I_cell * state.V_cell * dt
        if prev_phase is not None and state.bath_phase != prev_phase:
            phase_transitions += 1
        prev_phase = state.bath_phase
        final_state = state

    return {
        "O2_produced_mol": final_state.O2_produced_mol,
        "electrode_health_final": final_state.electrode_health,
        "phase_transitions": phase_transitions,
        "total_energy_J": total_energy_J,
    }


def test_open_loop_baseline_reproducible() -> None:
    """Re-running the same 1000-step sim must match the stored baseline within 1%."""
    assert _BASELINE_PATH.exists(), f"Baseline file not found: {_BASELINE_PATH}"

    baseline = json.loads(_BASELINE_PATH.read_text())
    params = baseline["run_params"]
    expected = baseline["metrics"]

    setpoints = {
        "heater_power": params["heater_power_W"],
        "I_cell_setpoint": params["I_cell_setpoint_A"],
    }

    actual = _run_simulation(
        dt=params["dt_s"],
        n_steps=params["n_steps"],
        setpoints=setpoints,
    )

    # Float metrics: relative tolerance 1 %
    for key in ("O2_produced_mol", "electrode_health_final", "total_energy_J"):
        assert actual[key] == pytest.approx(expected[key], rel=_TOL), (
            f"{key}: got {actual[key]}, expected {expected[key]}"
        )

    # Integer metric: must match exactly
    assert actual["phase_transitions"] == expected["phase_transitions"], (
        f"phase_transitions: got {actual['phase_transitions']}, "
        f"expected {expected['phase_transitions']}"
    )
