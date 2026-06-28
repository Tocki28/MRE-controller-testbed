"""Generate open-loop baseline metrics for M4.2.

Run from the testbed root:
    python3.11 baselines/generate_baseline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the package is importable when run as a script from any cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.plant import PlantSimulator

# ── Run parameters ────────────────────────────────────────────────────────────
DT_S = 1.0
N_STEPS = 1000
HEATER_POWER_W = 5000.0
I_CELL_SETPOINT_A = 100.0
PLANT_SEED = 42  # baked into PlantSimulator.__init__

SETPOINTS = {
    "heater_power": HEATER_POWER_W,
    "I_cell_setpoint": I_CELL_SETPOINT_A,
}


def run_simulation() -> dict:
    """Run 1000-step open-loop sim and return metrics dict."""
    sim = PlantSimulator()

    total_energy_J = 0.0
    phase_transitions = 0
    prev_phase: str | None = None
    final_state = None

    for _ in range(N_STEPS):
        state = sim.step(DT_S, SETPOINTS)

        # Accumulate energy: P = I * V, E = P * dt
        total_energy_J += state.I_cell * state.V_cell * DT_S

        # Count bath phase transitions
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


def main() -> None:
    metrics = run_simulation()

    baseline = {
        "run_params": {
            "dt_s": DT_S,
            "n_steps": N_STEPS,
            "heater_power_W": HEATER_POWER_W,
            "I_cell_setpoint_A": I_CELL_SETPOINT_A,
            "plant_seed": PLANT_SEED,
        },
        "metrics": metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(__file__).parent / "open_loop_baseline.json"
    out_path.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline written to {out_path}")
    print(f"  O2_produced_mol      : {metrics['O2_produced_mol']:.6f}")
    print(f"  electrode_health_final: {metrics['electrode_health_final']:.6f}")
    print(f"  phase_transitions     : {metrics['phase_transitions']}")
    print(f"  total_energy_J        : {metrics['total_energy_J']:.2f}")


if __name__ == "__main__":
    main()
