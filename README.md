# moe-controller-testbed

Autonomous controller for molten oxide electrolysis (MOE) — live simulation with fault detection, EIS-based electrode health monitoring, and sequential recipe execution.

## What it does

Runs a simulated MOE cell and an autonomous brain in the same process. The brain monitors electrode health via EIS, manages temperature with a PID loop, adapts current to electrode degradation, detects and recovers from anode-effect faults, and executes a multi-phase voltage recipe across a simulated regolith batch — tracking O₂ production (anode) and bath oxide evolution throughout. No operator decisions after start.

## Quick start

```bash
# Python 3.11+ required
python3.11 -m pip install -e ".[dev]"
python3.11 app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Dashboard

Five panels update every second:

| Panel | What it shows |
|-------|--------------|
| Plant state | T_bulk (°C), I_cell (A), V_cell (V), electrode health — rolling 60 s time series |
| EIS — Electrode health | Nyquist plot from the last synthetic EIS sweep; health gauge (0–100%) |
| OES Composition | Bath oxide fractions estimated from mock OES |
| O₂ Production | Sequential recipe phase (Fe₂O₃ → SiO₂ → Al₂O₃+TiO₂), cumulative O₂ (g), rate (g/hr), Faradaic efficiency, bath oxide depletion chart |
| Fault injection | Inject an anode-effect fault (severity 1–7); watch mode badge flip RUN_NOMINAL → FAULT_RECOVERY → RUN_NOMINAL |

## Code layout

```
app.py                  Dash dashboard entry point
testbed/
  plant.py              MOE cell simulator (Euler integration, thread-safe)
  scheduler.py          Background sim loop — plant → inference → control → fault detection
  controllers.py        Heater PID + electrode-degradation-aware current controller
  inference.py          Synthetic EIS sweep, Randles fit, health estimate, OES mock
  faults.py             Anode-effect injector + detector
  mode_manager.py       State machine: IDLE → HEATING → RUN_NOMINAL ⇄ FAULT_RECOVERY
  interfaces.py         ABCs for Inference, Control, FaultDetector
  logging_setup.py      Structured JSON event log (structlog)
tests/
  test_plant.py
  test_mode_manager.py
  test_fault_trip.py
  test_extraction_phases.py
```

## Tests

```bash
python3.11 -m pytest tests/ -v
```

17 tests, all passing.

## Context

This is M0.2 in the [autonomous MOE controller project](https://github.com/mingw). The thesis: nobody has built the autonomous brain that manages a full MOE cell — electrode health, process control, fault recovery, and recipe sequencing — without an operator. This testbed is the first simulation of that brain running end-to-end.

Chemistry note: MOE cathode output is a mixed Fe/Si/Al/Ti alloy, not sequential pure metals. The sequential recipe controls which oxides are preferentially reduced at each voltage step. The anode output — O₂ — is high-purity and mission-critical for life support and propellant on Mars.
