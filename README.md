# Testbed - the integration platform every other piece plugs into

> The first build artifact. Not an algorithm; the platform algorithms run on. Demonstrates system-architect thinking AND industrial-integration thinking in one piece of code.

## Quick start

```bash
# Requires Python 3.11+
python3.11 -m pip install -e ".[dev]"
python3.11 app.py
# Open http://127.0.0.1:8050 in your browser
```

The dashboard opens. Watch the electrode health gauge decline as the anode degrades. The O₂ production panel shows real-time anode output (cumulative grams, production rate, Faradaic efficiency) and bath oxide depletion. Inject an anode effect fault and watch the mode badge flip RUN_NOMINAL → FAULT_RECOVERY → RUN_NOMINAL automatically.

To run tests:

```bash
python3.11 -m pytest tests/ -v
```

## Why this first

The hard problem in autonomous process control is not any individual algorithm - it is getting EIS, OES, temperature, current, and voltage to feed a single coherent state estimate, and having that estimate drive actuator commands in real time without an operator in the loop. A standalone EIS fit or OES peak-finder cannot demonstrate this. The testbed can.

It also serves as the integration bench. Every subsequent module (EIS pipeline, OES inference, fault detector, recovery state machine) is developed and validated by plugging it into this harness. Nothing is built in isolation.

## What it is

A Python codebase that:

1. **Simulates an MOE cell as a multi-rate plant.** Low-fidelity physics is fine - the point is to exercise the controller architecture, not to predict a real cell. Multiple sensor streams at their natural rates (V/I @ ~kHz, T @ 1 Hz, level @ 1 Hz, EIS sweep every minute, OES inference @ 1 Hz). Realistic noise on each.
2. **Runs a controller harness in real time** (or accelerated wall-clock) against the simulator. The harness loads pluggable control / inference / fault-detection modules from configuration.
3. **Exposes sensors + setpoints over a standard industrial protocol** - OPC UA preferred (free Python servers exist), MQTT acceptable as a fallback. This is the integration-story artifact. A real DCS / SCADA / Boston-Metal-style stack speaks OPC UA. Your testbed should too.
4. **Supports fault injection.** A CLI or notebook lets you trigger anode effect, electrode degradation trajectories, melt freeze, contamination events. The fault catalog (F1) is the source of truth for what scenarios the testbed must support.
5. **Emits an event log + telemetry stream** in the same format the real on-orbit version would. Append-only, idempotent, replayable.

## Architecture (mirrors the system architecture doc)

```
   ┌────────────────────────────────────────────────────────────────┐
   │  TESTBED PROCESS (Python)                                       │
   │                                                                 │
   │   ┌──────────────────┐         ┌────────────────────────────┐  │
   │   │   PLANT SIM      │ samples │   CONTROLLER HARNESS       │  │
   │   │  (multi-rate     │────────►│                            │  │
   │   │   physics +      │         │  ┌──────────────────────┐  │  │
   │   │   noise model)   │ ◄────── │  │  Sensing HAL (mock)  │  │  │
   │   └──────────────────┘ acts    │  ├──────────────────────┤  │  │
   │           ▲                    │  │  Inference (plug-in) │  │  │
   │           │ fault injection    │  ├──────────────────────┤  │  │
   │   ┌──────────────────┐         │  │  Control (plug-in)   │  │  │
   │   │ FAULT INJECTOR   │────────►│  ├──────────────────────┤  │  │
   │   │  (CLI/notebook)  │         │  │  Fault SM (plug-in)  │  │  │
   │   └──────────────────┘         │  ├──────────────────────┤  │  │
   │                                │  │  Mode manager        │  │  │
   │                                │  └──────────────────────┘  │  │
   │                                └──────────────┬─────────────┘  │
   │                                               │                │
   │                  ┌────────────────────────────▼────────────┐   │
   │                  │  OPC UA / MQTT broker (in-process)       │   │
   │                  └────────────────────────────┬────────────┘   │
   │                                               │                │
   │                                ┌──────────────▼─────────────┐  │
   │                                │  TELEMETRY + EVENT LOG     │  │
   │                                │  (append-only, structured) │  │
   │                                └────────────────────────────┘  │
   └────────────────────────────────────────────────────────────────┘
                                ▲
                                │  external OPC UA / MQTT clients can
                                │  observe and (later) command
                                ▼
   ┌──────────────────────────────────────────────┐
   │  External viewer (Grafana? Jupyter? Streamlit?)│
   └──────────────────────────────────────────────┘
```

## Concrete deliverables (the contents of this folder when done)

```
04-build/testbed/
├── README.md                  This file.
├── pyproject.toml             Package, deps pinned.
├── plant/
│   ├── cell.py                Plant model: state, dynamics, noise.
│   ├── degradation.py         Electrode degradation parameterization.
│   └── fault_models.py        Per-fault dynamics (anode effect, melt freeze, etc.)
├── harness/
│   ├── scheduler.py           Multi-rate task scheduler (inner kHz, mid 1 Hz, slow 1/60).
│   ├── interfaces.py          Pluggable subsystem ABCs (Inference, Control, FaultDetector, etc.)
│   └── mode_manager.py        Global state machine - IDLE/HEAT/RUN/DEGRADED/FAULT/...
├── adapters/
│   ├── opcua_server.py        OPC UA server exposing sensor + setpoint tags.
│   ├── mqtt_bridge.py         Optional MQTT bridge.
│   └── log_sink.py            Append-only event log + telemetry packing.
├── inject/
│   └── fault_cli.py           CLI: inject anode-effect, force degradation rate, etc.
├── examples/
│   ├── 01_temp_holder.ipynb   Minimal demo: PID holds temperature setpoint.
│   ├── 02_fault_trip.ipynb    Demo: anode-effect injection → trip → recovery.
│   └── 03_eis_plugin.ipynb    Demo: plug in an EIS-based degradation tracker.
├── tests/
│   └── ...                    Unit tests for plant dynamics, fault SM, mode transitions.
└── docs/
    ├── architecture.md        Mirrors `00-thesis/system-architecture.md` at code level.
    ├── protocol.md            OPC UA tag taxonomy and units.
    └── extension.md           How to plug in a new algorithm.
```

## Key design choices (made explicit so they're debatable)

| Choice | Decision | Why | Worth revisiting if |
|--------|----------|-----|---------------------|
| Language | Python | Fast prototyping, scientific stack, OPC UA libs (asyncua), accessible | If real-time perf becomes the bottleneck - but it won't, this is a simulator |
| Plant fidelity | Low | Point is to exercise architecture, not predict reality | Once we have real lab data and want to validate algorithms quantitatively |
| Time model | Accelerated wall-clock | A 24 hr campaign runs in minutes for testing | Real-time mode also supported for soak testing |
| Industrial protocol | OPC UA (server-side) | This is what real DCS/SCADA stacks speak | If we discover the customer wedge speaks something else (Modbus, EtherCAT) |
| Pluggable interfaces | ABC + config | Lets you swap algorithms without touching harness | Never - this is the whole point |
| State machine library | Custom or `transitions` lib | Mode logic is small enough to roll our own | If complexity grows |
| Logging format | JSON Lines + Parquet rollup | Replayable, queryable | Standard industry choice |

## Scope (what is and isn't in v1)

**In scope for v1 (~3 weeks, solo, laptop):**
- Plant simulator with multi-rate sensor outputs, plausible noise, configurable fault injection
- Harness with the three pluggable slots (Inference, Control, FaultDetector) and the mode manager
- OPC UA server exposing the sensor + setpoint tag space
- One worked example: heater PID holds temperature; one anode-effect injection trips into FAULT_RECOVERY and back to RUN_NOMINAL
- Event log + a Jupyter notebook that plots a run end-to-end

**Out of scope for v1:**
- Realistic high-fidelity plant physics
- Actual EIS algorithm (that's I1, plugged in later)
- Actual OES algorithm (that's I2, plugged in later)
- Real hardware in the loop
- Rad-hard considerations (Earth Python prototype only)
- External Grafana / Streamlit dashboard (Jupyter is fine for v1)

## What this gives you

After v1, every subsequent piece (I1 EIS, I2 OES, I3 fusion, F2 detectors, F3 recovery SM, C1 control loops) has:

- A place to live (plug-in slot)
- A clear input contract (which sensor streams it reads)
- A clear output contract (what state it produces or what command it issues)
- A way to be validated (run it against the testbed, with and without fault injection)
- A way to be demonstrated (a Jupyter notebook showing the algorithm running inside the full system)

This is what the "integration story" looks like in practice. It is also exactly the artifact a controls-engineering buyer (Boston Metal, Helios, Blue Origin) recognizes as serious - because it's the artifact they would build themselves.

## How this changes the build order

Original order (from `00-thesis/problem-decomposition.md`):
1. F1 fault catalog
2. I1 EIS pipeline
3. I3 + C1 + C5 fusion + adaptive control
4. F2 + F3 detector + recovery SM
5. Write-up

Revised order:
1. **Testbed v1** (this folder)
2. F1 fault catalog - now becomes the spec for fault injection scenarios the testbed must support
3. I1 EIS pipeline - plug into the testbed's inference slot
4. I3 + C1 + C5 - plug into the testbed's fusion/control slots
5. F2 + F3 - plug into the testbed's detector/recovery slots
6. Write-up - describes the full integrated system running in the testbed, with screenshots/plots from the notebooks

Same scope, same time budget (~3 months); a coherent integrated artifact at the end instead of five disconnected algorithm notebooks.

## Linkage to other workspace docs

- `00-thesis/system-architecture.md` - the architecture this testbed implements
- `00-thesis/problem-decomposition.md` - the pieces that plug in
- `01-subproblems/P1`–`P5` - each gets a corresponding plug-in module here
- `03-why-me/experiments-log.md` - dated entries as each algorithm gets plugged in and validated
