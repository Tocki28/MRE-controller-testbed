# Testbed v1 MVP — maximum features, minimum work

> The "see something tangible" first build. A live dashboard demo of an autonomous MOE controller running on a simulated cell, with EIS-based health inference, adaptive control, fault injection, fault detection, mode transitions, and event logging. ~20–25 hours of focused work. End artifact: a GitHub repo + a 30-second screen capture you can paste into a Substack post.

## The goal

You want to **fructify** — see an outcome, fast. The constraint: maximum impressive features per hour of work. The deliverable: something you can demo in 30 seconds on your laptop, screen-record, and link to.

This is not a full testbed (that's the broader `README.md`). This is the smallest version that demonstrates the entire autonomy story end-to-end.

## What you see when you run it

`streamlit run app.py` opens a browser tab. One page, four panels:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Autonomous MOE Controller — Live Simulation                            │
├─────────────────────────────────────────────────────────────────────────┤
│  MODE: [RUN_NOMINAL]    Uptime: 02:14:33    Cell #1                    │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌── Plant state ──────────────────┐  ┌── Electrode health (EIS) ──┐   │
│  │ T_bulk:    1597.3 C   ▁▂▃▄▅      │  │  Health: 87% [degrading]   │   │
│  │ I_cell:     142 A     ▆▆▆▆▆      │  │  Pred. failure: 14:32 hrs  │   │
│  │ V_cell:    3.21 V     ─────      │  │  Nyquist plot:             │   │
│  │ Level:     12.4 cm    ─────      │  │  [impedance fit shown]     │   │
│  │ (live time series, 60s rolling)  │  │                            │   │
│  └─────────────────────────────────┘  └────────────────────────────┘   │
│                                                                          │
│  ┌── Composition (OES, mock) ──────┐  ┌── Fault Injection ──────────┐  │
│  │ Si: 32%  Al: 21%  Fe: 18%       │  │ [▼ Anode effect      ]      │  │
│  │ Ti: 14%  Other: 15%             │  │ Severity: ▮▮▮▯▯▯▯  [Inject]│  │
│  │ Yield: 87 g/hr  (Faradaic 76%)  │  │                              │  │
│  └─────────────────────────────────┘  └─────────────────────────────┘  │
│                                                                          │
│  ┌── Event Log ─────────────────────────────────────────────────────┐  │
│  │ 02:14:33  INFO   T_bulk in band                                    │  │
│  │ 02:14:30  INFO   EIS sweep complete, health=87.0%                  │  │
│  │ 02:13:00  INFO   Current setpoint adjusted (degradation-aware)     │  │
│  │ 01:58:45  WARN   Electrode health declining 0.3%/hr                │  │
│  │ 01:42:11  INFO   FAULT_RECOVERY → RUN_NOMINAL (anode-effect cleared)│  │
│  │ 01:41:47  ERROR  Anode-effect detected (V_cell spike)              │  │
│  │ 01:41:46  EVENT  Fault injected by user (anode-effect, sev 4)      │  │
│  │ ...                                                                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

What the user (you, demo viewer) does:
1. Watches the system run nominally. Plant state oscillates in band, EIS slowly tracks gradual electrode degradation, current setpoint adjusts subtly to compensate.
2. Clicks "Inject Anode Effect, severity 4."
3. Watches V_cell spike. Within 1–2 seconds the fault detector flags it. Mode transitions RUN_NOMINAL → FAULT_RECOVERY. Current ramps down. After recovery sequence completes, mode returns to RUN_NOMINAL.
4. Event log scrolls the whole story.

That is the demo. ~60 seconds end-to-end. Every key feature of the autonomy story is visible.

## Features covered (the "maximum features" claim)

| Feature | Where in the demo | Maps to system arch |
|---------|-------------------|---------------------|
| Multi-rate simulation (kHz + Hz + slow) | All plots updating at correct cadences | Section 5 of `system-architecture.md` |
| Sensor fusion across multiple modalities | Plant + EIS + OES + V/I all feeding one state estimator | Subsystem: Inference |
| Real EIS pipeline (synthetic but using `impedance.py`) | Nyquist plot panel + health gauge | I1 in `problem-decomposition.md` |
| EIS-based health inference + predicted time-to-failure | "Health: 87%" + "Pred. failure: 14:32 hrs" | I1 + I4 |
| OES composition mock | Si/Al/Fe/Ti percentages | I2 |
| Closed-loop control (heater PID) | T_bulk stays in band | C1, C2 |
| Adaptive control as electrode ages | Current setpoint adjusting visibly to degradation | C5 |
| Fault injection (user-driven) | Dropdown + severity + button | Testbed feature |
| Fault detection (predictive + reactive) | Log entries for "declining 0.3%/hr" + "Anode-effect detected" | F2 |
| Recovery state machine | Mode transitions during the demo | F3 |
| Global mode manager state machine | "MODE: [...]" badge updating | Subsystem: Mode manager |
| Event log persistence | Log panel scrolling | Subsystem: Comms + Logging |
| Yield + Faradaic efficiency display | "Yield: 87 g/hr  (Faradaic 76%)" | I5 |

13 of the 25 problem-decomposition pieces are touched (some lightly, all visibly). For ~20–25 hours of work.

## Explicitly NOT in v1 MVP (deferred to later builds)

| Feature | Why deferred |
|---------|--------------|
| OPC UA server | Section 3 of testbed README. Adds 4–8 hrs and isn't visible in the dashboard. Add in v2. |
| Modbus TCP adapter | v2 work, after OPC UA |
| Real hardware (AD5933 / Rodeostat) | Month 2 work |
| Realistic high-fidelity plant physics | Demo doesn't need it; low fidelity is enough to exercise the architecture |
| Persistent storage / replayable logs | Streaming buffer is fine for demo; persistence is a 1-day add later |
| Multiple cells / array | Single-cell only |
| Full fault catalog | One fault (anode-effect) is enough to demo the pattern |
| Authentication / security | Local-only demo; not needed |
| Rad-hard considerations | Earth Python only |

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11+ | Scientific stack, fast prototyping |
| Web UI | Streamlit | Single Python script, no frontend code, live-updating widgets, perfect for demos |
| Async runtime | `asyncio` | Standard library; sufficient for multi-rate scheduling |
| EIS analysis | `impedance.py` | Open-source EIS lib, Randles circuit fit built-in |
| Plotting | `plotly` | Streamlit's recommended chart lib for interactivity |
| State machine | `transitions` library OR a 50-line custom SM | Either is fine; library is faster to start |
| Logging | `structlog` | Structured (JSON Lines), easy to filter |
| Packaging | `uv` or `pip` + `pyproject.toml` | Modern Python tooling |
| Testing | `pytest` | Standard |

No databases, no Docker, no broker, no cloud. One process, one script, one browser tab.

## Code structure

```
04-build/testbed/
├── README.md              The full testbed vision
├── v1-mvp.md              This file
├── pyproject.toml
├── app.py                 The Streamlit entry point (~150 lines)
├── testbed/
│   ├── __init__.py
│   ├── plant.py           Plant simulator: ~100 lines. State, dynamics, noise, fault dynamics.
│   ├── scheduler.py       Multi-rate asyncio scheduler: ~80 lines. Task base class + 3 rate buckets.
│   ├── interfaces.py      ABCs for Inference, Control, FaultDetector: ~50 lines.
│   ├── mode_manager.py    State machine: ~80 lines. 5 modes for v1 (skip DEGRADED, HARD_STOP).
│   ├── controllers.py     Heater PID + adaptive current controller: ~80 lines.
│   ├── inference.py       EIS fit (using impedance.py) + simple OES mock + state estimator: ~100 lines.
│   ├── faults.py          Anode-effect injection + detector: ~60 lines.
│   └── logging_setup.py   structlog config: ~30 lines.
└── tests/
    ├── test_plant.py
    ├── test_mode_manager.py
    └── test_fault_trip.py
```

Roughly **~700 LOC of Python** total for the core. Plus the Streamlit `app.py` (~150 LOC) and tests. Easily a single git repo, deployable as a single Streamlit Cloud demo if you want to share a live URL later.

## Effort breakdown

Honest estimate for a developer with embedded + Python background, not the MOE domain:

| Day | Hours | Deliverable |
|-----|-------|-------------|
| 1 | 3 | `plant.py` + basic scheduler. Plant runs, sensor values print to console |
| 1 | 3 | `interfaces.py` + `mode_manager.py` + simple PID heater. T tracks setpoint in console |
| 2 | 3 | `inference.py` EIS sweep with `impedance.py`. Health estimate prints |
| 2 | 3 | `faults.py` anode-effect injection + detector. State machine transitions in console |
| 3 | 4 | `app.py` Streamlit dashboard. All four panels live |
| 3 | 3 | `controllers.py` adaptive control. Current setpoint adjusts to degradation |
| 4 | 3 | Polish: log panel scrolling, fault injection UI, mode badge, plot styling |
| 4 | 2 | Tests, README polish, GIF / screen recording |
| **Total** | **~24 hrs** | Working live demo + GitHub repo |

If everything goes perfectly: 3 days full-time, or 1 week of 3-4 hrs/day part-time.
Realistic with normal hiccups: 5–7 calendar days part-time.

## Acceptance criteria (so you know when you're done)

- [ ] `streamlit run app.py` opens browser, dashboard loads, plots update live
- [ ] Plant T_bulk holds within ±2 C of setpoint under PID control
- [ ] EIS health gauge shows gradual decline over time (degradation simulation working)
- [ ] Current setpoint visibly drops as health declines (adaptive control working)
- [ ] Clicking "Inject Anode Effect" causes a V_cell spike within 1 sec
- [ ] Fault detector flags it within 2 sec of injection
- [ ] Mode badge transitions RUN_NOMINAL → FAULT_RECOVERY → RUN_NOMINAL
- [ ] Event log shows the entire sequence with timestamps
- [ ] One unit test for each: plant dynamics, mode transitions, fault trip
- [ ] README has a 5-line "run me" section and one screenshot

If all of those tick, the v1 MVP is done.

## What to do with the demo once it exists

1. **Record a 30-second screen capture** (QuickTime → File → New Screen Recording on macOS). Show the system running, the fault inject, the recovery. Export as MP4.
2. **Convert to GIF** (or upload MP4 directly to GitHub). Embed in the README.
3. **Push to public GitHub.** Repo name: `moe-controller-testbed` or similar. README with one-paragraph thesis, one screenshot, run instructions, link to a longer Substack post.
4. **Substack draft.** Title: "I built an autonomous controller for high-temperature electrolysis. Here's why and how." Explain the thesis, the autonomy gap, the architecture, link to the repo. Post.
5. **Hacker News.** Submit the Substack post under "Show HN" tag. Be present in the comments for the first 4 hours.
6. **Now** — and only now — start the high-status outreach. The cold email to Helios / Hojong Kim / Adi Oltean opens with: "I built [link]. I think the autonomous control gap in MOE is unaddressed for the reasons in [Substack link]. I'd value your take on [specific question]."

That is the path from zero artifact today to a credible cold email in ~5 weeks.

## What this v1 MVP is NOT

- Not proof the algorithm works on a real cell. That requires bench hardware (Month 2) and lab access (Month 3).
- Not proof anyone will buy it. That requires the outreach conversations after the demo exists.
- Not a product. It's a portfolio artifact + the bench every subsequent piece runs on.

It is the first thing that turns thesis into evidence. Build this.

## Linkage

- Full testbed vision: [`04-build/testbed/README.md`](README.md)
- System architecture this implements: [`00-thesis/system-architecture.md`](../../00-thesis/system-architecture.md)
- Problem decomposition it touches: [`00-thesis/problem-decomposition.md`](../../00-thesis/problem-decomposition.md)
- Where the project stands overall: [`00-thesis/state-of-the-project.md`](../../00-thesis/state-of-the-project.md)
