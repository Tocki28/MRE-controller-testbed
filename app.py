"""MRE Feedback Controller - Live Dash Dashboard.

Entry point: python app.py
"""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from testbed.logging_setup import configure_logging, get_recent_events
from testbed.scheduler import SimLoop

import argparse
import os
import time
import threading
from datetime import datetime, timezone

import numpy as np
import serial

from testbed.eis_analysis import (
    EISBuffer,
    SweepParser,
    extract_analytical_params,
    fit_randles,
)
from testbed.randles_model import randles_impedance

# Boot the simulator once at module level (not inside a callback)
configure_logging()
sim = SimLoop()
sim.start()

# ── Mode badge colours ──────────────────────────────────────────────────────
MODE_COLORS = {
    "RUN_NOMINAL":         "#4CAF50",   # green
    "HEATING":             "#FF9800",   # orange
    "ELECTRODE_DEGRADING": "#FF5722",   # deep orange
    "ELECTRODE_SWAP":      "#795548",   # brown — inert swap, no current
    "BATH_DEPLETED":       "#9C27B0",   # purple
    "DRAINING":            "#2196F3",   # blue
    "CLEANOUT":            "#607D8B",   # blue-grey
    "FAULT_RECOVERY":      "#F44336",   # red
    "SAFE_SHUTDOWN":       "#F44336",   # red
    "IDLE":                "#9E9E9E",   # grey
}

# Faraday constant - used to compute O₂ production rate display
_FARADAY = 96_485.0
_O2_MOLAR_MASS = 32.0  # g/mol


# ── EIS serial configuration ─────────────────────────────────────────────────
def _get_eis_port() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eis-port", default=None)
    args, _ = parser.parse_known_args()
    return args.eis_port or os.environ.get("EIS_PORT", "/dev/ttyUSB0")


EIS_PORT = _get_eis_port()
EIS_BAUD = 115_200
STALE_THRESHOLD_S = 30


class SerialEISReader(threading.Thread):
    """Daemon thread: reads UART CSV from STM32, assembles sweeps into EISBuffer."""

    def __init__(self, port: str, baud: int, buffer: EISBuffer) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.buffer = buffer
        self._parser = SweepParser()

    def run(self) -> None:
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1.0) as ser:
                    self._parser = SweepParser()
                    self.buffer.set_status("NO_DATA")
                    while True:
                        raw = ser.readline().decode("ascii", errors="replace")
                        if not raw:
                            continue
                        self._parser.parse_line(raw)
                        if self._parser.is_complete_after(raw):
                            sweep = self._parser.get_sweep()
                            if sweep:
                                sweep["analytical"] = extract_analytical_params(
                                    sweep["freq"], sweep["Z_re"], sweep["Z_im"]
                                )
                                sweep["fit"] = fit_randles(
                                    sweep["freq"], sweep["Z_re"], sweep["Z_im"]
                                )
                                self.buffer.add_sweep(sweep)
            except Exception as exc:  # noqa: BLE001
                self.buffer.set_error(str(exc))
                time.sleep(2.0)


eis_buffer = EISBuffer()
_eis_reader = SerialEISReader(EIS_PORT, EIS_BAUD, eis_buffer)
_eis_reader.start()


# ── App ─────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

# ── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"fontFamily": "sans-serif", "padding": "16px 24px", "maxWidth": "1400px"},
    children=[
        # Interval driver
        dcc.Interval(id="tick", interval=1000, n_intervals=0),

        # ── Header ────────────────────────────────────────────────────────
        html.H2(
            "MRE Autonomous Controller — Lunar Regolith Electrolysis",
            style={"marginBottom": "8px"},
        ),
        html.Div(id="mode-badge", style={"marginBottom": "12px"}),

        # ── Bath Composition ───────────────────────────────────────────────
        html.Div(id="recipe-panel", style={"marginBottom": "12px"}),

        html.Hr(),

        # ── Section 1: Plant state ─────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "24px", "alignItems": "flex-start"},
            children=[
                # Left col - metric boxes (30%)
                html.Div(
                    style={"flex": "0 0 30%"},
                    children=[
                        html.H4("Plant State", style={"marginTop": "0", "marginBottom": "12px"}),
                        html.Div(
                            style={"marginBottom": "16px"},
                            children=[
                                html.Span("T_bulk (°C)", style={"color": "grey", "fontSize": "0.85em"}),
                                html.Div(id="metric-T", style={"fontSize": "1.8em", "fontWeight": "bold"}),
                            ],
                        ),
                        html.Div(
                            style={"marginBottom": "16px"},
                            children=[
                                html.Span("I_cell (A)", style={"color": "grey", "fontSize": "0.85em"}),
                                html.Div(id="metric-I", style={"fontSize": "1.8em", "fontWeight": "bold"}),
                            ],
                        ),
                        html.Div(
                            style={"marginBottom": "16px"},
                            children=[
                                html.Span("V_cell (V)", style={"color": "grey", "fontSize": "0.85em"}),
                                html.Div(id="metric-V", style={"fontSize": "1.8em", "fontWeight": "bold"}),
                            ],
                        ),
                        html.Div(
                            style={"marginBottom": "16px"},
                            children=[
                                html.Span("Electrode Health", style={"color": "grey", "fontSize": "0.85em"}),
                                html.Div(id="metric-H", style={"fontSize": "1.8em", "fontWeight": "bold"}),
                            ],
                        ),
                    ],
                ),
                # Right col - time series (70%)
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.H4("Rolling Time Series (last 60 s)", style={"marginTop": "0", "marginBottom": "4px"}),
                        dcc.Graph(id="timeseries", config={"displayModeBar": False}),
                    ],
                ),
            ],
        ),

        html.Hr(),

        # ── Section 2: Inference ────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "24px", "alignItems": "flex-start"},
            children=[
                # Left col - EIS / health gauge
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.H4("EIS - Electrode Health", style={"marginTop": "0", "marginBottom": "4px"}),
                        dcc.Graph(id="nyquist", config={"displayModeBar": False}),
                        dcc.Graph(id="health-gauge", config={"displayModeBar": False}),
                    ],
                ),
                # Right col - composition + OES metrics
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.H4("OES Composition (mock)", style={"marginTop": "0", "marginBottom": "4px"}),
                        dcc.Graph(id="composition", config={"displayModeBar": False}),
                        html.Div(id="oes-metrics", style={"marginTop": "8px"}),
                    ],
                ),
            ],
        ),

        html.Hr(),

        # ── Section 3: O₂ Production ────────────────────────────────────────
        html.H4("O₂ Production (Anode Output)", style={"marginBottom": "8px"}),
        html.Div(id="extraction-panel"),

        html.Hr(),

        # ── Section 4: Fault injection ──────────────────────────────────────
        html.H4("Fault Injection", style={"marginBottom": "8px"}),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "12px", "flexWrap": "wrap"},
            children=[
                dcc.Dropdown(
                    id="fault-type",
                    options=[
                        {"label": "Anode Burnout",    "value": "anode_burnout"},
                        {"label": "Power Loss",       "value": "power_loss"},
                        {"label": "Melt Freeze",      "value": "melt_freeze"},
                        {"label": "Electrode Short",  "value": "electrode_short"},
                        {"label": "Bath Depletion",   "value": "bath_depletion"},
                        {"label": "Sensor Dropout",   "value": "sensor_dropout"},
                        {"label": "Cathode Flooding", "value": "cathode_flooding"},
                        {"label": "Off-gas Blockage", "value": "offgas_blockage"},
                    ],
                    value="melt_freeze",
                    clearable=False,
                    style={"width": "200px"},
                ),
                dcc.Slider(
                    id="severity",
                    min=1, max=7, step=1, value=4,
                    marks={i: str(i) for i in range(1, 8)},
                    tooltip={"placement": "bottom"},
                ),
                html.Button(
                    "Inject",
                    id="inject-btn",
                    n_clicks=0,
                    style={
                        "background": "#EF5350", "color": "white",
                        "border": "none", "padding": "8px 20px",
                        "borderRadius": "4px", "cursor": "pointer",
                        "marginRight": "8px",
                    },
                ),
                html.Button(
                    "Clear Fault",
                    id="clear-btn",
                    n_clicks=0,
                    style={
                        "padding": "8px 20px", "borderRadius": "4px",
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    id="fault-status",
                    style={"marginTop": "8px", "color": "#EF5350"},
                ),
            ],
        ),

        html.Hr(),

        # ── Section 5: Event log ────────────────────────────────────────────
        html.H4("Event Log", style={"marginBottom": "8px"}),
        html.Div(
            id="event-log",
            style={
                "fontFamily": "monospace", "fontSize": "0.85em",
                "lineHeight": "1.8",
            },
        ),

        html.Hr(),

        # ── Footer: chemistry note ───────────────────────────────────────────
        html.Div(
            "Chemistry note: Cathode output is always a mixed Fe/Si/Al/Ti alloy — not pure "
            "sequential metals. Sequential voltage recipe controls which oxides are preferentially "
            "reduced. Pure metal separation requires downstream refining.",
            style={
                "color": "#888",
                "fontSize": "0.78em",
                "fontStyle": "italic",
                "background": "#F5F5F5",
                "padding": "10px 14px",
                "borderRadius": "4px",
                "marginTop": "12px",
            },
        ),

        html.Hr(),

        # ── EIS Status bar ──────────────────────────────────────────────────
        html.Div(id="eis-status-bar", style={"marginBottom": "12px"}),

        html.H3("EIS — Live Hardware Data", style={"marginBottom": "8px"}),

        # ── Section EIS-1: Impedance plots ──────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "flex-start", "marginBottom": "16px"},
            children=[
                html.Div(style={"flex": "1"}, children=[
                    html.H5("Nyquist", style={"margin": "0 0 4px 0", "color": "#555"}),
                    dcc.Graph(id="eis-nyquist", config={"displayModeBar": False}),
                ]),
                html.Div(style={"flex": "1"}, children=[
                    html.H5("Bode — |Z|", style={"margin": "0 0 4px 0", "color": "#555"}),
                    dcc.Graph(id="eis-bode-mag", config={"displayModeBar": False}),
                ]),
                html.Div(style={"flex": "1"}, children=[
                    html.H5("Bode — Phase", style={"margin": "0 0 4px 0", "color": "#555"}),
                    dcc.Graph(id="eis-bode-phase", config={"displayModeBar": False}),
                ]),
            ],
        ),

        html.Hr(),

        # ── Section EIS-2: Parameter cards ──────────────────────────────────
        html.Div(id="eis-params", style={"marginBottom": "16px"}),

        html.Hr(),

        # ── Section EIS-3: Time series ──────────────────────────────────────
        html.H4("EIS Parameter Timeline (last 100 sweeps)", style={"marginBottom": "8px"}),
        dcc.Graph(id="eis-timeseries", config={"displayModeBar": False}),

        # Interval driver for EIS (separate from simulation 1 s tick)
        dcc.Interval(id="eis-tick", interval=2000, n_intervals=0),
    ],
)


# ── Helper: build a blank figure ────────────────────────────────────────────
def _blank_fig(height: int = 250, annotation: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=40, r=20),
    )
    if annotation:
        fig.add_annotation(
            text=annotation,
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="grey"),
        )
    return fig


def _build_o2_panel(state, comp: dict) -> html.Div:
    """O₂ production panel: anode output metrics + bath oxide depletion chart.

    This machine runs one continuous process: electrolysis.
    Anode: O₂ (pure, mission-critical). Cathode: Fe/Si/Al/Ti alloy (structural).
    """
    o2_g = state.O2_produced_mol * _O2_MOLAR_MASS
    rate_g_hr = state.I_cell * state.faradaic_efficiency / (4.0 * _FARADAY) * _O2_MOLAR_MASS * 3600.0

    big = {"fontSize": "1.9em", "fontWeight": "bold", "color": "#42A5F5"}
    lbl = {"color": "grey", "fontSize": "0.8em", "display": "block", "marginBottom": "2px"}
    eff_color = "#66BB6A" if state.faradaic_efficiency > 0.7 else "#FFA726"

    metrics = html.Div(
        style={"display": "flex", "gap": "40px", "marginBottom": "16px", "alignItems": "flex-end"},
        children=[
            html.Div([
                html.Span("O₂ produced (anode)", style=lbl),
                html.Span(f"{o2_g:.2f} g", style=big),
            ]),
            html.Div([
                html.Span("Production rate", style=lbl),
                html.Span(f"{rate_g_hr:.1f} g / hr", style=big),
            ]),
            html.Div([
                html.Span("Faradaic efficiency", style=lbl),
                html.Span(f"{state.faradaic_efficiency * 100:.1f}%", style={**big, "color": eff_color}),
            ]),
        ],
    )

    note = html.Div(
        "Anode: O₂ (propellant + life support).  Cathode: Fe/Si/Al/Ti alloy (structural material).",
        style={"color": "#aaa", "fontSize": "0.78em", "marginTop": "6px", "fontStyle": "italic"},
    )

    comp_fig = go.Figure(go.Bar(
        x=list(comp.keys()),
        y=[v * 100 for v in comp.values()],
        marker_color=["#EF5350", "#42A5F5", "#FFA726", "#66BB6A", "#CE93D8"],
        showlegend=False,
        text=[f"{v * 100:.1f}%" for v in comp.values()],
        textposition="outside",
    ))
    comp_fig.update_layout(
        yaxis_title="Oxide fraction in bath (%)", yaxis_range=[0, 45], height=210,
        margin=dict(t=30, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Bath oxide fractions (depleting as electrolysis runs)", font=dict(size=12, color="grey"), x=0),
    )

    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "24px", "alignItems": "flex-start"},
            children=[
                html.Div(style={"flex": "0 0 38%"}, children=[metrics, note]),
                html.Div(
                    style={"flex": "1"},
                    children=[dcc.Graph(figure=comp_fig, config={"displayModeBar": False})],
                ),
            ],
        ),
    ])


def _build_bath_composition_panel(composition: dict) -> html.Div:
    SPECIES = [
        ("Fe",    "Fe₂O₃", "#EF5350", 0.04),
        ("Si",    "SiO₂",  "#42A5F5", 0.06),
        ("Al",    "Al₂O₃", "#FFA726", 0.04),
        ("Ti",    "TiO₂",  "#66BB6A", 0.04),
        ("Other", "Other",  "#BDBDBD", None),
    ]
    rows = []
    for key, label, color, threshold in SPECIES:
        frac = composition.get(key, 0.0)
        pct = frac * 100
        depleted = threshold is not None and frac <= threshold
        label_color = "#999" if depleted else "#333"
        badge = html.Span(" ⬤ depleted", style={"color": "#999", "fontSize": "0.72em"}) if depleted else ""
        rows.append(html.Div(style={"marginBottom": "8px"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "2px"}, children=[
                html.Span(label, style={"fontSize": "0.82em", "color": label_color, "fontWeight": "500"}),
                html.Span([f"{pct:.1f}%", badge], style={"fontSize": "0.82em", "color": label_color}),
            ]),
            html.Div(style={"position": "relative", "background": "#F0F0F0", "borderRadius": "4px", "height": "10px", "overflow": "visible"}, children=[
                html.Div(style={
                    "width": f"{max(0.0, min(pct, 100)):.1f}%",
                    "background": color if not depleted else "#CCCCCC",
                    "height": "10px", "borderRadius": "4px",
                    "transition": "width 0.3s ease",
                }),
                # Depletion threshold tick
                *([html.Div(style={
                    "position": "absolute", "top": "-3px", "bottom": "-3px",
                    "left": f"{max(0.0, min(threshold * 100, 100)):.1f}%",
                    "width": "2px", "background": "#999", "borderRadius": "1px",
                })] if threshold else []),
            ]),
        ]))
    return html.Div([
        html.Div("Bath Oxide Composition", style={"fontWeight": "600", "fontSize": "0.88em", "color": "#555", "marginBottom": "10px", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        *rows,
    ])


# ── Callback 1: Main update ──────────────────────────────────────────────────
@app.callback(
    [
        Output("mode-badge", "children"),
        Output("recipe-panel", "children"),
        Output("timeseries", "figure"),
        Output("nyquist", "figure"),
        Output("health-gauge", "figure"),
        Output("composition", "figure"),
        Output("extraction-panel", "children"),
        Output("metric-T", "children"),
        Output("metric-I", "children"),
        Output("metric-V", "children"),
        Output("metric-H", "children"),
        Output("oes-metrics", "children"),
        Output("event-log", "children"),
        Output("fault-status", "children"),
    ],
    Input("tick", "n_intervals"),
)
def update(n):  # noqa: ANN001
    snap = sim.get_snapshot()
    if not snap:
        raise dash.exceptions.PreventUpdate

    state = snap["state"]
    inferred = snap["inferred"]
    mode = snap["mode"]
    history = snap["history"]
    # ── Mode badge ─────────────────────────────────────────────────────────
    badge_color = MODE_COLORS.get(mode, "#9E9E9E")
    uptime_s = int(state.uptime_s)
    uptime_str = (
        f"{uptime_s // 3600:02d}:{(uptime_s % 3600) // 60:02d}:{uptime_s % 60:02d}"
    )
    mode_badge = html.Span(
        [
            html.Span(
                mode,
                style={
                    "background": badge_color, "color": "white",
                    "padding": "4px 14px", "borderRadius": "6px",
                    "fontWeight": "bold",
                },
            ),
            html.Span(
                f"  Uptime {uptime_str}",
                style={"color": "grey"},
            ),
        ]
    )

    # ── Timeseries ─────────────────────────────────────────────────────────
    if history:
        t_axis = list(range(len(history)))
        ts_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("T_bulk (°C)", "I_cell (A)", "V_cell (V)", "Health (%)"),
            vertical_spacing=0.15,
        )
        ts_fig.add_trace(
            go.Scatter(x=t_axis, y=[s.T_bulk for s in history],
                       line=dict(color="#EF5350"), showlegend=False),
            row=1, col=1,
        )
        ts_fig.add_trace(
            go.Scatter(x=t_axis, y=[s.I_cell for s in history],
                       line=dict(color="#42A5F5"), showlegend=False),
            row=1, col=2,
        )
        ts_fig.add_trace(
            go.Scatter(x=t_axis, y=[s.V_cell for s in history],
                       line=dict(color="#FFA726"), showlegend=False),
            row=2, col=1,
        )
        ts_fig.add_trace(
            go.Scatter(x=t_axis, y=[s.electrode_health * 100 for s in history],
                       line=dict(color="#66BB6A"), showlegend=False),
            row=2, col=2,
        )
        ts_fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(t=40, b=20, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        ts_fig.update_yaxes(range=[80, 200], row=1, col=2)
    else:
        ts_fig = _blank_fig(height=300, annotation="Waiting for history…")

    # ── Nyquist ────────────────────────────────────────────────────────────
    eis_re = inferred.get("eis_Z_re", [])
    eis_im = inferred.get("eis_Z_im", [])
    if eis_re and eis_im:
        nyq_fig = go.Figure()
        nyq_fig.add_trace(
            go.Scatter(
                x=eis_re, y=eis_im,
                mode="markers",
                marker=dict(size=4, color="#42A5F5"),
                showlegend=False,
            )
        )
        nyq_fig.update_layout(
            title="Nyquist Plot (last EIS sweep)",
            xaxis_title="Re(Z) / Ω",
            yaxis_title="−Im(Z) / Ω",
            height=200,
            margin=dict(t=40, b=40, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        nyq_fig = _blank_fig(height=200, annotation="EIS sweep pending…")

    # ── Health gauge ────────────────────────────────────────────────────────
    health_pct = inferred.get("electrode_health_est", state.electrode_health) * 100
    gauge_color = "#66BB6A" if health_pct > 60 else "#FFA726"
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_pct,
            title={"text": "Electrode Health (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 40],  "color": "#FFCDD2"},
                    {"range": [40, 70], "color": "#FFF9C4"},
                    {"range": [70, 100], "color": "#C8E6C9"},
                ],
            },
            number={"suffix": "%", "font": {"size": 28}},
        )
    )
    gauge_fig.update_layout(
        height=200,
        margin=dict(t=30, b=0, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # ── Composition bar (OES panel - static colours) ────────────────────────
    comp = inferred.get("composition_est", state.composition)
    comp_fig = go.Figure(
        go.Bar(
            x=list(comp.keys()),
            y=[v * 100 for v in comp.values()],
            marker_color=["#EF5350", "#42A5F5", "#FFA726", "#66BB6A", "#AB47BC"],
            showlegend=False,
        )
    )
    comp_fig.update_layout(
        yaxis_title="Fraction (%)",
        height=250,
        margin=dict(t=20, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # ── O₂ production panel ─────────────────────────────────────────────────
    extraction_panel = _build_o2_panel(state, comp)

    # ── Metric values ───────────────────────────────────────────────────────
    metric_T = f"{state.T_bulk:.1f}"
    metric_I = f"{state.I_cell:.1f}"
    metric_V = f"{state.V_cell:.3f}"
    metric_H = f"{state.electrode_health * 100:.1f}%"

    # ── OES metrics ─────────────────────────────────────────────────────────
    ttf = inferred.get("predicted_ttf_hrs", 999.0)
    ttf_str = f"{ttf:.1f} hrs" if ttf < 999 else "N/A"
    oes_metrics = html.Div([
        html.Div(
            style={"marginBottom": "10px"},
            children=[
                html.Span("Predicted failure in", style={"color": "grey", "fontSize": "0.85em", "display": "block"}),
                html.Span(ttf_str, style={"fontSize": "1.5em", "fontWeight": "bold"}),
            ],
        ),
        html.Div([
            html.Span("Faradaic efficiency", style={"color": "grey", "fontSize": "0.85em", "display": "block"}),
            html.Span(
                f"{state.faradaic_efficiency * 100:.1f}%",
                style={"fontSize": "1.5em", "fontWeight": "bold"},
            ),
        ]),
    ])

    # ── Event log ───────────────────────────────────────────────────────────
    events = get_recent_events(15)
    if events:
        rows = []
        for ev in reversed(events):
            ts = ev.get("timestamp", "")[:19].replace("T", " ")
            level = ev.get("log_level", "info").upper()
            msg = ev.get("event", "")
            rows.append(
                html.Div(
                    [
                        html.Span(ts, style={"color": "#888", "marginRight": "8px"}),
                        html.Span(
                            level,
                            style={
                                "fontWeight": "bold",
                                "color": "#F44336" if level == "ERROR" else "#FF9800" if level == "WARNING" else "#555",
                                "marginRight": "8px",
                            },
                        ),
                        html.Span(msg),
                    ]
                )
            )
        event_log = html.Div(rows)
    else:
        event_log = html.Span("No events yet…", style={"color": "grey"})

    # ── Fault status ────────────────────────────────────────────────────────
    active_fault = snap.get("active_fault")
    if active_fault:
        fault_status = f"Active fault: {active_fault} — V_cell = {state.V_cell:.2f} V  |  Mode: {mode}"
    elif state.fault_active:
        fault_status = f"Active fault: {state.fault_active} — V_cell = {state.V_cell:.2f} V"
    else:
        fault_status = ""

    return (
        mode_badge,
        _build_bath_composition_panel(state.composition),
        ts_fig,
        nyq_fig,
        gauge_fig,
        comp_fig,
        extraction_panel,
        metric_T,
        metric_I,
        metric_V,
        metric_H,
        oes_metrics,
        event_log,
        fault_status,
    )


# ── Callback 2: Inject fault ─────────────────────────────────────────────────
@app.callback(
    Output("fault-status", "children", allow_duplicate=True),
    Input("inject-btn", "n_clicks"),
    State("fault-type", "value"),
    State("severity", "value"),
    prevent_initial_call=True,
)
def on_inject(n_clicks, fault_type, severity):  # noqa: ANN001
    if n_clicks:
        sim.fault_injector.inject(fault_type, severity)
        trigger = f"{fault_type}_detected"
        sim.mode_manager.safe_trigger(trigger, fault_name=fault_type)
    return f"Injected: {fault_type} (severity {severity})"


# ── Callback 3: Clear fault ──────────────────────────────────────────────────
@app.callback(
    Output("fault-status", "children", allow_duplicate=True),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_clear(n_clicks):  # noqa: ANN001
    if n_clicks:
        sim.fault_injector.clear()
        sim.mode_manager.clear_active_fault()
    return "Fault cleared"


# ── EIS helpers ──────────────────────────────────────────────────────────────

def _fmt(val: object, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a numeric value or return '---' for None."""
    if val is None:
        return "---"
    return f"{val:{fmt}}{suffix}"


def _residual_badge(residual: float) -> html.Span:
    pct = residual * 100
    color = "#4CAF50" if pct < 5 else "#FF9800" if pct < 15 else "#F44336"
    return html.Span(
        f"Fit residual: {pct:.1f}%",
        style={
            "background": color, "color": "white",
            "padding": "2px 10px", "borderRadius": "4px",
            "fontSize": "0.85em", "marginLeft": "12px",
        },
    )


def _param_card(label: str, value: str, unit: str = "") -> html.Div:
    return html.Div(
        style={
            "background": "#F9F9F9", "border": "1px solid #E0E0E0",
            "borderRadius": "6px", "padding": "10px 14px",
            "minWidth": "120px", "textAlign": "center",
        },
        children=[
            html.Div(label, style={"color": "#888", "fontSize": "0.75em", "marginBottom": "4px"}),
            html.Div(
                value,
                style={"fontSize": "1.4em", "fontWeight": "bold", "color": "#333"},
            ),
            html.Div(unit, style={"color": "#aaa", "fontSize": "0.7em"}),
        ],
    )


# ── EIS Callback ─────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("eis-status-bar", "children"),
        Output("eis-nyquist", "figure"),
        Output("eis-bode-mag", "figure"),
        Output("eis-bode-phase", "figure"),
        Output("eis-params", "children"),
        Output("eis-timeseries", "figure"),
    ],
    Input("eis-tick", "n_intervals"),
)
def update_eis(n):  # noqa: ANN001
    snap = eis_buffer.get_snapshot()
    status = snap["status"]
    ts = snap["sweep_timestamp"]
    count = snap["sweep_count"]
    err = snap["error_msg"]

    # ── Derive effective status (STALE if last sweep was too long ago) ──────
    effective_status = status
    age_s: float | None = None
    if status == "LIVE" and ts is not None:
        age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        if age_s > STALE_THRESHOLD_S:
            effective_status = "STALE"

    # ── Status bar ──────────────────────────────────────────────────────────
    STATUS_COLORS = {
        "LIVE": "#4CAF50", "STALE": "#FF9800", "NO_DATA": "#9E9E9E", "ERROR": "#F44336",
    }
    status_color = STATUS_COLORS.get(effective_status, "#9E9E9E")
    age_str = f"  Last: {int(age_s)}s ago" if age_s is not None else ""
    err_str = f"  {err}" if err and effective_status == "ERROR" else ""

    latest = snap["latest"]
    fit_badge = html.Span("")
    if latest and latest.get("fit", {}).get("fit_ok"):
        fit_badge = _residual_badge(latest["fit"]["residual"])

    status_bar = html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap", "marginBottom": "4px"},
        children=[
            html.Span(
                effective_status,
                style={
                    "background": status_color, "color": "white",
                    "padding": "3px 12px", "borderRadius": "5px",
                    "fontWeight": "bold", "fontSize": "0.9em",
                },
            ),
            html.Span(
                f"Port: {EIS_PORT} | {EIS_BAUD} baud | Sweeps: {count}{age_str}{err_str}",
                style={"color": "#666", "fontSize": "0.85em"},
            ),
            fit_badge,
        ],
    )

    # ── Blank figures when no data ────────────────────────────────────────
    if latest is None:
        blank = _blank_fig(height=250, annotation="No hardware data")
        blank_ts = _blank_fig(height=200, annotation="No sweep history")
        no_cards = html.Div(
            "No EIS data — connect STM32 hardware",
            style={"color": "#aaa", "fontStyle": "italic", "padding": "8px"},
        )
        return status_bar, blank, blank, blank, no_cards, blank_ts

    freq = latest["freq"]
    Z_re = latest["Z_re"]
    Z_im = latest["Z_im"]
    an = latest.get("analytical") or {}
    ft = latest.get("fit") or {}

    # ── Nyquist plot ─────────────────────────────────────────────────────
    nyq_fig = go.Figure()

    # Previous sweep (ghost trace)
    prev = snap["previous"]
    if prev:
        nyq_fig.add_trace(go.Scatter(
            x=prev["Z_re"], y=[-v for v in prev["Z_im"]],
            mode="markers+lines",
            marker=dict(size=5, color="#BDBDBD"),
            line=dict(color="#BDBDBD", width=1, dash="dot"),
            name="Previous", showlegend=True,
        ))

    # Current sweep (solid)
    nyq_fig.add_trace(go.Scatter(
        x=Z_re, y=[-v for v in Z_im],
        mode="markers+lines",
        marker=dict(size=6, color="#42A5F5"),
        line=dict(color="#42A5F5", width=2),
        name="Current", showlegend=True,
    ))

    # Randles fit overlay (dashed orange) — T5
    if ft.get("fit_ok") and ft.get("Rs_fit") is not None:
        omega_dense = np.logspace(
            np.log10(2 * np.pi * min(freq)),
            np.log10(2 * np.pi * max(freq)),
            50,
        )
        Cdl_F = (ft["Cdl_uF"] or 0) / 1e6
        Z_fit_dense = randles_impedance(
            omega_dense,
            ft["Rs_fit"],
            ft["Rct"],
            Cdl_F,
            ft["sigma"],
        )
        nyq_fig.add_trace(go.Scatter(
            x=Z_fit_dense.real, y=-Z_fit_dense.imag,
            mode="lines",
            line=dict(color="#FF9800", width=2, dash="dash"),
            name="Randles fit", showlegend=True,
        ))

    nyq_fig.update_layout(
        xaxis_title="Re(Z) / Ω", yaxis_title="−Im(Z) / Ω",
        height=260,
        margin=dict(t=20, b=40, l=50, r=20),
        legend=dict(orientation="h", y=1.08, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )

    # ── Bode magnitude ───────────────────────────────────────────────────
    Z_abs = [np.hypot(r, i) for r, i in zip(Z_re, Z_im)]
    bode_mag = go.Figure()
    bode_mag.add_trace(go.Scatter(
        x=freq, y=Z_abs, mode="markers+lines",
        marker=dict(size=5, color="#66BB6A"), showlegend=False,
    ))
    bode_mag.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="|Z| (Ω)",
        xaxis_type="log", yaxis_type="log",
        height=260,
        margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )

    # ── Bode phase ───────────────────────────────────────────────────────
    phases = [np.degrees(np.arctan2(i, r)) for r, i in zip(Z_re, Z_im)]
    bode_phase = go.Figure()
    bode_phase.add_trace(go.Scatter(
        x=freq, y=phases, mode="markers+lines",
        marker=dict(size=5, color="#AB47BC"), showlegend=False,
    ))
    bode_phase.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Phase (°)",
        xaxis_type="log",
        height=260,
        margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )

    # ── Parameter cards ──────────────────────────────────────────────────
    CARD_STYLE = {"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}
    analytical_cards = html.Div(style=CARD_STYLE, children=[
        _param_card("Rs", _fmt(an.get("Rs")), "Ω"),
        _param_card("|Z| @ 1 Hz", _fmt(an.get("Z_abs_1Hz")), "Ω"),
        _param_card("Phase @ 100 Hz", _fmt(an.get("phase_100Hz"), ".1f"), "°"),
        _param_card("f_peak", _fmt(an.get("f_peak"), ".1f"), "Hz"),
        _param_card("τ", _fmt(an.get("tau_ms"), ".1f"), "ms"),
    ])
    fitted_cards = html.Div(style=CARD_STYLE, children=[
        _param_card("Rct", _fmt(ft.get("Rct")), "Ω"),
        _param_card("Cdl", _fmt(ft.get("Cdl_uF"), ".2f"), "µF"),
        _param_card("Warburg σ", _fmt(ft.get("sigma"), ".2f"), "Ω·s⁻⁰·⁵"),
        _param_card("i₀ proxy", _fmt(ft.get("i0_proxy_mA_cm2"), ".3f"), "mA/cm²"),
    ])
    param_section = html.Div([
        html.Div([
            html.Span("Analytical", style={"fontWeight": "600", "fontSize": "0.82em", "color": "#555", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        ], style={"marginBottom": "6px"}),
        analytical_cards,
        html.Div([
            html.Span("Randles fit", style={"fontWeight": "600", "fontSize": "0.82em", "color": "#555", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Span(
                " fit failed" if not ft.get("fit_ok") else "",
                style={"background": "#EF5350", "color": "white", "padding": "1px 8px", "borderRadius": "4px", "fontSize": "0.75em", "marginLeft": "8px"},
            ) if not ft.get("fit_ok") else "",
        ], style={"marginBottom": "6px"}),
        fitted_cards,
    ])

    # ── Time series ──────────────────────────────────────────────────────
    history = snap["history"]
    if len(history) > 1:
        xs = list(range(len(history)))
        rs_vals = [h.get("analytical", {}).get("Rs") for h in history]
        rct_vals = [h.get("fit", {}).get("Rct") for h in history]
        cdl_vals = [h.get("fit", {}).get("Cdl_uF") for h in history]
        ph_vals  = [h.get("analytical", {}).get("phase_100Hz") for h in history]

        ts_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Rs (Ω)", "Rct (Ω)", "Cdl (µF)", "Phase @ 100 Hz (°)"),
            vertical_spacing=0.18,
        )
        ts_fig.add_trace(go.Scatter(x=xs, y=rs_vals, line=dict(color="#42A5F5"), showlegend=False), row=1, col=1)
        ts_fig.add_trace(go.Scatter(x=xs, y=rct_vals, line=dict(color="#66BB6A"), showlegend=False), row=1, col=2)
        ts_fig.add_trace(go.Scatter(x=xs, y=cdl_vals, line=dict(color="#FFA726"), showlegend=False), row=2, col=1)
        ts_fig.add_trace(go.Scatter(x=xs, y=ph_vals, line=dict(color="#AB47BC"), showlegend=False), row=2, col=2)
        ts_fig.update_layout(
            height=280, showlegend=False,
            margin=dict(t=40, b=20, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        ts_fig = _blank_fig(height=280, annotation="Waiting for sweep history…")

    return status_bar, nyq_fig, bode_mag, bode_phase, param_section, ts_fig


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
