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

        # ── Recipe phase indicator ─────────────────────────────────────────
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
                    options=[{"label": "Anode Effect", "value": "anode_effect"}],
                    value="anode_effect",
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
            "Chemistry note: Cathode output is always a mixed Fe/Si/Al/Ti alloy — voltage does "
            "not select which metal is reduced. The brain adapts current setpoint as the bath "
            "depletes. Pure metal separation requires a second downstream process.",
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


_RECIPE_STEPS = [
    ("Fe",    "Phase 1", "Fe-rich bath — brain sets low current, O₂ production begins",    " 80 A",  "~2.4 V", "#EF5350"),
    ("Si",    "Phase 2", "Si-dominant bath — brain raises current, O₂ rate increasing",    "120 A",  "~3.6 V", "#42A5F5"),
    ("Al_Ti", "Phase 3", "Al/Ti-rich bath — brain at peak current, propellant precursor yield", "160 A", "~4.8 V", "#FFA726"),
]
_PHASE_ORDER = ["Fe", "Si", "Al_Ti", "complete"]


def _build_recipe_panel(bath_phase: str) -> html.Div:
    current_idx = _PHASE_ORDER.index(bath_phase) if bath_phase in _PHASE_ORDER else 3
    boxes = []
    for step_phase, label, desc, current, voltage, color in _RECIPE_STEPS:
        step_idx = _PHASE_ORDER.index(step_phase)
        is_active = step_idx == current_idx
        is_done = step_idx < current_idx

        if is_active:
            bg, border, text_color = color + "22", f"2px solid {color}", color
            status = "▶ ACTIVE"
        elif is_done:
            bg, border, text_color = "#F5F5F5", "2px solid #CCC", "#999"
            status = "✓ DONE"
        else:
            bg, border, text_color = "#FAFAFA", "1px dashed #DDD", "#BBB"
            status = "PENDING"

        boxes.append(html.Div(
            style={
                "flex": "1", "padding": "10px 14px",
                "background": bg, "border": border,
                "borderRadius": "6px", "minWidth": "160px",
            },
            children=[
                html.Div(label, style={"fontWeight": "bold", "color": text_color, "fontSize": "0.9em"}),
                html.Div(desc, style={"color": text_color, "fontSize": "0.8em", "margin": "2px 0"}),
                html.Div(
                    f"{current} · {voltage}",
                    style={"color": text_color, "fontSize": "0.78em", "fontFamily": "monospace"},
                ),
                html.Div(status, style={"color": text_color, "fontSize": "0.75em", "marginTop": "4px", "fontWeight": "bold"}),
            ],
        ))

    return html.Div(
        style={"display": "flex", "gap": "10px", "alignItems": "stretch"},
        children=boxes,
    )


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
    bath_phase = snap.get("bath_phase", state.bath_phase)

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
    if state.fault_active:
        fault_status = f"Active fault: {state.fault_active} - V_cell = {state.V_cell:.2f} V"
    else:
        fault_status = ""

    return (
        mode_badge,
        _build_recipe_panel(bath_phase),
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
    return "Fault cleared"


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
