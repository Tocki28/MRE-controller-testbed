"""Architecture diagram — full structure with brain internals and feedback loop."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

BG       = "#1a1a1a"
CELL_BG  = "#0d2137"; CELL_ED  = "#4a9eff"
BRAIN_BG = "#0d2a0d"; BRAIN_ED = "#4aaa4a"
INF_BG   = "#1a3a1a"; INF_ED   = "#66cc66"
CTL_BG   = "#1a2a3a"; CTL_ED   = "#66aaff"
FSM_BG   = "#2a1a2a"; FSM_ED   = "#cc88ff"
MM_BG    = "#2a2a0d"; MM_ED    = "#ddcc44"
FI_BG    = "#3a1a1a"; FI_ED    = "#cc4444"
LOG_BG   = "#2a2a2a"; LOG_ED   = "#888888"
WHITE    = "#ffffff";  LGRAY    = "#aaaaaa"

fig, ax = plt.subplots(figsize=(24, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 24)
ax.set_ylim(0, 9)
ax.axis("off")

def box(x, y, w, h, fc, ec, lw=2.0, r=0.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))

def arrow(x0, y0, x1, y1, color, lw=2.0, label="", ldy=0.28, ldx=0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=18,
                        connectionstyle="arc3,rad=0.0"),
        zorder=5)
    if label:
        mx, my = (x0+x1)/2 + ldx, (y0+y1)/2 + ldy
        ax.text(mx, my, label, color=color, fontsize=9.5,
                ha="center", va="center", fontweight="bold",
                fontfamily="DejaVu Sans", zorder=6)

def txt(x, y, s, color=WHITE, size=11, ha="center", va="center", bold=False):
    ax.text(x, y, s, color=color, fontsize=size,
            ha=ha, va=va, fontweight="bold" if bold else "normal",
            fontfamily="DejaVu Sans", linespacing=1.6, zorder=4)

def line(xs, ys, color, lw=2.0):
    ax.plot(xs, ys, color=color, lw=lw, zorder=3)

# ── Fault Injector ─────────────────────────────────────────────────────────────
ax.add_patch(Ellipse((2.5, 7.5), 3.2, 1.3,
    facecolor=FI_BG, edgecolor=FI_ED, linewidth=2.0, zorder=2))
txt(2.5, 7.65, "Fault Injector", FI_ED, 11, bold=True)
txt(2.5, 7.25, "simulates problems", "#ff8888", 9.5)

# ── MRE Cell ───────────────────────────────────────────────────────────────────
box(0.6, 2.0, 4.2, 4.5, CELL_BG, CELL_ED, lw=2.5)
txt(2.7, 6.0,  "MRE Cell",              CELL_ED, 14, bold=True)
txt(2.7, 5.5,  "Regolith + electricity",         WHITE, 10)
txt(2.7, 5.05, "electrolysis running",            WHITE, 10)
txt(2.7, 4.45, "Sensors:",              WHITE, 10, bold=True)
txt(2.7, 4.0,  "T · V · I · EIS",               WHITE, 10)
txt(2.7, 3.4,  "Outputs:",              WHITE, 10, bold=True)
txt(2.7, 2.95, "O₂  +  alloy",                   WHITE, 10)

# ── Feedback Controller container ─────────────────────────────────────────────
box(6.2, 1.2, 14.5, 6.3, BRAIN_BG, BRAIN_ED, lw=2.5)
txt(13.45, 7.1, "Feedback Controller", BRAIN_ED, 14, bold=True)

# ── Inference ──────────────────────────────────────────────────────────────────
box(6.8, 3.2, 3.2, 2.8, INF_BG, INF_ED, lw=1.8)
txt(8.4, 5.45, "Inference",        INF_ED, 11, bold=True)
txt(8.4, 5.0,  "EIS health fit",   WHITE,  9.5)
txt(8.4, 4.6,  "state estimate",   WHITE,  9.5)
txt(8.4, 4.1,  "OES composition",  WHITE,  9.5)
txt(8.4, 3.65, "(mock)",           LGRAY,  8.5)

# ── Control ────────────────────────────────────────────────────────────────────
box(11.2, 5.0, 3.0, 2.0, CTL_BG, CTL_ED, lw=1.8)
txt(12.7, 6.45, "Control",          CTL_ED, 11, bold=True)
txt(12.7, 5.95, "PID temperature",  WHITE,  9.5)
txt(12.7, 5.5,  "adaptive current", WHITE,  9.5)

# ── Fault detection ────────────────────────────────────────────────────────────
box(11.2, 2.4, 3.0, 2.1, FSM_BG, FSM_ED, lw=1.8)
txt(12.7, 3.95, "Fault Detection",  FSM_ED, 11, bold=True)
txt(12.7, 3.45, "anode effect",     WHITE,  9.5)
txt(12.7, 3.0,  "recovery SM",      WHITE,  9.5)

# ── Mode Manager ───────────────────────────────────────────────────────────────
box(15.5, 3.2, 4.4, 3.4, MM_BG, MM_ED, lw=1.8)
txt(17.7, 6.15, "Mode Manager",      MM_ED,  11, bold=True)
txt(17.7, 5.7,  "IDLE",              WHITE,  9.5)
txt(17.7, 5.3,  "HEATING",           WHITE,  9.5)
txt(17.7, 4.9,  "RUN_NOMINAL",       WHITE,  9.5)
txt(17.7, 4.5,  "FAULT_RECOVERY",    WHITE,  9.5)
txt(17.7, 4.0,  "SAFE_SHUTDOWN",     WHITE,  9.5)

# ── Event Log ──────────────────────────────────────────────────────────────────
box(20.6, 3.5, 2.8, 1.8, LOG_BG, LOG_ED, lw=1.6)
txt(22.0, 4.7, "Event Log",           LGRAY, 10, bold=True)
txt(22.0, 4.25, "all decisions",      LGRAY, 9)
txt(22.0, 3.85, "recorded",           LGRAY, 9)

# ── Arrows ─────────────────────────────────────────────────────────────────────
# FI → Cell
line([2.5, 2.5], [6.85, 6.5], FI_ED, lw=2.0)
line([2.5, 2.5], [6.5, 6.5], FI_ED, lw=2.0)
arrow(2.5, 6.5, 2.5, 6.5 - 0.01, FI_ED, lw=0)
ax.annotate("", xy=(2.5, 6.5), xytext=(2.5, 6.85),
    arrowprops=dict(arrowstyle="-|>", color=FI_ED, lw=2.0, mutation_scale=16), zorder=5)

# Cell → Inference  (sensor readings)
arrow(4.8, 4.6, 6.8, 4.6, CELL_ED, lw=2.5,
      label="sensor readings", ldy=0.3)

# Inference → Control
arrow(10.0, 5.5, 11.2, 5.9, INF_ED, lw=1.8, label="health / state", ldy=0.28)

# Inference → Fault Detection
arrow(10.0, 3.8, 11.2, 3.6, FSM_ED, lw=1.8, label="anomaly signal", ldy=-0.28)

# Control → Mode Manager
arrow(14.2, 6.0, 15.5, 5.5, CTL_ED, lw=1.8)

# Fault Detection → Mode Manager
arrow(14.2, 3.5, 15.5, 4.2, FSM_ED, lw=1.8)

# Mode Manager → Event Log
arrow(19.9, 4.4, 20.6, 4.4, LOG_ED, lw=1.6)

# Mode Manager → Cell  (setpoints & commands — the negative feedback)
# Route: right side of MM → down → left along bottom → up into Cell
line([17.7, 17.7], [3.2, 1.5], MM_ED, lw=2.5)   # down from MM
line([17.7, 2.7],  [1.5, 1.5], MM_ED, lw=2.5)   # left along bottom
ax.annotate("", xy=(2.7, 2.0), xytext=(2.7, 1.5),
    arrowprops=dict(arrowstyle="-|>", color=MM_ED,
                    lw=2.5, mutation_scale=20), zorder=5)
txt(10.2, 1.2, "setpoints & commands  (negative feedback)", MM_ED, 11, bold=True)

plt.tight_layout(pad=0)
plt.savefig("docs/architecture_v3.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print("Generated docs/architecture_v3.png")
