"""Generate architecture diagram using matplotlib for precise layout control."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

BG      = "#1a1a1a"
CELL_BG = "#0d2137"
CELL_ED = "#4a9eff"
BRAIN_BG= "#0d2a0d"
BRAIN_ED= "#4aaa4a"
FI_BG   = "#3a1a1a"
FI_ED   = "#cc4444"
FI_TXT  = "#ff8888"
LOG_BG  = "#2a2a2a"
LOG_ED  = "#666666"
WHITE   = "#ffffff"
GRAY    = "#888888"

fig, ax = plt.subplots(figsize=(20, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, lw=2.5, radius=0.25):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    ))

def oval(cx, cy, rx, ry, fc, ec, lw=2.0):
    ax.add_patch(mpatches.Ellipse(
        (cx, cy), rx * 2, ry * 2,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    ))

def arrow(x0, y0, x1, y1, color, lw=2.5):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=18,
            connectionstyle="arc3,rad=0",
        ),
        zorder=3,
    )

def txt(x, y, s, color=WHITE, size=11, ha="center", va="center", bold=False):
    ax.text(x, y, s, color=color, fontsize=size, ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            fontfamily="Helvetica Neue", zorder=4)

# ── Fault Injector ────────────────────────────────────────────────────────────
oval(1.5, 4.0, 1.3, 0.85, FI_BG, FI_ED)
txt(1.5, 4.25, "Fault Injector", FI_TXT, 10, bold=True)
txt(1.5, 3.78, "(simulates problems)", FI_TXT, 8.5)

# ── MRE Cell box ─────────────────────────────────────────────────────────────
CX, CY, CW, CH = 3.3, 1.2, 4.4, 5.6
box(CX, CY, CW, CH, CELL_BG, CELL_ED)
txt(CX + CW/2, CY + CH - 0.45, "MRE Cell", CELL_ED, 13, bold=True)
cell_body = (
    "Regolith melted by electricity\n\n"
    "Sensors read:\n"
    "Temperature · Current · Voltage\n"
    "Electrode condition\n\n"
    "Outputs:\n"
    "O₂ at anode  +  alloy at cathode"
)
ax.text(CX + CW/2, CY + CH/2 - 0.2, cell_body,
        color=WHITE, fontsize=10.5, ha="center", va="center",
        linespacing=1.7, fontfamily="Helvetica Neue", zorder=4)

# ── Autonomous Brain box ──────────────────────────────────────────────────────
BX, BY, BW, BH = 9.3, 1.2, 4.9, 5.6
box(BX, BY, BW, BH, BRAIN_BG, BRAIN_ED)
txt(BX + BW/2, BY + BH - 0.45, "Autonomous Brain", BRAIN_ED, 13, bold=True)
brain_body = (
    "Reads sensor data every second\n\n"
    "•  Is the electrode healthy?\n"
    "•  Is the cell running efficiently?\n"
    "•  Has a fault occurred?\n\n"
    "Decides what to do next\n"
    "and sends commands back"
)
ax.text(BX + BW/2, BY + BH/2 - 0.2, brain_body,
        color=WHITE, fontsize=10.5, ha="center", va="center",
        linespacing=1.7, fontfamily="Helvetica Neue", zorder=4)

# ── Event Log ─────────────────────────────────────────────────────────────────
LX, LY, LW, LH = 15.7, 2.8, 2.8, 2.4
box(LX, LY, LW, LH, LOG_BG, LOG_ED, lw=1.5)
txt(LX + LW/2, LY + LH/2 + 0.2, "Event Log", GRAY, 11, bold=True)
txt(LX + LW/2, LY + LH/2 - 0.3, "records every decision", GRAY, 9.5)

# ── Gap between Cell and Brain ────────────────────────────────────────────────
GAP_MID = (CX + CW + BX) / 2   # midpoint of the gap

# Sensor readings arrow: right side of Cell → left side of Brain (top)
arrow(CX + CW, CY + CH * 0.72, BX, BY + BH * 0.72, CELL_ED)
txt(GAP_MID, CY + CH * 0.72 + 0.35, "sensor readings", CELL_ED, 11, bold=True)

# Commands arrow: left side of Brain → right side of Cell (bottom)
arrow(BX, BY + BH * 0.28, CX + CW, BY + BH * 0.28, BRAIN_ED)
txt(GAP_MID, BY + BH * 0.28 - 0.38,
    "commands: adjust current, recover from fault",
    BRAIN_ED, 11, bold=True)

# Fault injector → Cell
arrow(1.5 + 1.3, 4.0, CX, CY + CH * 0.5, FI_ED, lw=2.0)
txt(2.7, 4.3, "injects fault", FI_TXT, 9.5)

# Brain → Event Log
arrow(BX + BW, BY + BH * 0.5, LX, LY + LH * 0.5, LOG_ED, lw=1.8)

plt.tight_layout(pad=0)
plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print("Generated docs/architecture.png")
