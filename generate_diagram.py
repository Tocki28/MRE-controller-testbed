"""Architecture diagram — dark background, two boxes, clean feedback loop."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG       = "#1a1a1a"
CELL_BG  = "#0d2137"
CELL_ED  = "#4a9eff"
BRAIN_BG = "#0d2a0d"
BRAIN_ED = "#4aaa4a"
FI_BG    = "#3a1a1a"
FI_ED    = "#cc4444"
FI_TXT   = "#ff8888"
LOG_BG   = "#2a2a2a"
LOG_ED   = "#666666"
WHITE    = "#ffffff"
LGRAY    = "#aaaaaa"

fig, ax = plt.subplots(figsize=(20, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 9)
ax.axis("off")

def box(x, y, w, h, fc, ec, lw=2.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=2,
    ))

def arrow(x0, y0, x1, y1, color, lw=2.8):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            mutation_scale=22,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )

def txt(x, y, s, color=WHITE, size=12, ha="center", va="center", bold=False):
    ax.text(x, y, s, color=color, fontsize=size,
            ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            linespacing=1.7, zorder=4)

# ── Fault Injector ─────────────────────────────────────────────────────────────
from matplotlib.patches import Ellipse
ax.add_patch(Ellipse((1.6, 6.5), 2.6, 1.4,
    facecolor=FI_BG, edgecolor=FI_ED, linewidth=2.0, zorder=2))
txt(1.6, 6.65, "Fault Injector",      FI_TXT, 11, bold=True)
txt(1.6, 6.25, "simulates problems",  FI_TXT, 9.5)

# ── MRE Cell ───────────────────────────────────────────────────────────────────
CX, CY, CW, CH = 3.2, 1.0, 5.0, 7.0
box(CX, CY, CW, CH, CELL_BG, CELL_ED)
txt(CX + CW/2, CY + CH - 0.55,  "MRE Cell",             CELL_ED, 15, bold=True)
txt(CX + CW/2, CY + CH - 1.15,  "Regolith + electricity",       WHITE, 11)
txt(CX + CW/2, CY + CH - 1.65,  "electrolysis running continuously", WHITE, 10.5)
txt(CX + CW/2, CY + CH - 2.45,  "Sensors read:",                WHITE, 11, bold=True)
txt(CX + CW/2, CY + CH - 3.0,   "Temperature · Current · Voltage", WHITE, 10.5)
txt(CX + CW/2, CY + CH - 3.5,   "Electrode condition (EIS)",    WHITE, 10.5)
txt(CX + CW/2, CY + CH - 4.3,   "Outputs:",                     WHITE, 11, bold=True)
txt(CX + CW/2, CY + CH - 4.85,  "O₂ at anode  +  alloy at cathode", WHITE, 10.5)

# ── Autonomous Brain ────────────────────────────────────────────────────────────
BX, BY, BW, BH = 10.3, 1.0, 5.5, 7.0
box(BX, BY, BW, BH, BRAIN_BG, BRAIN_ED)
txt(BX + BW/2, BY + BH - 0.55,  "Autonomous Brain",             BRAIN_ED, 15, bold=True)
txt(BX + BW/2, BY + BH - 1.15,  "Reads sensors every second",   WHITE, 11)
txt(BX + BW/2, BY + BH - 2.0,   "Asks:",                        WHITE, 11, bold=True)
txt(BX + BW/2, BY + BH - 2.6,   "•  Is the electrode healthy?",       WHITE, 10.5)
txt(BX + BW/2, BY + BH - 3.15,  "•  Is the cell running efficiently?", WHITE, 10.5)
txt(BX + BW/2, BY + BH - 3.7,   "•  Has a fault occurred?",           WHITE, 10.5)
txt(BX + BW/2, BY + BH - 4.55,  "Decides:",                     WHITE, 11, bold=True)
txt(BX + BW/2, BY + BH - 5.1,   "Adjust current  ·  Trigger recovery", WHITE, 10.5)
txt(BX + BW/2, BY + BH - 5.65,  "and sends commands back",      WHITE, 10.5)

# ── Event Log ──────────────────────────────────────────────────────────────────
LX, LY, LW, LH = 16.8, 3.5, 2.8, 2.0
box(LX, LY, LW, LH, LOG_BG, LOG_ED, lw=1.8)
txt(LX + LW/2, LY + LH/2 + 0.3,  "Event Log",             LGRAY, 11, bold=True)
txt(LX + LW/2, LY + LH/2 - 0.25, "records every decision", LGRAY, 9.5)

# ── Arrow: Fault Injector → MRE Cell ──────────────────────────────────────────
ax.plot([1.6, 1.6, CX], [5.8, CY + CH * 0.78, CY + CH * 0.78],
        color=FI_ED, lw=2.0, zorder=3)
arrow(CX, CY + CH * 0.78, CX + 0.01, CY + CH * 0.78, FI_ED, lw=2.0)
txt(2.4, CY + CH * 0.78 + 0.3, "injects fault", FI_TXT, 9.5)

# ── Gap between boxes ──────────────────────────────────────────────────────────
GAP_X0 = CX + CW          # 8.2
GAP_X1 = BX               # 10.3
MID_X  = (GAP_X0 + GAP_X1) / 2

# Top arrow: sensor readings Cell → Brain
S_Y = CY + CH * 0.72      # ~6.04
arrow(GAP_X0, S_Y, GAP_X1, S_Y, CELL_ED, lw=3.0)
txt(MID_X, S_Y + 0.38, "Sensor Readings", CELL_ED, 12, bold=True)

# Bottom arrow: commands Brain → Cell  (negative feedback)
C_Y = CY + CH * 0.28      # ~2.96
arrow(GAP_X1, C_Y, GAP_X0, C_Y, BRAIN_ED, lw=3.0)
txt(MID_X, C_Y - 0.4, "Commands  (negative feedback)", BRAIN_ED, 12, bold=True)

# ── Arrow: Brain → Event Log ───────────────────────────────────────────────────
MID_Y = BY + BH / 2
ax.plot([BX + BW, LX], [MID_Y, MID_Y], color=LOG_ED, lw=1.8, zorder=3)
arrow(LX - 0.01, MID_Y, LX, MID_Y, LOG_ED, lw=1.8)

plt.tight_layout(pad=0)
plt.savefig("docs/architecture_v2.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print("Generated docs/architecture_v2.png")
