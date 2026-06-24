"""Architecture diagram — clean block diagram style with negative feedback loop."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10))
BG = "#f5f5f5"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE_F  = "#d6e8f7"
BLUE_E  = "#2a7ab8"
GREEN_F = "#d5ecd8"
GREEN_E = "#2e8b57"
RED_F   = "#fddede"
RED_E   = "#c0392b"
GRAY_F  = "#e8e8e8"
GRAY_E  = "#888888"
BLACK   = "#1a1a1a"
DKBLUE  = "#1a4f78"
DKGREEN = "#1a5c35"
DKRED   = "#8b1a1a"

def box(x, y, w, h, fc, ec, lw=2.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=2,
    ))

def txt(x, y, s, color=BLACK, size=12, ha="center", va="center",
        bold=False, wrap=False):
    ax.text(x, y, s, color=color, fontsize=size,
            ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            fontfamily="DejaVu Sans",
            linespacing=1.6, zorder=4)

def arrow_h(x0, y, x1, color, lw=2.5, label="", label_above=True):
    """Horizontal arrow from (x0,y) to (x1,y) with centred label."""
    ax.annotate("",
        xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=20),
        zorder=3)
    if label:
        mid = (x0 + x1) / 2
        dy  = 0.28 if label_above else -0.28
        txt(mid, y + dy, label, color, 11, bold=True)

def arrow_v(x, y0, y1, color, lw=2.5, label="", label_right=True):
    ax.annotate("",
        xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=20),
        zorder=3)
    if label:
        dx = 0.3 if label_right else -0.3
        txt(x + dx, (y0 + y1) / 2, label, color, 10,
            ha="left" if label_right else "right", bold=True)

# ── Fault Injector ─────────────────────────────────────────────────────────────
box(3.8, 8.0, 2.8, 1.4, RED_F, RED_E)
txt(5.2, 8.7, "Fault Injector", DKRED, 13, bold=True)
txt(5.2, 8.25, "simulates disturbances", DKRED, 10)

# ── MRE Cell ───────────────────────────────────────────────────────────────────
box(1.0, 2.8, 5.2, 4.6, BLUE_F, BLUE_E)
txt(3.6, 6.9,  "MRE Cell",            DKBLUE, 15, bold=True)
txt(3.6, 6.35, "Regolith + electricity",         BLACK, 11)
txt(3.6, 5.9,  "→ electrolysis running",         BLACK, 11)
txt(3.6, 5.3,  "Sensors read:",                  BLACK, 11, bold=True)
txt(3.6, 4.85, "Temperature · Current · Voltage", BLACK, 10)
txt(3.6, 4.45, "Electrode condition (EIS)",       BLACK, 10)
txt(3.6, 3.85, "Outputs:",                        BLACK, 11, bold=True)
txt(3.6, 3.4,  "O₂ at anode  +  alloy at cathode", BLACK, 10)

# ── Autonomous Brain ────────────────────────────────────────────────────────────
box(9.8, 2.8, 6.0, 4.6, GREEN_F, GREEN_E)
txt(12.8, 6.9,  "Autonomous Brain",              DKGREEN, 15, bold=True)
txt(12.8, 6.35, "Reads all sensors every second", BLACK, 11)
txt(12.8, 5.75, "Asks:",                          BLACK, 11, bold=True)
txt(12.8, 5.3,  "•  Is the electrode healthy?",        BLACK, 10.5)
txt(12.8, 4.85, "•  Is the cell running efficiently?", BLACK, 10.5)
txt(12.8, 4.4,  "•  Has a fault occurred?",            BLACK, 10.5)
txt(12.8, 3.85, "Decides:",                       BLACK, 11, bold=True)
txt(12.8, 3.4,  "Adjust current  ·  Trigger recovery", BLACK, 10.5)

# ── Event Log ──────────────────────────────────────────────────────────────────
box(14.0, 8.0, 3.2, 1.4, GRAY_F, GRAY_E, lw=2.0)
txt(15.6, 8.7,  "Event Log",            GRAY_E, 13, bold=True)
txt(15.6, 8.25, "records every decision", "#555", 10)

# ── Connection: Fault Injector → MRE Cell ──────────────────────────────────────
arrow_v(5.2, 8.0, 7.4, RED_E, label="injects fault", label_right=True)
# line down to cell top
ax.annotate("",
    xy=(3.6, 7.4), xytext=(3.6, 7.0),
    arrowprops=dict(arrowstyle="-|>", color=RED_E, lw=2.5, mutation_scale=18),
    zorder=3)
ax.plot([5.2, 5.2, 3.6], [8.0, 7.4, 7.4], color=RED_E, lw=2.5, zorder=3)

# ── Feedback loop ──────────────────────────────────────────────────────────────
# Top arrow: Cell → Brain  (sensor readings)
arrow_h(6.2, 5.7, 9.8, BLUE_E, lw=3.0,
        label="Sensor Readings", label_above=True)

# Bottom arrow: Brain → Cell  (commands — the negative feedback)
arrow_h(9.8, 3.9, 6.2, GREEN_E, lw=3.0,
        label="Commands (negative feedback)", label_above=False)

# ── Brain → Event Log ──────────────────────────────────────────────────────────
ax.plot([15.8, 15.8, 15.6], [7.4, 8.0, 8.0], color=GRAY_E, lw=2.0, zorder=3)
ax.annotate("",
    xy=(15.6, 8.0), xytext=(15.6, 8.0),
    arrowprops=dict(arrowstyle="-|>", color=GRAY_E, lw=2.0, mutation_scale=16),
    zorder=3)
arrow_v(15.8, 7.4, 8.0, GRAY_E, lw=2.0)

# ── "Negative Feedback Loop" brace label ───────────────────────────────────────
ax.annotate("",
    xy=(1.0, 2.1), xytext=(15.8, 2.1),
    arrowprops=dict(arrowstyle="<->", color="#aaaaaa", lw=1.5,
                    mutation_scale=14),
    zorder=3)
txt(8.4, 1.75, "NEGATIVE FEEDBACK LOOP", "#aaaaaa", 10, bold=True)

plt.tight_layout(pad=0.3)
plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print("Generated docs/architecture.png")
