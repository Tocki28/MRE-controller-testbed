"""Generate architecture diagram as PNG using Graphviz."""

from graphviz import Digraph

dot = Digraph(engine="dot")
dot.attr(
    rankdir="LR",
    splines="ortho",
    nodesep="0.7",
    ranksep="1.4",
    bgcolor="#1e1e1e",
    fontname="Helvetica",
    fontcolor="white",
    pad="0.4",
)
dot.attr("node",
    shape="box",
    style="rounded,filled",
    fillcolor="#2d2d2d",
    fontcolor="white",
    fontname="Helvetica",
    fontsize="11",
    penwidth="1.5",
    color="#555555",
    margin="0.25,0.18",
)
dot.attr("edge",
    color="#aaaaaa",
    fontcolor="#aaaaaa",
    fontname="Helvetica",
    fontsize="10",
    penwidth="1.5",
)

# Fault injector at source rank
with dot.subgraph() as s:
    s.attr(rank="source")
    s.node("FI", "Fault Injector", shape="ellipse")

# MRE Cell subgraph
with dot.subgraph(name="cluster_cell") as c:
    c.attr(label="MRE Cell (simulated)", style="rounded,filled",
           fillcolor="#252525", color="#555555", fontcolor="white",
           fontname="Helvetica", fontsize="12", margin="16")
    c.node("PLANT", "Temperature · Current · Voltage\nEIS impedance · bath composition")

# Autonomous Brain subgraph
with dot.subgraph(name="cluster_brain") as b:
    b.attr(label="Autonomous Brain", style="rounded,filled",
           fillcolor="#252525", color="#555555", fontcolor="white",
           fontname="Helvetica", fontsize="12", margin="16")
    b.node("INF", "Inference\nelectrode health · state estimate")
    b.node("CTL", "Control\nPID · adaptive current")
    b.node("FSM", "Fault detection & recovery")
    b.node("MM", "Mode manager\nIDLE · HEATING · RUN_NOMINAL · FAULT_RECOVERY")
    b.edge("INF", "CTL")
    b.edge("INF", "FSM")
    b.edge("CTL", "MM")
    b.edge("FSM", "MM")

# Event log — rightmost
dot.node("LOG", "Event log\n& telemetry", shape="cylinder")

# Cross-cluster edges (xlabels avoid the ortho label bug)
# Feedback edge reversed so Graphviz keeps PLANT on the left rank
dot.edge("FI", "PLANT", xlabel="disturbance")
dot.edge("PLANT", "INF", xlabel="sensor readings")
dot.edge("PLANT", "MM", xlabel="setpoints & commands", dir="back")
dot.edge("MM", "LOG")

dot.render("docs/architecture", format="png", cleanup=True)
print("Generated docs/architecture.png")
