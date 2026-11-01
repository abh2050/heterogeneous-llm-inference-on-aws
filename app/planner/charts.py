"""Charts preserve evidence labels and metric definitions."""

import plotly.graph_objects as go

from planner.calibration import context_matches
from planner.models import Assumptions, BenchmarkRecord, Candidate, Hardware, ModelSpec, Workload
from planner.sizing import estimate

COLORS = ["#0d786b", "#3c76b9", "#b17137", "#7f67ad"]


def tpot_chart(
    candidates: list[Candidate],
    runs: list[BenchmarkRecord],
    model: ModelSpec,
    hardware: list[Hardware],
    w: Workload,
    a: Assumptions,
) -> go.Figure:
    fig = go.Figure()
    for i, h in enumerate(hardware):
        for tp in h.allowed_tp:
            rows = sorted(
                [
                    c
                    for c in candidates
                    if c.hardware_id == h.id and c.tp == tp and c.tpot_service_s is not None
                ],
                key=lambda c: c.concurrency,
            )
            if not rows:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[c.concurrency for c in rows],
                    y=[c.tpot_service_s * 1000 for c in rows],
                    mode="lines",
                    line={"color": COLORS[i % len(COLORS)], "dash": "solid" if tp == 1 else "dot"},
                    name=f"{h.name.split(' ·')[0]} · TP{tp} · estimate/fit",
                    customdata=[[c.evidence, c.validation] for c in rows],
                    hovertemplate="Concurrency %{x}<br>Service/mean TPOT %{y:.2f} ms<br>%{customdata[0]}<extra>%{fullData.name}</extra>",
                )
            )
    relevant = []
    for r in runs:
        for h in hardware:
            dummy = estimate(model, h, w, a, r.config.tp, r.config.max_sequences)
            if context_matches(r, model, h, a, dummy, w, exact_workload=False):
                relevant.append(r)
                break
    points = [r for r in relevant if getattr(r.metrics.tpot_s, w.tpot_percentile) is not None]
    if points:
        fig.add_trace(
            go.Scatter(
                x=[r.concurrency / r.config.replicas for r in points],
                y=[getattr(r.metrics.tpot_s, w.tpot_percentile) * 1000 for r in points],
                mode="markers",
                marker={"size": 11, "symbol": "diamond", "color": "#172d38"},
                name=f"Measured {w.tpot_percentile} TPOT · context matched",
                text=[
                    f"{r.run_id} · input {r.workload.input_avg}, output {r.workload.output_avg}"
                    for r in points
                ],
                hovertemplate="%{text}<br>Per-replica concurrency %{x}<br>TPOT %{y:.2f} ms<extra>Measured</extra>",
            )
        )
    fig.add_hline(
        y=w.tpot_s * 1000,
        line_dash="dash",
        line_color="#d26354",
        annotation_text=f"{w.tpot_percentile} TPOT target · {w.tpot_s * 1000:g} ms",
    )
    fig.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 35, "b": 15},
        xaxis_title="Active sequences per replica",
        yaxis_title="Time per output token (ms)",
        xaxis={"type": "log", "dtick": 0.30103},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.22},
        font={"color": "#172d38"},
    )
    return fig


def memory_chart(candidate: Candidate):
    fig = go.Figure()
    for name, color in zip(
        ("weights", "kv", "activations", "runtime", "communication", "safety"),
        ("#0d786b", "#65aba2", "#3c76b9", "#91b5d7", "#7f67ad", "#d5dfe3"),
    ):
        fig.add_trace(
            go.Bar(
                y=["Per device"],
                x=[candidate.memory.get(name, 0) / 2**30],
                name=name.capitalize(),
                orientation="h",
                marker_color=color,
                hovertemplate=f"{name}: %{{x:.2f}} GiB<extra></extra>",
            )
        )
    fig.add_vline(x=candidate.memory.get("budget", 0) / 2**30, line_dash="dash", line_color="#d26354")
    fig.update_layout(
        barmode="stack",
        height=170,
        margin={"l": 5, "r": 5, "t": 10, "b": 0},
        xaxis_title="GiB · dashed line is utilization budget",
        legend={"orientation": "h"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
