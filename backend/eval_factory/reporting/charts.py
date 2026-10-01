"""Four dated Plotly charts for a completed evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go

from eval_factory.domain.models import EvaluationRun, ScoreMode

_DET = "#1e4d3a"
_PROB = "#243e6a"
_COPPER = "#b85c38"
_INK = "#1c1915"
_PAPER = "#faf7f3"


def write_charts(run: EvaluationRun, directory: Path, day: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    names = [
        f"score-by-metric-{day}.html",
        f"mode-comparison-{day}.html",
        f"case-risk-{day}.html",
        f"case-heatmap-{day}.html",
    ]
    figures = [
        _score_by_metric(run),
        _mode_comparison(run),
        _case_risk(run),
        _case_heatmap(run),
    ]
    for name, figure in zip(names, figures, strict=True):
        figure.write_html(directory / name, include_plotlyjs="cdn", full_html=True)
    return names


def _layout(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=_PAPER,
        plot_bgcolor="#ffffff",
        font={"family": "IBM Plex Sans, sans-serif", "color": _INK},
        title={"text": title, "x": 0.02},
        margin={"l": 64, "r": 24, "t": 72, "b": 72},
    )
    return figure


def _score_by_metric(run: EvaluationRun) -> go.Figure:
    metrics = run.aggregate.metrics if run.aggregate else []
    colors = [_DET if metric.mode == ScoreMode.DETERMINISTIC else _PROB for metric in metrics]
    figure = go.Figure(
        data=[
            go.Bar(
                x=[metric.name for metric in metrics],
                y=[metric.score for metric in metrics],
                marker_color=colors,
                text=[f"{metric.score:.2f}" for metric in metrics],
                textposition="outside",
            )
        ]
    )
    if metrics:
        figure.add_hline(y=0.7, line_dash="dot", line_color="#8c3e22", annotation_text="0.70 reference")
    figure.update_yaxes(range=[0, 1.15], title="Score")
    return _layout(figure, "Score by metric")


def _mode_comparison(run: EvaluationRun) -> go.Figure:
    aggregate = run.aggregate
    labels = ["Deterministic", "Probabilistic", "Overall"]
    values = [
        aggregate.deterministic if aggregate and aggregate.deterministic is not None else 0,
        aggregate.probabilistic if aggregate and aggregate.probabilistic is not None else 0,
        aggregate.overall if aggregate else 0,
    ]
    figure = go.Figure(
        data=[go.Bar(x=labels, y=values, marker_color=[_DET, _PROB, _COPPER], text=[f"{value:.2f}" for value in values], textposition="outside")]
    )
    figure.update_yaxes(range=[0, 1.15], title="Score")
    return _layout(figure, "Deterministic, probabilistic, and overall")


def _case_risk(run: EvaluationRun) -> go.Figure:
    means = _case_means(run)
    bands = {"0.00–0.50": 0, "0.50–0.70": 0, "0.70–0.85": 0, "0.85–1.00": 0}
    for score in means.values():
        if score < 0.5:
            bands["0.00–0.50"] += 1
        elif score < 0.7:
            bands["0.50–0.70"] += 1
        elif score < 0.85:
            bands["0.70–0.85"] += 1
        else:
            bands["0.85–1.00"] += 1
    figure = go.Figure(
        data=[
            go.Bar(
                x=list(bands.keys()),
                y=list(bands.values()),
                marker_color=["#8f2d2d", _COPPER, "#c4a15a", _DET],
            )
        ]
    )
    figure.update_yaxes(title="Cases", rangemode="tozero")
    return _layout(figure, "Case score distribution")


def _case_heatmap(run: EvaluationRun) -> go.Figure:
    applicable = [result for result in run.results if result.applicable]
    metric_ids = []
    for result in applicable:
        if result.metric_id not in metric_ids:
            metric_ids.append(result.metric_id)
    case_ids = []
    for result in applicable:
        if result.case_id not in case_ids:
            case_ids.append(result.case_id)
    lookup = {(result.case_id, result.metric_id): result.score for result in applicable}
    names = {metric.metric_id: metric.name for metric in (run.aggregate.metrics if run.aggregate else [])}
    z = [[lookup.get((case_id, metric_id), None) for metric_id in metric_ids] for case_id in case_ids]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=[names.get(metric_id, metric_id) for metric_id in metric_ids],
                y=case_ids,
                zmin=0,
                zmax=1,
                colorscale=[[0, "#8f2d2d"], [0.5, "#e6d3a3"], [1, "#1e4d3a"]],
                colorbar={"title": "Score"},
            )
        ]
    )
    return _layout(figure, "Case by metric")


def _case_means(run: EvaluationRun) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in run.results:
        if result.applicable:
            grouped[result.case_id].append(result.score)
    return {case_id: sum(scores) / len(scores) for case_id, scores in grouped.items()}
