"""Decision-ready Markdown for a completed run."""

from __future__ import annotations

from pathlib import Path

from eval_factory.domain.models import EvaluationRun, RiskLevel
from eval_factory.reporting.charts import write_charts
from eval_factory.scoring.aggregate import aggregate_results


def publish_outputs(run: EvaluationRun, directory: Path) -> EvaluationRun:
    directory.mkdir(parents=True, exist_ok=True)
    run.aggregate = aggregate_results(run)
    day = run.updated_at.astimezone().date().isoformat()
    run.chart_files = write_charts(run, directory / "charts", day)
    run.report_markdown = render_report(run)
    (directory / "report.md").write_text(run.report_markdown, encoding="utf-8")
    return run


def render_report(run: EvaluationRun) -> str:
    aggregate = run.aggregate
    if aggregate is None:
        raise ValueError("publish_outputs must aggregate the run before rendering")
    analysis = run.analysis
    lines = [
        f"# Evaluation report: {run.name}",
        "",
        f"- System type: `{run.system_type.value}`",
        f"- Provider: `{run.provider.provider.value}`",
        f"- Cases: {aggregate.case_count}",
        f"- Risk: **{aggregate.risk_level.value}**",
        f"- Overall: **{aggregate.overall:.2f}**",
        f"- Deterministic: {_fmt(aggregate.deterministic)}",
        f"- Probabilistic: {_fmt(aggregate.probabilistic)}",
        "",
        "## Decision",
        "",
        aggregate.decision,
        "",
        "## Recommendations",
        "",
    ]
    for note in aggregate.recommendations:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Scores",
            "",
            "| Metric | Mode | Score | Threshold | Weight | Result |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    thresholds = (
        {metric.id: metric.threshold for metric in run.metric_plan.metrics} if run.metric_plan else {}
    )
    for metric in aggregate.metrics:
        result = "pass" if metric.passed else "fail"
        lines.append(
            f"| {metric.name} | {metric.mode.value} | {metric.score:.2f} | "
            f"{thresholds.get(metric.metric_id, 0):.2f} | {metric.normalized_weight:.2f} | {result} |"
        )
    lines.extend(["", "## Architecture", ""])
    if analysis is None:
        lines.append("No architecture analysis was stored.")
    else:
        lines.extend([analysis.summary, "", "### Components", ""])
        for component in analysis.components:
            lines.append(f"- **{component.name}** ({component.role}): {component.description}")
        if analysis.tools:
            lines.extend(["", "### Tools", "", ", ".join(f"`{tool}`" for tool in analysis.tools)])
        lines.extend(["", "### Risks", ""])
        for risk in analysis.risks:
            lines.append(
                f"- **{risk.name}** ({risk.severity.value}): {risk.description} Evidence: {risk.evidence}"
            )
    lines.extend(["", "## Charts", ""])
    if run.chart_files:
        for name in run.chart_files:
            lines.append(f"- `{name}`")
    else:
        lines.append("No chart files were written.")
    lines.extend(["", "## How to read this", ""])
    lines.append(
        "Deterministic scores come from coded checks and stay the same for the same input. "
        "Probabilistic scores come from LLM judges, or from the offline heuristic judge when no model provider is configured. "
        f"The release stance for this run is {aggregate.risk_level.value}."
    )
    if aggregate.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        lines.append("Share this report with the reviewer who approved the plan before changing the system.")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"**{value:.2f}**"
