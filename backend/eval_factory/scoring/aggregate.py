"""Turn per-case scores into one decision."""

from __future__ import annotations

from collections import defaultdict

from eval_factory.domain.models import (
    AggregateReport,
    EvaluationRun,
    MetricScore,
    MetricSpec,
    RiskLevel,
    ScoreMode,
)
from eval_factory.domain.ontology import recommendation_for
from eval_factory.errors import FactoryError
from eval_factory.textutil import clamp_score

_DECISIONS = {
    RiskLevel.LOW: "The system is inside the configured quality bar. Ship it with routine monitoring.",
    RiskLevel.MEDIUM: "The system is usable, with gaps to close. Revise the failed metrics before a broad release.",
    RiskLevel.HIGH: "Do not treat this build as decision-ready. Fix the high-severity gaps and run the evaluation again.",
    RiskLevel.CRITICAL: "Block release. Critical failures need a design, data, or prompt change before another run.",
}


def aggregate_results(run: EvaluationRun) -> AggregateReport:
    if run.metric_plan is None:
        raise FactoryError("A metric plan is required before scores can be aggregated")
    enabled = {metric.id: metric for metric in run.metric_plan.metrics if metric.enabled}
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in run.results:
        if result.applicable and result.metric_id in enabled:
            grouped[result.metric_id].append(result.score)

    raw: list[tuple[MetricSpec, float]] = []
    for metric_id, metric in enabled.items():
        scores = grouped.get(metric_id)
        if not scores:
            continue
        raw.append((metric, sum(scores) / len(scores)))

    weight_total = sum(metric.weight for metric, _score in raw)
    metric_scores: list[MetricScore] = []
    for metric, score in raw:
        normalized = (metric.weight / weight_total) if weight_total else 0.0
        bounded = clamp_score(score)
        metric_scores.append(
            MetricScore(
                metric_id=metric.id,
                name=metric.name,
                mode=metric.mode,
                score=bounded,
                passed=bounded >= metric.threshold,
                weight=metric.weight,
                normalized_weight=round(normalized, 4),
            )
        )

    deterministic = _weighted([item for item in metric_scores if item.mode == ScoreMode.DETERMINISTIC])
    probabilistic = _weighted([item for item in metric_scores if item.mode == ScoreMode.PROBABILISTIC])
    overall = _weighted(metric_scores)
    overall_value = 0.0 if overall is None else overall
    risk = _risk_level(overall_value, metric_scores)
    recommendations = _recommendations(run, metric_scores, deterministic, probabilistic, risk)
    return AggregateReport(
        overall=overall_value,
        deterministic=deterministic,
        probabilistic=probabilistic,
        risk_level=risk,
        metrics=metric_scores,
        recommendations=recommendations,
        case_count=len(run.cases),
        decision=_DECISIONS[risk],
    )


def _weighted(items: list[MetricScore]) -> float | None:
    total = sum(item.weight for item in items)
    if not items or total <= 0:
        return None
    return clamp_score(sum(item.score * item.weight for item in items) / total)


def _risk_level(overall: float, metrics: list[MetricScore]) -> RiskLevel:
    if not metrics:
        return RiskLevel.CRITICAL
    heavy_failure = any(item.normalized_weight >= 0.15 and item.score < 0.35 for item in metrics)
    if overall < 0.45 or heavy_failure:
        return RiskLevel.CRITICAL
    if overall < 0.65:
        return RiskLevel.HIGH
    if overall < 0.8:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _recommendations(run, metrics, deterministic, probabilistic, risk) -> list[str]:
    notes: list[str] = []
    failed = [item for item in metrics if not item.passed]
    if not failed:
        notes.append("Every enabled metric cleared its threshold.")
    for item in failed:
        if item.metric_id.startswith("bundle:"):
            notes.append(
                f"Revise responses that fail the '{item.name}' rule bundle, or edit the rules if the contract changed."
            )
            continue
        if item.metric_id.startswith("rubric:"):
            notes.append(f"Review '{item.name}' failures against the rubric before accepting the judge score.")
            continue
        template = recommendation_for(run.system_type, item.metric_id)
        notes.append(template or f"Improve '{item.name}' and run the evaluation again.")
    if deterministic is not None and probabilistic is not None and abs(deterministic - probabilistic) >= 0.2:
        gap = abs(deterministic - probabilistic)
        notes.append(
            f"Deterministic and probabilistic scores differ by {gap:.2f}. "
            "Inspect cases where the rules and the judge disagree before treating the overall score as final."
        )
    notes.append(_DECISIONS[risk])
    return notes
