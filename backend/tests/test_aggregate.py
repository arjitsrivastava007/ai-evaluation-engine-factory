"""Risk levels follow the weighted score, with a hard stop for a heavy failure."""

from eval_factory.domain.models import (
    EvalCase,
    EvaluationRun,
    EvaluatorResult,
    ScoreMode,
    SystemType,
)
from eval_factory.planning.planner import default_plan
from eval_factory.scoring.aggregate import aggregate_results


def _result(metric_id: str, mode: ScoreMode, score: float, threshold: float = 0.7) -> EvaluatorResult:
    return EvaluatorResult(
        case_id="c1",
        metric_id=metric_id,
        evaluator_id=metric_id,
        mode=mode,
        score=score,
        passed=score >= threshold,
        rationale="fixture",
    )


def _run_with(scores: dict[str, float]) -> EvaluationRun:
    plan = default_plan(SystemType.RAG)
    for metric in plan.metrics:
        if metric.id in scores:
            metric.threshold = 0.7
    run = EvaluationRun(
        name="demo",
        system_type=SystemType.RAG,
        metric_plan=plan,
        cases=[EvalCase(id="c1", input="question")],
        results=[
            _result(metric.id, metric.mode, scores.get(metric.id, 0.9), metric.threshold)
            for metric in plan.metrics
        ],
    )
    return run


def test_strong_scores_are_a_low_risk_ship_decision():
    report = aggregate_results(_run_with({}))
    assert report.risk_level.value == "low"
    assert report.overall >= 0.8
    assert report.decision.startswith("The system is inside")
    assert report.deterministic is not None
    assert report.probabilistic is not None


def test_a_heavy_low_score_is_critical_even_when_other_metrics_pass():
    report = aggregate_results(_run_with({"faithfulness": 0.2}))
    faithfulness = next(item for item in report.metrics if item.metric_id == "faithfulness")
    assert faithfulness.normalized_weight >= 0.15
    assert report.risk_level.value == "critical"
    assert any("regenerate" in note.lower() or "unsupported" in note.lower() for note in report.recommendations)


def test_mode_disagreement_is_called_out():
    scores = {metric.id: 0.95 for metric in default_plan(SystemType.RAG).metrics}
    for metric in default_plan(SystemType.RAG).metrics:
        if metric.mode == ScoreMode.PROBABILISTIC:
            scores[metric.id] = 0.55
    report = aggregate_results(_run_with(scores))
    assert any("differ by" in note for note in report.recommendations)
