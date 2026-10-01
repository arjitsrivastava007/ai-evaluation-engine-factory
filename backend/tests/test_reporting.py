"""A finished run writes a Markdown report and four dated chart files."""

from eval_factory.domain.models import EvalCase, EvaluationRun, EvaluatorResult, ScoreMode, SystemType
from eval_factory.planning.planner import default_plan
from eval_factory.reporting.markdown import publish_outputs


def test_publish_writes_report_and_four_charts(tmp_path):
    plan = default_plan(SystemType.RAG)
    run = EvaluationRun(
        name="Acme RAG",
        system_type=SystemType.RAG,
        metric_plan=plan,
        cases=[EvalCase(id="c1", input="What is the refund window?")],
        results=[
            EvaluatorResult(
                case_id="c1",
                metric_id=metric.id,
                evaluator_id=metric.evaluator_id,
                mode=metric.mode,
                score=0.9 if metric.mode == ScoreMode.DETERMINISTIC else 0.8,
                passed=True,
                rationale="fixture",
            )
            for metric in plan.metrics
        ],
    )
    publish_outputs(run, tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Evaluation report: Acme RAG")
    assert "Decision" in report
    assert len(run.chart_files) == 4
    for name in run.chart_files:
        html = (tmp_path / "charts" / name).read_text(encoding="utf-8")
        assert "plotly" in html.lower()
        assert name.split("-")[0] in {
            "score",
            "mode",
            "case",
        }
    assert any(name.startswith("score-by-metric-") for name in run.chart_files)
    assert any(name.startswith("mode-comparison-") for name in run.chart_files)
    assert any(name.startswith("case-risk-") for name in run.chart_files)
    assert any(name.startswith("case-heatmap-") for name in run.chart_files)
