"""The guided flow pauses for review, then scores and publishes."""

import pytest

from eval_factory.config import Settings
from eval_factory.domain.models import RunPhase, SystemType
from eval_factory.errors import FactoryError
from eval_factory.storage.store import RunStore
from eval_factory.workflow.service import EvaluationService

POLICY = b"""
# Acme Policy Retriever

The generator answers only from retrieved context and must cite the chunk id.
Refunds are accepted within 30 days of purchase.
The index is updated nightly.

Tools: search_policies
"""


def _service(tmp_path) -> EvaluationService:
    settings = Settings(data_dir=tmp_path / "data")
    return EvaluationService(RunStore(settings.data_dir), settings)


def test_review_checkpoints_block_scoring_until_both_are_approved(tmp_path):
    service = _service(tmp_path)
    run = service.create_run("Acme RAG", SystemType.RAG)
    with pytest.raises(FactoryError, match="artifact"):
        service.analyze(run.id)
    run = service.add_artifact(run.id, "policy.md", POLICY)
    run = service.analyze(run.id)
    assert run.phase is RunPhase.AWAITING_ANALYSIS_REVIEW
    assert run.metric_plan is None
    assert run.analysis is not None
    assert "search_policies" in run.analysis.tools

    run.analysis.summary = "Reviewed: a retriever grounds the generator."
    run = service.update_analysis(run.id, run.analysis)
    assert run.analysis is not None and run.analysis.revised is True

    with pytest.raises(FactoryError, match="plan"):
        service.approve_plan(run.id)

    run = service.approve_analysis(run.id)
    assert run.phase is RunPhase.AWAITING_PLAN_REVIEW
    assert run.metric_plan is not None
    assert run.results == []

    with pytest.raises(FactoryError, match="dataset"):
        service.approve_plan(run.id)

    run = service.set_dataset(
        run.id,
        {
            "cases": [
                {
                    "id": "refund",
                    "input": "What is the refund window?",
                    "expected": "30 days",
                    "context": ["Refunds are accepted within 30 days of purchase."],
                    "output": {
                        "response": "Refunds are accepted within 30 days of purchase. [1]",
                        "contexts": ["Refunds are accepted within 30 days of purchase."],
                    },
                }
            ]
        },
    )
    run = service.approve_plan(run.id)
    assert run.phase is RunPhase.COMPLETED
    assert run.aggregate is not None
    assert run.aggregate.overall > 0
    assert len(run.chart_files) == 4
    assert run.report_markdown is not None
    assert "Acme RAG" in run.report_markdown
    stored = service.store.get(run.id)
    assert stored.phase is RunPhase.COMPLETED


def test_synthesis_can_supply_the_dataset(tmp_path):
    service = _service(tmp_path)
    run = service.create_run("Synthesized", SystemType.CHATBOT)
    run = service.add_artifact(
        run.id,
        "bot.md",
        b"The billing assistant answers invoice questions in a calm tone and a professional voice.\n\n"
        b"It tells the user the next step when a payment failed and how to retry it safely.",
    )
    run = service.analyze(run.id)
    run = service.approve_analysis(run.id)
    run = service.synthesize(run.id)
    assert run.cases
    assert len(run.outputs) == len(run.cases)
    run = service.approve_plan(run.id)
    assert run.phase is RunPhase.COMPLETED
    assert run.data_strategy == "synthesize"
