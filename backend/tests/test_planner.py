"""The planner proposes a weighted plan and keeps custom checks attached to it."""

import pytest

from eval_factory.domain.models import LLMRubric, RubricCriterion, Rule, RuleBundle, RuleKind, SystemType
from eval_factory.domain.ontology import profile_for
from eval_factory.errors import FactoryError
from eval_factory.planning.planner import (
    attach_bundle,
    attach_rubric,
    default_plan,
    validate_bundle,
    validate_plan,
    validate_rubric,
)


def test_default_plan_copies_the_ontology():
    plan = default_plan(SystemType.RAG)
    profile = profile_for(SystemType.RAG)
    assert [metric.id for metric in plan.metrics] == [metric.id for metric in profile.metrics]
    assert all(metric.enabled for metric in plan.metrics)
    validate_plan(plan, [], [])


def test_custom_bundle_and_rubric_become_metrics():
    plan = default_plan(SystemType.CHATBOT)
    bundle = RuleBundle(
        name="Must mention the ticket id",
        rules=[Rule(name="ticket", kind=RuleKind.CONTAINS, params={"value": "ticket"})],
    )
    rubric = LLMRubric(
        name="Empathy",
        prompt="Reward a reply that acknowledges the problem before the fix.",
        criteria=[RubricCriterion(name="Acknowledgement", description="Names the user's problem.")],
    )
    validate_bundle(bundle)
    validate_rubric(rubric)
    attach_bundle(plan, bundle)
    attach_rubric(plan, rubric)
    attach_bundle(plan, bundle)
    assert sum(metric.id == f"bundle:{bundle.id}" for metric in plan.metrics) == 1
    assert any(metric.evaluator_id == "llm_rubric" for metric in plan.metrics)
    validate_plan(plan, [bundle], [rubric])


def test_plan_rejects_unknown_evaluators_and_empty_enablement():
    plan = default_plan(SystemType.AGENTIC)
    plan.metrics[0].evaluator_id = "made_up"
    with pytest.raises(FactoryError, match="Unknown evaluator"):
        validate_plan(plan, [], [])
    plan = default_plan(SystemType.AGENTIC)
    for metric in plan.metrics:
        metric.enabled = False
    with pytest.raises(FactoryError, match="at least one"):
        validate_plan(plan, [], [])


def test_bundle_rules_require_their_parameters():
    bundle = RuleBundle(name="Lengths", rules=[Rule(name="min", kind=RuleKind.MIN_LENGTH, params={})])
    with pytest.raises(FactoryError, match="integer"):
        validate_bundle(bundle)
