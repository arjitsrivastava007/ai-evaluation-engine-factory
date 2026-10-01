"""Build and update the metric plan reviewers edit before a run."""

from __future__ import annotations

from eval_factory.domain.models import (
    LLMRubric,
    MetricPlan,
    MetricSpec,
    RuleBundle,
    RuleKind,
    ScoreMode,
    SystemType,
)
from eval_factory.domain.ontology import KNOWN_EVALUATORS, profile_for
from eval_factory.errors import FactoryError


def default_plan(system_type: SystemType) -> MetricPlan:
    profile = profile_for(system_type)
    metrics = [
        MetricSpec(
            id=template.id,
            name=template.name,
            description=template.description,
            weight=template.weight,
            mode=template.mode,
            evaluator_id=template.evaluator_id,
            threshold=template.threshold,
        )
        for template in profile.metrics
    ]
    return MetricPlan(system_type=system_type, metrics=metrics, notes="")


def attach_bundle(plan: MetricPlan, bundle: RuleBundle) -> MetricPlan:
    metric_id = f"bundle:{bundle.id}"
    if any(metric.id == metric_id for metric in plan.metrics):
        return plan
    plan.metrics.append(
        MetricSpec(
            id=metric_id,
            name=bundle.name,
            description=bundle.description or "Custom deterministic rule bundle.",
            weight=0.1,
            mode=ScoreMode.DETERMINISTIC,
            evaluator_id="rule_bundle",
            threshold=0.7,
            config={"bundle_id": bundle.id},
        )
    )
    return plan


def attach_rubric(plan: MetricPlan, rubric: LLMRubric) -> MetricPlan:
    metric_id = f"rubric:{rubric.id}"
    if any(metric.id == metric_id for metric in plan.metrics):
        return plan
    plan.metrics.append(
        MetricSpec(
            id=metric_id,
            name=rubric.name,
            description=rubric.prompt,
            weight=0.1,
            mode=ScoreMode.PROBABILISTIC,
            evaluator_id="llm_rubric",
            threshold=0.7,
            config={"rubric_id": rubric.id},
        )
    )
    return plan


def validate_plan(plan: MetricPlan, bundles: list[RuleBundle], rubrics: list[LLMRubric]) -> None:
    if not plan.metrics:
        raise FactoryError("The metric plan needs at least one metric")
    seen: set[str] = set()
    enabled = 0
    bundle_ids = {bundle.id for bundle in bundles}
    rubric_ids = {rubric.id for rubric in rubrics}
    for metric in plan.metrics:
        if metric.id in seen:
            raise FactoryError(f"Metric id '{metric.id}' is duplicated")
        seen.add(metric.id)
        if metric.evaluator_id not in KNOWN_EVALUATORS:
            raise FactoryError(f"Unknown evaluator '{metric.evaluator_id}' on metric '{metric.id}'")
        if metric.enabled:
            enabled += 1
            if metric.weight <= 0:
                raise FactoryError(f"Enabled metric '{metric.name}' needs a weight above zero")
        if metric.evaluator_id == "rule_bundle":
            bundle_id = str(metric.config.get("bundle_id", ""))
            if bundle_id not in bundle_ids:
                raise FactoryError(f"Metric '{metric.name}' points at a missing rule bundle")
        if metric.evaluator_id == "llm_rubric":
            rubric_id = str(metric.config.get("rubric_id", ""))
            if rubric_id not in rubric_ids:
                raise FactoryError(f"Metric '{metric.name}' points at a missing rubric")
    if enabled == 0:
        raise FactoryError("Enable at least one metric before running evaluation")


def validate_bundle(bundle: RuleBundle) -> None:
    if not bundle.name.strip():
        raise FactoryError("A rule bundle needs a name")
    if not bundle.rules:
        raise FactoryError(f"Rule bundle '{bundle.name}' needs at least one rule")
    for rule in bundle.rules:
        _validate_rule(bundle.name, rule.kind, rule.params)


def validate_rubric(rubric: LLMRubric) -> None:
    if not rubric.name.strip() or not rubric.prompt.strip():
        raise FactoryError("A rubric needs a name and a prompt")
    if not rubric.criteria:
        raise FactoryError(f"Rubric '{rubric.name}' needs at least one criterion")
    for criterion in rubric.criteria:
        if not criterion.name.strip() or not criterion.description.strip():
            raise FactoryError(f"Rubric '{rubric.name}' has a criterion without a name or description")


def _validate_rule(bundle_name: str, kind: RuleKind, params: dict) -> None:
    if kind in {RuleKind.CONTAINS, RuleKind.NOT_CONTAINS} and not str(params.get("value", "")).strip():
        raise FactoryError(f"Bundle '{bundle_name}' has a {kind.value} rule without text")
    if kind is RuleKind.REGEX and not str(params.get("pattern", "")).strip():
        raise FactoryError(f"Bundle '{bundle_name}' has a regex rule without a pattern")
    if kind in {RuleKind.MIN_LENGTH, RuleKind.MAX_LENGTH}:
        try:
            limit = int(params.get("value"))
        except (TypeError, ValueError) as exc:
            raise FactoryError(f"Bundle '{bundle_name}' has a length rule without an integer") from exc
        if limit < 0:
            raise FactoryError(f"Bundle '{bundle_name}' has a negative length limit")
    if kind is RuleKind.REQUIRED_TOOLS:
        tools = params.get("tools")
        if not isinstance(tools, list) or not tools:
            raise FactoryError(f"Bundle '{bundle_name}' has a required-tools rule without tools")
