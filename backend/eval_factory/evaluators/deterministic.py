"""Coded evaluators. The same inputs always produce the same scores."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from eval_factory.domain.models import (
    EvalCase,
    EvaluationRun,
    EvaluatorResult,
    MetricSpec,
    Rule,
    RuleKind,
    ScoreMode,
    SystemOutput,
)
from eval_factory.errors import FactoryError
from eval_factory.evaluators.signals import CITATION, safety_hits
from eval_factory.textutil import clamp_score, content_tokens, recall

EvaluatorFn = Callable[[MetricSpec, EvalCase, SystemOutput, EvaluationRun], EvaluatorResult]


def run_deterministic(
    run: EvaluationRun,
    *,
    registry: dict[str, EvaluatorFn] | None = None,
    max_workers: int = 8,
) -> list[EvaluatorResult]:
    if run.metric_plan is None:
        raise FactoryError("A metric plan is required before evaluation")
    selected = registry or DETERMINISTIC_REGISTRY
    metrics = [
        metric
        for metric in run.metric_plan.metrics
        if metric.enabled and metric.mode == ScoreMode.DETERMINISTIC
    ]
    pairs = list(_pairs(run))
    tasks = [(metric, case, output) for case, output in pairs for metric in metrics]
    if not tasks:
        return []
    workers = max(1, min(max_workers, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda item: _invoke(selected, *item, run), tasks))
    results.sort(key=lambda result: (result.case_id, result.metric_id))
    return results


def _invoke(
    registry: dict[str, EvaluatorFn],
    metric: MetricSpec,
    case: EvalCase,
    output: SystemOutput,
    run: EvaluationRun,
) -> EvaluatorResult:
    function = registry.get(metric.evaluator_id)
    if function is None:
        raise FactoryError(f"No deterministic evaluator registered for '{metric.evaluator_id}'")
    return function(metric, case, output, run)


def _pairs(run: EvaluationRun):
    outputs = {output.case_id: output for output in run.outputs}
    for case in run.cases:
        yield case, outputs.get(case.id) or SystemOutput(case_id=case.id, response="")


def _result(
    metric: MetricSpec,
    case: EvalCase,
    score: float,
    rationale: str,
    *,
    applicable: bool = True,
    evidence: dict | None = None,
) -> EvaluatorResult:
    bounded = clamp_score(score)
    return EvaluatorResult(
        case_id=case.id,
        metric_id=metric.id,
        evaluator_id=metric.evaluator_id,
        mode=metric.mode,
        score=bounded,
        passed=applicable and bounded >= metric.threshold,
        rationale=rationale,
        applicable=applicable,
        evidence=evidence or {},
    )


def _contexts(case: EvalCase, output: SystemOutput) -> list[str]:
    return output.contexts or case.context


def score_context_overlap(metric, case, output, run) -> EvaluatorResult:
    del run
    answer = content_tokens(output.response)
    context = content_tokens(" ".join(_contexts(case, output)))
    if not answer:
        return _result(metric, case, 0, "The response has no content words to compare with the context.")
    if not context:
        return _result(metric, case, 0, "No retrieved context was supplied.")
    score = recall(answer, context)
    return _result(
        metric,
        case,
        score,
        f"{score:.0%} of the answer's content words appear in the retrieved context.",
        evidence={"answer_terms": len(answer), "supported_terms": round(score * len(set(answer)))},
    )


def score_citation_support(metric, case, output, run) -> EvaluatorResult:
    del run
    if CITATION.search(output.response):
        return _result(metric, case, 1, "The response includes a citation marker.")
    return _result(metric, case, 0, "The response does not cite a source.")


def score_answer_relevance(metric, case, output, run) -> EvaluatorResult:
    del run
    question = content_tokens(case.input)
    answer = content_tokens(output.response)
    expected = content_tokens(case.expected or "")
    if not answer:
        return _result(metric, case, 0, "The response is empty.")
    question_score = recall(question, answer) if question else 0.0
    if expected:
        score = (question_score + recall(expected, answer)) / 2
        rationale = "Relevance blends overlap with the question and the expected answer."
    else:
        score = question_score
        rationale = "Relevance is the share of question terms that appear in the response."
    return _result(metric, case, score, rationale)


def score_expected_coverage(metric, case, output, run) -> EvaluatorResult:
    del run
    if not case.expected:
        return _result(
            metric,
            case,
            1,
            "No expected answer was labeled, so this check does not apply.",
            applicable=False,
        )
    expected = content_tokens(case.expected)
    if not expected:
        return _result(metric, case, 1, "The expected answer has no content words.", applicable=False)
    score = recall(expected, content_tokens(output.response))
    return _result(metric, case, score, f"{score:.0%} of the expected content words appear in the response.")


def score_safety_screen(metric, case, output, run) -> EvaluatorResult:
    del run
    hits = safety_hits(output.response)
    if hits:
        return _result(metric, case, 0, "Safety screen matched: " + ", ".join(hits) + ".", evidence={"hits": hits})
    return _result(metric, case, 1, "No personal-data pattern or profanity was found.")


def score_format_bounds(metric, case, output, run) -> EvaluatorResult:
    del run
    length = len(output.response.strip())
    minimum = int(metric.config.get("min_length", 20))
    maximum = int(metric.config.get("max_length", 4000))
    if length == 0:
        return _result(metric, case, 0, "The response is empty.")
    if length < minimum:
        return _result(metric, case, 0.4, f"The response is shorter than {minimum} characters.")
    if length > maximum:
        return _result(metric, case, 0.5, f"The response is longer than {maximum} characters.")
    return _result(metric, case, 1, f"The response length ({length}) is inside the configured bounds.")


def score_instruction_adherence(metric, case, output, run) -> EvaluatorResult:
    del run
    phrases = [str(phrase) for phrase in case.metadata.get("required_phrases") or [] if str(phrase).strip()]
    response = output.response.lower()
    if phrases:
        hits = sum(1 for phrase in phrases if phrase.lower() in response)
        score = hits / len(phrases)
        return _result(metric, case, score, f"{hits} of {len(phrases)} required phrases are present.")
    if case.expected:
        score = recall(content_tokens(case.expected), content_tokens(output.response))
        return _result(metric, case, score, "Adherence is the share of expected words present in the response.")
    return _result(
        metric,
        case,
        1,
        "No required phrases or expected answer were provided.",
        applicable=False,
    )


def score_consistency(metric, case, output, run) -> EvaluatorResult:
    del run
    lowered = output.response.lower()
    if re.search(r"\balways\b", lowered) and re.search(r"\bnever\b", lowered):
        return _result(metric, case, 0.4, "The response says both always and never.")
    positives = {tail.strip() for negated, tail in re.findall(r"\bit is (not )?([a-z0-9 ]{3,40})", lowered) if not negated}
    negatives = {tail.strip() for negated, tail in re.findall(r"\bit is (not )?([a-z0-9 ]{3,40})", lowered) if negated}
    if positives & negatives:
        return _result(metric, case, 0.3, "The response both asserts and denies the same claim.")
    return _result(metric, case, 1, "No direct contradiction was found.")


def score_tool_selection(metric, case, output, run) -> EvaluatorResult:
    del run
    reference = [tool for tool in case.reference_tools if tool]
    if not reference:
        return _result(metric, case, 1, "No reference tools were labeled.", applicable=False)
    used = {call.name for call in output.tool_calls}
    score = len(set(reference) & used) / len(set(reference))
    missing = sorted(set(reference) - used)
    rationale = "All reference tools were called." if not missing else "Missing tools: " + ", ".join(missing) + "."
    return _result(metric, case, score, rationale, evidence={"missing": missing})


def score_tool_schema(metric, case, output, run) -> EvaluatorResult:
    del run
    if not output.tool_calls:
        if case.reference_tools:
            return _result(metric, case, 0, "The case expects tool calls, but none were recorded.")
        return _result(metric, case, 1, "No tool calls were expected.", applicable=False)
    valid = sum(1 for call in output.tool_calls if call.name.strip() and isinstance(call.arguments, dict))
    score = valid / len(output.tool_calls)
    return _result(metric, case, score, f"{valid} of {len(output.tool_calls)} tool calls have a name and arguments.")


def score_step_budget(metric, case, output, run) -> EvaluatorResult:
    del run
    limit = int(metric.config.get("max_steps", 8))
    count = len(output.steps) or len(output.tool_calls)
    if count == 0:
        return _result(metric, case, 0.5, "No steps were recorded.")
    if count <= limit:
        return _result(metric, case, 1, f"{count} steps are within the budget of {limit}.")
    score = max(0.0, 1 - (count - limit) / limit)
    return _result(metric, case, score, f"{count} steps exceed the budget of {limit}.")


def score_final_answer(metric, case, output, run) -> EvaluatorResult:
    del run
    if output.response.strip():
        return _result(metric, case, 1, "The trace includes a final answer.")
    return _result(metric, case, 0, "The trace has no final answer.")


def score_rule_bundle(metric, case, output, run) -> EvaluatorResult:
    bundle_id = str(metric.config.get("bundle_id", ""))
    bundle = next((item for item in run.rule_bundles if item.id == bundle_id), None)
    if bundle is None or not bundle.rules:
        return _result(metric, case, 0, "The rule bundle for this metric is missing.")
    weighted = 0.0
    weight_total = 0.0
    notes: list[str] = []
    for rule in bundle.rules:
        passed, note = _apply_rule(rule, case, output)
        weight = rule.weight if rule.weight > 0 else 1.0
        weighted += (1.0 if passed else 0.0) * weight
        weight_total += weight
        notes.append(f"{rule.name}: {note}")
    score = weighted / weight_total if weight_total else 0.0
    return _result(metric, case, score, " ".join(notes))


def _apply_rule(rule: Rule, case: EvalCase, output: SystemOutput) -> tuple[bool, str]:
    response = output.response
    if rule.kind is RuleKind.CONTAINS:
        value = str(rule.params.get("value", ""))
        ok = value.lower() in response.lower()
        return ok, f"contains '{value}'" if ok else f"missing '{value}'"
    if rule.kind is RuleKind.NOT_CONTAINS:
        value = str(rule.params.get("value", ""))
        ok = value.lower() not in response.lower()
        return ok, f"does not contain '{value}'" if ok else f"contains forbidden '{value}'"
    if rule.kind is RuleKind.REGEX:
        pattern = str(rule.params.get("pattern", ""))
        try:
            ok = re.search(pattern, response) is not None
        except re.error as exc:
            raise FactoryError(f"Rule '{rule.name}' has an invalid regex: {exc}") from exc
        return ok, "regex matched" if ok else "regex did not match"
    if rule.kind is RuleKind.MIN_LENGTH:
        limit = int(rule.params.get("value", 0))
        ok = len(response.strip()) >= limit
        return ok, f"length >= {limit}" if ok else f"shorter than {limit}"
    if rule.kind is RuleKind.MAX_LENGTH:
        limit = int(rule.params.get("value", 0))
        ok = len(response.strip()) <= limit
        return ok, f"length <= {limit}" if ok else f"longer than {limit}"
    if rule.kind is RuleKind.JSON_OBJECT:
        ok = _is_json_object(response)
        return ok, "valid JSON object" if ok else "not a JSON object"
    if rule.kind is RuleKind.REQUIRED_TOOLS:
        required = {str(tool) for tool in rule.params.get("tools") or []}
        used = {call.name for call in output.tool_calls}
        ok = required <= used
        return ok, "required tools called" if ok else "missing " + ", ".join(sorted(required - used))
    if rule.kind is RuleKind.CITATION_REQUIRED:
        ok = CITATION.search(response) is not None
        return ok, "citation present" if ok else "citation missing"
    raise FactoryError(f"Unsupported rule kind '{rule.kind}'")


def _is_json_object(response: str) -> bool:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


DETERMINISTIC_REGISTRY: dict[str, EvaluatorFn] = {
    "context_overlap": score_context_overlap,
    "citation_support": score_citation_support,
    "answer_relevance": score_answer_relevance,
    "expected_coverage": score_expected_coverage,
    "safety_screen": score_safety_screen,
    "format_bounds": score_format_bounds,
    "instruction_adherence": score_instruction_adherence,
    "consistency": score_consistency,
    "tool_selection": score_tool_selection,
    "tool_schema": score_tool_schema,
    "step_budget": score_step_budget,
    "final_answer": score_final_answer,
    "rule_bundle": score_rule_bundle,
}
