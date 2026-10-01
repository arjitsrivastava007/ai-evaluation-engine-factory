"""Load a supplied dataset of cases and optional system outputs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from eval_factory.domain.models import EvalCase, SystemOutput, ToolCall
from eval_factory.errors import FactoryError


def parse_dataset(payload: Any) -> tuple[list[EvalCase], list[SystemOutput]]:
    if isinstance(payload, dict) and "cases" in payload:
        raw_cases = payload["cases"]
    elif isinstance(payload, list):
        raw_cases = payload
    else:
        raise FactoryError("Dataset must be a list of cases or an object with a cases array")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise FactoryError("Dataset must contain at least one case")

    cases: list[EvalCase] = []
    outputs: list[SystemOutput] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise FactoryError(f"Case {index} must be an object")
        case_payload = {key: value for key, value in raw.items() if key != "output"}
        case_payload.setdefault("id", f"case-{index}")
        try:
            case = EvalCase.model_validate(case_payload)
        except ValidationError as exc:
            raise FactoryError(f"Case {index} is invalid: {exc}") from exc
        if not case.input.strip():
            raise FactoryError(f"Case '{case.id}' is missing input")
        if case.id in seen:
            raise FactoryError(f"Case id '{case.id}' is duplicated")
        seen.add(case.id)
        cases.append(case)
        if "output" in raw and raw["output"] is not None:
            outputs.append(_parse_output(case.id, raw["output"]))
    return cases, outputs


def missing_output_ids(cases: list[EvalCase], outputs: list[SystemOutput]) -> list[str]:
    present = {output.case_id for output in outputs}
    return [case.id for case in cases if case.id not in present]


def _parse_output(case_id: str, raw: Any) -> SystemOutput:
    if isinstance(raw, str):
        return SystemOutput(case_id=case_id, response=raw)
    if not isinstance(raw, dict):
        raise FactoryError(f"Output for case '{case_id}' must be a string or an object")
    payload = dict(raw)
    payload["case_id"] = case_id
    payload.setdefault("response", "")
    try:
        output = SystemOutput.model_validate(payload)
    except ValidationError as exc:
        raise FactoryError(f"Output for case '{case_id}' is invalid: {exc}") from exc
    if not isinstance(output.tool_calls, list):
        raise FactoryError(f"Output for case '{case_id}' has invalid tool calls")
    return output


def output_from_payload(
    case_id: str,
    response: str,
    *,
    contexts: list[str] | None = None,
    tool_calls: list[dict] | None = None,
    steps: list[str] | None = None,
) -> SystemOutput:
    calls = [ToolCall.model_validate(call) for call in tool_calls or []]
    return SystemOutput(
        case_id=case_id,
        response=response,
        contexts=contexts or [],
        tool_calls=calls,
        steps=steps or [],
    )
