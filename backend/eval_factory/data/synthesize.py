"""Synthesize cases and reference outputs from artifact text.

The output-generation graph is a small LangGraph: propose cases, draft the
system outputs those cases need, then validate the pair.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from eval_factory.domain.models import EvalCase, EvaluationRun, SystemOutput, SystemType
from eval_factory.errors import FactoryError
from eval_factory.providers.llm import LLMProvider

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class SynthState(TypedDict):
    system_type: str
    corpus: str
    tools: list[str]
    cases: list[dict[str, Any]]
    outputs: list[dict[str, Any]]


def synthesize_dataset(
    run: EvaluationRun,
    provider: LLMProvider,
) -> tuple[list[EvalCase], list[SystemOutput]]:
    corpus = "\n\n".join(artifact.text for artifact in run.artifacts).strip()
    if len(corpus) < 20:
        raise FactoryError("Artifacts do not contain enough text to synthesize a dataset")
    tools = list(run.analysis.tools) if run.analysis else []
    graph = build_output_generation_graph(provider)
    result = graph.invoke(
        {
            "system_type": run.system_type.value,
            "corpus": corpus,
            "tools": tools,
            "cases": [],
            "outputs": [],
        }
    )
    try:
        cases = [EvalCase.model_validate(case) for case in result["cases"]]
        outputs = [SystemOutput.model_validate(output) for output in result["outputs"]]
    except ValidationError as exc:
        raise FactoryError(f"Synthesized dataset was invalid: {exc}") from exc
    return cases, outputs


def build_output_generation_graph(provider: LLMProvider):
    graph = StateGraph(SynthState)
    graph.add_node("propose_cases", lambda state: _propose_cases(state, provider))
    graph.add_node("draft_outputs", lambda state: _draft_outputs(state, provider))
    graph.add_node("validate", _validate)
    graph.add_edge(START, "propose_cases")
    graph.add_edge("propose_cases", "draft_outputs")
    graph.add_edge("draft_outputs", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


def _propose_cases(state: SynthState, provider: LLMProvider) -> dict[str, Any]:
    if provider.name == "heuristic":
        return {"cases": _heuristic_cases(state)}
    data = provider.complete_json(
        system=(
            "You draft evaluation cases for an AI system. Return JSON "
            '{"cases": [{"id": "synth-1", "input": "", "expected": "", "context": [""], '
            '"reference_tools": []}]} . Use stable ids synth-1, synth-2, and so on. '
            "Base every case on the artifacts. Do not include outputs."
        ),
        user=json.dumps(
            {
                "system_type": state["system_type"],
                "tools": state["tools"],
                "artifacts": state["corpus"][:8000],
            }
        ),
        max_tokens=1600,
    )
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise FactoryError("The model did not return any evaluation cases")
    return {"cases": cases}


def _draft_outputs(state: SynthState, provider: LLMProvider) -> dict[str, Any]:
    if provider.name == "heuristic":
        return {"outputs": _heuristic_outputs(state)}
    data = provider.complete_json(
        system=(
            "You simulate the system under test. Return JSON "
            '{"outputs": [{"case_id": "", "response": "", "contexts": [], "tool_calls": '
            '[{"name": "", "arguments": {}}], "steps": []}]} with one output per case.'
        ),
        user=json.dumps({"system_type": state["system_type"], "cases": state["cases"]}),
        max_tokens=1800,
    )
    outputs = data.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise FactoryError("The model did not return system outputs")
    return {"outputs": outputs}


def _validate(state: SynthState) -> dict[str, Any]:
    case_ids = [str(case.get("id", "")) for case in state["cases"]]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise FactoryError("Synthesized cases need ids")
    output_ids = [str(output.get("case_id", "")) for output in state["outputs"]]
    if set(output_ids) != set(case_ids):
        raise FactoryError("Synthesized outputs do not match the generated cases")
    for output in state["outputs"]:
        if not str(output.get("response", "")).strip():
            raise FactoryError(f"Synthesized output for '{output.get('case_id')}' is empty")
    return {}


def _heuristic_cases(state: SynthState) -> list[dict[str, Any]]:
    system_type = SystemType(state["system_type"])
    passages = _passages(state["corpus"])
    tools = state["tools"][:3]
    cases = []
    for index, passage in enumerate(passages, start=1):
        sentence = _first_sentence(passage)
        if system_type is SystemType.RAG:
            case = {
                "id": f"synth-{index}",
                "input": f"What does the source say about this: {sentence}",
                "expected": sentence,
                "context": [passage],
                "reference_tools": [],
            }
        elif system_type is SystemType.CHATBOT:
            case = {
                "id": f"synth-{index}",
                "input": f"Can you help me understand this: {sentence}",
                "expected": sentence,
                "context": [],
                "reference_tools": [],
            }
        else:
            case = {
                "id": f"synth-{index}",
                "input": f"Complete this task and finish with an answer: {sentence}",
                "expected": sentence,
                "context": [passage],
                "reference_tools": tools or ["lookup"],
            }
        cases.append(case)
    return cases


def _heuristic_outputs(state: SynthState) -> list[dict[str, Any]]:
    system_type = SystemType(state["system_type"])
    outputs = []
    last = len(state["cases"]) - 1
    for index, case in enumerate(state["cases"]):
        expected = str(case.get("expected") or "")
        if system_type is SystemType.RAG:
            response = expected
            if index == last:
                response = f"{expected} This applies to every customer worldwide without exception."
            output = {
                "case_id": case["id"],
                "response": response,
                "contexts": list(case.get("context") or []),
                "tool_calls": [],
                "steps": [],
            }
        elif system_type is SystemType.CHATBOT:
            output = {
                "case_id": case["id"],
                "response": f"I can help with that. {expected}",
                "contexts": [],
                "tool_calls": [],
                "steps": [],
            }
        else:
            tool_names = list(case.get("reference_tools") or ["lookup"])
            output = {
                "case_id": case["id"],
                "response": expected,
                "contexts": list(case.get("context") or []),
                "tool_calls": [{"name": name, "arguments": {"query": case["input"]}} for name in tool_names],
                "steps": [
                    "Read the task.",
                    "Call the required tools.",
                    "Write the final answer.",
                ],
            }
        outputs.append(output)
    return outputs


def _passages(corpus: str) -> list[str]:
    chunks = []
    for chunk in re.split(r"\n\s*\n", corpus):
        cleaned = chunk.strip()
        if len(cleaned) < 40 or cleaned.startswith("# Artifact:"):
            continue
        chunks.append(cleaned)
    if len(chunks) >= 2:
        return chunks[:4]
    compact = re.sub(r"\s+", " ", corpus).strip()
    return [compact[:800]]


def _first_sentence(passage: str) -> str:
    sentence = _SENTENCE.split(re.sub(r"\s+", " ", passage).strip(), maxsplit=1)[0]
    return sentence[:400]
