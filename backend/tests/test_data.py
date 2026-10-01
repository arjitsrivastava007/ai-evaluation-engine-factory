"""Supplied datasets, the output-generation graph, and API capture."""

import json

import httpx
import pytest

from eval_factory.data.capture import assert_safe_capture_url, capture_outputs
from eval_factory.data.datasets import missing_output_ids, parse_dataset
from eval_factory.data.synthesize import synthesize_dataset
from eval_factory.domain.models import (
    ArtifactKind,
    ArtifactRecord,
    CaptureSpec,
    EvaluationRun,
    SystemType,
)
from eval_factory.errors import FactoryError
from eval_factory.providers.llm import HeuristicProvider, OpenAIProvider


def _run(system_type: SystemType, text: str) -> EvaluationRun:
    return EvaluationRun(
        name="demo",
        system_type=system_type,
        artifacts=[
            ArtifactRecord(
                filename="source.md",
                kind=ArtifactKind.MARKDOWN,
                media_type="text/markdown",
                text=text,
                stored_path="source.md",
            )
        ],
    )


def test_dataset_parser_accepts_inline_outputs_and_rejects_duplicates():
    cases, outputs = parse_dataset(
        {
            "cases": [
                {
                    "id": "c1",
                    "input": "What is the refund window?",
                    "expected": "30 days",
                    "context": ["Refunds are accepted within 30 days."],
                    "output": {
                        "response": "Refunds are accepted within 30 days. [1]",
                        "contexts": ["Refunds are accepted within 30 days."],
                        "tool_calls": [{"name": "search_policies", "arguments": {}}],
                    },
                },
                {"input": "Second question with no id"},
            ]
        }
    )
    assert cases[1].id == "case-2"
    assert outputs[0].tool_calls[0].name == "search_policies"
    assert missing_output_ids(cases, outputs) == ["case-2"]
    with pytest.raises(FactoryError, match="duplicated"):
        parse_dataset([{"id": "c1", "input": "a"}, {"id": "c1", "input": "b"}])


def test_output_generation_graph_is_stable_for_rag():
    text = (
        "Refunds are accepted within 30 days of purchase from the original store.\n\n"
        "Shipping labels are issued after the warehouse confirms the item is in stock.\n\n"
        "Agents must cite the policy chunk they used in the answer."
    )
    run = _run(SystemType.RAG, text)
    first_cases, first_outputs = synthesize_dataset(run, HeuristicProvider())
    second_cases, second_outputs = synthesize_dataset(run, HeuristicProvider())
    assert [case.id for case in first_cases] == ["synth-1", "synth-2", "synth-3"]
    assert first_cases[-1].context
    assert "worldwide" in first_outputs[-1].response
    assert "worldwide" not in first_outputs[0].response
    assert [case.model_dump() for case in first_cases] == [case.model_dump() for case in second_cases]
    assert [output.model_dump() for output in first_outputs] == [
        output.model_dump() for output in second_outputs
    ]


def test_agent_synthesis_records_reference_tools():
    run = _run(
        SystemType.AGENTIC,
        "Lookup the claim and notify the adjuster before writing the decision letter to the customer.",
    )
    run.analysis = None
    cases, outputs = synthesize_dataset(run, HeuristicProvider())
    assert cases[0].reference_tools == ["lookup"]
    assert outputs[0].tool_calls[0].name == "lookup"
    assert len(outputs[0].steps) == 3


def test_model_synthesis_walks_propose_then_draft():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        system = json.loads(request.content)["messages"][0]["content"]
        if "draft evaluation cases" in system:
            calls.append("propose")
            payload = {"cases": [{"id": "synth-1", "input": "Question?", "expected": "Answer", "context": ["Answer"]}]}
        else:
            calls.append("draft")
            payload = {"outputs": [{"case_id": "synth-1", "response": "Answer", "tool_calls": [], "steps": []}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

    provider = OpenAIProvider(
        api_key="test",
        model="gpt-4o-mini",
        base_url="https://example.test/v1",
        temperature=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    cases, outputs = synthesize_dataset(_run(SystemType.CHATBOT, "The bot explains invoices in plain language."), provider)
    assert calls == ["propose", "draft"]
    assert cases[0].input == "Question?"
    assert outputs[0].response == "Answer"


def test_capture_reads_json_and_blocks_metadata_addresses():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == "Where is my order?"
        return httpx.Response(
            200,
            json={
                "data": {"answer": "It ships tomorrow."},
                "tool_calls": [{"name": "lookup_order", "arguments": {"id": "1"}}],
                "steps": ["lookup", "reply"],
            },
        )

    cases, _ = parse_dataset([{"id": "c1", "input": "Where is my order?"}])
    spec = CaptureSpec(url="http://127.0.0.1:9/run", response_path="data.answer")
    outputs = capture_outputs(
        cases,
        spec,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert outputs[0].response == "It ships tomorrow."
    assert outputs[0].tool_calls[0].name == "lookup_order"
    with pytest.raises(FactoryError, match="blocked"):
        assert_safe_capture_url("http://169.254.169.254/latest/meta-data")
    with pytest.raises(FactoryError, match="blocked"):
        assert_safe_capture_url("http://metadata.google.internal/computeMetadata/v1/")
