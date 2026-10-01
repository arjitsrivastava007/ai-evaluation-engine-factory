"""The analyzer proposes an architecture and risks a reviewer can edit."""

import json

import httpx

from eval_factory.analysis.analyzer import analyze_heuristically, analyze_solution
from eval_factory.domain.models import (
    ArtifactKind,
    ArtifactRecord,
    EvaluationRun,
    ProviderConfig,
    RiskLevel,
    SystemType,
)
from eval_factory.providers.llm import OpenAIProvider


def _artifact(name: str, text: str) -> ArtifactRecord:
    return ArtifactRecord(
        filename=name,
        kind=ArtifactKind.MARKDOWN,
        media_type="text/markdown",
        text=text,
        stored_path=name,
    )


RAG_DOC = """
# Acme Policy Retriever

The system embeds questions and searches a policy knowledge base.
The generator answers only from retrieved context and must cite the chunk id.

## Refresh job

The index is updated nightly.

Tools: search_policies
"""

CHAT_DOC = """
# Billing assistant

The bot answers invoice questions in a calm tone.
It does not describe a filter for harmful replies.
"""

AGENT_DOC = """
# Claims agent

The planner calls tools and should stop at a step limit of 6.
Mutations require approval.

Tools:
- lookup_claim
- notify_adjuster
"""


def test_rag_analysis_reads_grounding_citations_and_tools():
    analysis = analyze_heuristically(SystemType.RAG, [_artifact("policy.md", RAG_DOC)])
    by_id = {risk.id: risk for risk in analysis.risks}
    assert by_id["citation_gap"].severity is RiskLevel.LOW
    assert by_id["hallucination"].severity is RiskLevel.MEDIUM
    assert by_id["stale_knowledge"].severity is RiskLevel.LOW
    assert analysis.tools == ["search_policies"]
    assert any(component.name == "Retriever" for component in analysis.components)
    assert "Highest risks" in analysis.summary


def test_chatbot_without_a_safety_control_is_flagged():
    analysis = analyze_heuristically(SystemType.CHATBOT, [_artifact("bot.md", CHAT_DOC)])
    by_id = {risk.id: risk for risk in analysis.risks}
    assert by_id["unsafe_content"].severity is RiskLevel.HIGH
    assert by_id["persona_drift"].severity is RiskLevel.LOW
    assert "Not observed" in by_id["unsafe_content"].evidence


def test_agent_tools_and_step_budget_lower_the_matching_risks():
    analysis = analyze_heuristically(SystemType.AGENTIC, [_artifact("agent.md", AGENT_DOC)])
    by_id = {risk.id: risk for risk in analysis.risks}
    assert analysis.tools == ["lookup_claim", "notify_adjuster"]
    assert by_id["wrong_tool"].severity is RiskLevel.MEDIUM
    assert by_id["step_runaway"].severity is RiskLevel.LOW
    assert by_id["unauthorized_action"].severity is RiskLevel.MEDIUM


def test_heuristic_analysis_is_repeatable():
    first = analyze_heuristically(SystemType.RAG, [_artifact("policy.md", RAG_DOC)])
    second = analyze_heuristically(SystemType.RAG, [_artifact("policy.md", RAG_DOC)])
    assert first.model_dump() == second.model_dump()


def test_model_analyzer_uses_the_provider_json():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "Solution Analyzer" in body["messages"][0]["content"]
        payload = {
            "summary": "A retriever feeds a generator.",
            "components": [
                {"name": "Retriever", "role": "retrieval", "description": "Searches policies."}
            ],
            "interactions": ["Retriever then generator."],
            "tools": ["search_policies"],
            "risks": [
                {
                    "id": "citation_gap",
                    "name": "Citation gap",
                    "severity": "low",
                    "description": "Citations are required.",
                    "evidence": "must cite the chunk id",
                }
            ],
            "assumptions": ["Artifacts are complete."],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}]},
        )

    provider = OpenAIProvider(
        api_key="test",
        model="gpt-4o-mini",
        base_url="https://example.test/v1",
        temperature=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    run = EvaluationRun(
        name="Acme",
        system_type=SystemType.RAG,
        provider=ProviderConfig(),
        artifacts=[_artifact("policy.md", RAG_DOC)],
    )
    analysis = analyze_solution(run, provider)
    assert analysis.tools == ["search_policies"]
    assert analysis.components[0].name == "Retriever"
    assert analysis.revised is False
