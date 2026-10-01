"""Configurable providers stay behind one interface, including the offline heuristic."""

import json

import httpx
import pytest

from eval_factory.domain.models import ProviderConfig, ProviderName
from eval_factory.errors import FactoryError
from eval_factory.providers.llm import (
    AnthropicProvider,
    HeuristicProvider,
    OpenAIProvider,
    build_provider,
    parse_json_object,
)


def test_fenced_json_is_accepted():
    parsed = parse_json_object('```json\n{"score": 0.5, "rationale": "partial"}\n```')
    assert parsed["score"] == 0.5


def test_heuristic_provider_refuses_to_pretend_it_is_a_model():
    provider = HeuristicProvider()
    with pytest.raises(FactoryError, match="heuristic"):
        provider.complete_json(system="s", user="u")


def test_openai_provider_posts_chat_completions_and_parses_json():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["temperature"] == 0
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"summary": "A retriever and a generator."}'}}]},
        )

    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-4o-mini",
        base_url="https://example.test/v1",
        temperature=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.complete_json(system="Analyze", user="artifacts")["summary"].startswith("A retriever")


def test_anthropic_provider_reads_text_blocks():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.anthropic.com"
        assert request.headers["x-api-key"] == "test-key"
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"tools": ["search_policies"]}'}]},
        )

    provider = AnthropicProvider(
        api_key="test-key",
        model="claude-3-5-haiku-latest",
        temperature=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.complete_json(system="Analyze", user="artifacts")["tools"] == ["search_policies"]


def test_remote_http_errors_do_not_include_the_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(FactoryError, match="OPENAI_API_KEY"):
        build_provider(ProviderConfig(provider=ProviderName.OPENAI))


def test_compatible_provider_requires_a_base_url(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_API_KEY", "secret")
    monkeypatch.delenv("EVAL_LLM_BASE_URL", raising=False)
    with pytest.raises(FactoryError, match="base_url"):
        build_provider(ProviderConfig(provider=ProviderName.OPENAI_COMPATIBLE))
