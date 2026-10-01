"""Model providers used by the analyzer, judges, and output-generation graph."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

import httpx

from eval_factory.domain.models import ProviderConfig, ProviderName
from eval_factory.errors import FactoryError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMProvider(Protocol):
    name: str

    def complete_text(self, *, system: str, user: str, max_tokens: int = 1200) -> str: ...

    def complete_json(self, *, system: str, user: str, max_tokens: int = 1200) -> dict[str, Any]: ...


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _FENCE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match is None:
            raise FactoryError("Model response was not JSON")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise FactoryError("Model response was not JSON") from exc
    if not isinstance(parsed, dict):
        raise FactoryError("Model response JSON must be an object")
    return parsed


class HeuristicProvider:
    """Offline stand-in. Scoring uses coded heuristics instead of a model call."""

    name = "heuristic"

    def complete_text(self, *, system: str, user: str, max_tokens: int = 1200) -> str:
        del system, user, max_tokens
        raise FactoryError(
            "The heuristic provider does not generate text. Choose OpenAI, Anthropic, "
            "or an OpenAI-compatible endpoint for model calls."
        )

    def complete_json(self, *, system: str, user: str, max_tokens: int = 1200) -> dict[str, Any]:
        del system, user, max_tokens
        raise FactoryError(
            "The heuristic provider does not call a model. Choose a configured provider "
            "when analysis or judging should use an LLM."
        )


class OpenAIProvider:
    """Chat completions client for OpenAI and compatible gateways."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float,
        client: httpx.Client | None = None,
        provider_name: str = "openai",
    ) -> None:
        self.name = provider_name
        self.model = model
        self.temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def complete_text(self, *, system: str, user: str, max_tokens: int = 1200) -> str:
        response = self._client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        if response.status_code >= 400:
            raise FactoryError(
                f"{self.name} request failed ({response.status_code}): {response.text[:400]}",
                status_code=502,
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise FactoryError(f"{self.name} returned an unexpected payload") from exc
        if not isinstance(content, str) or not content.strip():
            raise FactoryError(f"{self.name} returned an empty response")
        return content

    def complete_json(self, *, system: str, user: str, max_tokens: int = 1200) -> dict[str, Any]:
        text = self.complete_text(
            system=system + "\nReply with one JSON object and no other text.",
            user=user,
            max_tokens=max_tokens,
        )
        return parse_json_object(text)


class AnthropicProvider:
    """Messages client for Anthropic models."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def complete_text(self, *, system: str, user: str, max_tokens: int = 1200) -> str:
        response = self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        if response.status_code >= 400:
            raise FactoryError(
                f"anthropic request failed ({response.status_code}): {response.text[:400]}",
                status_code=502,
            )
        try:
            blocks = response.json()["content"]
            text = "".join(
                block.get("text", "") for block in blocks if block.get("type") == "text"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FactoryError("anthropic returned an unexpected payload") from exc
        if not text.strip():
            raise FactoryError("anthropic returned an empty response")
        return text

    def complete_json(self, *, system: str, user: str, max_tokens: int = 1200) -> dict[str, Any]:
        text = self.complete_text(
            system=system + "\nReply with one JSON object and no other text.",
            user=user,
            max_tokens=max_tokens,
        )
        return parse_json_object(text)


def build_provider(config: ProviderConfig) -> HeuristicProvider | OpenAIProvider | AnthropicProvider:
    if config.provider is ProviderName.HEURISTIC:
        return HeuristicProvider()
    if config.provider is ProviderName.OPENAI:
        return OpenAIProvider(
            api_key=_required_env("OPENAI_API_KEY"),
            model=config.model or "gpt-4o-mini",
            base_url=config.base_url or "https://api.openai.com/v1",
            temperature=config.temperature,
        )
    if config.provider is ProviderName.OPENAI_COMPATIBLE:
        base_url = config.base_url or os.environ.get("EVAL_LLM_BASE_URL", "")
        if not base_url:
            raise FactoryError(
                "Set base_url or EVAL_LLM_BASE_URL for the OpenAI-compatible provider"
            )
        return OpenAIProvider(
            api_key=_required_env("EVAL_LLM_API_KEY"),
            model=config.model or os.environ.get("EVAL_LLM_MODEL", "gpt-4o-mini"),
            base_url=base_url,
            temperature=config.temperature,
            provider_name="openai_compatible",
        )
    if config.provider is ProviderName.ANTHROPIC:
        return AnthropicProvider(
            api_key=_required_env("ANTHROPIC_API_KEY"),
            model=config.model or "claude-3-5-haiku-latest",
            temperature=config.temperature,
        )
    raise FactoryError(f"Unknown provider '{config.provider}'")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise FactoryError(
            f"{name} is not set. Add it to the environment or use the heuristic provider."
        )
    return value
