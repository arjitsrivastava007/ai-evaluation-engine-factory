"""Call the system under test and store what it actually returned."""

from __future__ import annotations

import ipaddress
import json
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from eval_factory.domain.models import CaptureSpec, EvalCase, SystemOutput, ToolCall
from eval_factory.errors import FactoryError

_BLOCKED_HOSTS = {"metadata.google.internal", "metadata.internal"}
_RESPONSE_KEYS = ("response", "output", "answer", "text", "content")


def capture_outputs(
    cases: list[EvalCase],
    spec: CaptureSpec,
    *,
    client: httpx.Client | None = None,
) -> list[SystemOutput]:
    if not cases:
        raise FactoryError("Capture needs at least one case")
    assert_safe_capture_url(spec.url)
    timeout = min(max(spec.timeout_seconds, 1), 60)
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        return [_capture_case(http, spec, case) for case in cases]
    finally:
        if owns_client:
            http.close()


def assert_safe_capture_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise FactoryError("Capture URL must use http or https")
    host = parsed.hostname
    if not host:
        raise FactoryError("Capture URL is missing a host")
    if host.lower() in _BLOCKED_HOSTS:
        raise FactoryError("Capture URL targets a blocked host")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise FactoryError(f"Could not resolve capture host '{host}'") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (
            ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
            or str(ip) == "169.254.169.254"
        ):
            raise FactoryError("Capture URL resolves to a blocked address")


def _capture_case(client: httpx.Client, spec: CaptureSpec, case: EvalCase) -> SystemOutput:
    mapping = {
        "input": case.input,
        "expected": case.expected or "",
        "context": "\n\n".join(case.context),
        "case_id": case.id,
    }
    try:
        if spec.method == "GET":
            response = client.get(spec.url, params={"input": case.input}, headers=spec.headers)
        else:
            body = _render(spec.body_template or {"input": "{{input}}", "context": "{{context}}"}, mapping)
            response = client.post(spec.url, json=body, headers=spec.headers)
    except httpx.HTTPError as exc:
        raise FactoryError(f"Capture request failed for case '{case.id}': {exc}") from exc
    if response.status_code >= 400:
        raise FactoryError(
            f"System under test returned {response.status_code} for case '{case.id}': {response.text[:300]}"
        )
    return _read_output(case.id, response.text, spec.response_path)


def _render(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        rendered = value
        for key, replacement in mapping.items():
            rendered = rendered.replace("{{" + key + "}}", replacement)
        return rendered
    if isinstance(value, dict):
        return {key: _render(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, mapping) for item in value]
    return value


def _read_output(case_id: str, body: str, response_path: str) -> SystemOutput:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        text = body.strip()
        if not text:
            raise FactoryError(f"Capture for case '{case_id}' returned an empty body")
        return SystemOutput(case_id=case_id, response=text)
    if not isinstance(payload, dict):
        return SystemOutput(case_id=case_id, response=json.dumps(payload))
    if response_path:
        response_text = _dig(payload, response_path, case_id)
    else:
        response_text = _first_text(payload)
    tool_calls = []
    for call in payload.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("name"):
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            tool_calls.append(ToolCall(name=str(call["name"]), arguments=arguments))
    contexts = [str(item) for item in payload.get("contexts") or [] if str(item).strip()]
    steps = [str(item) for item in payload.get("steps") or [] if str(item).strip()]
    return SystemOutput(
        case_id=case_id,
        response=response_text,
        contexts=contexts,
        tool_calls=tool_calls,
        steps=steps,
    )


def _dig(payload: dict, path: str, case_id: str) -> str:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise FactoryError(f"Response path '{path}' was not found for case '{case_id}'")
        current = current[part]
    return current if isinstance(current, str) else json.dumps(current)


def _first_text(payload: dict) -> str:
    for key in _RESPONSE_KEYS:
        if isinstance(payload.get(key), str) and payload[key].strip():
            return payload[key]
    return json.dumps(payload)
