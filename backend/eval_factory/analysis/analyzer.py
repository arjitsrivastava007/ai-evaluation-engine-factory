"""Map uploaded artifacts onto architecture, tools, and risks."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from eval_factory.domain.models import (
    ArchitectureAnalysis,
    ArtifactRecord,
    Component,
    EvaluationRun,
    RiskFinding,
    RiskLevel,
    SystemType,
)
from eval_factory.domain.ontology import SystemProfile, profile_for
from eval_factory.errors import FactoryError
from eval_factory.providers.llm import LLMProvider
from eval_factory.textutil import excerpt

_HEADING = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
_SKIP_HEADINGS = frozenset(
    {"overview", "introduction", "summary", "components", "risks", "tools", "architecture"}
)


def analyze_solution(run: EvaluationRun, provider: LLMProvider) -> ArchitectureAnalysis:
    if not run.artifacts:
        raise FactoryError("Upload at least one artifact before analysis")
    if provider.name == "heuristic":
        return analyze_heuristically(run.system_type, run.artifacts)
    return _analyze_with_model(run, provider)


def analyze_heuristically(
    system_type: SystemType,
    artifacts: list[ArtifactRecord],
) -> ArchitectureAnalysis:
    profile = profile_for(system_type)
    corpus = _corpus(artifacts)
    tools = extract_tools(corpus)
    components = _components(profile, corpus, tools)
    risks = _risks(profile, corpus, tools)
    names = ", ".join(artifact.filename for artifact in artifacts)
    highest = sorted(risks, key=lambda risk: _SEVERITY_RANK[risk.severity], reverse=True)[:2]
    highest_names = ", ".join(risk.name for risk in highest) if highest else "none"
    tool_clause = f" It names {len(tools)} tool(s): {', '.join(tools)}." if tools else ""
    summary = (
        f"The {profile.label} is described in {len(artifacts)} artifact(s) ({names}). "
        f"The analyzer mapped {len(components)} components.{tool_clause} "
        f"Highest risks: {highest_names}."
    )
    assumptions = [
        f"The user selected the system type '{system_type.value}'.",
        "This pass reads artifact text and file metadata. It does not call the system under test.",
    ]
    if any(artifact.kind.value == "image" for artifact in artifacts):
        assumptions.append("Image artifacts contributed dimensions and format, not a visual reading.")
    return ArchitectureAnalysis(
        system_type=system_type,
        summary=summary,
        components=components,
        interactions=_interactions(profile, tools),
        tools=tools,
        risks=risks,
        assumptions=assumptions,
        revised=False,
    )


def _tool_tokens(chunk: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", chunk)


def extract_tools(text: str) -> list[str]:
    found: list[str] = []
    lines = text.splitlines()
    capturing = False
    for line in lines:
        header = re.match(r"(?i)^tools?\s*:\s*(.*)$", line.strip())
        if header:
            found.extend(_tool_tokens(header.group(1)))
            capturing = True
            continue
        if capturing:
            bullet = re.match(r"^\s*[-*]\s+`?([A-Za-z][A-Za-z0-9_-]+)`?\s*$", line)
            if bullet:
                found.append(bullet.group(1))
                continue
            if line.strip():
                capturing = False
        inline = re.search(r"(?i)\btool\s+`([A-Za-z][A-Za-z0-9_-]+)`", line)
        if inline:
            found.append(inline.group(1))
    unique: list[str] = []
    for name in found:
        if name.lower() in {"the", "and", "with"}:
            continue
        if name not in unique:
            unique.append(name)
    return unique


def _analyze_with_model(run: EvaluationRun, provider: LLMProvider) -> ArchitectureAnalysis:
    profile = profile_for(run.system_type)
    payload = {
        "system_type": run.system_type.value,
        "known_risks": [
            {"id": risk.id, "name": risk.name, "description": risk.description}
            for risk in profile.risks
        ],
        "instructions": (
            "Describe only what the artifacts support. Prefer the known risk ids when they fit. "
            "severity must be low, medium, high, or critical."
        ),
        "artifacts": _corpus(run.artifacts)[:12_000],
    }
    data = provider.complete_json(
        system=(
            "You are the Solution Analyzer for an AI evaluation factory. "
            "Return one JSON object with keys summary, components, interactions, tools, "
            "risks, and assumptions. components items have name, role, and description. "
            "risks items have id, name, severity, description, and evidence."
        ),
        user=json.dumps(payload),
        max_tokens=1800,
    )
    data["system_type"] = run.system_type.value
    data["revised"] = False
    try:
        analysis = ArchitectureAnalysis.model_validate(data)
    except ValidationError as exc:
        raise FactoryError(f"Analyzer response did not match the expected shape: {exc}") from exc
    if not analysis.summary.strip() or not analysis.components:
        raise FactoryError("Analyzer response was missing a summary or components")
    return analysis


def _corpus(artifacts: list[ArtifactRecord]) -> str:
    blocks = [f"# Artifact: {artifact.filename}\n{artifact.text}" for artifact in artifacts]
    return "\n\n".join(blocks)


def _components(profile: SystemProfile, text: str, tools: list[str]) -> list[Component]:
    components = [
        Component(name=name, role=role, description=description)
        for name, role, description in profile.default_components
    ]
    known = {component.name.lower() for component in components}
    for heading in _HEADING.findall(text):
        title = heading.strip(" -:")
        if title.lower() in _SKIP_HEADINGS or title.lower() in known:
            continue
        if len(title) > 80:
            continue
        components.append(
            Component(
                name=title,
                role="described",
                description="Named in the uploaded artifacts.",
            )
        )
        known.add(title.lower())
    for tool in tools:
        if tool.lower() in known:
            continue
        components.append(
            Component(name=tool, role="tool", description="Tool named in the uploaded artifacts.")
        )
        known.add(tool.lower())
    return components


def _interactions(profile: SystemProfile, tools: list[str]) -> list[str]:
    if profile.system_type is SystemType.RAG:
        lines = [
            "A question is encoded and sent to the retriever.",
            "Retrieved passages are assembled into the context the generator may use.",
            "The generator writes an answer that should stay inside that context and cite it.",
        ]
    elif profile.system_type is SystemType.CHATBOT:
        lines = [
            "The router classifies the user message.",
            "Dialogue policy chooses the next move and the generator writes the reply.",
            "The safety filter is expected to stop unsafe or privacy-leaking text.",
        ]
    else:
        lines = [
            "The planner turns the goal into ordered steps.",
            "The tool router selects a tool for the current step and the executor runs it.",
            "Memory carries tool results forward until a final answer is produced.",
        ]
    if tools:
        lines.append("Named tools: " + ", ".join(tools) + ".")
    return lines


_SEVERITY_RANK = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def _risks(profile: SystemProfile, text: str, tools: list[str]) -> list[RiskFinding]:
    findings = []
    for template in profile.risks:
        severity, evidence = _risk_signal(profile.system_type, template.id, text, tools)
        findings.append(
            RiskFinding(
                id=template.id,
                name=template.name,
                severity=severity,
                description=template.description,
                evidence=evidence,
            )
        )
    return findings


def _risk_signal(
    system_type: SystemType,
    risk_id: str,
    text: str,
    tools: list[str],
) -> tuple[RiskLevel, str]:
    if system_type is SystemType.RAG:
        return _rag_signal(risk_id, text)
    if system_type is SystemType.CHATBOT:
        return _chat_signal(risk_id, text)
    return _agent_signal(risk_id, text, tools)


def _rag_signal(risk_id: str, text: str) -> tuple[RiskLevel, str]:
    if risk_id == "hallucination":
        grounded = _contains(text, "only from", "ground", "retrieved context", "from retrieved")
        return (
            RiskLevel.MEDIUM if grounded else RiskLevel.HIGH,
            _evidence(text, "ground", "retrieved", "context"),
        )
    if risk_id == "retrieval_miss":
        present = _contains(text, "retriev")
        return (
            RiskLevel.MEDIUM if present else RiskLevel.HIGH,
            _evidence(text, "retriev", "search", "index"),
        )
    if risk_id == "citation_gap":
        present = _contains(text, "cite", "citation", "source")
        return (
            RiskLevel.LOW if present else RiskLevel.HIGH,
            _evidence(text, "cite", "citation", "source"),
        )
    if risk_id == "stale_knowledge":
        fresh = _contains(text, "refresh", "reindex", "updated", "up to date")
        return (
            RiskLevel.LOW if fresh else RiskLevel.MEDIUM,
            _evidence(text, "refresh", "reindex", "updated"),
        )
    return RiskLevel.MEDIUM, "Not observed in the uploaded artifacts."


def _chat_signal(risk_id: str, text: str) -> tuple[RiskLevel, str]:
    if risk_id == "unsafe_content":
        guarded = _contains(text, "safety", "moderat", "redact", "privacy")
        return (
            RiskLevel.LOW if guarded else RiskLevel.HIGH,
            _evidence(text, "safety", "moderat", "privacy", "redact"),
        )
    if risk_id == "persona_drift":
        named = _contains(text, "tone", "persona", "voice")
        return (
            RiskLevel.LOW if named else RiskLevel.MEDIUM,
            _evidence(text, "tone", "persona", "voice"),
        )
    if risk_id == "prompt_injection":
        named = _contains(text, "injection", "untrusted")
        return (
            RiskLevel.LOW if named else RiskLevel.MEDIUM,
            _evidence(text, "injection", "untrusted"),
        )
    return RiskLevel.MEDIUM, _evidence(text, "help", "answer", "user")


def _agent_signal(risk_id: str, text: str, tools: list[str]) -> tuple[RiskLevel, str]:
    if risk_id == "wrong_tool":
        return (
            RiskLevel.MEDIUM if tools else RiskLevel.HIGH,
            "Named tools: " + ", ".join(tools) + "." if tools else "No tools were named in the artifacts.",
        )
    if risk_id == "step_runaway":
        bounded = _contains(text, "step limit", "max step", "step budget", "maximum steps")
        return (
            RiskLevel.LOW if bounded else RiskLevel.HIGH,
            _evidence(text, "step", "budget", "limit"),
        )
    if risk_id == "unauthorized_action":
        guarded = _contains(text, "permission", "allowlist", "authorized", "approval")
        return (
            RiskLevel.MEDIUM if guarded else RiskLevel.HIGH,
            _evidence(text, "permission", "allow", "auth"),
        )
    return RiskLevel.MEDIUM, _evidence(text, "goal", "finish", "complete")


def _contains(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _evidence(text: str, *needles: str) -> str:
    for piece in re.split(r"\n+|(?<=[.!?])\s+", text):
        lowered = piece.lower()
        if any(needle in lowered for needle in needles) and piece.strip():
            return excerpt(piece, 180)
    return "Not observed in the uploaded artifacts."
