"""Shared contracts for a single evaluation run."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid4().hex


class SystemType(str, Enum):
    RAG = "rag"
    CHATBOT = "chatbot"
    AGENTIC = "agentic"


class RunPhase(str, Enum):
    CREATED = "created"
    AWAITING_ANALYSIS_REVIEW = "awaiting_analysis_review"
    AWAITING_PLAN_REVIEW = "awaiting_plan_review"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactKind(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    DOCX = "docx"
    IMAGE = "image"
    TEXT = "text"


class ProviderName(str, Enum):
    HEURISTIC = "heuristic"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class RuleKind(str, Enum):
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    JSON_OBJECT = "json_object"
    REQUIRED_TOOLS = "required_tools"
    CITATION_REQUIRED = "citation_required"


class ScoreMode(str, Enum):
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProviderConfig(BaseModel):
    """Which model provider judges and generators should call.

    API keys stay in the environment. They are never stored on the run.
    """

    provider: ProviderName = ProviderName.HEURISTIC
    model: str = ""
    temperature: float = 0.0
    base_url: str | None = None

    @field_validator("temperature")
    @classmethod
    def _temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value


class ArtifactRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    filename: str
    kind: ArtifactKind
    media_type: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    stored_path: str


class Component(BaseModel):
    name: str
    role: str
    description: str


class RiskFinding(BaseModel):
    id: str
    name: str
    severity: RiskLevel
    description: str
    evidence: str


class ArchitectureAnalysis(BaseModel):
    """Solution Analyzer output. Experts can edit this before planning."""

    system_type: SystemType
    summary: str
    components: list[Component] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    risks: list[RiskFinding] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    revised: bool = False


class MetricSpec(BaseModel):
    id: str
    name: str
    description: str
    weight: float = 1.0
    mode: ScoreMode
    evaluator_id: str
    threshold: float = 0.7
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def _weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError("weight cannot be negative")
        return value

    @field_validator("threshold")
    @classmethod
    def _threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return value


class MetricPlan(BaseModel):
    system_type: SystemType
    metrics: list[MetricSpec]
    notes: str = ""


class Rule(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    kind: RuleKind
    params: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class RuleBundle(BaseModel):
    """A user-owned set of deterministic checks added to the metric plan."""

    id: str = Field(default_factory=new_id)
    name: str
    description: str = ""
    rules: list[Rule] = Field(default_factory=list)


class RubricCriterion(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    description: str
    weight: float = 1.0


class LLMRubric(BaseModel):
    """A user-owned LLM judge definition."""

    id: str = Field(default_factory=new_id)
    name: str
    prompt: str
    criteria: list[RubricCriterion] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    id: str
    input: str
    expected: str | None = None
    context: list[str] = Field(default_factory=list)
    reference_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemOutput(BaseModel):
    case_id: str
    response: str
    contexts: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaptureSpec(BaseModel):
    """How to call the system under test and read its response."""

    url: str
    method: Literal["GET", "POST"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: dict[str, Any] | None = None
    response_path: str = ""
    timeout_seconds: float = 30.0


class EvaluatorResult(BaseModel):
    case_id: str
    metric_id: str
    evaluator_id: str
    mode: ScoreMode
    score: float
    passed: bool
    rationale: str
    applicable: bool = True
    evidence: dict[str, Any] = Field(default_factory=dict)


class MetricScore(BaseModel):
    metric_id: str
    name: str
    mode: ScoreMode
    score: float
    passed: bool
    weight: float
    normalized_weight: float


class AggregateReport(BaseModel):
    overall: float
    deterministic: float | None
    probabilistic: float | None
    risk_level: RiskLevel
    metrics: list[MetricScore]
    recommendations: list[str]
    case_count: int
    decision: str


class EvaluationRun(BaseModel):
    """Durable state for one guided evaluation."""

    id: str = Field(default_factory=new_id)
    name: str
    system_type: SystemType
    phase: RunPhase = RunPhase.CREATED
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    analysis: ArchitectureAnalysis | None = None
    analysis_approved: bool = False
    metric_plan: MetricPlan | None = None
    plan_approved: bool = False
    rule_bundles: list[RuleBundle] = Field(default_factory=list)
    rubrics: list[LLMRubric] = Field(default_factory=list)
    cases: list[EvalCase] = Field(default_factory=list)
    outputs: list[SystemOutput] = Field(default_factory=list)
    data_strategy: Literal["unset", "upload", "synthesize", "capture"] = "unset"
    capture_spec: CaptureSpec | None = None
    results: list[EvaluatorResult] = Field(default_factory=list)
    aggregate: AggregateReport | None = None
    report_markdown: str | None = None
    chart_files: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def touch(self) -> None:
        self.updated_at = utcnow()
