"""Maps each system type to the metrics, risks, and evaluators it should use.

The Solution Analyzer and Evaluation Planner read this catalog so a new run
starts from an opinionated plan instead of an empty form.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval_factory.domain.models import ScoreMode, SystemType
from eval_factory.errors import FactoryError


@dataclass(frozen=True)
class MetricTemplate:
    id: str
    name: str
    description: str
    weight: float
    mode: ScoreMode
    evaluator_id: str
    threshold: float
    recommendation: str


@dataclass(frozen=True)
class RiskTemplate:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class SystemProfile:
    system_type: SystemType
    label: str
    summary: str
    default_components: tuple[tuple[str, str, str], ...]
    metrics: tuple[MetricTemplate, ...]
    risks: tuple[RiskTemplate, ...]


def _metric(
    metric_id: str,
    name: str,
    description: str,
    weight: float,
    mode: ScoreMode,
    evaluator_id: str,
    recommendation: str,
    threshold: float = 0.7,
) -> MetricTemplate:
    return MetricTemplate(
        id=metric_id,
        name=name,
        description=description,
        weight=weight,
        mode=mode,
        evaluator_id=evaluator_id,
        threshold=threshold,
        recommendation=recommendation,
    )


_RAG = SystemProfile(
    system_type=SystemType.RAG,
    label="RAG pipeline",
    summary=(
        "A retrieval-augmented generator. Quality depends on whether the answer "
        "stays inside retrieved context, cites it, and actually addresses the question."
    ),
    default_components=(
        ("Query encoder", "embedding", "Turns the user question into a retrieval query."),
        ("Retriever", "retrieval", "Returns the passages the generator is allowed to use."),
        ("Context builder", "assembly", "Orders and trims passages before generation."),
        ("Answer generator", "generation", "Writes the answer from the assembled context."),
    ),
    metrics=(
        _metric(
            "context_overlap",
            "Context overlap",
            "Share of answer content words that also appear in the retrieved context.",
            0.15,
            ScoreMode.DETERMINISTIC,
            "context_overlap",
            "Ground the generator in retrieved passages and drop claims that are not in context.",
        ),
        _metric(
            "citation_support",
            "Citation support",
            "The answer points at a source, chunk, or URL the reader can check.",
            0.10,
            ScoreMode.DETERMINISTIC,
            "citation_support",
            "Require an inline citation whenever the answer uses retrieved context.",
        ),
        _metric(
            "answer_relevance",
            "Answer relevance",
            "The answer covers the terms of the question and any expected answer.",
            0.10,
            ScoreMode.DETERMINISTIC,
            "answer_relevance",
            "Tighten the prompt so the answer resolves the question before adding extra detail.",
        ),
        _metric(
            "expected_coverage",
            "Expected coverage",
            "Required facts from the labeled answer show up in the response.",
            0.10,
            ScoreMode.DETERMINISTIC,
            "expected_coverage",
            "Add the missing labeled facts to the context or correct the generator's extraction.",
        ),
        _metric(
            "faithfulness",
            "Faithfulness",
            "A judge scores how completely the answer stays inside the supplied context.",
            0.25,
            ScoreMode.PROBABILISTIC,
            "faithfulness_judge",
            "Treat unsupported sentences as failures and regenerate from context only.",
            0.75,
        ),
        _metric(
            "relevance",
            "Relevance",
            "A judge scores whether the answer is about the user's actual question.",
            0.15,
            ScoreMode.PROBABILISTIC,
            "relevance_judge",
            "Rewrite the answer so the first sentences resolve the question.",
        ),
        _metric(
            "completeness",
            "Completeness",
            "A judge scores whether the answer includes the material a reviewer expects.",
            0.15,
            ScoreMode.PROBABILISTIC,
            "completeness_judge",
            "Extend the answer until each expected clause is covered, without adding new facts.",
        ),
    ),
    risks=(
        RiskTemplate(
            "hallucination",
            "Ungrounded answer",
            "The generator can state facts that never appear in retrieved context.",
        ),
        RiskTemplate(
            "retrieval_miss",
            "Retrieval miss",
            "The right passage may never reach the generator.",
        ),
        RiskTemplate(
            "citation_gap",
            "Citation gap",
            "Readers cannot trace a claim back to a passage.",
        ),
        RiskTemplate(
            "stale_knowledge",
            "Stale knowledge",
            "The index can lag the source of truth the business now uses.",
        ),
    ),
)

_CHATBOT = SystemProfile(
    system_type=SystemType.CHATBOT,
    label="Chatbot",
    summary=(
        "A conversational system. Quality depends on helpfulness, tone, "
        "safety, and whether the reply finishes the user's task."
    ),
    default_components=(
        ("Intent router", "routing", "Classifies what the user is trying to do."),
        ("Dialogue policy", "policy", "Decides the next conversational move."),
        ("Response generator", "generation", "Writes the user-facing reply."),
        ("Safety filter", "safety", "Blocks unsafe or privacy-leaking replies."),
    ),
    metrics=(
        _metric(
            "safety_screen",
            "Safety screen",
            "Coded checks for profanity and common personal-data patterns.",
            0.15,
            ScoreMode.DETERMINISTIC,
            "safety_screen",
            "Block replies that leak personal data or use abusive language before they reach the user.",
        ),
        _metric(
            "format_bounds",
            "Format bounds",
            "The reply is non-empty and stays inside a usable length.",
            0.10,
            ScoreMode.DETERMINISTIC,
            "format_bounds",
            "Set minimum and maximum reply lengths so answers are neither empty nor rambling.",
        ),
        _metric(
            "instruction_adherence",
            "Instruction adherence",
            "Required phrases and labeled expectations appear in the reply.",
            0.15,
            ScoreMode.DETERMINISTIC,
            "instruction_adherence",
            "Put required phrases in the system prompt and check them before sending.",
        ),
        _metric(
            "consistency",
            "Consistency",
            "The reply does not assert both sides of the same claim.",
            0.05,
            ScoreMode.DETERMINISTIC,
            "consistency",
            "Regenerate replies that say both always and never, or yes and no, about the same answer.",
        ),
        _metric(
            "helpfulness",
            "Helpfulness",
            "A judge scores whether the reply moves the user forward.",
            0.25,
            ScoreMode.PROBABILISTIC,
            "helpfulness_judge",
            "Lead with the action the user can take, then the supporting detail.",
        ),
        _metric(
            "tone",
            "Tone",
            "A judge scores whether the reply stays calm, clear, and professional.",
            0.10,
            ScoreMode.PROBABILISTIC,
            "tone_judge",
            "Lower the temperature and ban all-caps or excessive punctuation in the style guide.",
        ),
        _metric(
            "task_completion",
            "Task completion",
            "A judge scores whether the reply finished the requested task.",
            0.20,
            ScoreMode.PROBABILISTIC,
            "task_completion_judge",
            "Do not end on a refusal or a clarifying question when the request was already answerable.",
            0.75,
        ),
    ),
    risks=(
        RiskTemplate(
            "unsafe_content",
            "Unsafe content",
            "The bot can emit abusive language or personal data.",
        ),
        RiskTemplate(
            "persona_drift",
            "Persona drift",
            "Tone and commitments can change from one turn to the next.",
        ),
        RiskTemplate(
            "incomplete_help",
            "Incomplete help",
            "The reply can acknowledge the user without resolving the task.",
        ),
        RiskTemplate(
            "prompt_injection",
            "Prompt injection",
            "User text can try to override the bot's instructions.",
        ),
    ),
)

_AGENT = SystemProfile(
    system_type=SystemType.AGENTIC,
    label="Agent",
    summary=(
        "A multi-step agent. Quality depends on tool choice, a bounded plan, "
        "and whether the run actually finishes the goal."
    ),
    default_components=(
        ("Planner", "planning", "Breaks the goal into ordered steps."),
        ("Tool router", "routing", "Chooses which tool each step may call."),
        ("Tool executor", "execution", "Runs the selected tool and records the result."),
        ("Memory", "state", "Keeps prior steps available to the next decision."),
    ),
    metrics=(
        _metric(
            "tool_selection",
            "Tool selection",
            "The reference tools for the task were actually called.",
            0.15,
            ScoreMode.DETERMINISTIC,
            "tool_selection",
            "Constrain the router to the tools required for this task and fail closed when one is skipped.",
        ),
        _metric(
            "tool_schema",
            "Tool schema",
            "Every tool call has a name and an arguments object.",
            0.10,
            ScoreMode.DETERMINISTIC,
            "tool_schema",
            "Validate tool calls against their schema before execution.",
        ),
        _metric(
            "step_budget",
            "Step budget",
            "The agent finishes inside the configured step limit.",
            0.10,
            ScoreMode.DETERMINISTIC,
            "step_budget",
            "Stop the loop at the step budget and return the best partial result with a clear failure.",
            0.8,
        ),
        _metric(
            "final_answer",
            "Final answer",
            "The trace ends with a user-facing answer.",
            0.10,
            ScoreMode.DETERMINISTIC,
            "final_answer",
            "Require a final message even when tools fail, stating what was and was not done.",
        ),
        _metric(
            "plan_quality",
            "Plan quality",
            "A judge scores whether the steps are ordered, specific, and tool-aware.",
            0.20,
            ScoreMode.PROBABILISTIC,
            "plan_quality_judge",
            "Ask the planner for a short ordered plan before the first tool call.",
        ),
        _metric(
            "goal_completion",
            "Goal completion",
            "A judge scores whether the trace achieved the user's goal.",
            0.20,
            ScoreMode.PROBABILISTIC,
            "goal_completion_judge",
            "Compare the final answer to the goal and rerun only the missing steps.",
            0.75,
        ),
        _metric(
            "action_safety",
            "Action safety",
            "A judge scores whether the actions stay inside the task's authority.",
            0.15,
            ScoreMode.PROBABILISTIC,
            "safety_judge",
            "Deny tools that mutate data unless the task explicitly allows them.",
        ),
    ),
    risks=(
        RiskTemplate(
            "wrong_tool",
            "Wrong tool",
            "The agent can pick a tool that does not serve the goal.",
        ),
        RiskTemplate(
            "step_runaway",
            "Step runaway",
            "A loop can keep calling tools without reaching a final answer.",
        ),
        RiskTemplate(
            "unauthorized_action",
            "Unauthorized action",
            "A tool can change state the task never authorized.",
        ),
        RiskTemplate(
            "partial_completion",
            "Partial completion",
            "The trace can stop after a plausible step and still miss the goal.",
        ),
    ),
)

PROFILES: dict[SystemType, SystemProfile] = {
    SystemType.RAG: _RAG,
    SystemType.CHATBOT: _CHATBOT,
    SystemType.AGENTIC: _AGENT,
}

KNOWN_EVALUATORS = frozenset(
    metric.evaluator_id for profile in PROFILES.values() for metric in profile.metrics
) | {"rule_bundle", "llm_rubric"}


def profile_for(system_type: SystemType | str) -> SystemProfile:
    try:
        key = system_type if isinstance(system_type, SystemType) else SystemType(system_type)
    except ValueError as exc:
        raise FactoryError(
            "system_type must be one of: rag, chatbot, agentic",
            status_code=400,
        ) from exc
    return PROFILES[key]


def recommendation_for(system_type: SystemType, metric_id: str) -> str | None:
    profile = PROFILES[system_type]
    for metric in profile.metrics:
        if metric.id == metric_id:
            return metric.recommendation
    return None
