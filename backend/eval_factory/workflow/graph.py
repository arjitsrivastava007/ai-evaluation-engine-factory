"""Guided evaluation flow with review pauses after analysis and planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from eval_factory.analysis.analyzer import analyze_solution
from eval_factory.data.capture import capture_outputs
from eval_factory.data.datasets import missing_output_ids
from eval_factory.data.synthesize import synthesize_dataset
from eval_factory.domain.models import EvaluationRun, RunPhase, ScoreMode
from eval_factory.errors import FactoryError
from eval_factory.evaluators.deterministic import run_deterministic
from eval_factory.evaluators.judges import run_judges
from eval_factory.planning.planner import attach_bundle, attach_rubric, default_plan, validate_plan
from eval_factory.providers.llm import build_provider
from eval_factory.reporting.markdown import publish_outputs


class GraphState(TypedDict):
    run: dict[str, Any]


def build_evaluation_graph(data_dir: Path):
    """Compile the factory flow.

    A fresh invoke routes from the durable run. Interrupts stop the invoke
    after analysis and after planning so a reviewer can edit before execution.
    """

    def analyze_node(state: GraphState) -> dict[str, Any]:
        run = _load(state)
        run.analysis = analyze_solution(run, build_provider(run.provider))
        run.analysis_approved = False
        run.phase = RunPhase.AWAITING_ANALYSIS_REVIEW
        run.error = None
        return _dump(run)

    def plan_node(state: GraphState) -> dict[str, Any]:
        run = _load(state)
        if run.metric_plan is None:
            run.metric_plan = default_plan(run.system_type)
            for bundle in run.rule_bundles:
                attach_bundle(run.metric_plan, bundle)
            for rubric in run.rubrics:
                attach_rubric(run.metric_plan, rubric)
        run.plan_approved = False
        run.phase = RunPhase.AWAITING_PLAN_REVIEW
        run.error = None
        return _dump(run)

    def prepare_node(state: GraphState) -> dict[str, Any]:
        run = _load(state)
        provider = build_provider(run.provider)
        if not run.cases:
            if run.data_strategy != "synthesize":
                raise FactoryError("Add a dataset or synthesize one before running evaluation")
            cases, outputs = synthesize_dataset(run, provider)
            run.cases = cases
            run.outputs = outputs
        missing = missing_output_ids(run.cases, run.outputs)
        if missing and run.data_strategy == "capture" and run.capture_spec is not None:
            captured = {item.case_id: item for item in capture_outputs(run.cases, run.capture_spec)}
            current = {item.case_id: item for item in run.outputs}
            current.update(captured)
            run.outputs = [current[case.id] for case in run.cases if case.id in current]
            missing = missing_output_ids(run.cases, run.outputs)
        if missing:
            raise FactoryError("These cases have no system output: " + ", ".join(missing))
        if run.metric_plan is None:
            raise FactoryError("The metric plan is missing")
        validate_plan(run.metric_plan, run.rule_bundles, run.rubrics)
        run.phase = RunPhase.RUNNING
        run.error = None
        return _dump(run)

    def deterministic_node(state: GraphState) -> dict[str, Any]:
        run = _load(state)
        coded = run_deterministic(run)
        judged = [result for result in run.results if result.mode == ScoreMode.PROBABILISTIC]
        run.results = coded + judged
        return _dump(run)

    def judge_node(state: GraphState) -> dict[str, Any]:
        run = _load(state)
        judged = run_judges(run)
        coded = [result for result in run.results if result.mode == ScoreMode.DETERMINISTIC]
        run.results = coded + judged
        return _dump(run)

    def publish_node(state: GraphState) -> dict[str, Any]:
        run = _load(state)
        publish_outputs(run, data_dir / run.id)
        run.phase = RunPhase.COMPLETED
        run.error = None
        return _dump(run)

    graph = StateGraph(GraphState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("plan", plan_node)
    graph.add_node("prepare", prepare_node)
    graph.add_node("evaluate_deterministic", deterministic_node)
    graph.add_node("evaluate_judges", judge_node)
    graph.add_node("publish", publish_node)
    graph.add_conditional_edges(
        START,
        _route,
        {"analyze": "analyze", "plan": "plan", "prepare": "prepare", END: END},
    )
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "prepare")
    graph.add_edge("prepare", "evaluate_deterministic")
    graph.add_edge("evaluate_deterministic", "evaluate_judges")
    graph.add_edge("evaluate_judges", "publish")
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_after=["analyze", "plan"])


def _route(state: GraphState) -> str:
    run = state["run"]
    if run.get("phase") == RunPhase.COMPLETED.value:
        return END
    if not run.get("analysis"):
        return "analyze"
    if not run.get("analysis_approved"):
        return END
    if not run.get("metric_plan"):
        return "plan"
    if not run.get("plan_approved"):
        return END
    return "prepare"


def _load(state: GraphState) -> EvaluationRun:
    return EvaluationRun.model_validate(state["run"])


def _dump(run: EvaluationRun) -> dict[str, Any]:
    return {"run": run.model_dump(mode="json")}
