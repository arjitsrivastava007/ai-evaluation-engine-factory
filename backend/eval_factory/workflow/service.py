"""Application service for one evaluation, from upload through the report."""

from __future__ import annotations

import logging

from eval_factory.artifacts.ingest import ingest_bytes
from eval_factory.config import Settings
from eval_factory.data.capture import capture_outputs
from eval_factory.data.datasets import parse_dataset
from eval_factory.data.synthesize import synthesize_dataset
from eval_factory.domain.models import (
    ArchitectureAnalysis,
    ArtifactRecord,
    CaptureSpec,
    EvaluationRun,
    LLMRubric,
    MetricPlan,
    ProviderConfig,
    RuleBundle,
    RunPhase,
    SystemType,
    new_id,
)
from eval_factory.errors import FactoryError
from eval_factory.planning.planner import attach_bundle, attach_rubric, validate_bundle, validate_plan, validate_rubric
from eval_factory.providers.llm import build_provider
from eval_factory.storage.store import RunStore
from eval_factory.workflow.graph import build_evaluation_graph

logger = logging.getLogger(__name__)

_ANALYSIS_PHASES = {RunPhase.CREATED, RunPhase.AWAITING_ANALYSIS_REVIEW, RunPhase.FAILED}


class EvaluationService:
    def __init__(self, store: RunStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.graph = build_evaluation_graph(store.root)

    def create_run(self, name: str, system_type: SystemType, provider: ProviderConfig | None = None) -> EvaluationRun:
        cleaned = name.strip()
        if not cleaned:
            raise FactoryError("A run needs a name")
        run = EvaluationRun(name=cleaned[:120], system_type=system_type, provider=provider or ProviderConfig())
        self.store.save(run)
        return run

    def add_artifact(self, run_id: str, filename: str, data: bytes) -> EvaluationRun:
        run = self.store.get(run_id)
        if run.analysis_approved:
            raise FactoryError("Artifacts are locked after analysis is approved")
        if run.phase not in _ANALYSIS_PHASES:
            raise FactoryError("Artifacts can be added only before analysis is approved")
        if len(data) > self.settings.max_upload_bytes:
            raise FactoryError("Artifact exceeds the 10 MB upload limit")
        parsed = ingest_bytes(filename, data)
        stored = self.store.artifact_path(run.id, f"{new_id()[:8]}_{parsed.filename}")
        stored.write_bytes(data)
        run.artifacts.append(
            ArtifactRecord(
                filename=parsed.filename,
                kind=parsed.kind,
                media_type=parsed.media_type,
                text=parsed.text,
                metadata=parsed.metadata,
                stored_path=str(stored),
            )
        )
        run.analysis = None
        run.analysis_approved = False
        run.metric_plan = None
        run.plan_approved = False
        run.phase = RunPhase.CREATED
        run.error = None
        self.store.save(run)
        return run

    def analyze(self, run_id: str) -> EvaluationRun:
        run = self.store.get(run_id)
        if run.analysis_approved:
            raise FactoryError("Analysis is already approved")
        if run.phase not in _ANALYSIS_PHASES:
            raise FactoryError("Analysis can only run before it is approved")
        if not run.artifacts:
            raise FactoryError("Upload at least one artifact before analysis")
        run.analysis = None
        run.analysis_approved = False
        run.metric_plan = None
        run.plan_approved = False
        run.phase = RunPhase.CREATED
        run.error = None
        self.store.save(run)
        return self._invoke(run.id)

    def update_analysis(self, run_id: str, analysis: ArchitectureAnalysis) -> EvaluationRun:
        run = self.store.get(run_id)
        if run.phase != RunPhase.AWAITING_ANALYSIS_REVIEW:
            raise FactoryError("Analysis can be edited only while it is awaiting review")
        analysis.system_type = run.system_type
        analysis.revised = True
        if not analysis.summary.strip() or not analysis.components:
            raise FactoryError("Analysis needs a summary and at least one component")
        run.analysis = analysis
        self.store.save(run)
        return run

    def approve_analysis(self, run_id: str) -> EvaluationRun:
        run = self.store.get(run_id)
        if run.phase != RunPhase.AWAITING_ANALYSIS_REVIEW or run.analysis is None:
            raise FactoryError("Run the analyzer before approving it")
        run.analysis_approved = True
        run.error = None
        self.store.save(run)
        return self._invoke(run.id)

    def update_plan(self, run_id: str, plan: MetricPlan) -> EvaluationRun:
        run = self._plan_review(run_id)
        if plan.system_type != run.system_type:
            raise FactoryError("The plan system type does not match the run")
        validate_plan(plan, run.rule_bundles, run.rubrics)
        run.metric_plan = plan
        self.store.save(run)
        return run

    def add_bundle(self, run_id: str, bundle: RuleBundle) -> EvaluationRun:
        run = self._plan_review(run_id)
        validate_bundle(bundle)
        run.rule_bundles.append(bundle)
        if run.metric_plan is not None:
            attach_bundle(run.metric_plan, bundle)
        self.store.save(run)
        return run

    def add_rubric(self, run_id: str, rubric: LLMRubric) -> EvaluationRun:
        run = self._plan_review(run_id)
        validate_rubric(rubric)
        run.rubrics.append(rubric)
        if run.metric_plan is not None:
            attach_rubric(run.metric_plan, rubric)
        self.store.save(run)
        return run

    def set_dataset(self, run_id: str, payload: object) -> EvaluationRun:
        run = self._plan_review(run_id)
        cases, outputs = parse_dataset(payload)
        run.cases = cases
        run.outputs = outputs
        run.data_strategy = "upload"
        self.store.save(run)
        return run

    def synthesize(self, run_id: str) -> EvaluationRun:
        run = self._plan_review(run_id)
        cases, outputs = synthesize_dataset(run, build_provider(run.provider))
        run.cases = cases
        run.outputs = outputs
        run.data_strategy = "synthesize"
        self.store.save(run)
        return run

    def capture(self, run_id: str, spec: CaptureSpec) -> EvaluationRun:
        run = self._plan_review(run_id)
        if not run.cases:
            raise FactoryError("Upload or synthesize cases before capturing outputs")
        run.outputs = capture_outputs(run.cases, spec)
        run.capture_spec = spec
        run.data_strategy = "capture"
        self.store.save(run)
        return run

    def approve_plan(self, run_id: str) -> EvaluationRun:
        run = self.store.get(run_id)
        if run.phase == RunPhase.FAILED and run.analysis_approved and run.metric_plan is not None:
            run.phase = RunPhase.AWAITING_PLAN_REVIEW
            run.error = None
        if run.phase != RunPhase.AWAITING_PLAN_REVIEW or run.metric_plan is None:
            raise FactoryError("Review the plan before running evaluation")
        if not run.cases and run.data_strategy != "synthesize":
            raise FactoryError("Add a dataset or synthesize one before running evaluation")
        validate_plan(run.metric_plan, run.rule_bundles, run.rubrics)
        run.plan_approved = True
        run.error = None
        self.store.save(run)
        return self._invoke(run.id)

    def _plan_review(self, run_id: str) -> EvaluationRun:
        run = self.store.get(run_id)
        if run.phase == RunPhase.FAILED and run.analysis_approved and run.metric_plan is not None:
            run.phase = RunPhase.AWAITING_PLAN_REVIEW
            run.plan_approved = False
            run.error = None
            self.store.save(run)
        if run.phase != RunPhase.AWAITING_PLAN_REVIEW:
            raise FactoryError("Change the plan only while it is awaiting review")
        return run

    def _invoke(self, run_id: str) -> EvaluationRun:
        run = self.store.get(run_id)
        try:
            result = self.graph.invoke(
                {"run": run.model_dump(mode="json")},
                {"configurable": {"thread_id": f"{run.id}-{new_id()[:8]}"}},
            )
        except Exception as exc:
            failure = _root_error(exc)
            message = failure.message if isinstance(failure, FactoryError) else "Evaluation failed"
            logger.exception("evaluation advance failed for %s", run_id)
            run.error = message
            run.phase = RunPhase.FAILED
            self.store.save(run)
            if isinstance(failure, FactoryError):
                raise failure
            raise FactoryError(message) from exc
        updated = EvaluationRun.model_validate(result["run"])
        self.store.save(updated)
        return updated


def _root_error(exc: Exception) -> Exception:
    current: BaseException = exc
    seen: set[int] = set()
    while current.__cause__ is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, FactoryError):
            return current
        current = current.__cause__
    return current if isinstance(current, Exception) else exc
