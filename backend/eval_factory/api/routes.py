"""FastAPI routes. The service owns the workflow; these handlers only translate HTTP."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from eval_factory.domain.models import (
    ArchitectureAnalysis,
    CaptureSpec,
    LLMRubric,
    MetricPlan,
    ProviderConfig,
    RuleBundle,
    SystemType,
)
from eval_factory.domain.ontology import PROFILES
from eval_factory.errors import FactoryError
from eval_factory.workflow.service import EvaluationService

router = APIRouter()


class CreateRunRequest(BaseModel):
    name: str
    system_type: SystemType
    provider: ProviderConfig = Field(default_factory=ProviderConfig)


def _service(request: Request) -> EvaluationService:
    return request.app.state.service


def _summary(run) -> dict:
    return {
        "id": run.id,
        "name": run.name,
        "system_type": run.system_type.value,
        "phase": run.phase.value,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "overall": run.aggregate.overall if run.aggregate else None,
        "risk_level": run.aggregate.risk_level.value if run.aggregate else None,
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-evaluation-engine-factory"}


@router.get("/ontology")
def ontology() -> list[dict]:
    profiles = []
    for profile in PROFILES.values():
        profiles.append(
            {
                "system_type": profile.system_type.value,
                "label": profile.label,
                "summary": profile.summary,
                "metrics": [asdict(metric) | {"mode": metric.mode.value} for metric in profile.metrics],
                "risks": [asdict(risk) for risk in profile.risks],
            }
        )
    return profiles


@router.post("/runs", status_code=201)
def create_run(body: CreateRunRequest, request: Request):
    run = _service(request).create_run(body.name, body.system_type, body.provider)
    return run


@router.get("/runs")
def list_runs(request: Request) -> list[dict]:
    return [_summary(run) for run in _service(request).store.list_runs()]


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    return _service(request).store.get(run_id)


@router.post("/runs/{run_id}/artifacts")
async def upload_artifact(run_id: str, request: Request, file: UploadFile = File(...)):
    service = _service(request)
    data = await file.read(service.settings.max_upload_bytes + 1)
    if len(data) > service.settings.max_upload_bytes:
        raise FactoryError("Artifact exceeds the 10 MB upload limit")
    if not file.filename:
        raise FactoryError("The upload needs a filename")
    return service.add_artifact(run_id, file.filename, data)


@router.post("/runs/{run_id}/analyze")
def analyze(run_id: str, request: Request):
    return _service(request).analyze(run_id)


@router.put("/runs/{run_id}/analysis")
def update_analysis(run_id: str, body: ArchitectureAnalysis, request: Request):
    return _service(request).update_analysis(run_id, body)


@router.post("/runs/{run_id}/analysis/approve")
def approve_analysis(run_id: str, request: Request):
    return _service(request).approve_analysis(run_id)


@router.put("/runs/{run_id}/plan")
def update_plan(run_id: str, body: MetricPlan, request: Request):
    return _service(request).update_plan(run_id, body)


@router.post("/runs/{run_id}/rule-bundles")
def add_bundle(run_id: str, body: RuleBundle, request: Request):
    return _service(request).add_bundle(run_id, body)


@router.post("/runs/{run_id}/rubrics")
def add_rubric(run_id: str, body: LLMRubric, request: Request):
    return _service(request).add_rubric(run_id, body)


@router.post("/runs/{run_id}/dataset")
async def set_dataset(run_id: str, request: Request):
    payload = await request.json()
    return _service(request).set_dataset(run_id, payload)


@router.post("/runs/{run_id}/dataset/synthesize")
def synthesize(run_id: str, request: Request):
    return _service(request).synthesize(run_id)


@router.post("/runs/{run_id}/capture")
def capture(run_id: str, body: CaptureSpec, request: Request):
    return _service(request).capture(run_id, body)


@router.post("/runs/{run_id}/plan/approve")
def approve_plan(run_id: str, request: Request):
    return _service(request).approve_plan(run_id)


@router.get("/runs/{run_id}/report", response_class=PlainTextResponse)
def report(run_id: str, request: Request) -> PlainTextResponse:
    return PlainTextResponse(_service(request).store.report_text(run_id), media_type="text/markdown")


@router.get("/runs/{run_id}/charts/{filename}")
def chart(run_id: str, filename: str, request: Request) -> FileResponse:
    path = _service(request).store.chart_path(run_id, filename)
    return FileResponse(path, media_type="text/html")
