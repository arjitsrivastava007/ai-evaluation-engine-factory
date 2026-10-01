"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from eval_factory import __version__
from eval_factory.api.routes import router
from eval_factory.config import Settings
from eval_factory.errors import FactoryError
from eval_factory.storage.store import RunStore
from eval_factory.workflow.service import EvaluationService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="AI Evaluation Engine Factory",
        version=__version__,
        summary="Turn artifacts and datasets into scored evaluations for RAG, chatbot, and agentic systems.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = EvaluationService(RunStore(settings.data_dir), settings)
    app.include_router(router, prefix="/api")

    @app.exception_handler(FactoryError)
    def _factory_error(_request, exc: FactoryError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    return app
