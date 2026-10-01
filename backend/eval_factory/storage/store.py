"""JSON files for evaluation runs. The directory is the source of truth."""

from __future__ import annotations

import re
from pathlib import Path

from eval_factory.domain.models import EvaluationRun
from eval_factory.errors import FactoryError

_RUN_ID = re.compile(r"^[a-f0-9]{32}$")
_CHART_NAME = re.compile(r"^[a-z0-9-]+\.html$")


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: EvaluationRun) -> None:
        run.touch()
        directory = self._directory(run.id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "run.json.tmp"
        temporary.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(directory / "run.json")

    def get(self, run_id: str) -> EvaluationRun:
        path = self._directory(run_id) / "run.json"
        if not path.is_file():
            raise FactoryError("Evaluation run was not found", status_code=404)
        return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[EvaluationRun]:
        runs: list[EvaluationRun] = []
        for path in self.root.glob("*/run.json"):
            runs.append(EvaluationRun.model_validate_json(path.read_text(encoding="utf-8")))
        runs.sort(key=lambda run: run.created_at, reverse=True)
        return runs

    def artifact_path(self, run_id: str, filename: str) -> Path:
        directory = self._directory(run_id) / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    def chart_path(self, run_id: str, filename: str) -> Path:
        if not _CHART_NAME.fullmatch(filename):
            raise FactoryError("Chart was not found", status_code=404)
        path = self._directory(run_id) / "charts" / filename
        if not path.is_file():
            raise FactoryError("Chart was not found", status_code=404)
        return path

    def report_text(self, run_id: str) -> str:
        run = self.get(run_id)
        if not run.report_markdown:
            raise FactoryError("This run has no report yet", status_code=404)
        return run.report_markdown

    def _directory(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise FactoryError("Evaluation run was not found", status_code=404)
        return self.root / run_id
