# AI Evaluation Engine Factory

Stop guessing whether an AI system is good enough. This factory turns artifacts and datasets into a scored evaluation for a RAG pipeline, a chatbot, or an agent.

One workflow covers all three. Coded checks run in parallel and stay repeatable. LLM judges run one response at a time. Reviewers approve the architecture and the metric plan before anything is scored. The run ends as a Markdown report plus four dated Plotly charts.

## What a run does

1. **Analyze.** Upload PDF, Markdown, JSON, YAML, DOCX, or image artifacts. The Solution Analyzer maps components, tools, and risks, then pauses so you can edit that analysis.
2. **Plan.** The factory proposes a metric plan for the system type. Edit weights and thresholds, add your own deterministic rule bundles and LLM rubrics, then supply a dataset, synthesize one, or capture live outputs from an HTTP API.
3. **Evaluate and publish.** Deterministic evaluators and LLM judges score the outputs. The factory aggregates a risk level and recommendations, then writes `report.md` and four HTML charts.

The default provider is `heuristic`. It needs no API key, and the same input produces the same judge score. Point a run at OpenAI, Anthropic, or any OpenAI-compatible endpoint when you want a live model.

## Run it locally

Python 3.11+ and Node 20+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e "backend[dev]"
PYTHONPATH=backend .venv/bin/uvicorn eval_factory.main:app --host 127.0.0.1 --port 8000
```

In another shell:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. The dev server proxies `/api` to port 8000.

An example RAG artifact and dataset live in `examples/`. Paste the Markdown into a run, or upload `examples/acme_policy.md` and `examples/acme_dataset.json`.

## Providers

| Provider | Key | Default model |
| --- | --- | --- |
| `heuristic` | none | offline judges |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-latest` |
| `openai_compatible` | `EVAL_LLM_API_KEY` | `EVAL_LLM_MODEL` or `gpt-4o-mini` |

Compatible endpoints also need `base_url` on the run or `EVAL_LLM_BASE_URL`. Keys stay in the environment. They are not written into the run file.

Other settings use the `EVAL_` prefix: `EVAL_DATA_DIR` and `EVAL_CORS_ORIGINS`.

## Tests

```bash
cd backend
../.venv/bin/pytest
```

## Docker

The API image is `backend/Dockerfile`. From the repo root:

```bash
docker compose up --build
```

Run the frontend dev server separately and point it at http://127.0.0.1:8000.

## Layout

- `backend/eval_factory/domain` — run state and the evaluation ontology
- `backend/eval_factory/analysis` — solution analyzer
- `backend/eval_factory/planning` — metric plans, rule bundles, rubrics
- `backend/eval_factory/data` — datasets, the output-generation graph, API capture
- `backend/eval_factory/evaluators` — parallel coded checks and sequential judges
- `backend/eval_factory/workflow` — LangGraph flow with the two review pauses
- `backend/eval_factory/reporting` — Markdown report and Plotly charts
- `frontend` — the guided review UI
