# Evaluation factory reference

Read this when the task needs an endpoint, a phase, or a file path. The workflow itself is in [SKILL.md](SKILL.md).

## Phases

`created` → `awaiting_analysis_review` → `awaiting_plan_review` → `running` → `completed`

`failed` can reopen plan review when the analysis was already approved and a plan exists.

Graph nodes: `analyze`, `plan`, `prepare`, `evaluate_deterministic`, `evaluate_judges`, `publish`. The graph interrupts after `analyze` and after `plan`. Each advance uses a fresh thread id. Durable state is `data/{run_id}/run.json`, not the LangGraph checkpoint.

## HTTP

| Method | Path | Effect |
| --- | --- | --- |
| GET | `/api/health` | Liveness |
| GET | `/api/ontology` | Profiles for `rag`, `chatbot`, `agentic` |
| POST | `/api/runs` | Create. Body: `name`, `system_type`, `provider` object |
| GET | `/api/runs` | Summaries |
| GET | `/api/runs/{id}` | Full run |
| POST | `/api/runs/{id}/artifacts` | Multipart file. PDF, Markdown, JSON, YAML, DOCX, image, txt |
| POST | `/api/runs/{id}/analyze` | Solution Analyzer, then pause |
| PUT | `/api/runs/{id}/analysis` | Edit the analysis |
| POST | `/api/runs/{id}/analysis/approve` | Approve and generate the metric plan |
| PUT | `/api/runs/{id}/plan` | Edit weights, thresholds, enabled flags |
| POST | `/api/runs/{id}/rule-bundles` | Add a deterministic bundle |
| POST | `/api/runs/{id}/rubrics` | Add an LLM rubric |
| POST | `/api/runs/{id}/dataset` | JSON list of cases, or `{cases: [...]}` |
| POST | `/api/runs/{id}/dataset/synthesize` | Output-generation graph |
| POST | `/api/runs/{id}/capture` | POST or GET live outputs. Body template may use `{{input}}`, `{{expected}}`, `{{context}}`, `{{case_id}}` |
| POST | `/api/runs/{id}/plan/approve` | Score and publish |
| GET | `/api/runs/{id}/report` | `text/markdown` |
| GET | `/api/runs/{id}/charts/{filename}` | One HTML chart |

Chart names: `score-by-metric-YYYY-MM-DD.html`, `mode-comparison-YYYY-MM-DD.html`, `case-risk-YYYY-MM-DD.html`, `case-heatmap-YYYY-MM-DD.html`. The date comes from `run.updated_at`.

## Case shape

`id`, `input`, `expected`, `context`, `reference_tools`, `metadata`, and optional `output` with `response`, `contexts`, `tool_calls[{name, arguments}]`, and `steps`.

## Providers

| Provider | Key | Default model |
| --- | --- | --- |
| `heuristic` | none | offline judges |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-latest` |
| `openai_compatible` | `EVAL_LLM_API_KEY` | `EVAL_LLM_MODEL` or `gpt-4o-mini` |

Compatible endpoints also need `base_url` on the run or `EVAL_LLM_BASE_URL`. Temperature is 0 to 2 and defaults to 0.

Other settings use the `EVAL_` prefix: `EVAL_DATA_DIR`, `EVAL_CORS_ORIGINS`. Upload limit is 10 MB. Extracted text is capped at 200,000 characters. Images store format, size, and mode only.

## Layout

- `backend/eval_factory/domain` — models and ontology
- `backend/eval_factory/artifacts` — ingest
- `backend/eval_factory/analysis` — solution analyzer
- `backend/eval_factory/planning` — metric plans, bundles, rubrics
- `backend/eval_factory/data` — datasets, synthesis, capture
- `backend/eval_factory/evaluators` — parallel coded checks and sequential judges
- `backend/eval_factory/scoring` — weighted aggregate and risk
- `backend/eval_factory/reporting` — Markdown and Plotly
- `backend/eval_factory/workflow` — LangGraph service
- `frontend` — review UI
- `examples` — sample RAG artifact and dataset
