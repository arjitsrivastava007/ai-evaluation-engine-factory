---
name: ai-evaluation-factory
description: >-
  Runs and extends the AI Evaluation Engine Factory for RAG pipelines,
  chatbots, and agentic systems. Use when evaluating an AI system, reviewing
  architecture or a metric plan, adding deterministic rule bundles or LLM
  rubrics, capturing or synthesizing a dataset, or changing analyzers,
  evaluators, scoring, reports, or the LangGraph flow in this repository.
---

# AI Evaluation Engine Factory

Score a RAG pipeline, chatbot, or agent with one flow. Coded checks run in parallel. LLM judges run one response at a time. Reviewers approve the analysis and the metric plan before anything is scored. A run ends as `report.md` plus four dated Plotly HTML charts.

System types are `rag`, `chatbot`, and `agentic`. The default provider is `heuristic`. It needs no API key, and the same input produces the same judge score.

## Run an evaluation

From the repo root, with the API on port 8000:

```bash
python3 -m venv .venv
.venv/bin/pip install -e "backend[dev]"
PYTHONPATH=backend .venv/bin/uvicorn eval_factory.main:app --host 127.0.0.1 --port 8000
```

UI, in another shell: `cd frontend && npm install && npm run dev`, then open http://127.0.0.1:5173. The dev server proxies `/api` to port 8000.

API prefix is `/api`. Create a run, upload artifacts, analyze, approve the analysis, attach data, then approve the plan. Approving the plan is what starts scoring. Do not call evaluators before both approvals.

```bash
curl -sS -X POST http://127.0.0.1:8000/api/runs \
  -H 'content-type: application/json' \
  -d '{"name":"Acme policy RAG","system_type":"rag","provider":{"provider":"heuristic","model":"","temperature":0}}'
```

`provider` is an object, not a string. Keys stay in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EVAL_LLM_API_KEY`). Never write them into `data/{run_id}/run.json`.

Example inputs: `examples/acme_policy.md` and `examples/acme_dataset.json`.

## Guardrails

- Artifacts and re-analysis are allowed only before analysis approval.
- Plan edits, rule bundles, rubrics, datasets, synthesis, and capture are allowed only during plan review.
- A failed run that already has an approved analysis and a plan reopens plan review.
- Approve the plan only when cases exist, unless `data_strategy` is `synthesize`.
- Deterministic metrics use `mode == "deterministic"`. Do not use identity checks. `model_copy` can leave the mode as a string.
- Enabled metric weights must be above zero. The planner rejects unknown evaluator ids and duplicate metric ids.
- Capture allows http and https, including loopback and private addresses. It blocks cloud metadata hosts, link-local, and redirects.

## Change the factory

| Change | Where |
| --- | --- |
| Default metrics, risks, recommendations | `backend/eval_factory/domain/ontology.py` |
| Coded check | `backend/eval_factory/evaluators/deterministic.py` registry |
| LLM judge | `backend/eval_factory/evaluators/judges.py` `HEURISTIC_JUDGES` |
| One-off rule or rubric | A rule bundle or LLM rubric on the run, not a new ontology profile |
| Review pauses | `backend/eval_factory/workflow/graph.py` |
| HTTP shape | `backend/eval_factory/api/routes.py` |

A new built-in metric needs a template in the ontology and a function under the same `evaluator_id`. `KNOWN_EVALUATORS` is derived from the ontology plus `rule_bundle` and `llm_rubric`.

Rule kinds: `contains`, `not_contains`, `regex`, `min_length`, `max_length`, `json_object`, `required_tools`, `citation_required`.

After a behavior change, run `cd backend && ../.venv/bin/pytest`.

## More detail

Endpoint list, phases, and chart names are in [reference.md](reference.md).
