# Design

These are the choices that keep a run repeatable and reviewable. Change them only when the product decision changes with them.

## The saved run is the source of truth

LangGraph's checkpointer is `MemorySaver`. It is process-local and is not resumed across requests. Every advance uses a new thread id. `_route` looks at the durable run: phase, whether analysis exists, whether it is approved, whether a plan exists, and whether that plan is approved.

`RunStore` writes `run.json` through a temporary file in the same directory, then replaces the destination. A crash during the write leaves the previous run intact. Run ids are `uuid4` hex. Chart downloads only allow filenames that match `^[a-z0-9-]+\.html$`.

API keys never enter the run file. Provider selection stores the provider name, model, temperature, and optional base URL.

## Reviews are phase gates, not comments

Uploading an artifact clears any previous analysis and plan and returns the run to `created`. Analysis can be edited only in `awaiting_analysis_review`, and the edit must keep a summary and at least one component. Approving analysis is the call that builds the default plan and attaches bundles and rubrics already on the run.

Plan mutations require plan review. A failed run that already has an approved analysis and a plan clears `plan_approved` and reopens that review. A completed run leaves both the analysis and the plan locked.

The UI uses the same rules. Analysis controls disable once `analysis_approved` is set. Plan controls disable when the plan is approved and the phase is not `failed`.

## Evaluators stay pure functions of the case

A coded evaluator receives the metric, the case, and the output. It returns a score in `[0, 1]`, rounded to four decimals, plus a rationale and whether the metric applied. Mode filters use equality. `model_copy` can store `"deterministic"` as a string, so an identity check against the enum would drop every coded metric.

New built-in metrics are added in two places that must agree: a template in the ontology, and a function registered under the same `evaluator_id`. `KNOWN_EVALUATORS` is derived from the ontology plus `rule_bundle` and `llm_rubric`. The planner rejects an unknown evaluator, a duplicate metric id, an enabled weight of zero, and a bundle or rubric metric whose id is missing.

Prefer a rule bundle or a rubric for a one-off engagement. Extend the ontology when every future run of that system type should start with the metric.

## Scoring is explicit

The overall score is not a blend chosen at publish time. It is the weight-normalized mean of the enabled metrics that returned an applicable score. Deterministic and probabilistic means are computed the same way on their own subsets, so a report can show both rigor and judgment without hiding either.

Risk thresholds live in `scoring/aggregate.py` and are the product rule, not a per-run setting. Recommendations prefer the ontology sentence for a failed built-in metric.

## Providers are interchangeable

`LLMProvider` is a small protocol: `complete_text` and `complete_json`. The analyzer, the output-generation graph, and the judges all call it. The heuristic implementation does not leave the process. Live providers use `httpx` against the vendor URL. Temperature defaults to 0 and must sit between 0 and 2.

Heuristic judges are labeled probabilistic so the pipeline shape matches a live model. Their numbers do not vary between runs. A live provider can.

## Ingest records what the file contains

Supported kinds are PDF, Markdown, JSON, YAML, DOCX, png, jpg, jpeg, webp, gif, and plain text. YAML uses `safe_load`. Extracted text is capped at 200,000 characters. Uploads are capped at 10 MB. Filenames are sanitized. An image contributes format, size, and mode. The analyzer does not invent a visual description.

## The UI is a review surface

The workspace is one page with three steps: Analyze, Plan, Report. It does not hide later steps. Before approval, the plan section tells the reviewer to approve the analysis first, and the report section says nothing runs before plan approval. After publish, the same page shows the score, the risk stamp, recommendations, the four chart iframes, and the Markdown report.

The home page lists recent runs from `GET /api/runs`. Summaries include id, name, system type, phase, timestamps, overall, and risk. They do not include the full case payload.

## Tests follow the seams

`backend/tests` covers the ontology, ingest, providers, analyzer, planner, datasets, evaluators, aggregation, reporting, the workflow pauses, and the HTTP API. Run them with `cd backend && ../.venv/bin/pytest`. A workflow or scoring change belongs with a test that would fail if the pause, the weight, or the risk band moved.
