# Evaluation approach

The factory measures a system the way a review would, with the checks written down first. The system type selects the metrics. A person approves that selection. Only then do the evaluators run.

## One engine, three system types

`rag`, `chatbot`, and `agentic` share the graph. They do not share the scorecard. The ontology in `backend/eval_factory/domain/ontology.py` maps each type to default components, risks, metrics, evaluator ids, thresholds, and recommendation text. Enabled weights on a fresh plan sum to 1.

| Type | What the scorecard asks |
| --- | --- |
| RAG | Did the answer stay inside retrieved context, cite it, and cover the question? |
| Chatbot | Was the reply helpful, on tone, safe, and finished? |
| Agent | Did the agent pick the right tools, stay inside a step budget, and meet the goal? |

The Solution Analyzer reads the uploaded artifacts and fills components, tools, interactions, and risks. The heuristic analyzer uses the ontology plus cues in the text, such as a `Tools:` heading or words like "cite" and "step limit". A live provider must return an analysis with a summary and at least one component. The reviewer can edit that analysis before approval. Approval is what generates the metric plan.

## Two modes, one score

Every metric is either deterministic or probabilistic.

Deterministic evaluators are ordinary Python functions. They run in parallel. The same case and the same rules produce the same score. Built-in checks cover overlap with context, citations, expected facts, safety screens, format bounds, tool choice, tool arguments, and step budget. A rule bundle is also deterministic. Its kinds are `contains`, `not_contains`, `regex`, `min_length`, `max_length`, `json_object`, `required_tools`, and `citation_required`.

Probabilistic evaluators are judges. They run strictly in series so one judgment does not race another. The heuristic provider uses fixed formulas and is repeatable. OpenAI, Anthropic, and OpenAI-compatible endpoints are real model judgments. A custom LLM rubric is a judge metric. The model is asked for a JSON object with `score` and `rationale`.

Aggregation keeps both means and a single overall. The overall is the weighted average of enabled metrics that actually produced a score. Weights are renormalized across that set. A metric passes when its score is at least its threshold.

## Risk and the decision

| Risk | When |
| --- | --- |
| Critical | No scored metrics, overall below 0.45, or any metric with normalized weight of at least 0.15 scoring below 0.35 |
| High | Overall below 0.65 |
| Medium | Overall below 0.80 |
| Low | Otherwise |

The decision sentence is fixed for each risk level. Recommendations come from the ontology text for failed built-in metrics, plus a note for a failed bundle or rubric. If the deterministic mean and the probabilistic mean differ by 0.20 or more, the report says to inspect that disagreement before treating the overall as final.

## Data, without blocking the run

A dataset is a list of cases, or `{cases: [...]}`. A case has `id`, `input`, `expected`, `context`, `reference_tools`, `metadata`, and an optional output. The output carries `response`, `contexts`, `tool_calls`, and `steps`.

Three ways to get those outputs:

1. Upload a dataset that already includes them.
2. Synthesize cases and outputs from the artifact text. The heuristic path uses stable ids `synth-1`, `synth-2`, and so on. For RAG, the last synthetic answer appends an unsupported sentence so the scores can spread.
3. Capture live outputs. The factory calls the system under test and reads the response with dot-notation, or from `response`, `output`, `answer`, `text`, or `content`.

Approve plan requires cases unless the data strategy is `synthesize`.

## What a person still decides

The graph will not score a run while analysis or the plan is unapproved. During plan review the user can change weights and thresholds, disable a metric, add a rule bundle, add a rubric, and choose the dataset. That is the definition of "good" for the engagement. The factory proposes it. The reviewer accepts it.
