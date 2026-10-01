import { ChangeEvent, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  addBundle,
  addRubric,
  analyze,
  approveAnalysis,
  approvePlan,
  captureOutputs,
  getRun,
  saveAnalysis,
  savePlan,
  synthesize,
  uploadArtifact,
  uploadDataset,
} from "../api";
import type { Analysis, EvaluationRun, MetricPlan } from "../types";

export function Workspace() {
  const { runId = "" } = useParams();
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [plan, setPlan] = useState<MetricPlan | null>(null);
  const [note, setNote] = useState("");
  const [bundleName, setBundleName] = useState("");
  const [bundleValue, setBundleValue] = useState("");
  const [rubricName, setRubricName] = useState("");
  const [rubricPrompt, setRubricPrompt] = useState("");
  const [criterion, setCriterion] = useState("");
  const [captureUrl, setCaptureUrl] = useState("");
  const [responsePath, setResponsePath] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  useEffect(() => {
    let cancelled = false;
    getRun(runId)
      .then((loaded) => {
        if (cancelled) return;
        setRun(loaded);
        setAnalysis(loaded.analysis);
        setPlan(loaded.metric_plan);
      })
      .catch((err: Error) => setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [runId]);

  function apply(next: EvaluationRun) {
    setRun(next);
    setAnalysis(next.analysis);
    setPlan(next.metric_plan);
  }

  async function act(label: string, action: () => Promise<EvaluationRun>) {
    setBusy(label);
    setError("");
    try {
      apply(await action());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy("");
    }
  }

  if (!run) {
    return <main className="page">{error ? <p className="alert" role="alert">{error}</p> : <p>Loading evaluation…</p>}</main>;
  }

  const step = run.aggregate ? 2 : run.analysis_approved ? 1 : 0;
  const locked = run.analysis_approved;
  const planLocked = run.plan_approved && run.phase !== "failed";

  return (
    <main className="page">
      <p className="kicker">{run.system_type} · {run.provider.provider}</p>
      <h1>{run.name}</h1>
      <ol className="stepper">
        {["Analyze", "Plan", "Report"].map((label, index) => (
          <li key={label} className={index === step ? "step active" : index < step ? "step done" : "step"}>
            <strong>0{index + 1}</strong>
            {label}
          </li>
        ))}
      </ol>
      {error ? <p className="alert" role="alert">{error}</p> : null}
      {run.error && run.phase === "failed" ? <p className="alert" role="alert">{run.error}</p> : null}

      <section className="panel">
        <h2>1. Analyze the system under test</h2>
        <p>Add PDF, Markdown, JSON, YAML, DOCX, or an image. The analyzer maps components, tools, and risks, then waits for you.</p>
        <div className="row">
          <label>
            Upload artifact
            <input
              type="file"
              aria-label="Upload artifact"
              disabled={locked || busy !== ""}
              onChange={(event) => onFile(event)}
            />
          </label>
        </div>
        <label>
          Or paste Markdown
          <textarea aria-label="Paste Markdown" value={note} onChange={(event) => setNote(event.target.value)} disabled={locked} />
        </label>
        <div className="actions">
          <button className="button-secondary" type="button" disabled={locked || !note.trim() || busy !== ""} onClick={() => addNote()}>
            {busy === "note" ? "Adding…" : "Add pasted note"}
          </button>
          <button className="button" type="button" disabled={locked || run.artifacts.length === 0 || busy !== ""} onClick={() => act("analyze", () => analyze(run.id))}>
            {busy === "analyze" ? "Analyzing…" : "Run analyzer"}
          </button>
        </div>
        {run.artifacts.map((artifact) => (
          <article key={artifact.id} className="artifact">
            <strong>{artifact.filename}</strong> <span className="pill">{artifact.kind}</span>
            <p className="excerpt">{artifact.text.slice(0, 220)}</p>
          </article>
        ))}
        {analysis ? (
          <div>
            <h3>Architecture review</h3>
            <label>
              Summary
              <textarea
                aria-label="Analysis summary"
                value={analysis.summary}
                disabled={locked}
                onChange={(event) => setAnalysis({ ...analysis, summary: event.target.value })}
              />
            </label>
            <label>
              Tools
              <input
                aria-label="Tools"
                value={analysis.tools.join(", ")}
                disabled={locked}
                onChange={(event) =>
                  setAnalysis({
                    ...analysis,
                    tools: event.target.value.split(",").map((item) => item.trim()).filter(Boolean),
                  })
                }
              />
            </label>
            <label>
              Interactions
              <textarea
                aria-label="Interactions"
                value={analysis.interactions.join("\n")}
                disabled={locked}
                onChange={(event) =>
                  setAnalysis({
                    ...analysis,
                    interactions: event.target.value.split("\n").map((item) => item.trim()).filter(Boolean),
                  })
                }
              />
            </label>
            {analysis.risks.map((risk, index) => (
              <div className="row" key={risk.id}>
                <label>
                  {risk.name}
                  <select
                    aria-label={`${risk.name} severity`}
                    value={risk.severity}
                    disabled={locked}
                    onChange={(event) => {
                      const risks = analysis.risks.slice();
                      risks[index] = { ...risk, severity: event.target.value as RiskSeverity };
                      setAnalysis({ ...analysis, risks });
                    }}
                  >
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                    <option value="critical">critical</option>
                  </select>
                </label>
                <p className="excerpt">{risk.evidence}</p>
              </div>
            ))}
            <div className="actions">
              <button className="button-secondary" type="button" disabled={locked || busy !== ""} onClick={() => act("save-analysis", () => saveAnalysis(run.id, analysis))}>
                Save analysis
              </button>
              <button className="button" type="button" disabled={locked || busy !== ""} onClick={() => act("approve-analysis", () => approveAnalysis(run.id))}>
                {busy === "approve-analysis" ? "Planning…" : "Approve analysis"}
              </button>
            </div>
          </div>
        ) : null}
      </section>

      <section className="panel">
        <h2>2. Plan evaluation and prepare data</h2>
        <p>Edit weights, add a rule bundle or rubric, then bring a dataset, synthesize one, or capture live outputs.</p>
        {plan ? (
          <>
            <div className="metric-head">
              <span></span><span>Metric</span><span>Weight</span><span>Threshold</span><span>Mode</span>
            </div>
            {plan.metrics.map((metric, index) => (
              <div className="metric" key={metric.id}>
                <input
                  type="checkbox"
                  aria-label={`Enable ${metric.name}`}
                  checked={metric.enabled}
                  disabled={planLocked}
                  onChange={(event) => updateMetric(index, { enabled: event.target.checked })}
                />
                <div>
                  <strong>{metric.name}</strong>
                  <div className="tiny">{metric.description}</div>
                </div>
                <input
                  aria-label={`${metric.name} weight`}
                  type="number"
                  min={0}
                  step={0.05}
                  value={metric.weight}
                  disabled={planLocked}
                  onChange={(event) => updateMetric(index, { weight: Number(event.target.value) })}
                />
                <input
                  aria-label={`${metric.name} threshold`}
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={metric.threshold}
                  disabled={planLocked}
                  onChange={(event) => updateMetric(index, { threshold: Number(event.target.value) })}
                />
                <span className={metric.mode === "deterministic" ? "pill det" : "pill"}>{metric.mode}</span>
              </div>
            ))}
            <div className="actions">
              <button className="button-secondary" type="button" disabled={!run.analysis_approved || planLocked || busy !== ""} onClick={() => act("save-plan", () => savePlan(run.id, plan))}>
                Save metric plan
              </button>
            </div>
          </>
        ) : <p>Approve the analysis to generate the metric plan.</p>}

        <div className="grid-2">
          <div>
            <h3>Rule bundle</h3>
            <label>
              Bundle name
              <input aria-label="Bundle name" value={bundleName} onChange={(event) => setBundleName(event.target.value)} />
            </label>
            <label>
              Response must contain
              <input aria-label="Required text" value={bundleValue} onChange={(event) => setBundleValue(event.target.value)} />
            </label>
            <button
              className="button-secondary"
              type="button"
              disabled={!run.analysis_approved || planLocked || busy !== ""}
              onClick={() =>
                act("bundle", async () => {
                  const next = await addBundle(run.id, {
                    name: bundleName,
                    description: `Response must contain “${bundleValue}”.`,
                    rules: [{ name: "contains", kind: "contains", params: { value: bundleValue } }],
                  });
                  setBundleName("");
                  setBundleValue("");
                  return next;
                })
              }
            >
              Add rule bundle
            </button>
          </div>
          <div>
            <h3>LLM rubric</h3>
            <label>
              Rubric name
              <input aria-label="Rubric name" value={rubricName} onChange={(event) => setRubricName(event.target.value)} />
            </label>
            <label>
              Prompt
              <input aria-label="Rubric prompt" value={rubricPrompt} onChange={(event) => setRubricPrompt(event.target.value)} />
            </label>
            <label>
              Criterion
              <input aria-label="Criterion" value={criterion} onChange={(event) => setCriterion(event.target.value)} />
            </label>
            <button
              className="button-secondary"
              type="button"
              disabled={!run.analysis_approved || planLocked || busy !== ""}
              onClick={() =>
                act("rubric", async () => {
                  const next = await addRubric(run.id, {
                    name: rubricName,
                    prompt: rubricPrompt,
                    criteria: [{ name: criterion, description: criterion }],
                  });
                  setRubricName("");
                  setRubricPrompt("");
                  setCriterion("");
                  return next;
                })
              }
            >
              Add rubric
            </button>
          </div>
        </div>

        <h3>Dataset</h3>
        <div className="row">
          <label>
            Upload dataset JSON
            <input aria-label="Upload dataset" type="file" accept="application/json,.json" disabled={!run.analysis_approved || planLocked || busy !== ""} onChange={(event) => onDataset(event)} />
          </label>
          <label>
            Capture URL
            <input aria-label="Capture URL" value={captureUrl} onChange={(event) => setCaptureUrl(event.target.value)} placeholder="http://127.0.0.1:9000/run" />
          </label>
          <label>
            Response path
            <input aria-label="Response path" value={responsePath} onChange={(event) => setResponsePath(event.target.value)} placeholder="data.answer" />
          </label>
        </div>
        <div className="actions">
          <button className="button-secondary" type="button" disabled={!run.analysis_approved || planLocked || busy !== ""} onClick={() => act("synthesize", () => synthesize(run.id))}>
            {busy === "synthesize" ? "Synthesizing…" : "Synthesize dataset"}
          </button>
          <button
            className="button-secondary"
            type="button"
            disabled={!run.analysis_approved || planLocked || run.cases.length === 0 || !captureUrl || busy !== ""}
            onClick={() => act("capture", () => captureOutputs(run.id, { url: captureUrl, method: "POST", response_path: responsePath }))}
          >
            Capture outputs
          </button>
          <button className="button" type="button" disabled={!run.analysis_approved || planLocked || run.cases.length === 0 || busy !== ""} onClick={() => act("evaluate", () => approvePlan(run.id))}>
            {busy === "evaluate" ? "Evaluating…" : "Approve plan and evaluate"}
          </button>
        </div>
        <p className="tiny">{run.cases.length} cases · strategy {run.data_strategy}</p>
        {run.cases.slice(0, 4).map((item) => (
          <article key={item.id} className="case">
            <strong>{item.id}</strong>
            <p className="excerpt">{item.input}</p>
          </article>
        ))}
      </section>

      <section className="panel">
        <h2>3. Scores and report</h2>
        {run.aggregate ? (
          <>
            <div className="score-row">
              <div className="dial" style={{ ["--p" as string]: String(run.aggregate.overall) }}>
                <span>{run.aggregate.overall.toFixed(2)}</span>
              </div>
              <div>
                <span className={`stamp risk-${run.aggregate.risk_level}`}>{run.aggregate.risk_level} risk</span>
                <p>{run.aggregate.decision}</p>
                <p className="tiny">
                  Deterministic {formatScore(run.aggregate.deterministic)} · Probabilistic {formatScore(run.aggregate.probabilistic)} · {run.aggregate.case_count} cases
                </p>
              </div>
            </div>
            <ul>
              {run.aggregate.recommendations.map((item) => <li key={item}>{item}</li>)}
            </ul>
            <div className="charts">
              {run.chart_files.map((name) => (
                <iframe key={name} title={name} src={`/api/runs/${run.id}/charts/${name}`} />
              ))}
            </div>
            <h3>Markdown report</h3>
            <pre className="report">{run.report_markdown}</pre>
          </>
        ) : (
          <p>Scores appear here after you approve the plan. Nothing runs before that.</p>
        )}
        <p><Link to="/">Back to evaluations</Link></p>
      </section>
    </main>
  );

  function updateMetric(index: number, patch: Partial<MetricPlan["metrics"][number]>) {
    if (!plan) return;
    const metrics = plan.metrics.slice();
    metrics[index] = { ...metrics[index], ...patch };
    setPlan({ ...plan, metrics });
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    void act("upload", () => uploadArtifact(runId, file));
  }

  function addNote() {
    const file = new File([note], "notes.md", { type: "text/markdown" });
    void act("note", async () => {
      const next = await uploadArtifact(runId, file);
      setNote("");
      return next;
    });
  }

  function onDataset(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    void act("dataset", async () => {
      const payload = JSON.parse(await file.text()) as unknown;
      return uploadDataset(runId, payload);
    });
  }
}

type RiskSeverity = "low" | "medium" | "high" | "critical";

function formatScore(value: number | null) {
  return value === null ? "n/a" : value.toFixed(2);
}
