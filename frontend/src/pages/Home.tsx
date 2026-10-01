import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createRun, getOntology, listRuns } from "../api";
import type { OntologyProfile, RunSummary, SystemType } from "../types";

const FALLBACK: OntologyProfile[] = [
  {
    system_type: "rag",
    label: "RAG pipeline",
    summary: "Grounding, citations, and whether the answer stayed inside retrieved context.",
  },
  {
    system_type: "chatbot",
    label: "Chatbot",
    summary: "Helpfulness, tone, safety, and whether the reply finished the task.",
  },
  {
    system_type: "agentic",
    label: "Agent",
    summary: "Tool choice, a bounded plan, and whether the run actually met the goal.",
  },
];

export function Home() {
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<OntologyProfile[]>(FALLBACK);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [systemType, setSystemType] = useState<SystemType>("rag");
  const [name, setName] = useState("");
  const [provider, setProvider] = useState("heuristic");
  const [model, setModel] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getOntology().then(setProfiles).catch(() => setProfiles(FALLBACK));
    listRuns().then(setRuns).catch(() => setRuns([]));
  }, []);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const run = await createRun({ name, system_type: systemType, provider, model });
      navigate(`/runs/${run.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start the run");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="page">
      <p className="kicker">One engine, three system types</p>
      <h1>Stop guessing whether your AI is good enough.</h1>
      <p className="lede">
        Upload the artifacts, review the architecture, edit the metric plan, then score the system
        with coded checks and LLM judges. The result is a risk level, a recommendation, and a report.
      </p>
      <form onSubmit={onSubmit}>
        <div className="grid-3">
          {profiles.map((profile) => (
            <button
              key={profile.system_type}
              type="button"
              className={profile.system_type === systemType ? "card choice selected" : "card choice"}
              onClick={() => setSystemType(profile.system_type)}
            >
              <p className="kicker">{profile.label}</p>
              <h2>{profile.system_type === "agentic" ? "Agent" : profile.label.split(" ")[0]}</h2>
              <p>{profile.summary}</p>
            </button>
          ))}
        </div>
        <div className="panel">
          <h2>Start an evaluation</h2>
          <div className="row">
            <label>
              Run name
              <input value={name} onChange={(event) => setName(event.target.value)} required placeholder="Acme policy RAG" />
            </label>
            <label>
              Provider
              <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                <option value="heuristic">Heuristic (offline)</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="openai_compatible">OpenAI-compatible</option>
              </select>
            </label>
            <label>
              Model
              <input value={model} onChange={(event) => setModel(event.target.value)} placeholder="Optional override" />
            </label>
          </div>
          {error ? <p className="alert" role="alert">{error}</p> : null}
          <div className="actions">
            <button className="button" type="submit" disabled={busy}>
              {busy ? "Starting…" : "Start evaluation"}
            </button>
          </div>
          <p className="tiny">
            The heuristic provider scores judges locally, so a run does not need an API key. Set
            OPENAI_API_KEY, ANTHROPIC_API_KEY, or EVAL_LLM_API_KEY when you want a live model.
          </p>
        </div>
      </form>
      <section className="modes">
        <div className="mode det">
          <span>Deterministic</span>
          <h3>Same input, same score</h3>
          <p>Coded checks run in parallel. Citations, tool schemas, safety screens, and your rule bundles do not drift.</p>
        </div>
        <div className="mode prob">
          <span>Probabilistic</span>
          <h3>Judgment, one response at a time</h3>
          <p>LLM judges run in series over faithfulness, tone, plan quality, and any rubric you add.</p>
        </div>
      </section>
      <section className="panel">
        <h2>Recent runs</h2>
        {runs.length === 0 ? <p>No evaluations yet.</p> : (
          <ul className="run-list">
            {runs.map((run) => (
              <li key={run.id}>
                <Link to={`/runs/${run.id}`}>
                  <span>{run.name}</span>
                  <span className="phase">
                    {run.system_type} · {run.phase.replaceAll("_", " ")}
                    {run.overall !== null ? ` · ${run.overall.toFixed(2)}` : ""}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
