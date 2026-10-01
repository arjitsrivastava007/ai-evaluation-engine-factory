import type { Analysis, EvaluationRun, MetricPlan, OntologyProfile, RunSummary, SystemType } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const contentType = response.headers.get("content-type") ?? "";
  if (!response.ok) {
    let detail = response.statusText;
    if (contentType.includes("json")) {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    }
    throw new Error(detail || "Request failed");
  }
  if (contentType.includes("text/markdown") || contentType.startsWith("text/plain")) {
    return (await response.text()) as T;
  }
  return (await response.json()) as T;
}

export function listRuns(): Promise<RunSummary[]> {
  return request("/api/runs");
}

export function getOntology(): Promise<OntologyProfile[]> {
  return request("/api/ontology");
}

export function createRun(input: {
  name: string;
  system_type: SystemType;
  provider: string;
  model: string;
}): Promise<EvaluationRun> {
  return request("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: input.name,
      system_type: input.system_type,
      provider: { provider: input.provider, model: input.model, temperature: 0 },
    }),
  });
}

export function getRun(id: string): Promise<EvaluationRun> {
  return request(`/api/runs/${id}`);
}

export function uploadArtifact(id: string, file: File): Promise<EvaluationRun> {
  const body = new FormData();
  body.append("file", file);
  return request(`/api/runs/${id}/artifacts`, { method: "POST", body });
}

export function analyze(id: string): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/analyze`, { method: "POST" });
}

export function saveAnalysis(id: string, analysis: Analysis): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/analysis`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(analysis),
  });
}

export function approveAnalysis(id: string): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/analysis/approve`, { method: "POST" });
}

export function savePlan(id: string, plan: MetricPlan): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/plan`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(plan),
  });
}

export function addBundle(
  id: string,
  bundle: { name: string; description: string; rules: { name: string; kind: string; params: Record<string, unknown> }[] },
): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/rule-bundles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bundle),
  });
}

export function addRubric(
  id: string,
  rubric: { name: string; prompt: string; criteria: { name: string; description: string }[] },
): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/rubrics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rubric),
  });
}

export function uploadDataset(id: string, payload: unknown): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/dataset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function synthesize(id: string): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/dataset/synthesize`, { method: "POST" });
}

export function captureOutputs(
  id: string,
  spec: { url: string; method: "GET" | "POST"; response_path: string },
): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
}

export function approvePlan(id: string): Promise<EvaluationRun> {
  return request(`/api/runs/${id}/plan/approve`, { method: "POST" });
}
