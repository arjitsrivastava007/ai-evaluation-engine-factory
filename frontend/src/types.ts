export type SystemType = "rag" | "chatbot" | "agentic";

export type Phase =
  | "created"
  | "awaiting_analysis_review"
  | "awaiting_plan_review"
  | "running"
  | "completed"
  | "failed";

export interface RunSummary {
  id: string;
  name: string;
  system_type: SystemType;
  phase: Phase;
  created_at: string;
  updated_at: string;
  overall: number | null;
  risk_level: string | null;
}

export interface Artifact {
  id: string;
  filename: string;
  kind: string;
  text: string;
}

export interface ComponentModel {
  name: string;
  role: string;
  description: string;
}

export interface RiskFinding {
  id: string;
  name: string;
  severity: "low" | "medium" | "high" | "critical";
  description: string;
  evidence: string;
}

export interface Analysis {
  system_type: SystemType;
  summary: string;
  components: ComponentModel[];
  interactions: string[];
  tools: string[];
  risks: RiskFinding[];
  assumptions: string[];
  revised: boolean;
}

export interface MetricSpec {
  id: string;
  name: string;
  description: string;
  weight: number;
  mode: "deterministic" | "probabilistic";
  evaluator_id: string;
  threshold: number;
  enabled: boolean;
  config: Record<string, unknown>;
}

export interface MetricPlan {
  system_type: SystemType;
  metrics: MetricSpec[];
  notes: string;
}

export interface EvalCase {
  id: string;
  input: string;
  expected: string | null;
  context: string[];
  reference_tools: string[];
}

export interface SystemOutput {
  case_id: string;
  response: string;
}

export interface Aggregate {
  overall: number;
  deterministic: number | null;
  probabilistic: number | null;
  risk_level: string;
  metrics: {
    metric_id: string;
    name: string;
    mode: string;
    score: number;
    passed: boolean;
    normalized_weight: number;
  }[];
  recommendations: string[];
  case_count: number;
  decision: string;
}

export interface EvaluationRun {
  id: string;
  name: string;
  system_type: SystemType;
  phase: Phase;
  provider: {
    provider: string;
    model: string;
    temperature: number;
    base_url: string | null;
  };
  artifacts: Artifact[];
  analysis: Analysis | null;
  analysis_approved: boolean;
  metric_plan: MetricPlan | null;
  plan_approved: boolean;
  cases: EvalCase[];
  outputs: SystemOutput[];
  data_strategy: string;
  aggregate: Aggregate | null;
  report_markdown: string | null;
  chart_files: string[];
  error: string | null;
  updated_at: string;
}

export interface OntologyProfile {
  system_type: SystemType;
  label: string;
  summary: string;
}
