// TypeScript mirrors of the Pydantic models and response shapes in src/api/app.py.

export type ModelName = "tfidf" | "codebert";
/** summarizer = CodeT5 LoRA explainer; code_summarizer = fully fine-tuned CodeT5 summary model. */
export type ComponentName = ModelName | "summarizer" | "code_summarizer";
export type ExplanationMode = "deterministic" | "lora";
export type Label = "BUGGY" | "CLEAN";

export type ModelStatus =
  | "loaded"
  | "artifacts_present_not_loaded"
  | "load_failed"
  | "unavailable";

export interface HealthResponse {
  status: string;
  models: Record<ComponentName, { status: ModelStatus }>;
}

/** AnalysisRequest (extra="forbid"): optional fields must be omitted, never null. */
export interface AnalysisRequest {
  code: string;
  model: ModelName;
  original_code?: string;
  explanation: ExplanationMode;
  /** Also generate a CodeT5 summary. Send only when true. */
  summarize?: boolean;
}

export type SummarySource = "codet5_finetuned" | "unavailable";

/** POST /summarize body (extra="forbid"). */
export interface SummaryRequest {
  code: string;
}

export interface SummaryResponse {
  summary: string;
  summary_source: "codet5_finetuned";
  model: "codet5_summarizer";
}

export type ExplanationSource =
  | "verified_code_pair"
  | "operator_flip"
  | "lora_generated_unverified"
  | "unavailable";

/** One operator-flip repair, ranked by how much it lowers the bug probability. */
export interface RepairCandidate {
  line: number;
  old: string;
  new: string;
  mutation_type: string;
  drop: number;
}

export interface AnalysisResponse {
  label: Label;
  bug_probability: number;
  confidence: number;
  model: ModelName;
  is_buggy: boolean;
  bug_type: string | null;
  explanation: string | null;
  suggested_fix: string | null;
  explanation_source: ExplanationSource;
  verified: boolean;
  /** Present only with explanation_source "operator_flip" (no original code given). */
  suspected_line?: number;
  repair_candidates?: RepairCandidate[];
  /** Present only when the request had summarize=true; null if the summarizer could not run. */
  summary?: string | null;
  summary_source?: SummarySource;
}

/** FeedbackRequest (extra="forbid"). */
export interface FeedbackRequest {
  code: string;
  model: ModelName;
  predicted_label: Label;
  actual_label: Label;
  user_explanation?: string;
  fixed_code?: string;
}

export interface FeedbackResponse {
  status: string;
  is_mistake: boolean;
}

/** One row of data/feedback/user_feedback.jsonl, as returned by GET /feedback. */
export interface FeedbackRecord {
  timestamp: string;
  code: string;
  model: ModelName;
  predicted_label: Label;
  actual_label: Label;
  is_mistake: boolean;
  user_explanation: string | null;
  fixed_code: string | null;
}

export interface FeedbackStats {
  user_feedback_count: number;
  mistakes_count: number;
  uncertain_cases_count: number;
  recent_feedback: FeedbackRecord[];
}

/** RetrainRequest: epochs is an int in [1, 10]. */
export interface RetrainRequest {
  epochs: number;
  mistakes_only: boolean;
}

export type RetrainJobStatus = "queued" | "running" | "done" | "failed" | "rejected";

/** Validation metrics the swap gate compares (BUGGY class). */
export interface GateMetrics {
  accuracy: number;
  bug_precision: number;
  bug_recall: number;
  bug_f1: number;
}

export interface RetrainProgress {
  phase: "queued" | "preparing" | "evaluating_current" | "training" | "evaluating_new" | "saving" | string;
  epoch?: number;
  epochs?: number;
  loss?: number;
}

/** A background retraining job, from POST /feedback/retrain (202) and GET /feedback/retrain/{job_id}. */
export interface RetrainJob {
  job_id: string;
  status: RetrainJobStatus;
  epochs: number;
  mistakes_only: boolean;
  progress: RetrainProgress;
  /** "accepted" or "rejected_worse" once finished. */
  result: string | null;
  metrics: { old: GateMetrics; new: GateMetrics } | null;
  output_dir: string | null;
  active_model_dir: string;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export const MAX_CODE_CHARS = 20_000;
export const MAX_EXPLANATION_CHARS = 5_000;
export const MAX_BODY_BYTES = 128_000;
export const UNCERTAIN_LOW = 0.4;
export const UNCERTAIN_HIGH = 0.6;
