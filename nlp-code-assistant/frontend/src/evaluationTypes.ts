// Shapes of results/model_comparison.json and results/summarization_metrics.json
// (written by src/evaluation/compare_models.py and src/summarization/evaluate_summaries.py).

export interface Interval {
  low: number;
  high: number;
  resamples?: number;
}

export interface MutationBlock {
  n: number;
  detected?: number;
  recall?: number | null;
  false_positives?: number;
  false_positive_rate?: number | null;
}

export interface GroupMetrics {
  n: number;
  accuracy: number;
  bug_f1: number;
  bug_recall: number;
  false_positive_rate: number | null;
}

export interface BugMetrics {
  n: number;
  accuracy: number;
  balanced_accuracy: number | null;
  bug_precision: number;
  bug_recall: number;
  bug_f1: number;
  macro_f1: number;
  mcc: number;
  roc_auc: number | null;
  pr_auc: number | null;
  pr_auc_baseline: number;
  calibration: { brier: number; ece: number; log_loss: number | null } | null;
  confusion_matrix: { counts: number[][]; row_normalized: (number | null)[][] };
  by_mutation_type?: Record<string, MutationBlock>;
  bootstrap: { ci: Record<string, Interval | null> } | null;
}

export interface ModelEvaluation {
  test: BugMetrics;
  threshold: {
    validation_tuned_threshold: number;
    analysis_on_test: Record<string, { threshold: number; metrics: BugMetrics }>;
  };
  by_code_length?: Record<string, GroupMetrics>;
  by_truncation?: Record<string, GroupMetrics>;
}

export interface Latency {
  latency_ms_p50: number;
  latency_ms_p95: number;
  latency_ms_mean: number;
  samples: number;
}

export interface ComparisonRow {
  model: string;
  status?: string;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  accuracy?: number | null;
  latency_ms_median?: number | null;
  artifact_bytes?: number;
  base_plus_adapter_parameter_bytes?: number;
  evaluated?: boolean;
  efficiency?: {
    cpu_batch1: Latency;
    cpu_throughput: { functions_per_second: number; batch_size: number };
    cuda_batch1: Latency | null;
    parameters: { trainable_parameters: number; total_parameters: number };
  };
}

export interface Comparison {
  models?: ComparisonRow[];
  protocol?: string;
  evaluation?: Partial<Record<"tfidf" | "codebert", ModelEvaluation>>;
  paired_comparison?: {
    mcnemar: { b: number; c: number; exact_p: number; chi2_p: number };
    paired_bootstrap_bug_f1: { observed: number; ci_low: number; ci_high: number };
  } | null;
  evaluation_protocol?: { test_rows: number; threshold_selection: string };
  notes?: string[];
}

export interface Prf {
  precision: number;
  recall: number;
  f1: number;
}

export interface SummarySystem {
  name: string;
  examples: number;
  bleu4: number;
  sentence_bleu4_smoothed: number;
  rouge_1: Prf;
  rouge_2: Prf;
  rouge_l: Prf;
  chrf: number;
  meteor: number | null;
  meteor_note: string | null;
  exact_match: number;
  empty_output_rate: number;
  copy_rate: number | null;
  bootstrap: { bleu4: Interval; rouge_l_f1: Interval } | null;
}

export interface SummaryResults {
  split?: string;
  examples?: number;
  systems: Record<string, SummarySystem>;
  not_evaluated?: boolean;
}
