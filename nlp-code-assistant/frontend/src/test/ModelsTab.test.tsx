import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { ModelsTab } from "../components/ModelsTab";
import type { BugMetrics, Comparison, SummaryResults } from "../evaluationTypes";

const metrics: BugMetrics = {
  n: 2211,
  accuracy: 0.6924,
  balanced_accuracy: 0.688,
  bug_precision: 0.517,
  bug_recall: 0.675,
  bug_f1: 0.5854,
  macro_f1: 0.67,
  mcc: 0.356,
  roc_auc: 0.763,
  pr_auc: 0.525,
  pr_auc_baseline: 0.3216,
  calibration: { brier: 0.199, ece: 0.134, log_loss: 0.58 },
  confusion_matrix: { counts: [[1051, 449], [231, 480]], row_normalized: [[0.7007, 0.2993], [0.3249, 0.6751]] },
  by_mutation_type: {
    comparison_operator: { n: 303, detected: 214, recall: 0.7063 },
    arithmetic_operator: { n: 277, detected: 194, recall: 0.7004 },
    boolean_operator: { n: 131, detected: 72, recall: 0.5496 },
    clean: { n: 1500, false_positives: 449, false_positive_rate: 0.2993 },
  },
  bootstrap: { ci: { accuracy: { low: 0.672, high: 0.711 }, bug_f1: { low: 0.554, high: 0.613 } } },
};

const latency = (p50: number) => ({
  cpu_batch1: { latency_ms_p50: p50, latency_ms_p95: p50 * 3, latency_ms_mean: p50, samples: 30 },
  cpu_throughput: { functions_per_second: 10, batch_size: 32 },
  cuda_batch1: null,
  parameters: { trainable_parameters: 1, total_parameters: 1 },
});

const comparison: Comparison = {
  models: [
    { model: "TF-IDF + LR", status: "measured", f1: 0.5854, evaluated: true, efficiency: latency(2.26) },
    { model: "CodeBERT + LoRA", status: "measured", precision: 0.7698, recall: 0.7384, f1: 0.7538, accuracy: 0.8449, evaluated: false, efficiency: latency(424.8) },
  ],
  evaluation: {
    tfidf: {
      test: metrics,
      threshold: {
        validation_tuned_threshold: 0.455,
        analysis_on_test: {
          "default_0.5": { threshold: 0.5, metrics },
          validation_tuned: { threshold: 0.455, metrics: { ...metrics, bug_f1: 0.605 } },
        },
      },
    },
  },
  paired_comparison: null,
  notes: ["CodeBERT predictions skipped: no CUDA here."],
};

// The first table on the page is the main metrics table.
const rowOf = (label: string) =>
  within(screen.getAllByRole("table")[0]).getByText(label, { selector: "span" }).closest("tr")!;

describe("ModelsTab", () => {
  it("shows the main table with CIs, recorded-only CodeBERT values and latency", () => {
    render(<ModelsTab health={null} comparison={comparison} summary={{ systems: {}, not_evaluated: true }} figures={[]} />);
    const f1 = within(rowOf("BUGGY F1"));
    expect(f1.getByText("0.585")).toBeInTheDocument();
    expect(f1.getByText("[0.554–0.613]")).toBeInTheDocument();
    expect(f1.getByText("0.754")).toBeInTheDocument(); // CodeBERT, recorded at training time
    expect(within(rowOf("MCC")).getByText("0.356")).toBeInTheDocument();
    expect(within(rowOf("MCC")).getAllByText("—")).toHaveLength(1); // CodeBERT has no MCC yet
    expect(within(rowOf("Latency p50")).getByText("425 ms")).toBeInTheDocument();
    expect(within(rowOf("Latency p50")).getByText("2.3 ms")).toBeInTheDocument();
    expect(screen.getByText(/Needs test predictions from both models/)).toBeInTheDocument();
    expect(screen.getByText("Not evaluated yet")).toBeInTheDocument();
    expect(screen.getByText("CodeBERT predictions skipped: no CUDA here.")).toBeInTheDocument();
    // Details are collapsed but present.
    expect(screen.getByText("Recall per mutation type").closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText("0.550 (72/131)")).toBeInTheDocument();
    expect(screen.getByText("1051 (70.1%)")).toBeInTheDocument();
  });

  it("shows McNemar and the summarizer metrics when available", () => {
    const summary: SummaryResults = {
      split: "test",
      examples: 100,
      systems: {
        ast: {
          name: "AST template", examples: 100, bleu4: 0.05, sentence_bleu4_smoothed: 0.1,
          rouge_1: { precision: 0.2, recall: 0.2, f1: 0.2 }, rouge_2: { precision: 0.1, recall: 0.1, f1: 0.1 },
          rouge_l: { precision: 0.2, recall: 0.2, f1: 0.19 }, chrf: 0.3, meteor: null,
          meteor_note: "METEOR skipped: nltk is not installed", exact_match: 0, empty_output_rate: 0, copy_rate: 0,
          bootstrap: { bleu4: { low: 0.04, high: 0.06 }, rouge_l_f1: { low: 0.18, high: 0.2 } },
        },
      },
    };
    render(
      <ModelsTab
        health={null}
        comparison={{
          ...comparison,
          paired_comparison: {
            mcnemar: { b: 120, c: 440, exact_p: 1.2e-40, chi2_p: 1e-39 },
            paired_bootstrap_bug_f1: { observed: 0.168, ci_low: 0.14, ci_high: 0.2 },
          },
        }}
        summary={summary}
        figures={[{ name: "roc_curves", url: "/roc.png" }]}
      />,
    );
    expect(screen.getByText("1.20e-40")).toBeInTheDocument();
    expect(screen.getByText(/Only TF-IDF right: 120/)).toBeInTheDocument();
    const ast = within(screen.getByText("AST template").closest("tr")!);
    expect(ast.getByText("0.050")).toBeInTheDocument();
    expect(ast.getByText("[0.040–0.060]")).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "METEOR" })).not.toBeInTheDocument();
    expect(screen.getByText(/METEOR skipped: nltk is not installed/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "ROC curves" })).toHaveAttribute("src", "/roc.png");
  });
});
