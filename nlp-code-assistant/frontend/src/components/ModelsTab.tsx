import type { ReactNode } from "react";
import comparisonData from "../data/model_comparison.json";
import summaryData from "../data/summarization_metrics.json";
import type {
  BugMetrics,
  Comparison,
  ComparisonRow,
  GroupMetrics,
  Interval,
  ModelEvaluation,
  SummaryResults,
  SummarySystem,
} from "../evaluationTypes";
import { STATUS_TEXT } from "../lib/format";
import type { HealthResponse, ModelName } from "../types";
import { StatusDot } from "./TopBar";
import { Card } from "./ui";

const FIGURE_ORDER = [
  "roc_curves",
  "pr_curves",
  "reliability",
  "confusion_matrices",
  "mutation_recall",
  "f1_vs_threshold",
  "latency",
  "summarization_metrics",
];
const figureModules = import.meta.glob("../data/figures/*.png", {
  eager: true,
  query: "?url",
  import: "default",
}) as Record<string, string>;

export const DEFAULT_FIGURES = Object.entries(figureModules)
  .map(([path, url]) => ({ name: path.split("/").pop()!.replace(".png", ""), url }))
  .sort((a, b) => FIGURE_ORDER.indexOf(a.name) - FIGURE_ORDER.indexOf(b.name));

const FIGURE_TITLES: Record<string, string> = {
  roc_curves: "ROC curves",
  pr_curves: "Precision-recall curves",
  reliability: "Reliability diagram",
  confusion_matrices: "Confusion matrices",
  mutation_recall: "Recall per mutation type",
  f1_vs_threshold: "F1 vs threshold",
  latency: "Latency",
  summarization_metrics: "Summarization metrics",
};

const MODELS: { key: ModelName; title: string; match: RegExp }[] = [
  { key: "tfidf", title: "TF-IDF + LR", match: /tf-?idf/i },
  { key: "codebert", title: "CodeBERT + LoRA", match: /codebert/i },
];

const f3 = (v: number | null | undefined) => (typeof v === "number" ? v.toFixed(3) : "—");
const ms = (v: number | null | undefined) =>
  typeof v === "number" ? (v >= 100 ? `${Math.round(v)} ms` : `${v.toFixed(1)} ms`) : "—";
const pctText = (v: number | null | undefined, digits = 1) =>
  typeof v === "number" ? `${(v * 100).toFixed(digits)}%` : "—";

function Value({ value, ci, format = f3 }: { value: number | null | undefined; ci?: Interval | null; format?: (v: number) => string }) {
  if (typeof value !== "number") return <span className="text-muted">—</span>;
  return (
    <span className="inline-flex flex-col items-end leading-tight">
      <span className="font-mono">{format(value)}</span>
      {ci && (
        <span className="font-mono text-[11px] text-muted">
          [{format(ci.low)}–{format(ci.high)}]
        </span>
      )}
    </span>
  );
}

type Getter = (m: BugMetrics) => number | null | undefined;
const MAIN_ROWS: { label: string; hint: string; get: Getter; ci?: string; recorded?: keyof ComparisonRow; lowerBetter?: boolean }[] = [
  { label: "Accuracy", hint: "Share of correct predictions", get: (m) => m.accuracy, ci: "accuracy", recorded: "accuracy" },
  { label: "Balanced accuracy", hint: "Mean of BUGGY and CLEAN recall; robust to the 68/32 imbalance", get: (m) => m.balanced_accuracy },
  { label: "BUGGY precision", hint: "Flagged functions that are really buggy", get: (m) => m.bug_precision, recorded: "precision" },
  { label: "BUGGY recall", hint: "Buggy functions that get flagged", get: (m) => m.bug_recall, recorded: "recall" },
  { label: "BUGGY F1", hint: "Harmonic mean of BUGGY precision and recall", get: (m) => m.bug_f1, ci: "bug_f1", recorded: "f1" },
  { label: "Macro-F1", hint: "Mean F1 over both classes", get: (m) => m.macro_f1 },
  { label: "MCC", hint: "Correlation between prediction and truth (−1 to 1); 0 = chance", get: (m) => m.mcc, ci: "mcc" },
  { label: "ROC-AUC", hint: "Chance a buggy function scores above a clean one", get: (m) => m.roc_auc, ci: "roc_auc" },
  { label: "PR-AUC", hint: "Average precision for BUGGY; random = positive rate", get: (m) => m.pr_auc, ci: "pr_auc" },
  { label: "Brier score", hint: "Mean squared error of P(BUGGY); lower is better", get: (m) => m.calibration?.brier, lowerBetter: true },
  { label: "ECE", hint: "Expected calibration error, 10 bins; lower is better", get: (m) => m.calibration?.ece, lowerBetter: true },
];

function Th({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <th className={`px-3 py-2 text-xs font-medium text-muted ${className}`}>{children}</th>;
}

function MainTable({ comparison, health }: { comparison: Comparison; health: HealthResponse | null }) {
  const rows = comparison.models ?? [];
  const columns = MODELS.map((m) => ({
    ...m,
    row: rows.find((r) => m.match.test(r.model)),
    evaluation: comparison.evaluation?.[m.key] as ModelEvaluation | undefined,
  }));
  return (
    <Card className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-line">
            <tr>
              <Th className="text-left">Metric (test, threshold 0.5)</Th>
              {columns.map((c) => {
                const live = health?.models[c.key]?.status;
                return (
                  <Th key={c.key} className="text-right">
                    <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-fg">
                      {live && <StatusDot status={live} />}
                      {c.title}
                    </span>
                    {live && <span className="block text-[11px] font-normal">{STATUS_TEXT[live]}</span>}
                  </Th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {MAIN_ROWS.map((r) => (
              <tr key={r.label} className="border-t border-line first:border-t-0">
                <td className="px-3 py-2">
                  <span className="font-medium">{r.label}</span>
                  <span className="block text-xs text-muted">{r.hint}</span>
                </td>
                {columns.map((c) => {
                  const m = c.evaluation?.test;
                  const value = m ? r.get(m) : r.recorded ? (c.row?.[r.recorded] as number | null | undefined) : undefined;
                  const ci = m && r.ci ? m.bootstrap?.ci[r.ci] : null;
                  return (
                    <td key={c.key} className="px-3 py-2 text-right align-top">
                      <Value value={value} ci={ci} />
                    </td>
                  );
                })}
              </tr>
            ))}
            {(["latency_ms_p50", "latency_ms_p95"] as const).map((key) => (
              <tr key={key} className="border-t border-line">
                <td className="px-3 py-2">
                  <span className="font-medium">Latency {key === "latency_ms_p50" ? "p50" : "p95"}</span>
                  <span className="block text-xs text-muted">Warm CPU, batch 1, per function</span>
                </td>
                {columns.map((c) => (
                  <td key={c.key} className="px-3 py-2 text-right font-mono">
                    {ms(c.row?.efficiency?.cpu_batch1[key] ?? (key === "latency_ms_p50" ? c.row?.latency_ms_median : null))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-line px-3 py-2 text-xs text-muted">
        Brackets: 95% bootstrap confidence interval (1000 resamples). PR-AUC of a random ranker ={" "}
        {pctText(columns.find((c) => c.evaluation)?.evaluation?.test.pr_auc_baseline)} (the BUGGY share).
        {columns.some((c) => c.row && !c.evaluation) &&
          " Models without per-row predictions show only the metrics recorded at training time."}
      </p>
    </Card>
  );
}

function PairedCard({ comparison }: { comparison: Comparison }) {
  const paired = comparison.paired_comparison;
  return (
    <Card className="p-4">
      <h3 className="text-sm font-medium">Is the difference significant?</h3>
      {paired ? (
        <div className="mt-2 grid gap-3 text-sm sm:grid-cols-2">
          <div>
            <div className="text-xs text-muted">McNemar exact p (same test rows)</div>
            <div className="font-mono text-lg">{paired.mcnemar.exact_p.toExponential(2)}</div>
            <div className="text-xs text-muted">
              Only TF-IDF right: {paired.mcnemar.b} · only CodeBERT right: {paired.mcnemar.c}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted">BUGGY F1 difference (CodeBERT − TF-IDF)</div>
            <div className="font-mono text-lg">{f3(paired.paired_bootstrap_bug_f1.observed)}</div>
            <div className="text-xs text-muted">
              95% paired bootstrap CI [{f3(paired.paired_bootstrap_bug_f1.ci_low)}, {f3(paired.paired_bootstrap_bug_f1.ci_high)}]
            </div>
          </div>
        </div>
      ) : (
        <p className="mt-1 text-sm text-muted">
          Needs test predictions from both models. CodeBERT predictions have not been generated yet; see the
          evaluation notes below.
        </p>
      )}
    </Card>
  );
}

function Details({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="group rounded border border-line bg-surface" open={defaultOpen}>
      <summary className="cursor-pointer select-none px-4 py-3 text-sm font-medium marker:text-muted">{title}</summary>
      <div className="border-t border-line p-4">{children}</div>
    </details>
  );
}

function SimpleTable({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line">
            {head.map((h, i) => (
              <Th key={h} className={i ? "text-right" : "text-left"}>
                {h}
              </Th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, i) => (
            <tr key={i} className="border-t border-line first:border-t-0">
              {cells.map((cell, j) => (
                <td key={j} className={`px-3 py-1.5 ${j ? "text-right font-mono" : ""}`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EvaluationDetails({ comparison }: { comparison: Comparison }) {
  const evaluated = MODELS.filter((m) => comparison.evaluation?.[m.key]);
  if (!evaluated.length) return null;
  const get = (key: ModelName) => comparison.evaluation![key] as ModelEvaluation;
  const types = ["comparison_operator", "arithmetic_operator", "boolean_operator"];
  const groups = (field: "by_code_length" | "by_truncation") =>
    Object.keys(get(evaluated[0].key)[field] ?? {});
  const groupCell = (g: GroupMetrics | undefined) => (g ? `${f3(g.bug_f1)} / ${f3(g.accuracy)}` : "—");
  return (
    <div className="space-y-3">
      <Details title="Recall per mutation type">
        <SimpleTable
          head={["Rows", ...evaluated.map((m) => m.title)]}
          rows={[
            ...types.map((t) => [
              `${t.replace("_operator", "")} operator recall`,
              ...evaluated.map((m) => {
                const b = get(m.key).test.by_mutation_type?.[t];
                return b ? `${f3(b.recall)} (${b.detected}/${b.n})` : "—";
              }),
            ]),
            [
              "Clean rows flagged BUGGY",
              ...evaluated.map((m) => {
                const b = get(m.key).test.by_mutation_type?.clean;
                return b ? `${f3(b.false_positive_rate)} (${b.false_positives}/${b.n})` : "—";
              }),
            ],
          ]}
        />
      </Details>
      <Details title="Confusion matrices">
        <div className="grid gap-4 sm:grid-cols-2">
          {evaluated.map((m) => {
            const cm = get(m.key).test.confusion_matrix;
            return (
              <div key={m.key}>
                <div className="mb-1 text-xs font-medium text-muted">{m.title} · rows = actual</div>
                <SimpleTable
                  head={["", "Pred. CLEAN", "Pred. BUGGY"]}
                  rows={["CLEAN", "BUGGY"].map((label, i) => [
                    `Actual ${label}`,
                    ...[0, 1].map((j) => `${cm.counts[i][j]} (${pctText(cm.row_normalized[i][j])})`),
                  ])}
                />
              </div>
            );
          })}
        </div>
      </Details>
      <Details title="Threshold tuned on validation">
        <SimpleTable
          head={["Model / threshold", "BUGGY P", "BUGGY R", "BUGGY F1", "MCC"]}
          rows={evaluated.flatMap((m) =>
            Object.entries(get(m.key).threshold.analysis_on_test).map(([name, block]) => [
              `${m.title} · ${name === "default_0.5" ? "default" : "tuned on validation"} ${block.threshold.toFixed(3)}`,
              f3(block.metrics.bug_precision),
              f3(block.metrics.bug_recall),
              f3(block.metrics.bug_f1),
              f3(block.metrics.mcc),
            ]),
          )}
        />
        <p className="mt-2 text-xs text-muted">The threshold maximizes BUGGY F1 on validation; test rows are only scored.</p>
      </Details>
      {groups("by_code_length").length > 0 && (
        <Details title="By code length and truncation">
          <SimpleTable
            head={["Group (CodeBERT tokens)", "n", ...evaluated.map((m) => `${m.title} F1 / acc.`)]}
            rows={(["by_code_length", "by_truncation"] as const).flatMap((field) =>
              groups(field).map((g) => [
                g,
                String(get(evaluated[0].key)[field]?.[g]?.n ?? "—"),
                ...evaluated.map((m) => groupCell(get(m.key)[field]?.[g])),
              ]),
            )}
          />
        </Details>
      )}
    </div>
  );
}

const SUMMARY_COLUMNS: { label: string; get: (s: SummarySystem) => number | null; ci?: (s: SummarySystem) => Interval | undefined }[] = [
  { label: "BLEU-4", get: (s) => s.bleu4, ci: (s) => s.bootstrap?.bleu4 },
  { label: "ROUGE-1", get: (s) => s.rouge_1.f1 },
  { label: "ROUGE-2", get: (s) => s.rouge_2.f1 },
  { label: "ROUGE-L", get: (s) => s.rouge_l.f1, ci: (s) => s.bootstrap?.rouge_l_f1 },
  { label: "chrF", get: (s) => s.chrf },
  { label: "METEOR", get: (s) => s.meteor },
];

function SummarizerCard({ summary }: { summary: SummaryResults }) {
  const systems = Object.entries(summary.systems ?? {});
  const columns = SUMMARY_COLUMNS.filter((c) => c.label !== "METEOR" || systems.some(([, s]) => s.meteor !== null));
  const meteorNote = systems.find(([, s]) => s.meteor_note)?.[1].meteor_note;
  return (
    <Card className="p-5">
      <h2 className="text-base font-semibold">CodeT5 summarizer</h2>
      <p className="mt-0.5 text-xs text-muted">
        Fine-tuned CodeT5-small vs. template and retrieval baselines, scored against reference docstrings.
      </p>
      {systems.length === 0 ? (
        <div className="mt-4 rounded border border-dashed border-line px-4 py-6 text-center">
          <p className="text-sm font-medium">Not evaluated yet</p>
          <p className="mt-1 text-xs text-muted">
            Needs data/processed/summarization_dataset.csv. Then run{" "}
            <code className="font-mono">python -m src.summarization.evaluate_summaries</code> and{" "}
            <code className="font-mono">npm run sync-metrics</code>.
          </p>
        </div>
      ) : (
        <div className="mt-4">
          <SimpleTable
            head={["System", ...columns.map((c) => c.label)]}
            rows={systems.map(([key, s]) => [
              <span key={key} className={key === "codet5" ? "font-semibold" : ""}>
                {s.name}
              </span>,
              ...columns.map((c) => <Value key={c.label} value={c.get(s)} ci={c.ci?.(s)} />),
            ])}
          />
          <p className="mt-2 text-xs text-muted">
            {summary.split} split, {summary.examples} rows. Scores in 0–1; brackets are 95% bootstrap CIs.
            {meteorNote ? ` ${meteorNote}.` : ""}
          </p>
        </div>
      )}
    </Card>
  );
}

function Figures({ figures }: { figures: { name: string; url: string }[] }) {
  if (!figures.length) {
    return <p className="text-sm text-muted">No figures yet. Run python -m src.evaluation.make_figures, then npm run sync-metrics.</p>;
  }
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {figures.map((f) => (
        <figure key={f.name} className="rounded border border-line bg-white p-2">
          <img src={f.url} alt={FIGURE_TITLES[f.name] ?? f.name} className="w-full" loading="lazy" />
          <figcaption className="mt-1 text-xs text-neutral-600">{FIGURE_TITLES[f.name] ?? f.name}</figcaption>
        </figure>
      ))}
    </div>
  );
}

export function ModelsTab({
  health,
  comparison = comparisonData as Comparison,
  summary = summaryData as SummaryResults,
  figures = DEFAULT_FIGURES,
}: {
  health: HealthResponse | null;
  comparison?: Comparison;
  summary?: SummaryResults;
  figures?: { name: string; url: string }[];
}) {
  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <h2 className="text-base font-semibold">Bug detection</h2>
        <MainTable comparison={comparison} health={health} />
        <PairedCard comparison={comparison} />
        <EvaluationDetails comparison={comparison} />
      </section>
      <SummarizerCard summary={summary} />
      <section className="space-y-3">
        <h2 className="text-base font-semibold">Evaluation</h2>
        <Details title={`Figures (${figures.length})`}>
          <Figures figures={figures} />
        </Details>
        {(comparison.notes?.length ?? 0) > 0 && (
          <Details title="Notes">
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted">
              {comparison.notes!.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </Details>
        )}
        <p className="text-xs text-muted">
          Snapshot of results/ copied at build time by <code className="font-mono">npm run sync-metrics</code>.
          {comparison.evaluation_protocol && ` ${comparison.evaluation_protocol.threshold_selection}.`}
        </p>
      </section>
    </div>
  );
}
