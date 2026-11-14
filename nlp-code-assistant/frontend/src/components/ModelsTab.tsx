import comparison from "../data/model_comparison.json";
import { pct, STATUS_TEXT } from "../lib/format";
import type { HealthResponse, ModelName } from "../types";
import { StatusDot } from "./TopBar";
import { Card } from "./ui";

interface ComparisonRow {
  model: string;
  status?: string;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  accuracy?: number | null;
  latency_ms_median?: number | null;
  samples?: number;
  artifact_bytes?: number;
  base_plus_adapter_parameter_bytes?: number;
}

const data = comparison as { models?: ComparisonRow[]; protocol?: string };

const CARDS: { key: ModelName; title: string; match: RegExp; blurb: string }[] = [
  { key: "tfidf", title: "TF-IDF + LR", match: /tf-?idf/i, blurb: "TF-IDF features with a logistic regression classifier." },
  { key: "codebert", title: "CodeBERT + LoRA", match: /codebert/i, blurb: "CodeBERT fine-tuned with a LoRA adapter." },
];

const METRICS: { key: "precision" | "recall" | "f1" | "accuracy"; label: string }[] = [
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "accuracy", label: "Accuracy" },
];

function mb(bytes: number) {
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function MetricRow({ label, value }: { label: string; value: number | null | undefined }) {
  const has = typeof value === "number";
  return (
    <div className="grid grid-cols-[84px_1fr_64px] items-center gap-3 text-sm">
      <span className="text-muted">{label}</span>
      <div className="h-1.5 rounded-full bg-subtle">
        {has && <div className="h-full rounded-full bg-accent" style={{ width: `${value * 100}%` }} />}
      </div>
      <span className="text-right font-mono">{has ? pct(value) : "—"}</span>
    </div>
  );
}

export function ModelsTab({ health }: { health: HealthResponse | null }) {
  const rows = data.models ?? [];
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        {CARDS.map((c) => {
          const row = rows.find((r) => c.match.test(r.model));
          const trained = row && METRICS.some((m) => typeof row[m.key] === "number");
          const live = health?.models[c.key]?.status;
          return (
            <Card key={c.key} className="p-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold">{c.title}</h2>
                  <p className="mt-0.5 text-xs text-muted">{c.blurb}</p>
                </div>
                {live && (
                  <span className="inline-flex shrink-0 items-center gap-1.5 text-xs text-muted" title={STATUS_TEXT[live]}>
                    <StatusDot status={live} /> {STATUS_TEXT[live]}
                  </span>
                )}
              </div>
              {trained ? (
                <div className="mt-5 space-y-2.5">
                  {METRICS.map((m) => (
                    <MetricRow key={m.key} label={m.label} value={row[m.key]} />
                  ))}
                  <div className="flex items-baseline justify-between border-t border-line pt-3 text-sm">
                    <span className="text-muted">Median latency</span>
                    <span className="font-mono">
                      {typeof row.latency_ms_median === "number" ? `${row.latency_ms_median.toFixed(2)} ms` : "—"}
                    </span>
                  </div>
                  {typeof row.artifact_bytes === "number" && (
                    <div className="flex items-baseline justify-between text-sm">
                      <span className="text-muted">Artifact size</span>
                      <span className="font-mono" title={row.base_plus_adapter_parameter_bytes ? `${mb(row.base_plus_adapter_parameter_bytes)} with the pretrained base loaded` : undefined}>
                        {mb(row.artifact_bytes)}
                        {row.base_plus_adapter_parameter_bytes ? (
                          <span className="text-muted"> ({mb(row.base_plus_adapter_parameter_bytes)} loaded)</span>
                        ) : null}
                      </span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="mt-5 rounded border border-dashed border-line px-4 py-8 text-center">
                  <p className="text-sm font-medium">Not trained yet</p>
                  <p className="mt-1 text-xs text-muted">Metrics appear here after training and running the comparison.</p>
                </div>
              )}
            </Card>
          );
        })}
      </div>
      {data.protocol && <p className="text-xs text-muted">Protocol: {data.protocol}.</p>}
      <p className="text-xs text-muted">
        Snapshot of results/model_comparison.json, copied at build time by <code className="font-mono">npm run sync-metrics</code>.
      </p>
    </div>
  );
}
