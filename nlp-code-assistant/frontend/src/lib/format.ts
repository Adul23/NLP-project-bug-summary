import type { ComponentName, ModelName, ModelStatus, RetrainJob } from "../types";
import { UNCERTAIN_HIGH, UNCERTAIN_LOW } from "../types";

export const MODEL_LABELS: Record<ComponentName, string> = {
  tfidf: "TF-IDF",
  codebert: "CodeBERT",
  summarizer: "Summarizer",
};

export const STATUS_TEXT: Record<ModelStatus, string> = {
  loaded: "Loaded",
  artifacts_present_not_loaded: "Artifacts present, not loaded yet",
  load_failed: "Failed to load",
  unavailable: "Not trained / unavailable",
};

/** A model can be selected unless /health says it cannot be loaded. */
export function isUsable(status: ModelStatus | undefined): boolean {
  return status === undefined || status === "loaded" || status === "artifacts_present_not_loaded";
}

export function isUncertain(p: number): boolean {
  return p >= UNCERTAIN_LOW && p <= UNCERTAIN_HIGH;
}

export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function modelLabel(model: ModelName | string): string {
  return MODEL_LABELS[model as ComponentName] ?? model;
}

export function humanize(slug: string): string {
  const s = slug.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** First "line N" mentioned in an explanation, if any. */
export function mentionedLine(text: string | null | undefined): number | null {
  if (!text) return null;
  const m = /\bline\s+(\d+)\b/i.exec(text);
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isInteger(n) && n > 0 ? n : null;
}

export function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** One line describing what the job is doing right now. */
export function describeProgress(job: RetrainJob): string {
  const { phase, epoch, epochs, loss } = job.progress;
  switch (phase) {
    case "queued":
      return "Queued";
    case "preparing":
      return "Loading feedback and the model";
    case "evaluating_current":
      return "Measuring the current model on validation";
    case "training":
      return `Training — epoch ${epoch} of ${epochs}${typeof loss === "number" ? `, loss ${loss.toFixed(4)}` : ""}`;
    case "evaluating_new":
      return "Checking the new model on validation";
    case "saving":
      return "Saving the new adapter";
    default:
      return "Working";
  }
}
