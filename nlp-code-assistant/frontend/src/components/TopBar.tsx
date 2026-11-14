import { MODEL_LABELS, STATUS_TEXT } from "../lib/format";
import type { ComponentName, HealthResponse, ModelStatus } from "../types";
import { BugIcon } from "./icons";

export type Tab = "analyze" | "feedback" | "models";

const TABS: { id: Tab; label: string }[] = [
  { id: "analyze", label: "Analyze" },
  { id: "feedback", label: "Feedback" },
  { id: "models", label: "Models" },
];

const DOT: Record<ModelStatus, string> = {
  loaded: "bg-ok",
  artifacts_present_not_loaded: "bg-warn",
  unavailable: "bg-muted/60",
  load_failed: "bg-bad",
};

export function StatusDot({ status }: { status: ModelStatus | "offline" }) {
  return <span className={`h-2 w-2 shrink-0 rounded-full ${status === "offline" ? "bg-muted/40" : DOT[status]}`} />;
}

const COMPONENTS: ComponentName[] = ["tfidf", "codebert", "summarizer"];

function StatusPills({ health, error }: { health: HealthResponse | null; error: string | null }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {COMPONENTS.map((name) => {
        const status = health?.models[name]?.status;
        const title = error && !health ? "Backend unreachable" : status ? STATUS_TEXT[status] : "Checking…";
        return (
          <span
            key={name}
            data-testid={`pill-${name}`}
            data-status={status ?? "offline"}
            title={`${MODEL_LABELS[name]}: ${title}`}
            className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-0.5 text-xs"
          >
            <StatusDot status={status ?? "offline"} />
            {MODEL_LABELS[name]}
          </span>
        );
      })}
    </div>
  );
}

export function TopBar({
  tab,
  onTab,
  health,
  healthError,
}: {
  tab: Tab;
  onTab: (t: Tab) => void;
  health: HealthResponse | null;
  healthError: string | null;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-surface/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5 md:px-6">
        <div className="flex items-center gap-2 font-semibold">
          <span className="flex h-7 w-7 items-center justify-center rounded bg-accent/10 text-accent">
            <BugIcon />
          </span>
          Bug detector
        </div>
        <nav role="tablist" aria-label="Sections" className="flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              type="button"
              aria-selected={tab === t.id}
              onClick={() => onTab(t.id)}
              className={`rounded px-3 py-1.5 text-sm transition-colors ${
                tab === t.id ? "bg-subtle font-medium text-fg" : "text-muted hover:text-fg"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          {healthError && !health && <span className="text-xs text-bad">Backend offline</span>}
          <StatusPills health={health} error={healthError} />
        </div>
      </div>
    </header>
  );
}
