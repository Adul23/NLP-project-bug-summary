import type { AnalysisResponse } from "../types";
import { humanize, modelLabel } from "../lib/format";
import { FeedbackRow } from "./FeedbackRow";
import { AlertIcon, CheckIcon } from "./icons";
import { ProbabilityBar } from "./ProbabilityBar";
import { Card, ErrorNote } from "./ui";

export interface AnalysisRun {
  result: AnalysisResponse;
  latencyMs: number;
  code: string;
}

interface Props {
  run: AnalysisRun | null;
  loading: boolean;
  error: string | null;
  stale?: boolean;
}

export function ResultCard({ run, loading, error, stale = false }: Props) {
  return (
    <Card className="p-5">
      {loading ? (
        <ResultSkeleton />
      ) : run ? (
        <ResultBody run={run} stale={stale} error={error} />
      ) : error ? (
        <ErrorNote message={error} />
      ) : (
        <EmptyState />
      )}
    </Card>
  );
}

function EmptyState() {
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-2 text-center">
      <p className="text-base font-medium">Paste a Python function and run an analysis</p>
      <p className="max-w-sm text-sm text-muted">
        You'll get a buggy/clean verdict, the model's bug probability, and an explanation. Add the original
        code to get a verified explanation of the exact change.
      </p>
      <p className="mt-2 text-xs text-muted">
        Tip: press <kbd className="rounded border border-line bg-subtle px-1 font-mono">Ctrl</kbd>+
        <kbd className="rounded border border-line bg-subtle px-1 font-mono">Enter</kbd> to analyze.
      </p>
    </div>
  );
}

function ResultSkeleton() {
  return (
    <div className="animate-pulse space-y-5" data-testid="result-skeleton" aria-busy="true">
      <div className="flex items-center gap-3">
        <div className="h-9 w-24 rounded bg-subtle" />
        <div className="h-4 w-32 rounded bg-subtle" />
      </div>
      <div className="h-2.5 w-full rounded-full bg-subtle" />
      <div className="flex gap-2">
        <div className="h-6 w-28 rounded-full bg-subtle" />
        <div className="h-6 w-20 rounded-full bg-subtle" />
      </div>
      <div className="space-y-2">
        <div className="h-3 w-full rounded bg-subtle" />
        <div className="h-3 w-4/5 rounded bg-subtle" />
      </div>
    </div>
  );
}

function SourceChip({ source }: { source: AnalysisResponse["explanation_source"] }) {
  if (source === "verified_code_pair") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-ok/40 bg-ok/10 px-2.5 py-0.5 text-xs font-medium text-ok">
        <CheckIcon width={12} height={12} /> Verified
      </span>
    );
  }
  if (source === "lora_generated_unverified") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-warn/40 bg-warn/10 px-2.5 py-0.5 text-xs font-medium text-warn">
        <AlertIcon width={12} height={12} /> Generated, unverified
      </span>
    );
  }
  return null;
}

function ResultBody({ run, stale, error }: { run: AnalysisRun; stale: boolean; error: string | null }) {
  const { result, latencyMs } = run;
  const buggy = result.label === "BUGGY";
  return (
    <div className="space-y-5">
      {error && <ErrorNote message={error} />}
      <div className="flex flex-wrap items-center gap-3">
        <span
          data-testid="verdict"
          className={`rounded px-3 py-1.5 text-lg font-semibold ${
            buggy ? "bg-bad/10 text-bad ring-1 ring-bad/30" : "bg-ok/10 text-ok ring-1 ring-ok/30"
          }`}
        >
          {buggy ? "Buggy" : "Clean"}
        </span>
        <span className="text-sm text-muted">
          via {modelLabel(result.model)} · {Math.round(latencyMs)} ms
        </span>
      </div>
      {stale && <p className="text-xs text-muted">The code has changed since this result. Run again to refresh.</p>}

      <ProbabilityBar value={result.bug_probability} buggy={buggy} />

      {(result.bug_type || result.explanation_source !== "unavailable") && (
        <div className="flex flex-wrap gap-2">
          {result.bug_type && (
            <span className="rounded-full border border-line bg-subtle px-2.5 py-0.5 text-xs font-medium">
              {humanize(result.bug_type)}
            </span>
          )}
          <SourceChip source={result.explanation_source} />
        </div>
      )}

      <div>
        <h3 className="mb-1 text-xs font-medium text-muted">Explanation</h3>
        {result.explanation ? (
          <p className="text-sm leading-relaxed">{result.explanation}</p>
        ) : (
          <p className="text-sm text-muted">
            No explanation available. Add the original code for a verified explanation
            {result.explanation_source === "unavailable" ? ", or pick LoRA if the summarizer is trained" : ""}.
          </p>
        )}
      </div>

      {result.suggested_fix && (
        <div>
          <h3 className="mb-1 text-xs font-medium text-muted">Suggested fix</h3>
          <pre className="overflow-x-auto rounded border border-line bg-subtle p-3 font-mono text-[13px]">
            <code>{result.suggested_fix}</code>
          </pre>
        </div>
      )}

      <div className="border-t border-line pt-4">
        <FeedbackRow key={`${run.code}:${latencyMs}`} result={result} code={run.code} />
      </div>
    </div>
  );
}
