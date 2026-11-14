import { Fragment, useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { formatTime, modelLabel } from "../lib/format";
import type { FeedbackRecord, FeedbackStats } from "../types";
import { AlertIcon, CheckIcon, ChevronIcon, Spinner } from "./icons";
import { RetrainPanel } from "./RetrainPanel";
import { Button, Card, ErrorNote } from "./ui";

interface Props {
  active: boolean;
  onRetrained: () => void;
}

function Metric({ label, value, hint }: { label: string; value: number | undefined; hint: string }) {
  return (
    <Card className="p-4">
      <div className="text-xs font-medium text-muted">{label}</div>
      <div className="mt-1 font-mono text-2xl font-semibold">{value ?? "—"}</div>
      <div className="mt-1 text-xs text-muted">{hint}</div>
    </Card>
  );
}

function firstLine(code: string): string {
  const line = code.split("\n").find((l) => l.trim()) ?? "";
  return line.trim();
}

function Row({ row }: { row: FeedbackRecord }) {
  const [open, setOpen] = useState(false);
  return (
    <Fragment>
      <tr
        className="cursor-pointer border-t border-line hover:bg-subtle/60"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <td className="whitespace-nowrap px-3 py-2 text-muted">
          <span className="inline-flex items-center gap-1.5">
            <ChevronIcon width={12} height={12} className={`transition-transform ${open ? "rotate-90" : ""}`} />
            {formatTime(row.timestamp)}
          </span>
        </td>
        <td className="whitespace-nowrap px-3 py-2">{modelLabel(row.model)}</td>
        <td className="whitespace-nowrap px-3 py-2 font-mono text-xs">
          {row.predicted_label.toLowerCase()} → {row.actual_label.toLowerCase()}
        </td>
        <td className="px-3 py-2">
          {row.is_mistake ? (
            <AlertIcon className="text-bad" aria-label="Mistake" />
          ) : (
            <CheckIcon className="text-ok" aria-label="Correct" />
          )}
        </td>
        <td className="max-w-0 w-full px-3 py-2">
          <code className="block truncate font-mono text-xs text-muted">{firstLine(row.code)}</code>
        </td>
      </tr>
      {open && (
        <tr className="bg-subtle/40">
          <td colSpan={5} className="space-y-3 px-3 py-3">
            <pre className="overflow-x-auto rounded border border-line bg-surface p-3 font-mono text-xs">{row.code}</pre>
            {row.user_explanation && (
              <p className="text-sm">
                <span className="text-muted">Note: </span>
                {row.user_explanation}
              </p>
            )}
            {row.fixed_code && (
              <div>
                <div className="mb-1 text-xs font-medium text-muted">Fixed code</div>
                <pre className="overflow-x-auto rounded border border-line bg-surface p-3 font-mono text-xs">
                  {row.fixed_code}
                </pre>
              </div>
            )}
          </td>
        </tr>
      )}
    </Fragment>
  );
}

export function FeedbackTab({ active, onRetrained }: Props) {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setStats(await api.feedbackStats());
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (active) void load();
  }, [active, load]);

  const rows = stats ? [...stats.recent_feedback].reverse() : [];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <Metric label="Feedback" value={stats?.user_feedback_count} hint="User reports saved" />
        <Metric label="Mistakes" value={stats?.mistakes_count} hint="Predictions marked wrong" />
        <Metric label="Uncertain" value={stats?.uncertain_cases_count} hint="Predictions at 40–60%" />
      </div>

      {error && <ErrorNote message={error} />}

      <Card>
        <div className="flex items-center justify-between border-b border-line px-4 py-3">
          <h2 className="text-sm font-medium">Recent feedback</h2>
          <Button variant="ghost" onClick={load} disabled={loading}>
            {loading && <Spinner />} Refresh
          </Button>
        </div>
        {rows.length === 0 ? (
          <p className="px-4 py-8 text-center text-sm text-muted">
            {stats ? "No feedback yet. Use “Was this right?” on the Analyze tab." : loading ? "Loading…" : "—"}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Time</th>
                  <th className="px-3 py-2 font-medium">Model</th>
                  <th className="px-3 py-2 font-medium">Predicted → actual</th>
                  <th className="px-3 py-2 font-medium">
                    <span className="sr-only">Mistake</span>
                  </th>
                  <th className="px-3 py-2 font-medium">Code</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <Row key={`${row.timestamp}-${i}`} row={row} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <RetrainPanel
        onDone={() => {
          onRetrained();
          void load();
        }}
      />
    </div>
  );
}
