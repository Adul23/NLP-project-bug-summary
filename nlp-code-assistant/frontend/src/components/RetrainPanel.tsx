import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import { describeProgress, pct } from "../lib/format";
import type { RetrainJob } from "../types";
import { AlertIcon, CheckIcon, Spinner } from "./icons";
import { Button, Card, ErrorNote } from "./ui";

const POLL_MS = 2000;

function elapsed(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

const isActive = (job: RetrainJob | null) => job?.status === "queued" || job?.status === "running";

function F1Change({ job }: { job: RetrainJob }) {
  if (!job.metrics) return null;
  return (
    <span className="font-mono">
      {pct(job.metrics.old.bug_f1)} → {pct(job.metrics.new.bug_f1)}
    </span>
  );
}

export function RetrainPanel({ onDone }: { onDone: () => void }) {
  const [epochs, setEpochs] = useState(3);
  const [mistakesOnly, setMistakesOnly] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [starting, setStarting] = useState(false);
  const [job, setJob] = useState<RetrainJob | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [error, setError] = useState<string | null>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  const running = starting || isActive(job);
  const jobId = isActive(job) ? job!.job_id : null;

  // Pick up a job that is already running (page reload, or started from another tab).
  useEffect(() => {
    const ctrl = new AbortController();
    api
      .latestRetrainJob(ctrl.signal)
      .then(({ job: latest }) => {
        if (isActive(latest)) setJob(latest);
      })
      .catch(() => {
        /* the Feedback tab already reports an unreachable backend */
      });
    return () => ctrl.abort();
  }, []);

  // Poll the job status until it reaches a final state.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const next = await api.retrainJob(jobId);
        if (cancelled) return;
        setJob(next);
        setError(null);
        if (isActive(next)) {
          timer = setTimeout(tick, POLL_MS);
        } else {
          onDoneRef.current();
        }
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          // The server restarted and forgot the job.
          setJob(null);
          setError("The retraining job was lost. The server may have restarted.");
          return;
        }
        setError((err as Error).message);
        timer = setTimeout(tick, POLL_MS);
      }
    };
    timer = setTimeout(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [running]);

  useEffect(() => {
    if (!confirming) return;
    confirmRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setConfirming(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirming]);

  const validEpochs = Number.isInteger(epochs) && epochs >= 1 && epochs <= 10;

  const start = useCallback(async () => {
    setConfirming(false);
    setStarting(true);
    setError(null);
    try {
      setJob(await api.startRetrain({ epochs, mistakes_only: mistakesOnly }));
      setNow(Date.now());
    } catch (err) {
      setError((err as Error).message);
      if (err instanceof ApiError && err.status === 409) {
        // Someone else started a job: follow that one.
        api
          .latestRetrainJob()
          .then(({ job: latest }) => latest && setJob(latest))
          .catch(() => {});
      }
    } finally {
      setStarting(false);
    }
  }, [epochs, mistakesOnly]);

  const startedAt = job ? Date.parse(job.started_at ?? job.created_at) : now;

  return (
    <Card className="p-4">
      <h2 className="text-sm font-medium">Retrain CodeBERT</h2>
      <p className="mt-1 text-xs text-muted">
        Fine-tunes the CodeBERT LoRA adapter on saved feedback mixed with original training rows. The new adapter
        replaces the active one only if its validation F1 does not drop by more than one point. Runs in the
        background.
      </p>
      <div className="mt-4 flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-muted">Epochs</span>
          <input
            type="number"
            min={1}
            max={10}
            step={1}
            value={Number.isNaN(epochs) ? "" : epochs}
            disabled={running}
            onChange={(e) => setEpochs(e.target.valueAsNumber)}
            className={`w-20 rounded border bg-surface px-2.5 py-1.5 font-mono text-sm outline-none focus:border-accent ${
              validEpochs ? "border-line" : "border-bad"
            }`}
          />
        </label>
        <label className="flex items-center gap-2 pb-1.5 text-sm">
          <input
            type="checkbox"
            checked={mistakesOnly}
            disabled={running}
            onChange={(e) => setMistakesOnly(e.target.checked)}
            className="h-4 w-4 accent-[rgb(var(--accent))]"
          />
          Mistakes only
        </label>
        <Button variant="primary" disabled={running || !validEpochs} onClick={() => setConfirming(true)}>
          {running && <Spinner />} {running ? "Retraining" : "Retrain"}
        </Button>
      </div>
      {!validEpochs && <p className="mt-2 text-xs text-bad">Epochs must be a whole number from 1 to 10.</p>}

      {job && isActive(job) && (
        <div className="mt-4" role="status">
          <div className="relative h-1.5 overflow-hidden rounded-full bg-subtle">
            {job.progress.phase === "training" && job.progress.epoch && job.progress.epochs ? (
              <div
                className="h-full rounded-full bg-accent transition-[width]"
                style={{ width: `${(job.progress.epoch / job.progress.epochs) * 100}%` }}
              />
            ) : (
              <div className="absolute inset-y-0 w-1/3 animate-[slide_1.4s_ease-in-out_infinite] rounded-full bg-accent" />
            )}
          </div>
          <p className="mt-1.5 text-xs text-muted">
            {describeProgress(job)} · {elapsed(now - startedAt)} elapsed. You can leave this tab; the job keeps
            running.
          </p>
        </div>
      )}
      {error && (
        <div className="mt-4">
          <ErrorNote message={error} />
        </div>
      )}
      {job?.status === "failed" && (
        <div className="mt-4">
          <ErrorNote message={job.error ?? "Retraining failed."} />
        </div>
      )}
      {job?.status === "done" && (
        <div className="mt-4 space-y-1 text-sm">
          <p className="flex flex-wrap items-center gap-1.5">
            <CheckIcon className="text-ok" /> Retrained and now active. Validation F1 <F1Change job={job} />
          </p>
          <p className="text-xs text-muted">
            Active model: <code className="rounded bg-subtle px-1.5 py-0.5 font-mono">{job.active_model_dir}</code>
          </p>
        </div>
      )}
      {job?.status === "rejected" && (
        <div className="mt-4 space-y-1 text-sm">
          <p className="flex flex-wrap items-center gap-1.5 text-warn">
            <AlertIcon /> Not applied: the new model was worse on validation. F1 <F1Change job={job} />
          </p>
          <p className="text-xs text-muted">
            The current model stays active. The rejected adapter was kept in{" "}
            <code className="rounded bg-subtle px-1.5 py-0.5 font-mono">{job.output_dir}</code>
          </p>
        </div>
      )}

      {confirming && (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/30 p-4"
          onClick={() => setConfirming(false)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="retrain-title"
            className="w-full max-w-sm rounded border border-line bg-surface p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 id="retrain-title" className="text-base font-semibold">
              Retrain CodeBERT?
            </h3>
            <p className="mt-2 text-sm text-muted">
              {epochs} epoch{epochs === 1 ? "" : "s"} on {mistakesOnly ? "mistakes only" : "all feedback"}. Training
              runs in the background and can take several minutes. The new adapter is applied only if it passes the
              validation check.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setConfirming(false)}>
                Cancel
              </Button>
              <Button ref={confirmRef} variant="primary" onClick={start}>
                Start retraining
              </Button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
