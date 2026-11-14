import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { EXAMPLES } from "../lib/examples";
import { isUsable, mentionedLine, STATUS_TEXT } from "../lib/format";
import { MAX_CODE_CHARS, type ExplanationMode, type HealthResponse, type ModelName } from "../types";
import { CodeEditor } from "./CodeEditor";
import { ChevronIcon, Spinner } from "./icons";
import { ResultCard, type AnalysisRun } from "./ResultCard";
import { Button, Segmented } from "./ui";

interface Props {
  health: HealthResponse | null;
  active: boolean;
  onAnalyzed?: () => void;
}

const fmt = (n: number) => n.toLocaleString("en-US");

function CharCounter({ count }: { count: number }) {
  const over = count > MAX_CODE_CHARS;
  return (
    <span data-testid="char-counter" className={`font-mono text-xs ${over ? "font-medium text-bad" : "text-muted"}`}>
      {fmt(count)} / {fmt(MAX_CODE_CHARS)}
    </span>
  );
}

export function AnalyzeTab({ health, active, onAnalyzed }: Props) {
  const [code, setCode] = useState("");
  const [original, setOriginal] = useState("");
  const [showOriginal, setShowOriginal] = useState(false);
  const [model, setModel] = useState<ModelName>("tfidf");
  const [explanation, setExplanation] = useState<ExplanationMode>("deterministic");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [examplesOpen, setExamplesOpen] = useState(false);
  const inflight = useRef<AbortController | null>(null);

  const status = health?.models;
  const tfidfOk = isUsable(status?.tfidf.status);
  const codebertOk = isUsable(status?.codebert.status);
  const summarizerOk = isUsable(status?.summarizer.status);

  // Fall back when /health says the current selection can't be used.
  useEffect(() => {
    if (model === "codebert" && !codebertOk && tfidfOk) setModel("tfidf");
    if (model === "tfidf" && !tfidfOk && codebertOk) setModel("codebert");
  }, [model, tfidfOk, codebertOk]);
  useEffect(() => {
    if (explanation === "lora" && !summarizerOk) setExplanation("deterministic");
  }, [explanation, summarizerOk]);

  const overLimit = code.length > MAX_CODE_CHARS || original.length > MAX_CODE_CHARS;
  const canSubmit = !loading && code.trim().length > 0 && !overLimit;
  const stale = run !== null && run.code !== code;
  const highlight = run && !stale ? mentionedLine(run.result.explanation) : null;

  const analyze = useCallback(async () => {
    if (!canSubmit) return;
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    setLoading(true);
    setError(null);
    const snapshot = code;
    const t0 = performance.now();
    try {
      const result = await api.analyze(
        {
          code: snapshot,
          model,
          explanation,
          original_code: showOriginal && original.trim() ? original : undefined,
        },
        ctrl.signal,
      );
      setRun({ result, latencyMs: performance.now() - t0, code: snapshot });
      onAnalyzed?.();
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError((err as Error).message);
    } finally {
      if (inflight.current === ctrl) setLoading(false);
    }
  }, [canSubmit, code, model, explanation, showOriginal, original, onAnalyzed]);

  // Ctrl/Cmd+Enter anywhere on the tab (editors handle it themselves and preventDefault).
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        void analyze();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, analyze]);

  function loadExample(i: number) {
    const ex = EXAMPLES[i];
    setCode(ex.code);
    setOriginal(ex.original ?? "");
    setShowOriginal(Boolean(ex.original));
    setExamplesOpen(false);
  }

  const unavailableTip = (name: "tfidf" | "codebert" | "summarizer") =>
    status ? `${STATUS_TEXT[status[name].status]} — see README to train it` : undefined;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="min-w-0 space-y-4">
        <div>
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <label className="text-sm font-medium">Python code</label>
            <div className="relative">
              <button
                type="button"
                className="text-sm text-accent hover:underline"
                aria-expanded={examplesOpen}
                onClick={() => setExamplesOpen((o) => !o)}
              >
                Load example
              </button>
              {examplesOpen && (
                <ul className="absolute right-0 z-20 mt-1 w-64 rounded border border-line bg-surface p-1 shadow-sm">
                  {EXAMPLES.map((ex, i) => (
                    <li key={ex.name}>
                      <button
                        type="button"
                        className="w-full rounded px-2.5 py-1.5 text-left text-sm hover:bg-subtle"
                        onClick={() => loadExample(i)}
                      >
                        {ex.name}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          <CodeEditor
            value={code}
            onChange={setCode}
            onSubmit={analyze}
            highlight={highlight}
            ariaLabel="Python code"
            placeholder={"def my_function(x):\n    ..."}
          />
          <div className="mt-1 flex justify-between">
            <span className="text-xs text-muted">{highlight ? `Line ${highlight} highlighted from the explanation` : ""}</span>
            <CharCounter count={code.length} />
          </div>
        </div>

        <div className="rounded border border-line bg-surface">
          <button
            type="button"
            aria-expanded={showOriginal}
            onClick={() => setShowOriginal((s) => !s)}
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm"
          >
            <ChevronIcon className={`text-muted transition-transform ${showOriginal ? "rotate-90" : ""}`} />
            Original code (optional, for verified diff)
          </button>
          {showOriginal && (
            <div className="border-t border-line p-3">
              <p className="mb-2 text-xs text-muted">
                Paste the version before the change. If exactly one operator differs, the explanation is verified.
              </p>
              <CodeEditor
                value={original}
                onChange={setOriginal}
                onSubmit={analyze}
                minHeight="140px"
                maxHeight="320px"
                ariaLabel="Original code"
              />
              <div className="mt-1 text-right">
                <CharCounter count={original.length} />
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <Segmented<ModelName>
            label="Model"
            value={model}
            onChange={setModel}
            options={[
              { value: "tfidf", label: "TF-IDF", disabled: !tfidfOk, tooltip: tfidfOk ? undefined : unavailableTip("tfidf") },
              {
                value: "codebert",
                label: "CodeBERT",
                disabled: !codebertOk,
                tooltip: codebertOk ? undefined : unavailableTip("codebert"),
              },
            ]}
          />
          <Segmented<ExplanationMode>
            label="Explanation"
            value={explanation}
            onChange={setExplanation}
            options={[
              { value: "deterministic", label: "Rule-based" },
              {
                value: "lora",
                label: "LoRA",
                disabled: !summarizerOk,
                tooltip: summarizerOk ? undefined : unavailableTip("summarizer"),
              },
            ]}
          />
          <Button variant="primary" className="ml-auto min-w-[120px]" disabled={!canSubmit} onClick={analyze}>
            {loading && <Spinner />}
            {loading ? "Analyzing" : "Analyze"}
          </Button>
        </div>
        {overLimit && <p className="text-xs text-bad">Code is over the 20,000 character limit.</p>}
      </div>

      <div className="min-w-0 lg:sticky lg:top-20 lg:self-start">
        <ResultCard run={run} loading={loading} error={error} stale={stale} />
      </div>
    </div>
  );
}
