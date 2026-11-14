import { useState } from "react";
import { api } from "../api";
import { MAX_EXPLANATION_CHARS, type AnalysisResponse, type Label } from "../types";
import { CodeEditor } from "./CodeEditor";
import { CheckIcon, Spinner, XIcon } from "./icons";
import { useToast } from "./Toast";
import { Button, ErrorNote } from "./ui";

interface Props {
  result: AnalysisResponse;
  code: string;
}

export function FeedbackRow({ result, code }: Props) {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [explanation, setExplanation] = useState("");
  const [fixed, setFixed] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState<"correct" | "wrong" | null>(null);

  const opposite: Label = result.label === "BUGGY" ? "CLEAN" : "BUGGY";

  async function send(actual: Label, kind: "correct" | "wrong") {
    setBusy(true);
    setError(null);
    try {
      await api.submitFeedback({
        code,
        model: result.model,
        predicted_label: result.label,
        actual_label: actual,
        user_explanation: explanation.trim() || undefined,
        fixed_code: fixed.trim() ? fixed : undefined,
      });
      setSent(kind);
      setOpen(false);
      toast("Feedback saved");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (sent) {
    return (
      <p className="flex items-center gap-1.5 text-sm text-muted">
        <CheckIcon className="text-ok" />
        Thanks — feedback saved{sent === "wrong" ? ` (marked as ${opposite.toLowerCase()})` : ""}.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-1 text-sm text-muted">Was this right?</span>
        <Button disabled={busy} onClick={() => send(result.label, "correct")}>
          <CheckIcon className="text-ok" /> Correct
        </Button>
        <Button disabled={busy} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
          <XIcon className="text-bad" /> Wrong
        </Button>
      </div>

      {open && (
        <div className="space-y-3 rounded border border-line bg-subtle/50 p-3">
          <p className="text-xs text-muted">
            This will be saved as <span className="font-medium text-fg">{opposite.toLowerCase()}</span>.
          </p>
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-muted">What's wrong? (optional)</span>
            <textarea
              value={explanation}
              maxLength={MAX_EXPLANATION_CHARS}
              onChange={(e) => setExplanation(e.target.value)}
              rows={2}
              placeholder="e.g. operator + was replaced with - on line 2"
              className="w-full resize-y rounded border border-line bg-surface px-2.5 py-1.5 text-sm outline-none focus:border-accent"
            />
          </label>
          <div>
            <span className="mb-1 block text-xs font-medium text-muted">Fixed code (optional)</span>
            <CodeEditor value={fixed} onChange={setFixed} minHeight="96px" maxHeight="240px" ariaLabel="Fixed code" />
          </div>
          {error && <ErrorNote message={error} />}
          <div className="flex gap-2">
            <Button variant="primary" disabled={busy} onClick={() => send(opposite, "wrong")}>
              {busy && <Spinner />} Submit feedback
            </Button>
            <Button variant="ghost" disabled={busy} onClick={() => setOpen(false)}>
              Cancel
            </Button>
          </div>
        </div>
      )}
      {!open && error && <ErrorNote message={error} />}
    </div>
  );
}
