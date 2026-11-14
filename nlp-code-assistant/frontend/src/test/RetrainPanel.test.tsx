import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RetrainPanel } from "../components/RetrainPanel";
import { describeProgress } from "../lib/format";
import type { RetrainJob } from "../types";

const metrics = { accuracy: 0.84, bug_precision: 0.76, bug_recall: 0.74, bug_f1: 0.746 };

function job(overrides: Partial<RetrainJob>): RetrainJob {
  return {
    job_id: "abc123",
    status: "queued",
    epochs: 3,
    mistakes_only: false,
    progress: { phase: "queued" },
    result: null,
    metrics: null,
    output_dir: null,
    active_model_dir: "models/codebert_lora",
    error: null,
    created_at: new Date().toISOString(),
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

/** Answers each request from a queue per "METHOD path"; the last entry repeats. */
function mockApi(routes: Record<string, Response[]>) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? "GET"} ${url}`;
      calls.push(key);
      const queue = routes[key];
      if (!queue) throw new Error(`unexpected request: ${key}`);
      return (queue.length > 1 ? queue.shift()! : queue[0]).clone();
    }),
  );
  return calls;
}

async function startRetrain() {
  await userEvent.click(screen.getByRole("button", { name: "Retrain" }));
  await userEvent.click(screen.getByRole("button", { name: "Start retraining" }));
}

afterEach(() => vi.unstubAllGlobals());

describe("RetrainPanel", () => {
  it("starts a background job, polls its status and reports the accepted result", async () => {
    const onDone = vi.fn();
    const calls = mockApi({
      "GET /api/feedback/retrain": [json({ job: null })],
      "POST /api/feedback/retrain": [json(job({}), 202)],
      "GET /api/feedback/retrain/abc123": [
        json(job({ status: "running", progress: { phase: "training", epoch: 1, epochs: 3, loss: 0.4321 } })),
        json(
          job({
            status: "done",
            result: "accepted",
            metrics: { old: metrics, new: { ...metrics, bug_f1: 0.761 } },
            output_dir: "models/codebert_lora_v2_20261114T093005Z",
            active_model_dir: "models/codebert_lora_v2_20261114T093005Z",
          }),
        ),
      ],
    });
    render(<RetrainPanel onDone={onDone} />);
    await startRetrain();

    expect(await screen.findByText(/Training — epoch 1 of 3, loss 0\.4321/, {}, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retraining" })).toBeDisabled();

    expect(await screen.findByText(/Retrained and now active/, {}, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByText("74.6% → 76.1%")).toBeInTheDocument();
    expect(screen.getByText("models/codebert_lora_v2_20261114T093005Z")).toBeInTheDocument();
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Retrain" })).toBeEnabled();
    // One short POST, then polling: no long-lived request.
    expect(calls.filter((c) => c.startsWith("POST")).length).toBe(1);
    expect(calls.filter((c) => c === "GET /api/feedback/retrain/abc123").length).toBe(2);
  }, 10_000);

  it("explains a rejected model and a 409 conflict", async () => {
    mockApi({
      "GET /api/feedback/retrain": [
        json({ job: null }),
        json({
          job: job({
            status: "rejected",
            result: "rejected_worse",
            metrics: { old: metrics, new: { ...metrics, bug_f1: 0.6 } },
            output_dir: "models/codebert_lora_v3_x",
          }),
        }),
      ],
      "POST /api/feedback/retrain": [json({ detail: "A retraining job is already running" }, 409)],
    });
    render(<RetrainPanel onDone={() => {}} />);
    await startRetrain();

    expect(await screen.findByRole("alert")).toHaveTextContent("A retraining job is already running");
    expect(await screen.findByText(/Not applied: the new model was worse on validation/)).toBeInTheDocument();
    expect(screen.getByText("74.6% → 60.0%")).toBeInTheDocument();
  });

  it("resumes a job that is already running when the page loads", async () => {
    mockApi({
      "GET /api/feedback/retrain": [json({ job: job({ status: "running", progress: { phase: "evaluating_new" } }) })],
      "GET /api/feedback/retrain/abc123": [json(job({ status: "failed", error: "Split dir is missing" }))],
    });
    render(<RetrainPanel onDone={() => {}} />);
    expect(await screen.findByText(/Checking the new model on validation/)).toBeInTheDocument();
    expect(await screen.findByText("Split dir is missing", {}, { timeout: 4000 })).toBeInTheDocument();
  });

  it("describes every progress phase in plain words", () => {
    expect(describeProgress(job({ progress: { phase: "evaluating_current" } }))).toMatch(/current model/);
    expect(describeProgress(job({ progress: { phase: "saving" } }))).toMatch(/Saving/);
    expect(describeProgress(job({ progress: { phase: "something_new" } }))).toBe("Working");
  });
});
