import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AnalyzeTab } from "../components/AnalyzeTab";
import { ToastProvider } from "../components/Toast";
import type { HealthResponse } from "../types";

const health: HealthResponse = {
  status: "ok",
  models: {
    tfidf: { status: "artifacts_present_not_loaded" },
    codebert: { status: "unavailable" },
    summarizer: { status: "unavailable" },
    code_summarizer: { status: "unavailable" },
  },
};

afterEach(() => vi.unstubAllGlobals());

describe("AnalyzeTab", () => {
  it("disables unavailable models and shows the backend 503 as a human message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "tfidf unavailable: train its artifacts" }), { status: 503 }),
      ),
    );
    render(
      <ToastProvider>
        <AnalyzeTab health={health} active />
      </ToastProvider>,
    );
    expect(screen.getByRole("radio", { name: "CodeBERT" })).toBeDisabled();
    expect(screen.getByRole("radio", { name: "LoRA" })).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: "Load example" }));
    await userEvent.click(screen.getByRole("button", { name: "Clean helper" }));
    await userEvent.click(screen.getByRole("button", { name: "Analyze" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Model not available");
    expect(alert.textContent).not.toContain("detail");
  });

  it("highlights the suspected line from an operator-flip result", async () => {
    const result = {
      label: "BUGGY",
      bug_probability: 0.9,
      confidence: 0.9,
      model: "tfidf",
      is_buggy: true,
      bug_type: "comparison_operator",
      explanation: "Possibly a flipped comparison",
      suggested_fix: "def f():\n    pass\n",
      explanation_source: "operator_flip",
      verified: false,
      suspected_line: 3,
      repair_candidates: [{ line: 3, old: "<", new: ">", mutation_type: "comparison_operator", drop: 0.5 }],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async () => new Response(JSON.stringify(result), { status: 200 })),
    );
    render(
      <ToastProvider>
        <AnalyzeTab health={health} active />
      </ToastProvider>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Load example" }));
    await userEvent.click(screen.getByRole("button", { name: "Clean helper" }));
    await userEvent.click(screen.getByRole("button", { name: "Analyze" }));

    // suspected_line wins even though the explanation text names no line
    expect(await screen.findByText("Line 3 highlighted from the explanation")).toBeInTheDocument();
  });
});
