import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ResultCard, type AnalysisRun } from "../components/ResultCard";
import { ToastProvider } from "../components/Toast";
import type { AnalysisResponse } from "../types";

function makeRun(overrides: Partial<AnalysisResponse>): AnalysisRun {
  return {
    code: "def f(a, b):\n    return a - b\n",
    latencyMs: 42.4,
    result: {
      label: "BUGGY",
      bug_probability: 0.89,
      confidence: 0.89,
      model: "tfidf",
      is_buggy: true,
      bug_type: null,
      explanation: null,
      suggested_fix: null,
      explanation_source: "unavailable",
      verified: false,
      ...overrides,
    },
  };
}

const renderCard = (props: Parameters<typeof ResultCard>[0]) =>
  render(
    <ToastProvider>
      <ResultCard {...props} />
    </ToastProvider>,
  );

describe("ResultCard", () => {
  it("shows an invitation before the first run", () => {
    renderCard({ run: null, loading: false, error: null });
    expect(screen.getByText(/Paste a Python function/)).toBeInTheDocument();
  });

  it("renders the BUGGY state with verified chip and bug type", () => {
    renderCard({
      run: makeRun({
        bug_type: "comparison_operator_change",
        explanation: "Synthetic mutation: comparison operator changed from `==` to `!=` on line 1.",
        explanation_source: "verified_code_pair",
        verified: true,
      }),
      loading: false,
      error: null,
    });
    expect(screen.getByTestId("verdict")).toHaveTextContent("Buggy");
    expect(screen.getByText("via TF-IDF · 42 ms")).toBeInTheDocument();
    expect(screen.getByText("Verified")).toBeInTheDocument();
    expect(screen.getByText("Comparison operator change")).toBeInTheDocument();
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "89");
    expect(screen.queryByText(/Uncertain zone/)).not.toBeInTheDocument();
  });

  it("renders the CLEAN state", () => {
    renderCard({
      run: makeRun({ label: "CLEAN", is_buggy: false, bug_probability: 0.12, confidence: 0.88 }),
      loading: false,
      error: null,
    });
    expect(screen.getByTestId("verdict")).toHaveTextContent("Clean");
    expect(screen.getByText("12.0%")).toBeInTheDocument();
    expect(screen.queryByText(/Uncertain zone/)).not.toBeInTheDocument();
  });

  it("flags the uncertain zone and unverified LoRA explanations", () => {
    renderCard({
      run: makeRun({
        bug_probability: 0.52,
        explanation: "Returns the difference of a and b.",
        explanation_source: "lora_generated_unverified",
      }),
      loading: false,
      error: null,
    });
    expect(screen.getByText("Uncertain zone — logged for review")).toBeInTheDocument();
    expect(screen.getByText("Generated, unverified")).toBeInTheDocument();
  });

  it("shows a skeleton while loading", () => {
    renderCard({ run: null, loading: true, error: null });
    expect(screen.getByTestId("result-skeleton")).toBeInTheDocument();
  });

  it("renders the 503 message as a readable alert", () => {
    renderCard({
      run: null,
      loading: false,
      error: "Model not available. Train its artifacts and install its dependencies, then try again.",
    });
    expect(screen.getByRole("alert")).toHaveTextContent(/^Model not available/);
  });
});
