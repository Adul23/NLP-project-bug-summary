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
});
