import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, friendlyError, omitNullish } from "../api";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("omitNullish", () => {
  it("drops null and undefined but keeps falsy values", () => {
    expect(omitNullish({ a: null, b: undefined, c: "", d: 0, e: false, f: "x" })).toEqual({
      c: "",
      d: 0,
      e: false,
      f: "x",
    });
  });
});

describe("api client", () => {
  it("omits null optionals from /analyze", async () => {
    const fetchMock = mockFetch(200, {});
    await api.analyze({
      code: "def f(): pass",
      model: "tfidf",
      explanation: "deterministic",
      original_code: null as unknown as undefined,
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/analyze");
    expect(JSON.parse(init.body)).toEqual({ code: "def f(): pass", model: "tfidf", explanation: "deterministic" });
    expect(init.body).not.toContain("original_code");
  });

  it("omits null optionals from /feedback", async () => {
    const fetchMock = mockFetch(200, { status: "saved", is_mistake: true });
    await api.submitFeedback({
      code: "x = 1",
      model: "tfidf",
      predicted_label: "CLEAN",
      actual_label: "BUGGY",
      user_explanation: undefined,
      fixed_code: null as unknown as undefined,
    });
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(Object.keys(body).sort()).toEqual(["actual_label", "code", "model", "predicted_label"]);
  });

  it("turns a 503 into a short human message", async () => {
    mockFetch(503, { detail: "codebert unavailable: train its artifacts and install its optional dependencies; see README" });
    const err = await api.analyze({ code: "x", model: "codebert", explanation: "deterministic" }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(503);
    expect(err.message).toMatch(/^Model not available/);
    expect(err.message).not.toContain("{");
  });

  it("formats list-style 422 details without raw JSON", () => {
    const msg = friendlyError(422, [
      { loc: ["body", "code"], msg: "Value error, Code must not be blank", type: "value_error" },
    ]);
    expect(msg).toBe("Invalid input — code: Code must not be blank");
  });

  it("rejects oversized bodies client-side with the 413 message", async () => {
    const fetchMock = mockFetch(200, {});
    const err = await api
      .analyze({ code: "é".repeat(70_000), model: "tfidf", explanation: "deterministic" })
      .catch((e) => e);
    expect(err.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
