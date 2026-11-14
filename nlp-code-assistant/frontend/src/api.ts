import {
  MAX_BODY_BYTES,
  type AnalysisRequest,
  type AnalysisResponse,
  type FeedbackRequest,
  type FeedbackResponse,
  type FeedbackStats,
  type HealthResponse,
  type RetrainJob,
  type RetrainRequest,
} from "./types";

export const API_BASE = "/api";
const DEFAULT_TIMEOUT_MS = 60_000;

/** An API failure carrying a short, human-readable message (never raw JSON). */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Drop keys whose value is null or undefined. The backend uses Pydantic with
 * extra="forbid" and strict=True, so optional fields must be omitted rather
 * than sent as null.
 */
export function omitNullish<T extends object>(obj: T): T {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value !== null && value !== undefined) out[key] = value;
  }
  return out as T;
}

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

function describeDetail(detail: unknown): string | null {
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (Array.isArray(detail) && detail.length > 0) {
    const issue = detail[0] as ValidationIssue;
    const field = issue.loc?.filter((p) => p !== "body").join(".");
    const msg = (issue.msg ?? "invalid value").replace(/^Value error, /, "");
    const more = detail.length > 1 ? ` (+${detail.length - 1} more)` : "";
    return field ? `${field}: ${msg}${more}` : `${msg}${more}`;
  }
  return null;
}

/** Map an HTTP status + FastAPI `detail` to a short message for the UI. */
export function friendlyError(status: number, detail: unknown): string {
  const text = describeDetail(detail);
  switch (status) {
    case 503:
      return "Model not available. Train its artifacts and install its dependencies, then try again.";
    case 413:
      return "Request too large. Keep the total input under 128 KB.";
    case 409:
      return "A retraining job is already running. Wait for it to finish.";
    case 404:
      return text ?? "Not found.";
    case 422:
      return text ? `Invalid input — ${text}` : "Invalid input.";
    case 500:
      return text ?? "The server hit an error.";
    case 502:
    case 504:
      return "Can't reach the backend. Is uvicorn running on port 8000?";
    default:
      return text ? `Request failed (${status}): ${text}` : `Request failed (${status}).`;
  }
}

interface RequestOptions {
  method?: "GET" | "POST";
  body?: object;
  timeoutMs?: number;
  signal?: AbortSignal;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = opts;
  let payload: string | undefined;
  if (body !== undefined) {
    payload = JSON.stringify(omitNullish(body));
    if (new TextEncoder().encode(payload).length > MAX_BODY_BYTES) {
      throw new ApiError(friendlyError(413, null), 413);
    }
  }

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: payload ? { "Content-Type": "application/json" } : undefined,
      body: payload,
      signal: controller.signal,
    });
  } catch (err) {
    if (timedOut) throw new ApiError("The request timed out.", 0);
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError("Can't reach the backend. Is uvicorn running on port 8000?", 0);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }

  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = ((await res.json()) as { detail?: unknown }).detail;
    } catch {
      /* non-JSON error body (e.g. proxy error page) */
    }
    throw new ApiError(friendlyError(res.status, detail), res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", { signal, timeoutMs: 10_000 }),

  analyze: (body: AnalysisRequest, signal?: AbortSignal) =>
    request<AnalysisResponse>("/analyze", { method: "POST", body, signal }),

  feedbackStats: (signal?: AbortSignal) => request<FeedbackStats>("/feedback", { signal }),

  submitFeedback: (body: FeedbackRequest) =>
    request<FeedbackResponse>("/feedback", { method: "POST", body }),

  /** Starts a background job and returns immediately (202). 409 if one is already running. */
  startRetrain: (body: RetrainRequest) => request<RetrainJob>("/feedback/retrain", { method: "POST", body }),

  retrainJob: (jobId: string, signal?: AbortSignal) =>
    request<RetrainJob>(`/feedback/retrain/${encodeURIComponent(jobId)}`, { signal }),

  /** The most recent job, so a reloaded page can pick up a running one. */
  latestRetrainJob: (signal?: AbortSignal) => request<{ job: RetrainJob | null }>("/feedback/retrain", { signal }),
};
