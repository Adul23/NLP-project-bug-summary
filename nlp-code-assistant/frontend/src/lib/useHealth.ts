import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { HealthResponse } from "../types";

const POLL_MS = 15_000;

export interface HealthState {
  health: HealthResponse | null;
  error: string | null;
  refresh: () => void;
}

export function useHealth(): HealthState {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    api
      .health(ctrl.signal)
      .then((h) => {
        setHealth(h);
        setError(null);
      })
      .catch((err: Error) => {
        if (err.name === "AbortError") return;
        setError(err.message);
      });
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => {
      clearInterval(id);
      inflight.current?.abort();
    };
  }, [refresh]);

  return { health, error, refresh };
}
