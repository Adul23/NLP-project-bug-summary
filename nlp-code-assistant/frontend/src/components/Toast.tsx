import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";

type Tone = "ok" | "bad";
interface ToastItem {
  id: number;
  text: string;
  tone: Tone;
}

const ToastContext = createContext<(text: string, tone?: Tone) => void>(() => {});

// eslint-disable-next-line react-refresh/only-export-components -- hook is colocated with its provider
export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const push = useCallback((text: string, tone: Tone = "ok") => {
    const id = nextId.current++;
    setItems((xs) => [...xs, { id, text, tone }]);
    setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), 3500);
  }, []);

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col gap-2" aria-live="polite">
        {items.map((t) => (
          <div
            key={t.id}
            role="status"
            className="pointer-events-auto flex items-center gap-2 rounded border border-line bg-surface px-3 py-2 text-sm shadow-sm"
          >
            <span className={`h-2 w-2 rounded-full ${t.tone === "ok" ? "bg-ok" : "bg-bad"}`} />
            {t.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
