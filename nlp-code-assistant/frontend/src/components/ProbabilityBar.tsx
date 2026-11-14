import { UNCERTAIN_HIGH, UNCERTAIN_LOW } from "../types";
import { isUncertain, pct } from "../lib/format";

export function ProbabilityBar({ value, buggy }: { value: number; buggy: boolean }) {
  const clamped = Math.min(1, Math.max(0, value));
  const uncertain = isUncertain(clamped);
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-xs text-muted">
        <span>Bug probability</span>
        <span className="font-mono text-sm font-medium text-fg">{pct(clamped)}</span>
      </div>
      <div
        className="relative h-2.5 rounded-full bg-subtle"
        role="meter"
        aria-label="Bug probability"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(clamped * 100)}
      >
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${buggy ? "bg-bad" : "bg-ok"}`}
          style={{ width: `${clamped * 100}%` }}
        />
        <div
          data-testid="uncertain-band"
          className="absolute -inset-y-1 border-x border-dashed border-muted/70 bg-muted/10"
          style={{ left: `${UNCERTAIN_LOW * 100}%`, width: `${(UNCERTAIN_HIGH - UNCERTAIN_LOW) * 100}%` }}
          title="Uncertain zone (40–60%)"
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted">
        <span>0</span>
        <span>50</span>
        <span>100</span>
      </div>
      {uncertain && (
        <p className="mt-1.5 text-xs font-medium text-warn">Uncertain zone — logged for review</p>
      )}
    </div>
  );
}
