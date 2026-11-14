import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { AlertIcon } from "./icons";

type Variant = "primary" | "secondary" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-accent-fg hover:bg-accent/90 border border-accent",
  secondary: "bg-surface text-fg border border-line hover:bg-subtle",
  ghost: "text-muted hover:text-fg hover:bg-subtle border border-transparent",
};

export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }>(
  function Button({ variant = "secondary", className = "", ...props }, ref) {
    return (
      <button
        ref={ref}
        type="button"
        className={`inline-flex items-center justify-center gap-2 rounded px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent/50 disabled:cursor-not-allowed disabled:opacity-50 ${VARIANTS[variant]} ${className}`}
        {...props}
      />
    );
  },
);

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`rounded border border-line bg-surface ${className}`}>{children}</section>;
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded border border-bad/40 bg-bad/5 px-3 py-2 text-sm text-bad">
      <AlertIcon className="mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
  tooltip?: string;
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: SegmentOption<T>[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-muted">{label}</span>
      <div role="radiogroup" aria-label={label} className="inline-flex rounded border border-line bg-subtle p-0.5">
        {options.map((o) => {
          const active = o.value === value;
          return (
            <span key={o.value} className="group relative" title={o.tooltip}>
              <button
                type="button"
                role="radio"
                aria-checked={active}
                disabled={o.disabled}
                onClick={() => onChange(o.value)}
                className={`rounded-[6px] px-3 py-1 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                  active ? "bg-surface font-medium text-fg shadow-[0_0_0_1px_rgb(var(--line))]" : "text-muted hover:text-fg"
                }`}
              >
                {o.label}
              </button>
              {o.tooltip && o.disabled && (
                <span
                  role="tooltip"
                  className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 hidden w-max max-w-[220px] -translate-x-1/2 rounded border border-line bg-surface px-2 py-1 text-xs text-fg group-hover:block"
                >
                  {o.tooltip}
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}
