// Small presentational building blocks (plain Tailwind, no UI library).
import type { ReactNode } from "react";

import { STATUS_LABEL } from "@/lib/labels";
import type { ApiError } from "@/lib/problem";
import type { DocumentStatus, Visibility } from "@/lib/types";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

const SIZE = { md: "px-3 py-1.5 text-sm", sm: "px-2 py-0.5 text-xs" };

export const btn = {
  base: "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
  primary: "bg-accent text-accent-fg hover:opacity-90",
  secondary: "border border-border bg-surface text-fg hover:bg-surface-2",
  ghost: "text-fg hover:bg-surface-2",
  danger: "bg-bad text-white hover:opacity-90 dark:text-bg",
  dangerOutline: "border border-bad/50 bg-surface text-bad hover:bg-bad/10",
};

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof Omit<typeof btn, "base">;
  size?: keyof typeof SIZE;
}) {
  return <button type="button" {...props} className={cx(btn.base, SIZE[size], btn[variant], className)} />;
}

type Tone = "neutral" | "ok" | "warn" | "bad" | "info" | "official" | "mine";

const TONE: Record<Tone, string> = {
  neutral: "border-border text-muted",
  ok: "border-ok/40 bg-ok/10 text-ok",
  warn: "border-warn/40 bg-warn/10 text-warn",
  bad: "border-bad/40 bg-bad/10 text-bad",
  info: "border-info/40 bg-info/10 text-info",
  official: "border-official/40 bg-official/10 text-official",
  mine: "border-mine/40 bg-mine/10 text-mine",
};

export function Badge({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-px text-xs font-medium",
        TONE[tone],
      )}
    >
      {children}
    </span>
  );
}

export function ScopeBadge({ scope }: { scope: Visibility }) {
  return scope === "public" ? (
    <Badge tone="official" title="From the shared official corpus">
      Official
    </Badge>
  ) : (
    <Badge tone="mine" title="Your private upload">
      Yours
    </Badge>
  );
}

const STATUS_TONE: Record<DocumentStatus, Tone> = {
  PENDING: "neutral",
  PROCESSING: "info",
  UPDATING: "info",
  READY: "ok",
  READY_STALE: "warn",
  FAILED: "bad",
  DELETING: "warn",
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  const busy = status === "PENDING" || status === "PROCESSING" || status === "UPDATING" || status === "DELETING";
  return (
    <Badge tone={STATUS_TONE[status]}>
      {busy && <Spinner small />}
      {STATUS_LABEL[status]}
    </Badge>
  );
}

export function Spinner({ small = false, label }: { small?: boolean; label?: string }) {
  return (
    <span role={label ? "status" : undefined} className="inline-flex items-center gap-2">
      <span
        aria-hidden="true"
        className={cx(
          "inline-block animate-spin rounded-full border-2 border-current border-t-transparent",
          small ? "h-3 w-3" : "h-4 w-4",
        )}
      />
      {label && <span>{label}</span>}
    </span>
  );
}

/** A problem+json error rendered inline: title, detail and the request id for support. */
export function ProblemAlert({ error, className }: { error: ApiError; className?: string }) {
  return (
    <div role="alert" className={cx("border border-bad/40 bg-bad/10 px-3 py-2 text-sm", className ?? "rounded-md")}>
      <p className="font-semibold text-bad">{error.title}</p>
      {error.detail && error.detail !== error.title && <p className="mt-0.5 text-fg">{error.detail}</p>}
      {error.requestId && <p className="mt-1 font-mono text-xs text-muted">Request ID: {error.requestId}</p>}
    </div>
  );
}
