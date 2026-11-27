"use client";

import Link from "next/link";
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

import { cx } from "./ui";

export type ToastInput = {
  tone?: "info" | "ok" | "warn" | "bad";
  title: string;
  detail?: string;
  link?: { href: string; label: string };
  /** ms; 0 keeps it until dismissed. Toasts with a link default to 10 s. */
  duration?: number;
};

type ToastItem = ToastInput & { id: number };

const ToastContext = createContext<(t: ToastInput) => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

const TONE = {
  info: "border-info/50",
  ok: "border-ok/50",
  warn: "border-warn/50",
  bad: "border-bad/50",
};
const TITLE_TONE = { info: "text-info", ok: "text-ok", warn: "text-warn", bad: "text-bad" };

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);

  const push = useCallback(
    (t: ToastInput) => {
      const id = nextId.current++;
      setItems((xs) => [...xs.slice(-3), { ...t, id }]);
      const duration = t.duration ?? (t.link ? 10_000 : 6_000);
      if (duration > 0) setTimeout(() => dismiss(id), duration);
    },
    [dismiss],
  );

  const value = useMemo(() => push, [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      {/* Polite live region: screen readers announce each toast without stealing focus. */}
      <div
        aria-live="polite"
        aria-relevant="additions"
        className="pointer-events-none fixed inset-x-4 bottom-4 z-50 flex flex-col items-end gap-2 sm:inset-x-auto sm:end-4"
      >
        {items.map((t) => {
          const tone = t.tone ?? "info";
          return (
            <div
              key={t.id}
              role={tone === "bad" ? "alert" : "status"}
              className={cx(
                "pointer-events-auto w-full max-w-sm rounded-lg border-s-4 border bg-surface p-3 text-sm shadow-lg",
                TONE[tone],
              )}
            >
              <div className="flex items-start gap-3">
                <div className="min-w-0 flex-1">
                  <p className={cx("font-semibold", TITLE_TONE[tone])}>{t.title}</p>
                  {t.detail && (
                    <p dir="auto" className="mt-0.5 break-words text-fg">
                      {t.detail}
                    </p>
                  )}
                  {t.link && (
                    <Link
                      href={t.link.href}
                      onClick={() => dismiss(t.id)}
                      className="mt-1 inline-block font-medium text-accent underline underline-offset-2"
                    >
                      {t.link.label}
                    </Link>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => dismiss(t.id)}
                  className="rounded px-1 text-muted hover:text-fg"
                  aria-label="Dismiss notification"
                >
                  ✕
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}
