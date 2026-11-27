"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { JURISDICTION_LABEL, formatDate } from "@/lib/labels";
import { isArabicScript } from "@/lib/text";
import type { Citation } from "@/lib/types";

import { Button, ScopeBadge } from "../ui";

const WIDTH = 360;

/**
 * Non-modal popover anchored to a citation chip. Escape or a click outside closes it and the
 * parent returns focus to the chip.
 */
export function CitationPopover({
  citation,
  anchor,
  onClose,
  onShowChunk,
}: {
  citation: Citation;
  anchor: HTMLElement;
  onClose: () => void;
  onShowChunk: (chunkId: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  useLayoutEffect(() => {
    const place = () => {
      const r = anchor.getBoundingClientRect();
      const h = ref.current?.offsetHeight ?? 260;
      const w = Math.min(WIDTH, window.innerWidth - 16);
      const left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
      const below = r.bottom + 8;
      const top = below + h > window.innerHeight - 8 && r.top - h - 8 > 8 ? r.top - h - 8 : below;
      setPos({ top, left });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [anchor]);

  useEffect(() => {
    ref.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!ref.current?.contains(t) && !anchor.contains(t)) onClose();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [anchor, onClose]);

  const urduQuote = isArabicScript(citation.quote);
  const where = [
    citation.page != null ? `page ${citation.page}` : null,
    citation.section ? citation.section : null,
  ].filter(Boolean);

  return (
    <div
      ref={ref}
      role="dialog"
      aria-label={`Source ${citation.marker}: ${citation.file_name}`}
      tabIndex={-1}
      dir="ltr"
      style={{ top: pos?.top ?? -9999, left: pos?.left ?? -9999, width: Math.min(WIDTH, typeof window === "undefined" ? WIDTH : window.innerWidth - 16) }}
      className="fixed z-40 rounded-lg border border-border bg-surface p-3 text-sm text-fg shadow-xl"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="font-semibold">
          <span className="me-1 rounded bg-accent px-1.5 py-px text-xs text-accent-fg">{citation.marker}</span>
          <span className="break-all">{citation.file_name}</span>
        </p>
        <button type="button" onClick={onClose} aria-label="Close source" className="rounded px-1 text-muted hover:text-fg">
          ✕
        </button>
      </div>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-muted">Source</dt>
        <dd>
          <ScopeBadge scope={citation.scope} />
        </dd>
        <dt className="text-muted">Agency</dt>
        <dd>{citation.agency ?? "—"}</dd>
        <dt className="text-muted">Province</dt>
        <dd>{citation.jurisdiction ? JURISDICTION_LABEL[citation.jurisdiction] : "—"}</dd>
        <dt className="text-muted">Date</dt>
        <dd>{formatDate(citation.effective_date)}</dd>
        <dt className="text-muted">Location</dt>
        <dd>{where.length ? where.join(" · ") : "—"}</dd>
      </dl>
      <blockquote
        dir="auto"
        lang={urduQuote ? "ur" : undefined}
        className="mt-2 border-s-4 border-accent bg-surface-2 px-3 py-1.5 text-fg"
      >
        <mark>{citation.quote}</mark>
      </blockquote>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button
          onClick={() => {
            onShowChunk(citation.chunk_id);
            onClose();
          }}
        >
          Show in chunks panel
        </Button>
        {citation.source_url && (
          <a
            href={citation.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-accent underline underline-offset-2"
          >
            Official source<span className="sr-only"> (opens in a new tab)</span> ↗
          </a>
        )}
      </div>
    </div>
  );
}
