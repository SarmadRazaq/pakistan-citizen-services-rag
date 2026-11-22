"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

import { JURISDICTION_LABEL, LANGUAGE_LABEL, formatDate } from "@/lib/labels";
import { highlightQuotes } from "@/lib/text";
import type { AskResponse, RetrievedChunk } from "@/lib/types";

import { Badge, ScopeBadge, cx } from "../ui";

type Filter = "all" | "used" | "cited";

function ScoreBar({ value, threshold }: { value: number; threshold: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const tick = Math.max(0, Math.min(1, threshold)) * 100;
  const pass = value >= threshold;
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 shrink-0 text-xs text-muted">similarity</span>
      <div aria-hidden="true" className="relative h-2 flex-1 overflow-visible rounded-full bg-surface-2">
        <div className={cx("h-2 rounded-full", pass ? "bg-ok" : "bg-muted")} style={{ width: `${pct}%` }} />
        <div
          className="absolute -top-1 h-4 w-0.5 bg-fg"
          style={{ insetInlineStart: `${tick}%` }}
          title={`threshold ${threshold}`}
        />
      </div>
      <span className={cx("w-12 shrink-0 text-end font-mono text-xs tabular-nums", pass ? "text-ok" : "text-muted")}>
        {value.toFixed(3)}
      </span>
      <span className="sr-only">{pass ? "at or above" : "below"} the threshold</span>
    </div>
  );
}

function ChunkCard({
  chunk,
  threshold,
  quotes,
  expanded,
  onToggle,
  highlighted,
  cardRef,
}: {
  chunk: RetrievedChunk;
  threshold: number;
  quotes: string[];
  expanded: boolean;
  onToggle: () => void;
  highlighted: boolean;
  cardRef: (el: HTMLLIElement | null) => void;
}) {
  const textId = useId();
  const parts = useMemo(() => highlightQuotes(chunk.text, quotes), [chunk.text, quotes]);
  const urdu = chunk.language === "ur";
  const meta = [
    chunk.agency,
    chunk.jurisdiction ? JURISDICTION_LABEL[chunk.jurisdiction] : null,
    chunk.effective_date ? formatDate(chunk.effective_date) : null,
  ].filter(Boolean);

  return (
    <li
      ref={cardRef}
      tabIndex={-1}
      className={cx(
        "rounded-lg border bg-surface p-3 text-sm transition-shadow",
        highlighted ? "border-accent ring-2 ring-accent" : chunk.used_in_context ? "border-border" : "border-border/60",
        !chunk.used_in_context && "opacity-90",
      )}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-xs font-semibold text-muted">#{chunk.rank}</span>
        <ScopeBadge scope={chunk.scope} />
        {chunk.used_in_context && <Badge tone="info">used</Badge>}
        {chunk.cited && <Badge tone="ok">cited</Badge>}
        {chunk.is_table && <Badge>table</Badge>}
        {chunk.is_ocr && <Badge>OCR</Badge>}
        {chunk.injection_flag && (
          <Badge tone="bad" title="This chunk contains text that looks like instructions; it is treated as data only">
            ⚠ injection-flagged
          </Badge>
        )}
      </div>
      <p className="mt-1 break-all font-medium text-fg">{chunk.file_name}</p>
      <p className="text-xs text-muted">
        {meta.length > 0 && <span>{meta.join(" · ")} · </span>}
        {chunk.page != null ? `p. ${chunk.page}` : "no page"} ·{" "}
        <span lang={urdu ? "ur" : undefined} className="ui-ur">
          {chunk.language ? LANGUAGE_LABEL[chunk.language] : "?"}
        </span>
      </p>

      <div className="mt-2 flex flex-col gap-1">
        <ScoreBar value={chunk.similarity} threshold={threshold} />
        <p className="text-xs text-muted">
          rerank{" "}
          <span className="font-mono tabular-nums text-fg">
            {chunk.rerank_score != null ? chunk.rerank_score.toFixed(3) : "—"}
          </span>
        </p>
        <p className="text-xs text-muted">
          sparse <span className="font-mono tabular-nums">{chunk.sparse_score?.toFixed(3) ?? "—"}</span> · RRF{" "}
          <span className="font-mono tabular-nums">{chunk.rrf_score?.toFixed(4) ?? "—"}</span> · chunk{" "}
          <span className="font-mono">#{chunk.chunk_index}</span>
        </p>
      </div>

      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-controls={textId}
        className="mt-2 text-xs font-medium text-accent underline underline-offset-2"
      >
        {expanded ? "Hide text" : quotes.length ? "Show text and cited quote" : "Show text"}
      </button>
      {expanded && (
        <p
          id={textId}
          dir="auto"
          lang={urdu ? "ur" : undefined}
          className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-surface-2 p-2 text-fg"
        >
          {parts.map((p, i) => (p.hit ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>))}
        </p>
      )}
      {expanded && quotes.length > 0 && !parts.some((p) => p.hit) && (
        <p className="mt-1 text-xs text-muted">
          Cited quote (fuzzy match): <q dir="auto">{quotes.join(" … ")}</q>
        </p>
      )}
    </li>
  );
}

export function ChunksPanel({
  response,
  focus,
}: {
  response: AskResponse | null;
  /** Set by "Show in chunks panel"; `nonce` lets the same chunk be focused twice. */
  focus: { chunkId: string; nonce: number } | null;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [highlight, setHighlight] = useState<string | null>(null);
  const cards = useRef(new Map<string, HTMLLIElement>());
  const headingId = useId();

  const retrieval = response?.retrieval;
  const chunks = useMemo(() => retrieval?.chunks ?? [], [retrieval]);
  const quotesByChunk = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const c of response?.citations ?? []) m.set(c.chunk_id, [...(m.get(c.chunk_id) ?? []), c.quote]);
    return m;
  }, [response]);

  // New answer: reset the view.
  useEffect(() => {
    setFilter("all");
    setExpanded(new Set());
    setHighlight(null);
  }, [response?.request_id]);

  useEffect(() => {
    if (!focus) return;
    setFilter("all");
    setExpanded((s) => new Set(s).add(focus.chunkId));
    setHighlight(focus.chunkId);
    const t = setTimeout(() => {
      const el = cards.current.get(focus.chunkId);
      el?.scrollIntoView({ behavior: "smooth", block: "center" });
      el?.focus({ preventScroll: true });
    }, 30);
    return () => clearTimeout(t);
  }, [focus]);

  if (!response || !retrieval) {
    return (
      <section aria-labelledby={headingId} className="rounded-xl border border-dashed border-border p-4 text-sm text-muted">
        <h2 id={headingId} className="font-semibold text-fg">
          Retrieved chunks
        </h2>
        <p className="mt-1">Ask a question to see every candidate chunk, its scores, and which ones were used and cited.</p>
      </section>
    );
  }

  const cited = chunks.filter((c) => c.cited).length;
  const visible = chunks.filter((c) => (filter === "used" ? c.used_in_context : filter === "cited" ? c.cited : true));

  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-2">
      <div className="rounded-xl border border-border bg-surface p-3">
        <h2 id={headingId} className="font-semibold">
          Retrieved chunks
        </h2>
        <p className="mt-0.5 text-sm" aria-live="polite">
          <span className="font-semibold">Fetched {retrieval.candidates_fetched}</span>
          <span className="text-muted"> · </span>
          <span className="font-semibold text-info">used {retrieval.chunks_used}</span>
          <span className="text-muted"> · </span>
          <span className="font-semibold text-ok">cited {cited}</span>
          <span className="text-muted">
            {" "}
            · threshold <span className="font-mono">{retrieval.threshold.toFixed(2)}</span>
          </span>
        </p>
        <div role="radiogroup" aria-label="Filter chunks" className="mt-2 flex gap-1">
          {(["all", "used", "cited"] as Filter[]).map((f) => (
            <label
              key={f}
              className={cx(
                "cursor-pointer rounded-full border px-2.5 py-0.5 text-xs font-medium has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-[var(--focus)]",
                filter === f ? "border-accent bg-accent text-accent-fg" : "border-border text-fg hover:bg-surface-2",
              )}
            >
              <input
                type="radio"
                name="chunk-filter"
                className="sr-only"
                checked={filter === f}
                onChange={() => setFilter(f)}
              />
              {f === "all" ? `All (${chunks.length})` : f === "used" ? `Used (${chunks.filter((c) => c.used_in_context).length})` : `Cited (${cited})`}
            </label>
          ))}
        </div>
      </div>

      {visible.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border p-3 text-sm text-muted">
          {chunks.length === 0 ? "No chunks were retrieved for this question." : "No chunks match this filter."}
        </p>
      ) : (
        <ol className="flex flex-col gap-2" aria-label="Chunks by rank">
          {visible.map((c) => (
            <ChunkCard
              key={c.chunk_id}
              chunk={c}
              threshold={retrieval.threshold}
              quotes={quotesByChunk.get(c.chunk_id) ?? []}
              expanded={expanded.has(c.chunk_id)}
              highlighted={highlight === c.chunk_id}
              onToggle={() =>
                setExpanded((s) => {
                  const n = new Set(s);
                  if (n.has(c.chunk_id)) n.delete(c.chunk_id);
                  else n.add(c.chunk_id);
                  return n;
                })
              }
              cardRef={(el) => {
                if (el) cards.current.set(c.chunk_id, el);
                else cards.current.delete(c.chunk_id);
              }}
            />
          ))}
        </ol>
      )}
    </section>
  );
}
