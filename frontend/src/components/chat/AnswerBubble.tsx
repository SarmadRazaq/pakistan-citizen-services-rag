"use client";

import { useCallback, useRef, useState } from "react";

import { LANGUAGE_LABEL, REFUSAL_LABEL } from "@/lib/labels";
import { ApiError, isGuardrailBlock } from "@/lib/problem";
import { isArabicScript, splitMarkers } from "@/lib/text";
import type { AskResponse, Citation } from "@/lib/types";

import { Badge, Button, ProblemAlert, Spinner, cx } from "../ui";
import { CitationPopover } from "./CitationPopover";
import { NoticeItem, NoticeStrip } from "./NoticeStrip";

export type Turn = {
  id: string;
  question: string;
  pending: boolean;
  response?: AskResponse;
  error?: ApiError;
};

export function QuestionBubble({ text }: { text: string }) {
  const urdu = isArabicScript(text);
  return (
    <div className="flex justify-end">
      <p
        dir="auto"
        lang={urdu ? "ur" : undefined}
        className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-ee-sm bg-accent px-4 py-2 text-accent-fg"
      >
        {text}
      </p>
    </div>
  );
}

function AnswerText({
  answer,
  citations,
  onShowChunk,
  urdu,
}: {
  answer: string;
  citations: Citation[];
  onShowChunk: (chunkId: string) => void;
  urdu: boolean;
}) {
  const [open, setOpen] = useState<{ index: number; marker: string; anchor: HTMLElement } | null>(null);
  const chipRefs = useRef(new Map<number, HTMLButtonElement>());
  const byMarker = new Map(citations.map((c) => [c.marker, c]));

  const openIndex = open?.index ?? null;
  const close = useCallback(() => {
    setOpen(null);
    // Return focus to the chip that opened the popover.
    if (openIndex !== null) chipRefs.current.get(openIndex)?.focus();
  }, [openIndex]);

  const segments = splitMarkers(answer);
  return (
    <>
      <p dir="auto" lang={urdu ? "ur" : undefined} className="whitespace-pre-wrap break-words text-fg">
        {segments.map((seg, i) => {
          if (seg.kind === "text") return <span key={i}>{seg.text}</span>;
          const c = byMarker.get(seg.marker);
          if (!c) {
            return (
              <span
                key={i}
                title="No verified citation for this marker"
                className="mx-0.5 inline-block rounded border border-dashed border-bad px-1 align-baseline font-mono text-xs text-bad"
                style={{ lineHeight: 1.4 }}
              >
                {seg.marker}?
              </span>
            );
          }
          const isOpen = open?.index === i;
          return (
            <button
              key={i}
              ref={(el) => {
                if (el) chipRefs.current.set(i, el);
                else chipRefs.current.delete(i);
              }}
              type="button"
              dir="ltr"
              aria-haspopup="dialog"
              aria-expanded={isOpen}
              aria-label={`Source ${seg.marker}: ${c.file_name}${c.page != null ? `, page ${c.page}` : ""}`}
              onClick={(e) => setOpen(isOpen ? null : { index: i, marker: seg.marker, anchor: e.currentTarget })}
              className={cx(
                "mx-0.5 inline-block rounded px-1.5 align-baseline font-mono text-xs font-semibold",
                isOpen ? "bg-accent text-accent-fg" : "bg-accent/15 text-accent hover:bg-accent/25",
              )}
              style={{ lineHeight: 1.5 }}
            >
              {seg.marker}
            </button>
          );
        })}
      </p>
      {open && byMarker.has(open.marker) && (
        <CitationPopover
          citation={byMarker.get(open.marker)!}
          anchor={open.anchor}
          onClose={close}
          onShowChunk={onShowChunk}
        />
      )}
    </>
  );
}

function Blocked({ title, detail, requestId }: { title: string; detail: string; requestId?: string | null }) {
  return (
    <div role="alert" className="rounded-2xl rounded-es-sm border-2 border-bad bg-bad/10 px-4 py-3">
      <p className="flex items-center gap-2 font-semibold text-bad">
        <span aria-hidden="true">⛔</span> Blocked by a safety check
      </p>
      <p className="mt-1 text-sm text-fg">
        <span className="font-medium">{title}.</span> {detail}
      </p>
      <p className="mt-1 text-xs text-muted">Rephrase the question as a plain question about a citizen service.</p>
      {requestId && <p className="mt-1 font-mono text-xs text-muted">Request ID: {requestId}</p>}
    </div>
  );
}

export function AnswerBubble({
  turn,
  selected,
  onSelect,
  onShowChunk,
  onOpenDebug,
}: {
  turn: Turn;
  selected: boolean;
  onSelect: () => void;
  onShowChunk: (chunkId: string) => void;
  onOpenDebug: () => void;
}) {
  if (turn.pending) {
    return (
      <div className="rounded-2xl rounded-es-sm border border-border bg-surface px-4 py-3 text-sm text-muted">
        <Spinner small label="Searching official sources and writing a cited answer…" />
      </div>
    );
  }

  if (turn.error) {
    if (isGuardrailBlock(turn.error)) {
      return <Blocked title={turn.error.title} detail={turn.error.detail} requestId={turn.error.requestId} />;
    }
    return <ProblemAlert error={turn.error} className="rounded-2xl rounded-es-sm" />;
  }

  const r = turn.response;
  if (!r) return null;

  if (r.guardrails?.input === "blocked") {
    return <Blocked title="Input blocked" detail={r.answer || "This question was not processed."} requestId={r.request_id} />;
  }

  const notices = r.notices ?? [];
  const emergency = notices.filter((n) => n.code === "emergency");
  const rest = notices.filter((n) => n.code !== "emergency");
  const answerLang = r.language.answer;
  const rtl = answerLang === "ur";
  const refused = !r.answerable;
  const outputRefused = r.guardrails?.output === "refused";
  const citations = r.citations ?? [];
  const unanswered = r.unanswered_parts ?? [];

  return (
    <article
      dir={rtl ? "rtl" : "ltr"}
      aria-label="Answer"
      className={cx(
        "flex flex-col gap-2 rounded-2xl rounded-es-sm border bg-surface px-4 py-3",
        selected ? "border-accent ring-1 ring-accent" : "border-border",
      )}
    >
      {emergency.length > 0 && <NoticeStrip notices={emergency} label="Emergency notice" />}

      {r.degraded && (
        <p role="status" className="rounded-md border border-warn/50 bg-warn/10 px-3 py-1.5 text-sm text-fg" dir="ltr">
          <span className="font-semibold text-warn">Degraded answer.</span> The hosted model was unavailable, so the
          local fallback ({r.provider.model ?? r.provider.endpoint}) answered. Quality may be lower.
        </p>
      )}

      {refused || outputRefused ? (
        <div className="rounded-md border-2 border-dashed border-warn/70 bg-warn/5 px-3 py-2">
          <p className="text-sm font-semibold text-warn" dir="ltr">
            Not answered · {r.refusal_reason ? REFUSAL_LABEL[r.refusal_reason] : outputRefused ? REFUSAL_LABEL.guardrail_output : "No answer"}
          </p>
          <p dir="auto" lang={rtl ? "ur" : undefined} className="mt-1 text-fg">
            {r.answer || "The available sources do not contain an answer to this question."}
          </p>
          {unanswered.length > 0 && (
            <>
              <p className="mt-1 text-xs font-semibold text-muted" dir="ltr">
                Not covered by the sources
              </p>
              <ul className="list-disc ps-5 text-sm text-fg">
                {unanswered.map((u) => (
                  <li key={u} dir="auto">
                    {u}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      ) : (
        <>
          <AnswerText answer={r.answer} citations={citations} onShowChunk={onShowChunk} urdu={rtl} />
          {unanswered.length > 0 && (
            <div className="rounded-md border border-dashed border-warn/60 px-3 py-1.5 text-sm" dir="ltr">
              <p className="font-semibold text-warn">Not covered by the sources</p>
              <ul className="list-disc ps-5 text-fg">
                {unanswered.map((u) => (
                  <li key={u} dir="auto">
                    {u}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      <NoticeStrip notices={rest} label="Notices about this answer" />

      <footer className="flex flex-wrap items-center gap-2 border-t border-border pt-2 text-xs text-muted" dir="ltr">
        <Badge title="Question language → answer language">
          <span lang={r.language.question === "ur" ? "ur" : undefined} className="ui-ur">
            {LANGUAGE_LABEL[r.language.question]}
          </span>
          →
          <span lang={answerLang === "ur" ? "ur" : undefined} className="ui-ur">
            {LANGUAGE_LABEL[answerLang]}
          </span>
        </Badge>
        {citations.length > 0 && <Badge tone="ok">{citations.length} cited</Badge>}
        {r.guardrails?.output === "repaired" && <Badge tone="warn">repaired by output check</Badge>}
        <span>
          {r.provider.endpoint === "none" ? "no model call" : r.provider.model ?? r.provider.endpoint}
          {r.latency_ms?.total != null && ` · ${(r.latency_ms.total / 1000).toFixed(1)} s`}
        </span>
        <span className="ms-auto flex gap-1">
          {!selected && (
            <Button variant="ghost" size="sm" onClick={onSelect}>
              Show chunks
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onOpenDebug}>
            Debug
          </Button>
        </span>
      </footer>
    </article>
  );
}
