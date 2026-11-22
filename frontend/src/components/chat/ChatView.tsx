"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, newRequestId } from "@/lib/client";
import { ApiError, toProblem } from "@/lib/problem";

import { AnswerBubble, QuestionBubble, type Turn } from "./AnswerBubble";
import { ChunksPanel } from "./ChunksPanel";
import { DebugDrawer } from "./DebugDrawer";
import { QuestionBox, type AskOptions } from "./QuestionBox";

export function ChatView() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [options, setOptions] = useState<AskOptions>({ answerLanguage: "auto", scope: "all", jurisdiction: "auto" });
  const [focus, setFocus] = useState<{ chunkId: string; nonce: number } | null>(null);
  const [debugOpen, setDebugOpen] = useState(false);
  const [liveStatus, setLiveStatus] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const chunksRef = useRef<HTMLDivElement>(null);

  const busy = turns.some((t) => t.pending);
  const selected = turns.find((t) => t.id === selectedId) ?? null;
  const selectedResponse = selected?.response ?? null;

  // Keep the newest turn in view: once when it is asked, again when its answer arrives.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, busy]);

  const ask = useCallback(
    async (question: string) => {
      const id = newRequestId();
      setTurns((ts) => [...ts, { id, question, pending: true }]);
      setLiveStatus("Searching the sources…");
      try {
        const response = await api.ask({
          question,
          scope: options.scope,
          answer_language: options.answerLanguage,
          jurisdictions: options.jurisdiction === "auto" ? null : [options.jurisdiction],
          agencies: null,
          doc_ids: null,
          top_k: 5,
        });
        setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, pending: false, response } : t)));
        setSelectedId(id);
        setLiveStatus(
          response.answerable
            ? `Answer ready with ${response.citations?.length ?? 0} citations.`
            : "The question was not answered. The reason is shown in the answer.",
        );
      } catch (err) {
        const error =
          err instanceof ApiError
            ? err
            : new ApiError(toProblem(0, { code: "client_error", title: "Something went wrong", detail: String(err) }));
        setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, pending: false, error } : t)));
        setLiveStatus(`Error: ${error.title}.`);
      }
    },
    [options],
  );

  const showChunk = useCallback((turnId: string, chunkId: string) => {
    setSelectedId(turnId);
    setFocus({ chunkId, nonce: Date.now() });
    if (window.matchMedia("(max-width: 1023px)").matches) {
      chunksRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, []);

  return (
    <div className="mx-auto grid max-w-[1600px] gap-4 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_minmax(340px,440px)]">
      <p className="sr-only" aria-live="polite" role="status">
        {liveStatus}
      </p>

      <div className="flex min-w-0 flex-col gap-4">
        {turns.length === 0 ? (
          <section className="rounded-xl border border-border bg-surface p-5">
            <h1 className="text-xl font-semibold">Ask about a citizen service</h1>
            <p className="mt-1 text-muted">
              Answers come only from official documents and your own uploads, with a citation for every fact. Ask in
              English, Urdu or Roman Urdu.
            </p>
            <p lang="ur" dir="rtl" className="mt-1 text-muted">
              سرکاری دستاویزات سے حوالہ جات کے ساتھ جوابات
            </p>
          </section>
        ) : (
          <ol className="flex flex-col gap-4" aria-label="Conversation">
            {turns.map((t) => (
              <li key={t.id} className="flex flex-col gap-2">
                <QuestionBubble text={t.question} />
                <div className="max-w-[95%]">
                  <AnswerBubble
                    turn={t}
                    selected={t.id === selectedId && !!t.response}
                    onSelect={() => setSelectedId(t.id)}
                    onShowChunk={(chunkId) => showChunk(t.id, chunkId)}
                    onOpenDebug={() => {
                      setSelectedId(t.id);
                      setDebugOpen(true);
                    }}
                  />
                </div>
              </li>
            ))}
          </ol>
        )}
        <div ref={endRef} />
        <div className="pb-2 [@media(min-width:1024px)_and_(min-height:820px)]:sticky [@media(min-width:1024px)_and_(min-height:820px)]:bottom-0 [@media(min-width:1024px)_and_(min-height:820px)]:z-10 [@media(min-width:1024px)_and_(min-height:820px)]:bg-bg">
          <QuestionBox busy={busy} onAsk={ask} options={options} onOptionsChange={setOptions} />
        </div>
      </div>

      <aside
        ref={chunksRef}
        aria-label="Retrieved chunks panel"
        className="min-w-0 lg:sticky lg:top-16 lg:max-h-[calc(100dvh-5rem)] lg:overflow-y-auto lg:pe-1"
      >
        <ChunksPanel response={selectedResponse} focus={focus} />
      </aside>

      <DebugDrawer response={selectedResponse} open={debugOpen} onClose={() => setDebugOpen(false)} />
    </div>
  );
}
