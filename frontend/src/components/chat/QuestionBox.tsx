"use client";

import { useId, useRef, useState, type KeyboardEvent } from "react";

import { MAX_QUESTION_CHARS } from "@/lib/config";
import { ANSWER_LANGUAGE_OPTIONS, JURISDICTION_LABEL, SCOPE_OPTIONS, TOPIC_STARTERS } from "@/lib/labels";
import { isArabicScript } from "@/lib/text";
import type { AnswerLanguage, Jurisdiction, Scope } from "@/lib/types";

import { Button, Spinner, cx } from "../ui";

export type AskOptions = {
  answerLanguage: AnswerLanguage;
  scope: Scope;
  jurisdiction: Jurisdiction | "auto";
};

export function QuestionBox({
  busy,
  onAsk,
  options,
  onOptionsChange,
}: {
  busy: boolean;
  onAsk: (question: string) => void;
  options: AskOptions;
  onOptionsChange: (o: AskOptions) => void;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const ids = { q: useId(), count: useId(), scope: useId(), prov: useId(), lang: useId() };
  const over = text.length > MAX_QUESTION_CHARS;
  const near = text.length > MAX_QUESTION_CHARS * 0.9;
  const blank = text.trim().length === 0;
  const urdu = isArabicScript(text);

  const submit = () => {
    if (busy || blank || over) return;
    onAsk(text.trim());
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form
      className="rounded-xl border border-border bg-surface p-3 shadow-sm"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2" role="group" aria-label="Example questions by topic">
        <span className="text-xs font-medium text-muted">Try:</span>
        {TOPIC_STARTERS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => {
              setText(t.question);
              ref.current?.focus();
            }}
            className="rounded-full border border-border px-2.5 py-0.5 text-xs font-medium text-fg hover:bg-surface-2"
            title={t.question}
          >
            {t.label}
          </button>
        ))}
      </div>

      <label htmlFor={ids.q} className="sr-only">
        Your question
      </label>
      <textarea
        ref={ref}
        id={ids.q}
        dir="auto"
        lang={urdu ? "ur" : undefined}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        rows={3}
        placeholder="Ask about ID cards, passports, BISP, tax or your rights, in English, اردو or Roman Urdu…"
        aria-describedby={ids.count}
        aria-invalid={over || undefined}
        className="block w-full resize-y rounded-md border border-border bg-bg px-3 py-2 text-base text-fg placeholder:text-muted"
      />

      <div className="mt-2 flex flex-wrap items-end gap-3">
        <fieldset className="flex flex-col gap-1">
          <legend id={ids.lang} className="text-xs font-medium text-muted">
            Answer in
          </legend>
          <div role="radiogroup" aria-labelledby={ids.lang} className="flex overflow-hidden rounded-md border border-border">
            {ANSWER_LANGUAGE_OPTIONS.map((o) => {
              const checked = options.answerLanguage === o.value;
              return (
                <label
                  key={o.value}
                  className={cx(
                    "cursor-pointer px-2.5 py-1 text-xs font-medium has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-[var(--focus)]",
                    checked ? "bg-accent text-accent-fg" : "bg-surface text-fg hover:bg-surface-2",
                  )}
                >
                  <input
                    type="radio"
                    name="answer-language"
                    value={o.value}
                    checked={checked}
                    onChange={() => onOptionsChange({ ...options, answerLanguage: o.value })}
                    className="sr-only"
                  />
                  <span lang={o.lang} className="ui-ur">
                    {o.label}
                  </span>
                </label>
              );
            })}
          </div>
        </fieldset>

        <div className="flex flex-col gap-1">
          <label htmlFor={ids.scope} className="text-xs font-medium text-muted">
            Search in
          </label>
          <select
            id={ids.scope}
            value={options.scope}
            onChange={(e) => onOptionsChange({ ...options, scope: e.target.value as Scope })}
            className="rounded-md border border-border bg-surface px-2 py-1 text-sm text-fg"
          >
            {SCOPE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor={ids.prov} className="text-xs font-medium text-muted">
            Province
          </label>
          <select
            id={ids.prov}
            value={options.jurisdiction}
            onChange={(e) => onOptionsChange({ ...options, jurisdiction: e.target.value as AskOptions["jurisdiction"] })}
            className="rounded-md border border-border bg-surface px-2 py-1 text-sm text-fg"
          >
            <option value="auto">Auto (from question)</option>
            {(Object.keys(JURISDICTION_LABEL) as Jurisdiction[]).map((j) => (
              <option key={j} value={j}>
                {JURISDICTION_LABEL[j]}
              </option>
            ))}
          </select>
        </div>

        <div className="ms-auto flex items-center gap-3">
          <span
            id={ids.count}
            className={cx("text-xs tabular-nums", over ? "font-semibold text-bad" : near ? "text-warn" : "text-muted")}
            aria-live={near ? "polite" : "off"}
          >
            {text.length.toLocaleString()} / {MAX_QUESTION_CHARS.toLocaleString()}
            {over && " — too long"}
          </span>
          <Button type="submit" variant="primary" disabled={busy || blank || over}>
            {busy ? <Spinner small label="Asking…" /> : "Ask"}
          </Button>
        </div>
      </div>
      <p className="mt-1 text-xs text-muted">Enter to send · Shift+Enter for a new line</p>
    </form>
  );
}
