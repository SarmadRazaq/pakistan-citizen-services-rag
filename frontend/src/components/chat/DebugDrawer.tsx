"use client";

import { useEffect, useId, useRef } from "react";

import { JURISDICTION_LABEL } from "@/lib/labels";
import { isArabicScript } from "@/lib/text";
import type { AskResponse } from "@/lib/types";

import { Badge, Button } from "../ui";

const VARIANT_LABEL = ["Original", "English", "Urdu script"];
const STEP_ORDER = ["understand", "retrieve", "rerank", "generate", "guard_output"];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-border py-3 first:border-t-0">
      <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">{title}</h3>
      {children}
    </section>
  );
}

/** Modal side drawer (native <dialog>) with the pipeline internals for one answer. */
export function DebugDrawer({ response, open, onClose }: { response: AskResponse | null; open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  }, [open]);

  const r = response;
  const latency = r?.latency_ms ?? {};
  const steps = [
    ...STEP_ORDER.filter((k) => k in latency),
    ...Object.keys(latency).filter((k) => k !== "total" && !STEP_ORDER.includes(k)),
  ];
  const maxStep = Math.max(1, ...steps.map((k) => latency[k] ?? 0));

  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
      className="ms-auto me-0 mt-0 mb-0 h-dvh max-h-dvh w-[min(28rem,100vw)] max-w-full border-s border-border bg-surface p-0 text-fg shadow-2xl"
    >
      {open && (
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <h2 id={titleId} className="font-semibold">
              Debug
            </h2>
            <Button variant="ghost" onClick={onClose} aria-label="Close debug drawer">
              ✕
            </Button>
          </div>
          {!r ? (
            <p className="p-4 text-sm text-muted">No answer selected.</p>
          ) : (
            <div className="flex-1 overflow-y-auto px-4 pb-6 text-sm">
              <Section title="Query variants">
                <ol className="flex flex-col gap-1">
                  {r.retrieval.query_variants.map((v, i) => (
                    <li key={i} className="rounded bg-surface-2 px-2 py-1">
                      <span className="me-2 text-xs font-medium text-muted">{VARIANT_LABEL[i] ?? `Variant ${i + 1}`}</span>
                      <span dir="auto" lang={isArabicScript(v) ? "ur" : undefined}>
                        {v}
                      </span>
                    </li>
                  ))}
                  {r.retrieval.query_variants.length === 0 && <li className="text-muted">none (stopped before retrieval)</li>}
                </ol>
                {(r.retrieval.sub_queries ?? []).length > 0 && (
                  <>
                    <p className="mt-2 text-xs font-medium text-muted">Sub-queries</p>
                    <ul className="list-disc ps-5">
                      {r.retrieval.sub_queries!.map((q) => (
                        <li key={q} dir="auto">
                          {q}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </Section>

              <Section title="Filters and flags">
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                  <dt className="text-muted">Agency</dt>
                  <dd>{(r.retrieval.agency_filter ?? []).join(", ") || "—"}</dd>
                  <dt className="text-muted">Province</dt>
                  <dd>{(r.retrieval.jurisdiction_filter ?? []).map((j) => JURISDICTION_LABEL[j]).join(", ") || "—"}</dd>
                  <dt className="text-muted">Advice request</dt>
                  <dd>{r.retrieval.advice_request ? <Badge tone="warn">yes</Badge> : "no"}</dd>
                  <dt className="text-muted">Guardrails</dt>
                  <dd className="flex flex-wrap gap-1">
                    <Badge tone={r.guardrails?.input === "passed" ? "ok" : "warn"}>input {r.guardrails?.input ?? "passed"}</Badge>
                    <Badge tone={r.guardrails?.output === "passed" ? "ok" : "warn"}>output {r.guardrails?.output ?? "passed"}</Badge>
                    {(r.guardrails?.flags ?? []).map((f) => (
                      <Badge key={f}>{f}</Badge>
                    ))}
                  </dd>
                  <dt className="text-muted">Threshold</dt>
                  <dd className="font-mono">{r.retrieval.threshold}</dd>
                </dl>
              </Section>

              <Section title={`Latency${latency.total != null ? ` · total ${latency.total} ms` : ""}`}>
                {steps.length === 0 ? (
                  <p className="text-muted">not reported</p>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {steps.map((k) => (
                      <li key={k} className="flex items-center gap-2">
                        <span className="w-24 shrink-0 text-xs text-muted">{k}</span>
                        <span aria-hidden="true" className="h-2 flex-1 rounded-full bg-surface-2">
                          <span
                            className="block h-2 rounded-full bg-info"
                            style={{ width: `${((latency[k] ?? 0) / maxStep) * 100}%` }}
                          />
                        </span>
                        <span className="w-16 shrink-0 text-end font-mono text-xs tabular-nums">{latency[k]} ms</span>
                      </li>
                    ))}
                  </ul>
                )}
              </Section>

              <Section title="Model and usage">
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                  <dt className="text-muted">Endpoint</dt>
                  <dd>{r.provider.endpoint}</dd>
                  <dt className="text-muted">Model</dt>
                  <dd className="break-all font-mono text-xs">{r.provider.model ?? "—"}</dd>
                  {r.provider.seller && (
                    <>
                      <dt className="text-muted">Seller</dt>
                      <dd>{r.provider.seller}</dd>
                    </>
                  )}
                  <dt className="text-muted">Attempts</dt>
                  <dd>{r.provider.attempts}</dd>
                  <dt className="text-muted">Tokens</dt>
                  <dd className="font-mono text-xs">
                    {r.usage?.input_tokens ?? 0} in · {r.usage?.output_tokens ?? 0} out
                  </dd>
                  <dt className="text-muted">Cost</dt>
                  <dd className="font-mono text-xs">
                    ${(r.usage?.cost_usd ?? 0).toFixed(4)} (list price ${(r.usage?.list_price_equivalent_usd ?? 0).toFixed(4)})
                  </dd>
                  <dt className="text-muted">Degraded</dt>
                  <dd>{r.degraded ? <Badge tone="warn">yes</Badge> : "no"}</dd>
                </dl>
              </Section>

              <Section title="Identifiers">
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                  <dt className="text-muted">Request ID</dt>
                  <dd className="break-all font-mono text-xs">{r.request_id}</dd>
                  <dt className="text-muted">Trace ID</dt>
                  <dd className="break-all font-mono text-xs">{r.trace_id ?? "—"}</dd>
                </dl>
              </Section>
            </div>
          )}
        </div>
      )}
    </dialog>
  );
}
