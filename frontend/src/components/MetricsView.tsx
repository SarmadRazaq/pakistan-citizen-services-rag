"use client";

import { useEffect, useState } from "react";

import { api, type MetricsWindow } from "@/lib/client";
import { fmtMs, fmtNum, fmtPct, fmtUsd } from "@/lib/metrics";
import { ApiError } from "@/lib/problem";
import type { MetricsSummary } from "@/lib/types";

import { Badge, ProblemAlert, Spinner, cx } from "./ui";

const WINDOWS: { id: MetricsWindow; label: string }[] = [
  { id: "1h", label: "Last hour" },
  { id: "24h", label: "Last 24 hours" },
  { id: "7d", label: "Last 7 days" },
];

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <dt className="text-xs font-medium uppercase tracking-wide text-muted">{label}</dt>
      <dd className="mt-1 text-2xl font-semibold tabular-nums">{value}</dd>
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </div>
  );
}

export function MetricsView() {
  const [window, setWindow] = useState<MetricsWindow>("24h");
  const [data, setData] = useState<MetricsSummary | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .metrics(window)
      .then((d) => {
        if (!alive) return;
        setData(d);
        setError(null);
      })
      .catch((err) => {
        if (alive) setError(err instanceof ApiError ? err : null);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [window]);

  const mix = data?.provider_mix ? Object.entries(data.provider_mix).sort((a, b) => b[1] - a[1]) : null;

  return (
    <div className="mx-auto max-w-5xl space-y-5 px-4 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Metrics</h1>
        {data && <Badge tone={data.scope === "global" ? "info" : "neutral"}>{data.scope === "global" ? "All users" : "My requests"}</Badge>}
        <div role="tablist" aria-label="Time window" className="ms-auto flex gap-1">
          {WINDOWS.map((w) => (
            <button
              key={w.id}
              role="tab"
              type="button"
              aria-selected={window === w.id}
              onClick={() => setWindow(w.id)}
              className={cx(
                "rounded-md px-3 py-1.5 text-sm font-medium",
                window === w.id ? "bg-accent text-accent-fg" : "text-fg hover:bg-surface-2",
              )}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>

      {error && <ProblemAlert error={error} />}
      {loading && !data && <Spinner label="Loading metrics" />}

      {data && (
        <div aria-busy={loading} className={cx("space-y-5", loading && "opacity-60")}>
          {data.requests === 0 ? (
            <p className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">No requests in this window yet.</p>
          ) : null}
          <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Tile label="Requests" value={fmtNum(data.requests)} />
            <Tile label="Latency p50" value={fmtMs(data.p50_latency_ms)} hint="Median request" />
            <Tile label="Latency p95" value={fmtMs(data.p95_latency_ms)} hint="Slowest 5% start here" />
            <Tile label="Error rate" value={fmtPct(data.error_rate)} hint="Server errors (5xx)" />
            <Tile label="Fallback rate" value={fmtPct(data.fallback_rate)} hint="Answered by the local model" />
            <Tile label="Avg tokens in" value={fmtNum(data.avg_input_tokens)} hint="Per question, all model calls" />
            <Tile label="Avg tokens out" value={fmtNum(data.avg_output_tokens)} hint="Per question, all model calls" />
            <Tile label="Cost / question" value={fmtUsd(data.cost_per_question_usd)} hint="Actually billed (free tier is $0)" />
            <Tile label="List price / question" value={fmtUsd(data.list_price_per_question_usd)} hint="What it would cost at list prices" />
          </dl>

          {mix && (
            <section aria-labelledby="mix-h" className="rounded-lg border border-border bg-surface p-4">
              <h2 id="mix-h" className="text-sm font-semibold">
                Provider mix
              </h2>
              {mix.length === 0 ? (
                <p className="mt-2 text-sm text-muted">No model calls in this window.</p>
              ) : (
                <table className="mt-2 w-full text-sm">
                  <thead>
                    <tr className="text-start text-xs text-muted">
                      <th className="py-1 text-start font-medium">Model</th>
                      <th className="py-1 text-end font-medium">Requests</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mix.map(([model, n]) => (
                      <tr key={model} className="border-t border-border">
                        <td className="py-1.5 font-mono text-xs">{model}</td>
                        <td className="py-1.5 text-end tabular-nums">{fmtNum(n)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
