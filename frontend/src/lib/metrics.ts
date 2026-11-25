// Display helpers for the metrics dashboard.

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

export function fmtPct(rate: number | null | undefined): string {
  if (rate == null) return "–";
  return `${(rate * 100).toFixed(rate > 0 && rate < 0.001 ? 2 : 1)}%`;
}

export function fmtNum(n: number | null | undefined): string {
  return n == null ? "–" : Math.round(n).toLocaleString("en-US");
}

/** Per-question dollar amounts are tiny: keep enough digits to show them, and a clean $0 for free. */
export function fmtUsd(usd: number | null | undefined): string {
  if (usd == null) return "–";
  if (usd === 0) return "$0.00";
  return usd < 0.01 ? `$${usd.toFixed(5)}` : `$${usd.toFixed(3)}`;
}
