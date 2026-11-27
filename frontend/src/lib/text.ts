// Pure text helpers for answers and chunks. No imports, so `npm test` runs them under plain Node.

export type AnswerSegment = { kind: "text"; text: string } | { kind: "cite"; marker: string };

const MARKER_GROUP = /\[((?:S\d+)(?:\s*[,;]\s*S\d+)*)\]/g;

/**
 * Split an answer into text and citation markers. Handles `[S1]`, `[S1][S2]` and `[S1, S2]`.
 * Anything else in brackets stays text.
 */
export function splitMarkers(answer: string): AnswerSegment[] {
  const out: AnswerSegment[] = [];
  let last = 0;
  for (const m of answer.matchAll(MARKER_GROUP)) {
    const start = m.index ?? 0;
    if (start > last) out.push({ kind: "text", text: answer.slice(last, start) });
    for (const marker of m[1].split(/\s*[,;]\s*/)) out.push({ kind: "cite", marker });
    last = start + m[0].length;
  }
  if (last < answer.length) out.push({ kind: "text", text: answer.slice(last) });
  return out;
}

/** Markers in order of first appearance, without duplicates. */
export function markersIn(answer: string): string[] {
  const seen: string[] = [];
  for (const seg of splitMarkers(answer)) {
    if (seg.kind === "cite" && !seen.includes(seg.marker)) seen.push(seg.marker);
  }
  return seen;
}

export type HighlightPart = { text: string; hit: boolean };

/**
 * Mark every occurrence of any quote inside `text`. Matching ignores case and collapses runs of
 * whitespace (chunk text keeps line breaks from tables; quotes usually do not).
 */
export function highlightQuotes(text: string, quotes: string[]): HighlightPart[] {
  const norm: string[] = [];
  const map: number[] = [];
  let prevSpace = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (/\s/.test(ch)) {
      if (prevSpace) continue;
      norm.push(" ");
      prevSpace = true;
    } else {
      norm.push(ch.toLowerCase());
      prevSpace = false;
    }
    map.push(i);
  }
  const haystack = norm.join("");

  const ranges: Array<[number, number]> = [];
  for (const raw of quotes) {
    const needle = raw.replace(/\s+/g, " ").trim().toLowerCase();
    if (needle.length < 3) continue;
    let from = 0;
    for (;;) {
      const at = haystack.indexOf(needle, from);
      if (at < 0) break;
      const endNorm = at + needle.length - 1;
      ranges.push([map[at], map[endNorm] + 1]);
      from = at + needle.length;
    }
  }
  if (ranges.length === 0) return [{ text, hit: false }];

  ranges.sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [];
  for (const r of ranges) {
    const top = merged[merged.length - 1];
    if (top && r[0] <= top[1]) top[1] = Math.max(top[1], r[1]);
    else merged.push([r[0], r[1]]);
  }

  const parts: HighlightPart[] = [];
  let pos = 0;
  for (const [s, e] of merged) {
    if (s > pos) parts.push({ text: text.slice(pos, s), hit: false });
    parts.push({ text: text.slice(s, e), hit: true });
    pos = e;
  }
  if (pos < text.length) parts.push({ text: text.slice(pos), hit: false });
  return parts;
}

/** True when most letters are Arabic script (Urdu in Nastaliq), so the text gets the Urdu font. */
export function isArabicScript(text: string): boolean {
  let arabic = 0;
  let latin = 0;
  for (const ch of text) {
    if (/[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/.test(ch)) arabic++;
    else if (/[A-Za-z]/.test(ch)) latin++;
  }
  return arabic > 0 && arabic >= latin;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10 * 1024 ? 1 : 0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
