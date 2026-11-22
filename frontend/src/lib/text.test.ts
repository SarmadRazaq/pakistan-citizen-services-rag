// Run with `npm test` (Node's built-in runner + type stripping; no extra packages).
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { highlightQuotes, isArabicScript, markersIn, splitMarkers } from "./text.ts";

describe("splitMarkers", () => {
  it("splits single, adjacent and grouped markers", () => {
    const segs = splitMarkers("Fee is Rs 9,000 [S1]. Takes 7 days [S2][S3] or [S4, S5].");
    assert.deepEqual(
      segs.filter((s) => s.kind === "cite").map((s) => (s.kind === "cite" ? s.marker : "")),
      ["S1", "S2", "S3", "S4", "S5"],
    );
    assert.equal(segs[0].kind === "text" && segs[0].text, "Fee is Rs 9,000 ");
  });

  it("leaves other brackets as text", () => {
    assert.deepEqual(splitMarkers("see [note] and [s1]"), [{ kind: "text", text: "see [note] and [s1]" }]);
  });

  it("handles Urdu text with markers", () => {
    const segs = splitMarkers("فیس 750 روپے ہے [S1]۔");
    assert.equal(segs.length, 3);
    assert.deepEqual(segs[1], { kind: "cite", marker: "S1" });
  });

  it("lists unique markers in order", () => {
    assert.deepEqual(markersIn("a [S2] b [S1] c [S2]"), ["S2", "S1"]);
  });
});

describe("highlightQuotes", () => {
  it("matches across collapsed whitespace and case", () => {
    const text = "Adult | 36 pages |\n Rs 4,500 | Rs 9,000\nFees are non-refundable.";
    const parts = highlightQuotes(text, ["adult | 36 pages | rs 4,500 | rs 9,000"]);
    const hit = parts.filter((p) => p.hit);
    assert.equal(hit.length, 1);
    assert.equal(hit[0].text, "Adult | 36 pages |\n Rs 4,500 | Rs 9,000");
    assert.equal(parts.map((p) => p.text).join(""), text);
  });

  it("merges overlapping quotes and ignores misses", () => {
    const parts = highlightQuotes("one two three four", ["two three", "three four", "absent"]);
    assert.deepEqual(parts, [
      { text: "one ", hit: false },
      { text: "two three four", hit: true },
    ]);
  });

  it("returns the whole text when nothing matches", () => {
    assert.deepEqual(highlightQuotes("abc", ["xyz"]), [{ text: "abc", hit: false }]);
  });
});

describe("isArabicScript", () => {
  it("detects Urdu vs Latin script", () => {
    assert.equal(isArabicScript("شناختی کارڈ کی فیس کتنی ہے؟"), true);
    assert.equal(isArabicScript("Urgent passport banwane ki fee kitni hai?"), false);
    assert.equal(isArabicScript("12345"), false);
  });
});
