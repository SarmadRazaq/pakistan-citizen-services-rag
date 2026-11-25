import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { fmtMs, fmtNum, fmtPct, fmtUsd } from "./metrics.ts";

describe("metrics formatting", () => {
  it("formats latency", () => {
    assert.equal(fmtMs(null), "–");
    assert.equal(fmtMs(850.4), "850 ms");
    assert.equal(fmtMs(2400), "2.4 s");
  });
  it("formats rates and counts", () => {
    assert.equal(fmtPct(0), "0.0%");
    assert.equal(fmtPct(0.014), "1.4%");
    assert.equal(fmtNum(2150.4), "2,150");
  });
  it("formats tiny dollar amounts", () => {
    assert.equal(fmtUsd(0), "$0.00");
    assert.equal(fmtUsd(0.0021), "$0.00210");
    assert.equal(fmtUsd(0.25), "$0.250");
  });
});
