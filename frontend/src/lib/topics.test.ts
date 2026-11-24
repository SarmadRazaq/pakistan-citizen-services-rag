// Run with `npm test` (Node's built-in runner + type stripping; no extra packages).
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { TOPIC_OPTIONS, isKnownTopic, normalizeTopic } from "./topics.ts";

describe("TOPIC_OPTIONS", () => {
  it("has 9 unique values with non-empty labels", () => {
    assert.equal(TOPIC_OPTIONS.length, 9);
    assert.equal(new Set(TOPIC_OPTIONS.map((o) => o.value)).size, 9);
    for (const o of TOPIC_OPTIONS) assert.ok(o.label.trim().length > 0);
  });

  it("labels driving_licence as Driving licence", () => {
    assert.equal(TOPIC_OPTIONS.find((o) => o.value === "driving_licence")?.label, "Driving licence");
  });
});

describe("normalizeTopic", () => {
  it("trims, lowercases and joins words with underscores", () => {
    assert.equal(normalizeTopic("  Social Protection "), "social_protection");
    assert.equal(normalizeTopic("Driving-Licence"), "driving_licence");
    assert.equal(normalizeTopic("PASSPORT"), "passport");
    assert.equal(normalizeTopic("consumer   protection"), "consumer_protection");
  });
});

describe("isKnownTopic", () => {
  it("accepts any spelling that normalises to a listed topic", () => {
    assert.equal(isKnownTopic("Social Protection"), true);
    assert.equal(isKnownTopic("passport"), true);
  });

  it("rejects topics outside the list", () => {
    assert.equal(isKnownTopic("cnic"), false);
    assert.equal(isKnownTopic(""), false);
  });
});
