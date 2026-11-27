import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiError, isDuplicate, isGuardrailBlock, toProblem } from "./problem.ts";

describe("toProblem", () => {
  it("passes a real problem+json through", () => {
    const p = toProblem(409, {
      type: "https://rag.example/errors/duplicate_document",
      title: "Duplicate document",
      status: 409,
      code: "duplicate_document",
      detail: "Already uploaded.",
      request_id: "r1",
      extra: { existing_document_id: "d_1" },
      errors: [],
    });
    assert.equal(p.code, "duplicate_document");
    assert.deepEqual(p.extra, { existing_document_id: "d_1" });
    assert.equal(p.request_id, "r1");
  });

  it("wraps FastAPI's default {detail} body", () => {
    const p = toProblem(404, { detail: "Not Found" }, "rid");
    assert.equal(p.status, 404);
    assert.equal(p.detail, "Not Found");
    assert.equal(p.request_id, "rid");
  });

  it("does not leak an HTML gateway page into the detail", () => {
    const p = toProblem(502, "<html><body>Bad gateway</body></html>");
    assert.equal(p.code, "backend_unreachable");
    assert.equal(p.detail, "Backend unreachable");
  });
});

describe("ApiError", () => {
  it("exposes field errors by their last loc segment", () => {
    const e = new ApiError(
      toProblem(422, {
        code: "validation_failed",
        title: "Request validation failed",
        detail: "One or more fields are invalid.",
        errors: [{ loc: ["body", "password"], msg: "too short", type: "string_too_short" }],
      }),
    );
    assert.deepEqual(e.fieldErrors(), { password: "too short" });
  });

  it("classifies duplicates and guardrail blocks", () => {
    const dup = new ApiError(toProblem(409, { code: "duplicate_public_document", title: "Dup" }));
    const block = new ApiError(toProblem(400, { code: "prompt_injection_detected", title: "Blocked" }));
    assert.equal(isDuplicate(dup), true);
    assert.equal(isGuardrailBlock(dup), false);
    assert.equal(isGuardrailBlock(block), true);
    assert.equal(isDuplicate(new Error("x")), false);
  });
});
