// RFC 9457 problem+json → one typed error the UI can render (title + detail + field errors).
// Self-contained (type-only imports) so `npm test` can run it under plain Node.
import type { FieldError, Problem } from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly title: string;
  readonly detail: string;
  readonly requestId: string | null;
  readonly extra: Record<string, unknown>;
  readonly errors: FieldError[];

  constructor(problem: Problem, requestId?: string | null) {
    super(problem.detail || problem.title);
    this.name = "ApiError";
    this.status = problem.status;
    this.code = problem.code;
    this.title = problem.title;
    this.detail = problem.detail;
    this.requestId = problem.request_id ?? requestId ?? null;
    this.extra = problem.extra ?? {};
    this.errors = problem.errors ?? [];
  }

  /** Field errors keyed by the last `loc` segment (`["body", "password"]` → `password`). */
  fieldErrors(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const e of this.errors) {
      const key = e.loc[e.loc.length - 1] ?? "_";
      out[key] ??= e.msg;
    }
    return out;
  }
}

const GUARDRAIL_CODES = new Set([
  "prompt_injection_detected",
  "jailbreak_detected",
  "obfuscated_input",
]);

/** A 400 from an input guardrail: rendered as a "blocked" bubble, not a generic error. */
export function isGuardrailBlock(err: unknown): err is ApiError {
  return err instanceof ApiError && GUARDRAIL_CODES.has(err.code);
}

export function isDuplicate(err: unknown): boolean {
  return (
    err instanceof ApiError &&
    (err.code === "duplicate_document" ||
      err.code === "duplicate_public_document" ||
      err.code === "duplicate_content")
  );
}

function statusTitle(status: number): string {
  if (status === 401) return "Authentication required";
  if (status === 403) return "Not allowed";
  if (status === 404) return "Not found";
  if (status === 413) return "Too large";
  if (status === 429) return "Too many requests";
  if (status === 501) return "Not implemented yet";
  if (status === 502 || status === 504) return "Backend unreachable";
  if (status === 503) return "Service unavailable";
  return status >= 500 ? "Server error" : "Request failed";
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Coerce any error body into a Problem: a real problem+json passes through; FastAPI's default
 * `{detail: ...}`, plain text or an HTML gateway page become a generic problem with the status.
 */
export function toProblem(status: number, body: unknown, requestId?: string | null): Problem {
  if (isRecord(body) && typeof body.code === "string" && typeof body.title === "string") {
    return {
      type: typeof body.type === "string" ? body.type : "about:blank",
      title: body.title,
      status: typeof body.status === "number" ? body.status : status,
      code: body.code,
      detail: typeof body.detail === "string" ? body.detail : body.title,
      request_id: typeof body.request_id === "string" ? body.request_id : (requestId ?? null),
      extra: isRecord(body.extra) ? body.extra : {},
      errors: Array.isArray(body.errors) ? (body.errors as FieldError[]) : [],
    };
  }
  const title = statusTitle(status);
  let detail = title;
  if (isRecord(body) && typeof body.detail === "string") detail = body.detail;
  else if (typeof body === "string" && body.trim() && !body.trimStart().startsWith("<")) {
    detail = body.trim().slice(0, 300);
  }
  return {
    type: "about:blank",
    title,
    status,
    code: status === 502 || status === 504 ? "backend_unreachable" : `http_${status}`,
    detail,
    request_id: requestId ?? null,
    extra: {},
    errors: [],
  };
}
