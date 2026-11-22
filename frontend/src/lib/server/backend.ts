// Server-only helpers shared by the route handlers (the BFF proxy and the session routes).
// Nothing here is imported by client components: the JWT never reaches browser JS.
import { toProblem } from "@/lib/problem";

/** FastAPI origin. Server-side env only (not NEXT_PUBLIC_), so it is never shipped to the browser. */
export const API_BASE_URL = (process.env.API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
/** Must match `Settings.api_prefix` in app/config.py. */
export const API_PREFIX = "/api/v1";

export const SESSION_COOKIE = "rag_session";

/** httpOnly + SameSite=Lax always; Secure in production (override with COOKIE_SECURE=0|1). */
export function sessionCookieOptions(maxAgeSeconds: number) {
  const secureEnv = process.env.COOKIE_SECURE;
  const secure = secureEnv ? secureEnv === "1" : process.env.NODE_ENV === "production";
  return {
    httpOnly: true,
    sameSite: "lax" as const,
    secure,
    path: "/",
    maxAge: Math.max(0, Math.floor(maxAgeSeconds)),
  };
}

const VALID_REQUEST_ID = /^[A-Za-z0-9._-]{1,64}$/;

/** Echo a well-formed client X-Request-ID (same rule as the backend middleware), else mint one. */
export function requestIdFrom(req: Request): string {
  const incoming = req.headers.get("x-request-id");
  return incoming && VALID_REQUEST_ID.test(incoming) ? incoming : crypto.randomUUID();
}

export function problemResponse(status: number, body: unknown, requestId: string | null): Response {
  return Response.json(toProblem(status, body, requestId), {
    status,
    headers: {
      "content-type": "application/problem+json",
      ...(requestId ? { "x-request-id": requestId } : {}),
    },
  });
}

/**
 * CSRF defence in depth on top of SameSite=Lax: a state-changing request whose Origin header
 * names another host is refused. Same-origin fetches from the app always pass.
 */
export function isCrossOrigin(req: Request): boolean {
  const origin = req.headers.get("origin");
  if (!origin) return false;
  try {
    return new URL(origin).host !== req.headers.get("host");
  } catch {
    return true;
  }
}

/** Read an upstream body as JSON when it is JSON, else as text (for problem normalisation). */
export async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  const type = res.headers.get("content-type") ?? "";
  if (type.includes("json")) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return text;
}
