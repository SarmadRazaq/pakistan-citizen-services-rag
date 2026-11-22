// BFF proxy: /api/<path> → ${API_BASE_URL}/api/v1/<path> (and /api/health/* → /health/*).
//
// - The JWT lives only in the httpOnly `rag_session` cookie and is turned into an
//   `Authorization: Bearer` header here, server-side. Browser JS never sees it.
// - X-Request-ID is passed through (or minted) and returned on every response.
// - Any upstream error body is normalised to application/problem+json, so the UI has one error type.
// - Only a fixed set of top-level resources is forwarded, and path segments are re-encoded,
//   so a caller cannot point the request (or the token) anywhere else.
import { cookies } from "next/headers";

import {
  API_BASE_URL,
  API_PREFIX,
  SESSION_COOKIE,
  isCrossOrigin,
  problemResponse,
  readBody,
  requestIdFrom,
} from "@/lib/server/backend";

export const dynamic = "force-dynamic";

const ALLOWED_ROOTS = new Set(["auth", "documents", "ask", "search", "feedback", "metrics", "health"]);
const FORWARD_REQUEST_HEADERS = ["accept", "accept-language", "content-type", "if-match", "if-none-match"];
const FORWARD_RESPONSE_HEADERS = ["content-type", "location", "retry-after", "etag", "x-request-id"];
const UPSTREAM_TIMEOUT_MS = Number(process.env.API_TIMEOUT_MS ?? 120_000);

type Ctx = { params: Promise<{ path: string[] }> };

function upstreamUrl(segments: string[], search: string): string | null {
  if (segments.length === 0 || !ALLOWED_ROOTS.has(segments[0])) return null;
  if (segments.some((s) => s === "" || s === "." || s === ".." || s.includes("/") || s.includes("\\"))) {
    return null;
  }
  const path = segments.map(encodeURIComponent).join("/");
  // Health routes are mounted at the root, everything else under the API prefix.
  const base = segments[0] === "health" ? "" : API_PREFIX;
  return `${API_BASE_URL}${base}/${path}${search}`;
}

async function forward(req: Request, ctx: Ctx): Promise<Response> {
  const requestId = requestIdFrom(req);
  const { path } = await ctx.params;
  const incoming = new URL(req.url);
  const target = upstreamUrl(path, incoming.search);
  if (!target) {
    return problemResponse(404, { code: "not_found", title: "Not found", detail: "Unknown API route." }, requestId);
  }
  if (req.method !== "GET" && req.method !== "HEAD" && isCrossOrigin(req)) {
    return problemResponse(
      403,
      { code: "cross_origin_rejected", title: "Cross-origin request rejected", detail: "Origin does not match." },
      requestId,
    );
  }

  const headers = new Headers({ "x-request-id": requestId });
  for (const name of FORWARD_REQUEST_HEADERS) {
    const value = req.headers.get(name);
    if (value) headers.set(name, value);
  }
  const jar = await cookies();
  const token = jar.get(SESSION_COOKIE)?.value;
  if (token) headers.set("authorization", `Bearer ${token}`);

  const hasBody = !["GET", "HEAD", "DELETE"].includes(req.method);
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      // Buffered: uploads are capped at 50 MB by the backend, and buffering keeps
      // Content-Length intact for the multipart parser.
      body: hasBody ? await req.arrayBuffer() : undefined,
      redirect: "manual",
      cache: "no-store",
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = err instanceof DOMException && err.name === "TimeoutError";
    return problemResponse(
      timedOut ? 504 : 502,
      {
        code: "backend_unreachable",
        title: timedOut ? "Backend timed out" : "Backend unreachable",
        detail: timedOut
          ? "The API did not answer in time. Try again in a moment."
          : `Could not reach the API at ${API_BASE_URL}. Is the backend running?`,
      },
      requestId,
    );
  }

  const upstreamRequestId = upstream.headers.get("x-request-id") ?? requestId;

  // An expired or revoked token: drop the cookie so the UI goes back to /login cleanly.
  if (upstream.status === 401 && token) jar.delete(SESSION_COOKIE);

  if (!upstream.ok && upstream.status !== 304) {
    const type = upstream.headers.get("content-type") ?? "";
    // /health/ready answers 503 with a HealthResponse body: keep it as-is for the status widget.
    if (!(path[0] === "health" && type.includes("application/json"))) {
      return problemResponse(upstream.status, await readBody(upstream), upstreamRequestId);
    }
  }

  const out = new Headers();
  for (const name of FORWARD_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  out.set("x-request-id", upstreamRequestId);
  out.set("cache-control", "no-store");
  const body = upstream.status === 204 || upstream.status === 304 ? null : upstream.body;
  return new Response(body, { status: upstream.status, headers: out });
}

export const GET = forward;
export const POST = forward;
export const PUT = forward;
export const PATCH = forward;
export const DELETE = forward;
