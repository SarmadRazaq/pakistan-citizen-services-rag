// Session login: the browser posts JSON {email, password}; this handler calls the backend's
// OAuth2 password form and stores the access token in an httpOnly cookie. The token is never
// returned to the browser.
import { cookies } from "next/headers";

import {
  API_BASE_URL,
  API_PREFIX,
  SESSION_COOKIE,
  isCrossOrigin,
  problemResponse,
  readBody,
  requestIdFrom,
  sessionCookieOptions,
} from "@/lib/server/backend";
import type { TokenResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const requestId = requestIdFrom(req);
  if (isCrossOrigin(req)) {
    return problemResponse(
      403,
      { code: "cross_origin_rejected", title: "Cross-origin request rejected", detail: "Origin does not match." },
      requestId,
    );
  }

  let email = "";
  let password = "";
  try {
    const body = (await req.json()) as { email?: unknown; password?: unknown };
    email = typeof body.email === "string" ? body.email.trim() : "";
    password = typeof body.password === "string" ? body.password : "";
  } catch {
    // fall through to the validation error below
  }
  if (!email || !password) {
    return problemResponse(
      422,
      {
        code: "validation_failed",
        title: "Request validation failed",
        detail: "Email and password are required.",
        errors: [
          ...(!email ? [{ loc: ["body", "email"], msg: "Enter your email.", type: "missing" }] : []),
          ...(!password ? [{ loc: ["body", "password"], msg: "Enter your password.", type: "missing" }] : []),
        ],
      },
      requestId,
    );
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${API_BASE_URL}${API_PREFIX}/auth/login`, {
      method: "POST",
      headers: {
        "content-type": "application/x-www-form-urlencoded",
        accept: "application/json",
        "x-request-id": requestId,
      },
      body: new URLSearchParams({ grant_type: "password", username: email, password }),
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    return problemResponse(
      502,
      {
        code: "backend_unreachable",
        title: "Backend unreachable",
        detail: `Could not reach the API at ${API_BASE_URL}. Is the backend running?`,
      },
      requestId,
    );
  }

  const upstreamRequestId = upstream.headers.get("x-request-id") ?? requestId;
  const body = await readBody(upstream);
  if (!upstream.ok) return problemResponse(upstream.status, body, upstreamRequestId);

  const token = body as Partial<TokenResponse>;
  if (typeof token.access_token !== "string") {
    return problemResponse(
      502,
      { code: "bad_upstream", title: "Unexpected login response", detail: "The API returned no access token." },
      upstreamRequestId,
    );
  }
  const expiresIn = typeof token.expires_in === "number" ? token.expires_in : 30 * 60;
  const jar = await cookies();
  jar.set(SESSION_COOKIE, token.access_token, sessionCookieOptions(expiresIn));
  return Response.json({ ok: true, expires_in: expiresIn }, { headers: { "x-request-id": upstreamRequestId } });
}
