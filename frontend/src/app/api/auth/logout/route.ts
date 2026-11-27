// Session logout: drop the httpOnly cookie. JWTs are stateless, so nothing is called upstream.
import { cookies } from "next/headers";

import { SESSION_COOKIE, isCrossOrigin, problemResponse, requestIdFrom } from "@/lib/server/backend";

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
  const jar = await cookies();
  jar.delete(SESSION_COOKIE);
  return new Response(null, { status: 204, headers: { "x-request-id": requestId } });
}
