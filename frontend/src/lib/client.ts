// Browser-side API client. Every call goes to the same-origin BFF proxy (/api/...), which adds the
// JWT from the httpOnly cookie. With NEXT_PUBLIC_USE_MOCKS=1 the typed mocks answer instead.
import { USE_MOCKS } from "./config";
import { mockApi } from "./mocks";
import { ApiError, toProblem } from "./problem";
import type {
  AskRequest,
  AskResponse,
  DocumentMetadata,
  DocumentOut,
  DocumentPage,
  HealthResponse,
  Jurisdiction,
  Scope,
  SignupRequest,
  UserOut,
  Visibility,
} from "./types";

export type UploadOptions = {
  visibility: Visibility;
  title?: string;
  agency?: string;
  jurisdiction?: Jurisdiction;
  topic?: string;
  effective_date?: string;
};

export type ListOptions = { scope: Scope; cursor?: string | null; limit?: number };

export interface Api {
  login(email: string, password: string): Promise<void>;
  logout(): Promise<void>;
  signup(body: SignupRequest): Promise<UserOut>;
  me(): Promise<UserOut>;
  health(): Promise<HealthResponse | { status: "unreachable"; version: string; components: [] }>;
  listDocuments(opts: ListOptions): Promise<DocumentPage>;
  getDocument(id: string): Promise<DocumentOut>;
  uploadDocument(file: File, opts: UploadOptions): Promise<DocumentOut>;
  patchDocument(id: string, body: DocumentMetadata): Promise<DocumentOut>;
  deleteDocument(id: string): Promise<void>;
  ask(body: AskRequest): Promise<AskResponse>;
}

/** Same format the backend accepts (`^[A-Za-z0-9._-]{1,64}$`); works outside secure contexts too. */
export function newRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    try {
      return crypto.randomUUID();
    } catch {
      // not a secure context: fall through
    }
  }
  return `r-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const requestId = newRequestId();
  const headers = new Headers(init.headers);
  headers.set("x-request-id", requestId);
  if (!headers.has("accept")) headers.set("accept", "application/json");
  let res: Response;
  try {
    res = await fetch(`/api/${path}`, { ...init, headers, credentials: "same-origin", cache: "no-store" });
  } catch {
    throw new ApiError(
      toProblem(0, { code: "network_error", title: "Network error", detail: "Could not reach the app server." }),
      requestId,
    );
  }
  const rid = res.headers.get("x-request-id") ?? requestId;
  const body = res.status === 204 ? null : await parseBody(res);
  if (!res.ok) throw new ApiError(toProblem(res.status, body, rid), rid);
  return body as T;
}

function json(method: string, body: unknown): RequestInit {
  return { method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
}

const realApi: Api = {
  async login(email, password) {
    await request("auth/login", json("POST", { email, password }));
  },
  async logout() {
    await request("auth/logout", { method: "POST" });
  },
  signup: (body) => request<UserOut>("auth/signup", json("POST", body)),
  me: () => request<UserOut>("auth/me"),
  async health() {
    try {
      const res = await fetch("/api/health/ready", { cache: "no-store" });
      const body = (await parseBody(res)) as HealthResponse | null;
      if (body && typeof body === "object" && "status" in body && "components" in body) return body;
    } catch {
      // fall through
    }
    return { status: "unreachable", version: "?", components: [] };
  },
  listDocuments({ scope, cursor, limit = 50 }) {
    const q = new URLSearchParams({ scope, limit: String(limit) });
    if (cursor) q.set("cursor", cursor);
    return request<DocumentPage>(`documents?${q}`);
  },
  getDocument: (id) => request<DocumentOut>(`documents/${encodeURIComponent(id)}`),
  uploadDocument(file, opts) {
    const form = new FormData();
    form.set("file", file, file.name);
    form.set("visibility", opts.visibility);
    for (const key of ["title", "agency", "jurisdiction", "topic", "effective_date"] as const) {
      const value = opts[key];
      if (value) form.set(key, value);
    }
    // No content-type header: the browser sets the multipart boundary.
    return request<DocumentOut>("documents", { method: "POST", body: form });
  },
  patchDocument: (id, body) => request<DocumentOut>(`documents/${encodeURIComponent(id)}`, json("PATCH", body)),
  async deleteDocument(id) {
    await request(`documents/${encodeURIComponent(id)}`, { method: "DELETE" });
  },
  ask: (body) => request<AskResponse>("ask", json("POST", body)),
};

export const api: Api = USE_MOCKS ? mockApi : realApi;
