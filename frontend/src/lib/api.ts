// Thin API client. Replace these hand-written types with the generated ones
// (`npm run gen:api` → src/lib/api-types.ts) as soon as work package A4 starts.

const API_BASE = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";

export type ComponentHealth = {
  name: string;
  ok: boolean;
  required: boolean;
  detail: string | null;
  latency_ms: number | null;
};

export type HealthResponse = {
  status: "ok" | "degraded" | "unavailable" | "unreachable";
  version: string;
  components: ComponentHealth[];
};

export async function getReadiness(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/health/ready`, { cache: "no-store" });
    return (await res.json()) as HealthResponse;
  } catch {
    return { status: "unreachable", version: "?", components: [] };
  }
}
