import { getReadiness } from "@/lib/api";

// Scaffold page: shows backend readiness. Work package A4 builds the documents panel,
// chat, citations and chunks panel (blueprint §13) on top of the generated API types.
export const dynamic = "force-dynamic";

export default async function Home() {
  const ready = await getReadiness();

  return (
    <main className="mx-auto max-w-2xl px-4 py-12">
      <h1 className="text-2xl font-semibold">Citizen Services Helpdesk</h1>
      <p className="mt-2 text-[color:var(--muted)]">
        Official answers with citations, in English, Urdu and Roman Urdu.
      </p>
      <p dir="auto" className="urdu mt-1 text-[color:var(--muted)]">
        سرکاری دستاویزات سے حوالہ جات کے ساتھ جوابات
      </p>

      <section className="mt-8 rounded-lg border border-[color:var(--border)] p-4">
        <h2 className="font-medium">
          Backend:{" "}
          <span style={{ color: ready.status === "ok" ? "var(--ok)" : "var(--bad)" }}>{ready.status}</span>
        </h2>
        <ul className="mt-3 space-y-1 text-sm">
          {ready.components.map((c) => (
            <li key={c.name}>
              <span style={{ color: c.ok ? "var(--ok)" : "var(--bad)" }}>{c.ok ? "●" : "○"}</span> {c.name}
              {!c.required && <span className="text-[color:var(--muted)]"> (optional)</span>}
              {c.detail && <span className="text-[color:var(--muted)]"> · {c.detail}</span>}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
