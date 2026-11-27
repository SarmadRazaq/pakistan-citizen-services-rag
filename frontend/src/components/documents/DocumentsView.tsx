"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";

import { api } from "@/lib/client";
import { JURISDICTION_LABEL, LANGUAGE_LABEL, SCOPE_OPTIONS, formatDate } from "@/lib/labels";
import { ApiError } from "@/lib/problem";
import { formatBytes } from "@/lib/text";
import type { DocumentOut, Scope } from "@/lib/types";

import { Dialog } from "../Dialog";
import { useSession } from "../Session";
import { useToast } from "../Toast";
import { Button, ProblemAlert, ScopeBadge, Spinner, StatusBadge, cx } from "../ui";
import { EditDialog } from "./EditDialog";
import { UploadPanel } from "./UploadPanel";

function inScope(d: DocumentOut, scope: Scope): boolean {
  return scope === "all" || (scope === "public" ? d.visibility === "public" : d.visibility === "private");
}

export function DocumentsView() {
  const { isAdmin } = useSession();
  const toast = useToast();
  const router = useRouter();
  const params = useSearchParams();
  const linkedId = params.get("doc");

  const [scope, setScope] = useState<Scope>("all");
  const [items, setItems] = useState<DocumentOut[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [editing, setEditing] = useState<DocumentOut | null>(null);
  const [deleting, setDeleting] = useState<DocumentOut | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const tabs = useRef<Array<HTMLButtonElement | null>>([]);
  const rows = useRef(new Map<string, HTMLTableRowElement>());

  const load = useCallback(async (s: Scope) => {
    setLoading(true);
    setError(null);
    try {
      const page = await api.listDocuments({ scope: s });
      setItems(page.items);
      setCursor(page.next_cursor ?? null);
    } catch (err) {
      setItems([]);
      setError(err instanceof ApiError ? err : null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(scope);
  }, [scope, load]);

  // A duplicate toast links here with ?doc=<id>: show All, then highlight and scroll to the row.
  const scrolledFor = useRef<string | null>(null);
  const fetchedFor = useRef<string | null>(null);

  useEffect(() => {
    if (!linkedId) return;
    scrolledFor.current = null;
    fetchedFor.current = null;
    setScope("all");
    setHighlight(linkedId);
  }, [linkedId]);

  useEffect(() => {
    if (!highlight || loading || scrolledFor.current === highlight) return;
    const row = rows.current.get(highlight);
    if (row) {
      scrolledFor.current = highlight;
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      row.focus({ preventScroll: true });
      return;
    }
    if (fetchedFor.current === highlight) return;
    fetchedFor.current = highlight;
    // Not on the first page: fetch it directly and pin it on top.
    api
      .getDocument(highlight)
      .then((d) => setItems((xs) => (xs.some((x) => x.id === d.id) ? xs : [d, ...xs])))
      .catch(() => toast({ tone: "warn", title: "Document not found", detail: "It may have been deleted." }));
  }, [highlight, loading, items, toast]);

  const upsert = useCallback(
    (doc: DocumentOut) => {
      setItems((xs) => {
        const i = xs.findIndex((x) => x.id === doc.id);
        if (i >= 0) return xs.map((x) => (x.id === doc.id ? doc : x));
        return inScope(doc, scope) ? [doc, ...xs] : xs;
      });
    },
    [scope],
  );

  const confirmDelete = async () => {
    const doc = deleting;
    if (!doc) return;
    setDeleting(null);
    const snapshot = items;
    setItems((xs) => xs.filter((x) => x.id !== doc.id)); // optimistic: the row goes immediately
    try {
      await api.deleteDocument(doc.id);
      toast({ tone: "ok", title: "Document deleted", detail: `${doc.file_name} and its chunks were removed from search.` });
    } catch (err) {
      setItems(snapshot);
      const e = err instanceof ApiError ? err : null;
      toast({ tone: "bad", title: e?.title ?? "Delete failed", detail: e?.detail });
    }
  };

  const loadMore = async () => {
    if (!cursor) return;
    setLoadingMore(true);
    try {
      const page = await api.listDocuments({ scope, cursor });
      setItems((xs) => [...xs, ...page.items.filter((d) => !xs.some((x) => x.id === d.id))]);
      setCursor(page.next_cursor ?? null);
    } catch (err) {
      if (err instanceof ApiError) toast({ tone: "bad", title: err.title, detail: err.detail });
    } finally {
      setLoadingMore(false);
    }
  };

  const onTabKey = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const n = SCOPE_OPTIONS.length;
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % n;
    else if (e.key === "ArrowLeft") next = (i - 1 + n) % n;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = n - 1;
    if (next < 0) return;
    e.preventDefault();
    tabs.current[next]?.focus();
    setScope(SCOPE_OPTIONS[next].value);
  };

  const canModify = (d: DocumentOut) => d.visibility === "private" || isAdmin;

  return (
    <div className="mx-auto flex max-w-[1600px] flex-col gap-4 px-4 py-4 xl:flex-row xl:items-start">
      <div className="xl:w-[380px] xl:shrink-0">
        <UploadPanel isAdmin={isAdmin} onDocument={upsert} />
      </div>

      <section aria-labelledby="docs-heading" className="min-w-0 flex-1 rounded-xl border border-border bg-surface p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h1 id="docs-heading" className="text-lg font-semibold">
            Documents
          </h1>
          <Button variant="ghost" onClick={() => void load(scope)} disabled={loading}>
            Refresh
          </Button>
        </div>

        <div role="tablist" aria-label="Document scope" className="mt-2 flex gap-1 border-b border-border">
          {SCOPE_OPTIONS.map((o, i) => {
            const selected = scope === o.value;
            return (
              <button
                key={o.value}
                ref={(el) => {
                  tabs.current[i] = el;
                }}
                role="tab"
                id={`tab-${o.value}`}
                type="button"
                aria-selected={selected}
                aria-controls="docs-panel"
                tabIndex={selected ? 0 : -1}
                onClick={() => {
                  setScope(o.value);
                  if (linkedId) router.replace("/documents");
                }}
                onKeyDown={(e) => onTabKey(e, i)}
                className={cx(
                  "-mb-px border-b-2 px-3 py-1.5 text-sm font-medium",
                  selected ? "border-accent text-accent" : "border-transparent text-muted hover:text-fg",
                )}
              >
                {o.label}
              </button>
            );
          })}
        </div>

        <div id="docs-panel" role="tabpanel" aria-labelledby={`tab-${scope}`} className="mt-3">
          <p className="sr-only" aria-live="polite">
            {loading ? "Loading documents" : `${items.length} documents`}
          </p>
          {error ? (
            <ProblemAlert error={error} />
          ) : loading ? (
            <p className="py-6 text-center text-sm text-muted">
              <Spinner small label="Loading documents…" />
            </p>
          ) : items.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted">
              {scope === "mine" ? "You have not uploaded anything yet." : "No documents in this scope."}
            </p>
          ) : (
            <div className="relative overflow-x-auto">
              <table className="w-full min-w-[960px] border-collapse text-sm">
                <caption className="sr-only">Documents you can read, {SCOPE_OPTIONS.find((o) => o.value === scope)?.label}</caption>
                <thead>
                  <tr className="border-b border-border text-start text-xs uppercase tracking-wide text-muted">
                    {["Source", "Document", "Agency", "Province", "Topic", "Date", "Ver.", "Chunks", "Lang.", "Size", "Status"].map(
                      (h) => (
                        <th key={h} scope="col" className="px-2 py-2 text-start font-semibold">
                          {h}
                        </th>
                      ),
                    )}
                    <th scope="col" className="px-2 py-2 text-end font-semibold">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((d) => (
                    <tr
                      key={d.id}
                      ref={(el) => {
                        if (el) rows.current.set(d.id, el);
                        else rows.current.delete(d.id);
                      }}
                      tabIndex={highlight === d.id ? -1 : undefined}
                      className={cx(
                        "border-b border-border/70 align-top",
                        highlight === d.id && "bg-accent/10 outline-2 outline-accent",
                      )}
                    >
                      <td className="px-2 py-2">
                        <ScopeBadge scope={d.visibility} />
                      </td>
                      <td className="min-w-[15rem] max-w-[22rem] px-2 py-2">
                        <p className="font-medium [overflow-wrap:anywhere]" dir="auto">
                          {d.file_name}
                        </p>
                        {d.title && (
                          <p className="text-xs text-muted" dir="auto" lang={d.language === "ur" ? "ur" : undefined}>
                            {d.title}
                          </p>
                        )}
                        {d.error && <p className="mt-0.5 text-xs text-bad">{d.error}</p>}
                        {d.source_url && (
                          <a
                            href={d.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-accent underline underline-offset-2"
                          >
                            source<span className="sr-only"> (opens in a new tab)</span> ↗
                          </a>
                        )}
                      </td>
                      <td className="px-2 py-2">{d.agency ?? "—"}</td>
                      <td className="px-2 py-2">{d.jurisdiction ? JURISDICTION_LABEL[d.jurisdiction] : "—"}</td>
                      <td className="px-2 py-2">{d.topic ?? "—"}</td>
                      <td className="whitespace-nowrap px-2 py-2">{formatDate(d.effective_date)}</td>
                      <td className="px-2 py-2 tabular-nums">v{d.version}</td>
                      <td className="px-2 py-2 tabular-nums">{d.chunk_count ?? "—"}</td>
                      <td className="px-2 py-2">
                        {d.language ? (
                          <span lang={d.language === "ur" ? "ur" : undefined} className="ui-ur">
                            {LANGUAGE_LABEL[d.language]}
                          </span>
                        ) : (
                          "—"
                        )}
                        {d.is_ocr && <span className="ms-1 text-xs text-muted">(OCR)</span>}
                      </td>
                      <td className="whitespace-nowrap px-2 py-2 tabular-nums">{formatBytes(d.size_bytes)}</td>
                      <td className="px-2 py-2">
                        <StatusBadge status={d.status} />
                        {d.progress && d.status !== "READY" && (
                          <p className="mt-0.5 text-xs text-muted">
                            {d.progress.pages_done}/{d.progress.pages_total} pages
                          </p>
                        )}
                      </td>
                      <td className="whitespace-nowrap px-2 py-2 text-end">
                        {canModify(d) && (
                          <span className="inline-flex gap-1">
                            <Button size="sm" onClick={() => setEditing(d)} aria-label={`Edit ${d.file_name}`}>
                              Edit
                            </Button>
                            <Button
                              variant="dangerOutline"
                              size="sm"
                              onClick={() => setDeleting(d)}
                              aria-label={`Delete ${d.file_name}`}
                              disabled={d.status === "DELETING"}
                            >
                              Delete
                            </Button>
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {cursor && !loading && (
            <div className="mt-3 text-center">
              <Button onClick={() => void loadMore()} disabled={loadingMore}>
                {loadingMore ? <Spinner small label="Loading…" /> : "Load more"}
              </Button>
            </div>
          )}
        </div>
      </section>

      <EditDialog doc={editing} onClose={() => setEditing(null)} onSaved={upsert} />

      <Dialog
        open={!!deleting}
        onClose={() => setDeleting(null)}
        title="Delete this document?"
        footer={
          <>
            <Button onClick={() => setDeleting(null)} autoFocus>
              Cancel
            </Button>
            <Button variant="danger" onClick={() => void confirmDelete()}>
              Delete
            </Button>
          </>
        }
      >
        {deleting && (
          <div className="text-sm">
            <p className="break-all font-medium" dir="auto">
              {deleting.file_name}
            </p>
            <p className="mt-2 text-fg">
              The document and {deleting.chunk_count != null ? `all ${deleting.chunk_count} of its chunks` : "all of its chunks"}{" "}
              are removed from search. Answers will no longer cite it. This cannot be undone.
            </p>
            {deleting.visibility === "public" && (
              <p className="mt-2 font-medium text-bad">This is an official source: it disappears for every user.</p>
            )}
          </div>
        )}
      </Dialog>
    </div>
  );
}
