"use client";

import { useEffect, useId, useState } from "react";

import { api } from "@/lib/client";
import { JURISDICTION_LABEL } from "@/lib/labels";
import { ApiError } from "@/lib/problem";
import type { DocumentMetadata, DocumentOut, Jurisdiction } from "@/lib/types";

import { Dialog } from "../Dialog";
import { Button, ProblemAlert, Spinner } from "../ui";

type Form = { title: string; agency: string; jurisdiction: Jurisdiction | ""; topic: string; effective_date: string };

function toForm(d: DocumentOut | null): Form {
  return {
    title: d?.title ?? "",
    agency: d?.agency ?? "",
    jurisdiction: d?.jurisdiction ?? "",
    topic: d?.topic ?? "",
    effective_date: d?.effective_date ?? "",
  };
}

export function EditDialog({
  doc,
  onClose,
  onSaved,
}: {
  doc: DocumentOut | null;
  onClose: () => void;
  onSaved: (doc: DocumentOut) => void;
}) {
  const [form, setForm] = useState<Form>(toForm(doc));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const id = useId();

  useEffect(() => {
    setForm(toForm(doc));
    setError(null);
  }, [doc]);

  const fieldErrors = error?.fieldErrors() ?? {};

  const save = async () => {
    if (!doc) return;
    setSaving(true);
    setError(null);
    const body: DocumentMetadata = {
      title: form.title.trim() || null,
      agency: form.agency.trim() || null,
      jurisdiction: form.jurisdiction || null,
      topic: form.topic.trim() || null,
      effective_date: form.effective_date || null,
    };
    try {
      onSaved(await api.patchDocument(doc.id, body));
      onClose();
    } catch (err) {
      if (err instanceof ApiError) setError(err);
    } finally {
      setSaving(false);
    }
  };

  const input = "rounded-md border border-border bg-bg px-2 py-1 text-fg";
  const fields: Array<{ key: Exclude<keyof Form, "jurisdiction">; label: string; type?: string; max?: number }> = [
    { key: "title", label: "Title", max: 300 },
    { key: "agency", label: "Agency", max: 80 },
    { key: "topic", label: "Topic", max: 60 },
    { key: "effective_date", label: "Effective date", type: "date" },
  ];

  return (
    <Dialog
      open={!!doc}
      onClose={onClose}
      title="Edit document metadata"
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={() => void save()} disabled={saving}>
            {saving ? <Spinner small label="Saving…" /> : "Save"}
          </Button>
        </>
      }
    >
      {doc && (
        <form
          className="flex flex-col gap-3 text-sm"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <p className="break-all text-muted">{doc.file_name}</p>
          {error && <ProblemAlert error={error} />}
          {fields.map((f) => (
            <div key={f.key} className="flex flex-col gap-1">
              <label htmlFor={`${id}-${f.key}`} className="text-xs font-medium text-muted">
                {f.label}
              </label>
              <input
                id={`${id}-${f.key}`}
                type={f.type ?? "text"}
                dir="auto"
                maxLength={f.max}
                value={form[f.key]}
                onChange={(e) => setForm({ ...form, [f.key]: e.target.value })}
                aria-invalid={!!fieldErrors[f.key] || undefined}
                aria-describedby={fieldErrors[f.key] ? `${id}-${f.key}-err` : undefined}
                className={input}
              />
              {fieldErrors[f.key] && (
                <p id={`${id}-${f.key}-err`} className="text-xs text-bad">
                  {fieldErrors[f.key]}
                </p>
              )}
            </div>
          ))}
          <div className="flex flex-col gap-1">
            <label htmlFor={`${id}-jur`} className="text-xs font-medium text-muted">
              Province
            </label>
            <select
              id={`${id}-jur`}
              value={form.jurisdiction}
              onChange={(e) => setForm({ ...form, jurisdiction: e.target.value as Jurisdiction | "" })}
              className={input}
            >
              <option value="">Not set</option>
              {(Object.keys(JURISDICTION_LABEL) as Jurisdiction[]).map((j) => (
                <option key={j} value={j}>
                  {JURISDICTION_LABEL[j]}
                </option>
              ))}
            </select>
          </div>
          <button type="submit" className="hidden" aria-hidden="true" tabIndex={-1} />
        </form>
      )}
    </Dialog>
  );
}
