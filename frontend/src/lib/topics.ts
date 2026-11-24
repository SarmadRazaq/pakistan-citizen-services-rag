/** The closed Topic list (GLOSSARY.md "Topic"); keep in sync with `TOPICS` in app/ingestion/metadata.py. */
export const TOPIC_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "identity", label: "Identity documents" },
  { value: "passport", label: "Passport" },
  { value: "tax", label: "Tax" },
  { value: "health", label: "Health" },
  { value: "social_protection", label: "Social protection" },
  { value: "consumer_protection", label: "Consumer protection" },
  { value: "driving_licence", label: "Driving licence" },
  { value: "complaints", label: "Complaints" },
  { value: "rights", label: "Rights" },
];

/** Same rule as the backend: trim, lowercase, join runs of whitespace/hyphens with one underscore. */
export function normalizeTopic(s: string): string {
  return s.trim().toLowerCase().replace(/[\s-]+/g, "_");
}

export function isKnownTopic(s: string): boolean {
  const slug = normalizeTopic(s);
  return TOPIC_OPTIONS.some((o) => o.value === slug);
}
