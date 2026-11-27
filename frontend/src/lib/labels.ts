import type {
  AnswerLanguage,
  DocumentStatus,
  Jurisdiction,
  LanguageCode,
  NoticeCode,
  RefusalReason,
  Scope,
} from "./types";

export const JURISDICTION_LABEL: Record<Jurisdiction, string> = {
  federal: "Federal",
  punjab: "Punjab",
  sindh: "Sindh",
  kp: "Khyber Pakhtunkhwa",
};

export const LANGUAGE_LABEL: Record<LanguageCode, string> = {
  en: "EN",
  ur: "اردو",
  roman_ur: "Roman Urdu",
};

export const ANSWER_LANGUAGE_OPTIONS: Array<{ value: AnswerLanguage; label: string; lang?: string }> = [
  { value: "auto", label: "Auto" },
  { value: "en", label: "EN" },
  { value: "ur", label: "اردو", lang: "ur" },
  { value: "roman_ur", label: "Roman Urdu" },
];

export const SCOPE_OPTIONS: Array<{ value: Scope; label: string }> = [
  { value: "all", label: "All" },
  { value: "public", label: "Official sources" },
  { value: "mine", label: "My uploads" },
];

export const REFUSAL_LABEL: Record<RefusalReason, string> = {
  no_documents: "No documents to search",
  off_topic: "Outside this helpdesk's topics",
  insufficient_context: "The sources do not answer this",
  guardrail_output: "Answer withheld by an output check",
};

export const NOTICE_LABEL: Record<NoticeCode, string> = {
  not_advice: "Information, not advice",
  referral: "Where to get help",
  dated_information: "Check the date",
  emergency: "Emergency",
  third_party_processing: "Your uploads were processed by a hosted model",
};

export const STATUS_LABEL: Record<DocumentStatus, string> = {
  PENDING: "Queued",
  PROCESSING: "Processing",
  READY: "Ready",
  FAILED: "Failed",
  UPDATING: "Updating",
  READY_STALE: "Ready (stale)",
  DELETING: "Deleting",
};

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

/** Example questions behind the topic starter chips (in mock mode each maps to one canned response). */
export const TOPIC_STARTERS = [
  { id: "id_card", label: "ID card", question: "شناختی کارڈ کی تجدید کی فیس کتنی ہے؟" },
  { id: "passport", label: "Passport", question: "Urgent passport banwane ki fee kitni hai?" },
  { id: "bisp", label: "BISP", question: "How can I check if I am eligible for BISP Kafaalat?" },
  { id: "tax", label: "Tax", question: "What is the income tax rate for freelancers in 2026?" },
  {
    id: "rights",
    label: "Rights & complaints",
    question: "Mera landlord mujhe bina notice ke ghar se nikal raha hai, mujhe kya karna chahiye?",
  },
] as const;
