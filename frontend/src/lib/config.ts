// Public (browser-visible) build-time switches. Server-only settings live in src/lib/server/.

/** `NEXT_PUBLIC_USE_MOCKS=1`: every API call is served by src/lib/mocks.ts; no backend needed. */
export const USE_MOCKS = process.env.NEXT_PUBLIC_USE_MOCKS === "1";

/** Mock mode only: `NEXT_PUBLIC_MOCK_ROLE=admin` shows the admin controls on official rows. */
export const MOCK_ROLE: "user" | "admin" = process.env.NEXT_PUBLIC_MOCK_ROLE === "admin" ? "admin" : "user";

/** Mirrors `Settings.max_question_chars` (app/config.py). */
export const MAX_QUESTION_CHARS = 2000;
/** Mirrors `Settings.max_upload_mb` (app/config.py). */
export const MAX_UPLOAD_MB = 50;

/** Ingestion supported by the MVP (blueprint §16.1, slice 1). */
export const SUPPORTED_EXTENSIONS = ["pdf", "docx", "html", "htm", "txt", "md"] as const;
/** Listed in the contract but not ingested yet (stretch items 2 and 8). */
export const COMING_SOON_EXTENSIONS = ["pptx", "jpg", "jpeg", "png"] as const;
