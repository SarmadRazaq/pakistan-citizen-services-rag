// Short names for the generated API schema (src/lib/api-types.ts, from `npm run gen:api:offline`).
// Never hand-write a backend shape here: alias the generated one so the contract cannot drift.
import type { components } from "./api-types";

type S = components["schemas"];

export type Problem = S["Problem"];
export type FieldError = S["FieldError"];

export type SignupRequest = S["SignupRequest"];
export type TokenResponse = S["TokenResponse"];
export type UserOut = S["UserOut"];
export type Role = S["Role"];

export type DocumentOut = S["DocumentOut"];
export type DocumentMetadata = S["DocumentMetadata"];
export type DocumentStatus = S["DocumentStatus"];
export type DocumentPage = S["Page_DocumentOut_"];
export type FileType = S["FileType"];
export type Visibility = S["Visibility"];
export type Scope = S["Scope"];
export type Jurisdiction = S["Jurisdiction"];
export type LanguageCode = S["LanguageCode"];

export type AskRequest = S["AskRequest"];
export type AskResponse = S["AskResponse"];
export type AnswerLanguage = S["AnswerLanguage"];
export type Citation = S["Citation"];
export type RetrievedChunk = S["RetrievedChunk"];
export type RetrievalInfo = S["RetrievalInfo"];
export type Notice = S["Notice"];
export type NoticeCode = S["NoticeCode"];
export type RefusalReason = S["RefusalReason"];

export type HealthResponse = S["HealthResponse"];
