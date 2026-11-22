"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useId, useState, type FormEvent } from "react";

import { api } from "@/lib/client";
import { USE_MOCKS } from "@/lib/config";
import { ApiError, toProblem } from "@/lib/problem";

import { Button, ProblemAlert, Spinner } from "./ui";

type Mode = "login" | "signup";

/** Only same-app relative paths are allowed as a post-login redirect (no open redirect). */
function safeNext(next: string | null): string {
  return next && next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/\\") ? next : "/";
}

export function AuthForm({ mode }: { mode: Mode }) {
  const router = useRouter();
  const params = useSearchParams();
  const ids = { email: useId(), password: useId(), name: useId(), hint: useId() };
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const fields = error?.fieldErrors() ?? {};
  // FastAPI's OAuth2 form names the email field `username`.
  const emailError = fields.email ?? fields.username;
  const passwordError = fields.password;
  const nameError = fields.display_name;
  // Field-level problems are shown next to the field; the banner carries the rest.
  const banner = error && !(error.status === 422 && (emailError || passwordError || nameError)) ? error : null;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") {
        await api.signup({ email: email.trim(), password, display_name: displayName.trim() || null });
      }
      await api.login(email.trim(), password);
      router.replace(safeNext(params.get("next")));
      router.refresh();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err
          : new ApiError(toProblem(0, { code: "client_error", title: "Something went wrong", detail: String(err) })),
      );
    } finally {
      setBusy(false);
    }
  };

  const input =
    "w-full rounded-md border border-border bg-bg px-3 py-2 text-fg aria-[invalid=true]:border-bad";

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-4 py-10">
      <h1 className="text-2xl font-semibold">{mode === "login" ? "Sign in" : "Create an account"}</h1>
      <p className="mt-1 text-sm text-muted">
        Citizen Services Helpdesk: cited answers from official Pakistani sources.
      </p>
      {USE_MOCKS && (
        <p className="mt-3 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-sm">
          Mock mode: any email and password signs in. Use <code>wrong-password</code> to see a 401, or sign up with{" "}
          <code>taken@example.com</code> to see a 409.
        </p>
      )}

      <form onSubmit={submit} noValidate className="mt-6 flex flex-col gap-4 rounded-xl border border-border bg-surface p-5">
        <div aria-live="assertive">{banner && <ProblemAlert error={banner} />}</div>

        {mode === "signup" && (
          <div className="flex flex-col gap-1">
            <label htmlFor={ids.name} className="text-sm font-medium">
              Display name <span className="font-normal text-muted">(optional)</span>
            </label>
            <input
              id={ids.name}
              dir="auto"
              value={displayName}
              maxLength={80}
              autoComplete="nickname"
              onChange={(e) => setDisplayName(e.target.value)}
              aria-invalid={!!nameError || undefined}
              aria-describedby={nameError ? `${ids.name}-err` : undefined}
              className={input}
            />
            {nameError && (
              <p id={`${ids.name}-err`} className="text-sm text-bad">
                {nameError}
              </p>
            )}
          </div>
        )}

        <div className="flex flex-col gap-1">
          <label htmlFor={ids.email} className="text-sm font-medium">
            Email
          </label>
          <input
            id={ids.email}
            type="email"
            required
            value={email}
            autoComplete="email"
            onChange={(e) => setEmail(e.target.value)}
            aria-invalid={!!emailError || undefined}
            aria-describedby={emailError ? `${ids.email}-err` : undefined}
            className={input}
          />
          {emailError && (
            <p id={`${ids.email}-err`} className="text-sm text-bad">
              {emailError}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor={ids.password} className="text-sm font-medium">
            Password
          </label>
          <input
            id={ids.password}
            type="password"
            required
            value={password}
            minLength={mode === "signup" ? 10 : undefined}
            maxLength={128}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            onChange={(e) => setPassword(e.target.value)}
            aria-invalid={!!passwordError || undefined}
            aria-describedby={[mode === "signup" ? ids.hint : "", passwordError ? `${ids.password}-err` : ""].filter(Boolean).join(" ") || undefined}
            className={input}
          />
          {mode === "signup" && (
            <p id={ids.hint} className="text-xs text-muted">
              At least 10 characters.
            </p>
          )}
          {passwordError && (
            <p id={`${ids.password}-err`} className="text-sm text-bad">
              {passwordError}
            </p>
          )}
        </div>

        <Button type="submit" variant="primary" disabled={busy} className="w-full">
          {busy ? <Spinner small label={mode === "login" ? "Signing in…" : "Creating account…"} /> : mode === "login" ? "Sign in" : "Create account"}
        </Button>
      </form>

      <p className="mt-4 text-center text-sm text-muted">
        {mode === "login" ? (
          <>
            No account?{" "}
            <Link href="/signup" className="font-medium text-accent underline underline-offset-2">
              Create one
            </Link>
          </>
        ) : (
          <>
            Already registered?{" "}
            <Link href="/login" className="font-medium text-accent underline underline-offset-2">
              Sign in
            </Link>
          </>
        )}
      </p>
    </div>
  );
}
