"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { api } from "@/lib/client";
import { USE_MOCKS } from "@/lib/config";

import { useSession } from "./Session";
import { Badge, Button, cx } from "./ui";

const NAV = [
  { href: "/", label: "Ask" },
  { href: "/documents", label: "Documents" },
];

function HealthDot() {
  const [status, setStatus] = useState<string>("checking");
  useEffect(() => {
    let alive = true;
    const check = () =>
      api.health().then((h) => {
        if (alive) setStatus(h.status);
      });
    check();
    const t = setInterval(check, 30_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  const tone = status === "ok" ? "bg-ok" : status === "degraded" ? "bg-warn" : status === "checking" ? "bg-muted" : "bg-bad";
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted" title={`Backend readiness: ${status}`}>
      <span aria-hidden="true" className={cx("h-2 w-2 rounded-full", tone)} />
      <span className="hidden sm:inline">API {status}</span>
      <span className="sr-only sm:hidden">API {status}</span>
    </span>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { user, error, isAdmin, logout } = useSession();

  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:start-2 focus:top-2 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-border bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2">
          <Link href="/" className="flex items-baseline gap-2 font-semibold">
            <span>Citizen Services Helpdesk</span>
            <span lang="ur" className="hidden text-sm font-normal text-muted md:inline" style={{ lineHeight: 1.6 }}>
              شہری خدمات
            </span>
          </Link>
          <nav aria-label="Main" className="flex gap-1">
            {NAV.map((n) => {
              const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
              return (
                <Link
                  key={n.href}
                  href={n.href}
                  aria-current={active ? "page" : undefined}
                  className={cx(
                    "rounded-md px-3 py-1.5 text-sm font-medium",
                    active ? "bg-accent text-accent-fg" : "text-fg hover:bg-surface-2",
                  )}
                >
                  {n.label}
                </Link>
              );
            })}
          </nav>
          <div className="ms-auto flex items-center gap-3">
            {USE_MOCKS && <Badge tone="warn">Mock data</Badge>}
            <HealthDot />
            {user && (
              <span className="hidden text-sm text-muted md:inline">
                {user.display_name || user.email}
                {isAdmin && (
                  <>
                    {" "}
                    <Badge tone="info">admin</Badge>
                  </>
                )}
              </span>
            )}
            <Button variant="ghost" onClick={() => void logout()}>
              Sign out
            </Button>
          </div>
        </div>
        {USE_MOCKS && (
          <p className="border-t border-border bg-warn/10 px-4 py-1 text-center text-xs text-fg">
            Mock mode: every answer, fee and document here is sample data, not official information.
          </p>
        )}
        {error && (
          <p role="alert" className="border-t border-border bg-bad/10 px-4 py-1 text-center text-xs text-fg">
            Could not load your account ({error.title}
            {error.detail && error.detail !== error.title ? `: ${error.detail}` : ""}). Some controls may be hidden.
          </p>
        )}
      </header>
      <main id="main" className="flex-1">
        {children}
      </main>
    </div>
  );
}
