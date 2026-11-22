"use client";

import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { api } from "@/lib/client";
import { ApiError } from "@/lib/problem";
import type { UserOut } from "@/lib/types";

type SessionValue = {
  user: UserOut | null;
  /** Set when /auth/me failed for a reason other than 401 (e.g. backend not implemented yet). */
  error: ApiError | null;
  loading: boolean;
  isAdmin: boolean;
  logout: () => Promise<void>;
};

const SessionContext = createContext<SessionValue>({
  user: null,
  error: null,
  loading: true,
  isAdmin: false,
  logout: async () => {},
});

export function useSession() {
  return useContext(SessionContext);
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<UserOut | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api
      .me()
      .then((u) => {
        if (!cancelled) setUser(u);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 401) {
          router.replace(`/login?next=${encodeURIComponent(pathname || "/")}`);
          return;
        }
        setError(err instanceof ApiError ? err : null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // Load once per mount; navigation between app pages keeps the provider.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      router.replace("/login");
      router.refresh();
    }
  }, [router]);

  const value = useMemo<SessionValue>(
    () => ({ user, error, loading, isAdmin: user?.role === "admin", logout }),
    [user, error, loading, logout],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}
