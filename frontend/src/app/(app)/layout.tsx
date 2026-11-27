import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { AppShell } from "@/components/AppShell";
import { SessionProvider } from "@/components/Session";
import { USE_MOCKS } from "@/lib/config";
import { SESSION_COOKIE } from "@/lib/server/backend";

// Signed-in area. Without a session cookie there is nothing to show, so go to /login before
// rendering. (The backend still validates the token on every call; this is only a shortcut.)
export default async function AppLayout({ children }: { children: React.ReactNode }) {
  if (!USE_MOCKS) {
    const jar = await cookies();
    if (!jar.has(SESSION_COOKIE)) redirect("/login");
  }
  return (
    <SessionProvider>
      <AppShell>{children}</AppShell>
    </SessionProvider>
  );
}
