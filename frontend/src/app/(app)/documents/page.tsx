import type { Metadata } from "next";
import { Suspense } from "react";

import { DocumentsView } from "@/components/documents/DocumentsView";

export const metadata: Metadata = { title: "Documents · Citizen Services Helpdesk" };

export default function DocumentsPage() {
  return (
    <Suspense>
      <DocumentsView />
    </Suspense>
  );
}
