import type { Metadata } from "next";

import { MetricsView } from "@/components/MetricsView";

export const metadata: Metadata = { title: "Metrics · Citizen Services Helpdesk" };

export default function MetricsPage() {
  return <MetricsView />;
}
