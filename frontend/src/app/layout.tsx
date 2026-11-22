import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Citizen Services Helpdesk",
  description: "Cited answers about Pakistani citizen services and rights, in English, Urdu and Roman Urdu.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
