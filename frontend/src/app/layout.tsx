import type { Metadata, Viewport } from "next";
import { Noto_Nastaliq_Urdu } from "next/font/google";

import { ToastProvider } from "@/components/Toast";

import "./globals.css";

// Nastaliq for every element marked lang="ur" (see globals.css). Only the Arabic subset is
// loaded, and no metric fallback is generated, so Latin text keeps the system UI font.
const nastaliq = Noto_Nastaliq_Urdu({
  subsets: ["arabic"],
  weight: ["400", "700"],
  display: "swap",
  variable: "--font-nastaliq",
  adjustFontFallback: false,
  preload: false,
});

export const metadata: Metadata = {
  title: "Citizen Services Helpdesk",
  description: "Cited answers about Pakistani citizen services and rights, in English, Urdu and Roman Urdu.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4f6fa" },
    { media: "(prefers-color-scheme: dark)", color: "#0b1220" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={nastaliq.variable}>
      <body className="min-h-screen antialiased">
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  );
}
