import type { Metadata } from "next";
import { Providers } from "@/components/layout/providers";
import { CommandMenu } from "@/components/layout/command-menu";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Kotha GPT",
    template: "%s · Kotha GPT",
  },
  description:
    "Bangla-first AI platform — models, datasets, training, evaluation, RAG and agents.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="bn" className="dark">
      <body className="min-h-svh bg-background font-sans antialiased">
        <Providers>
          <CommandMenu />
          {children}
        </Providers>
      </body>
    </html>
  );
}
