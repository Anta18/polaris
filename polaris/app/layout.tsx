import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Polaris News - Truth-Powered News Intelligence",
  description: "Experience news with AI-powered analysis. Detect bias, verify facts, and discover the truth behind every story with advanced sentiment analysis, clickbait detection, and fact-checking.",
  keywords: "news, AI analysis, bias detection, fact checking, sentiment analysis, journalism, media",
  authors: [{ name: "Polaris News" }],
  openGraph: {
    title: "Polaris News - Truth-Powered News Intelligence",
    description: "Experience news with AI-powered analysis. Detect bias, verify facts, and discover the truth behind every story.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
