import type { Metadata } from "next";
import { Geist, Geist_Mono, Orbitron } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const orbitron = Orbitron({
  variable: "--font-orbitron",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
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
    <html lang="en" className="bg-black">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${orbitron.variable} antialiased bg-black`}
      >
        {children}
      </body>
    </html>
  );
}
