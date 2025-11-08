import SearchBar from "@/components/SearchBar";
import ArticleCard from "@/components/ArticleCard";
import StatsSection from "@/components/StatsSection";
import { NewsSidebar } from "@/components/NewsSidebar";
import { ArticleDoc } from "@/models/articles";
import { Newspaper, TrendingUp, Shield, Sparkles } from "lucide-react";
import { Highlight } from "@/components/ui/hero-highlight";
import { TypewriterEffectSmooth } from "@/components/ui/typewriter-effect";

async function getTopArticles(): Promise<ArticleDoc[]> {
  try {
    const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000";
    console.log("yes")
    const response = await fetch(`${baseUrl}/api/articles`, {
      cache: "no-store",
    });
      console.log("response", response);
      if (!response.ok) {
        console.error("Failed to fetch articles");
        return [];
      }
      
      const data = await response.json();
      console.log("Received data:", data);
      console.log("Articles count:", data.articles?.length);
      console.log("Articles:", data.articles);
      return data.articles || [];
  } catch (error) {
    console.error("Error fetching articles:", error);
    return [];
  }
}

export default async function Home() {
  const articles = await getTopArticles();

  return (
    <NewsSidebar>
      <div className="min-h-screen bg-black">
      {/* Hero Section */}
      <section className="bg-zinc-900 border-b border-zinc-800">
        <div className="max-w-7xl mx-auto px-6 py-16 lg:py-24">
          {/* Logo/Brand */}
          <div className="flex justify-center mb-8">
            <div className="flex items-center gap-4">
              <Newspaper className="w-10 h-10 text-blue-400" />
              <TypewriterEffectSmooth 
                words={[
                  { text: "POLARIS", className: "text-gray-100 font-[family-name:var(--font-orbitron)]" },
                  { text: "News", className: "text-blue-400 font-[family-name:var(--font-orbitron)]" }
                ]}
                className="text-5xl"
                cursorClassName="bg-blue-400"
              />
            </div>
          </div>

          {/* Headline */}
          <div className="text-center mb-10">
            <h1 className="text-4xl md:text-6xl font-bold text-gray-100 mb-6 leading-tight">
              <Highlight className="px-3 py-1 rounded-lg bg-gradient-to-r from-[#1d4ed8] via-[#2563eb] to-[#1e3a8a] dark:from-[#2563eb] dark:via-[#1e40af] dark:to-[#172554] text-white ring-1 ring-white/10 shadow-[0_8px_24px_-8px_rgba(37,99,235,.45)]">Professional News</Highlight> Intelligence
              <span className="block text-blue-400 mt-6">
                Powered by <Highlight className="px-3 py-1 rounded-lg bg-gradient-to-r from-[#1e40af]/60 via-[#1d4ed8]/60 to-[#1e3a8a]/60 dark:from-[#1e40af]/50 dark:via-[#1d4ed8]/50 dark:to-[#1e3a8a]/50 text-blue-300 ring-1 ring-blue-500/40 shadow-[0_6px_20px_-6px_rgba(37,99,235,.5)]">AI Analysis</Highlight>
              </span>
            </h1>
            <p className="text-lg md:text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Access comprehensive news analysis with bias detection, sentiment analysis, and fact-checking capabilities.
            </p>
          </div>

          {/* Feature Pills */}
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            <div className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800 rounded-lg border border-zinc-700">
              <Shield className="w-5 h-5 text-green-400" />
              <span className="text-sm font-semibold text-gray-300">Bias Detection</span>
            </div>
            <div className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800 rounded-lg border border-zinc-700">
              <TrendingUp className="w-5 h-5 text-blue-400" />
              <span className="text-sm font-semibold text-gray-300">Sentiment Analysis</span>
            </div>
            <div className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800 rounded-lg border border-zinc-700">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              <span className="text-sm font-semibold text-gray-300">Fact Checking</span>
            </div>
          </div>

          {/* Search Bar */}
          <SearchBar />
        </div>
      </section>

      {/* Stats Section */}
      {/* <StatsSection totalArticles={articles.length > 0 ? articles.length * 167 : 1000} /> */}

      {/* Featured Articles Section */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="flex items-center justify-between mb-10">
          <div>
            <h2 className="text-3xl font-bold text-gray-100 mb-2">
              Latest Stories
            </h2>
            <p className="text-base text-gray-400">
              Top verified and analyzed news articles
            </p>
          </div>
          <div className="hidden md:block">
            <div className="px-5 py-2 bg-blue-600 text-white rounded-lg font-semibold shadow-sm">
              {articles.length} Articles
            </div>
          </div>
        </div>

        {articles.length === 0 ? (
          <div className="text-center py-20 bg-zinc-900 rounded-xl border border-zinc-800">
            <div className="w-24 h-24 mx-auto mb-6 bg-zinc-800 rounded-full flex items-center justify-center">
              <Newspaper className="w-12 h-12 text-gray-500" />
            </div>
            <h3 className="text-xl font-bold text-gray-100 mb-2">
              No articles yet
            </h3>
            <p className="text-gray-400">
              Start adding articles to your database to see them here
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {articles.map((article) => (
              <ArticleCard key={article.id || (article as any)._id?.toString()} article={article} />
            ))}
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="bg-zinc-900 border-t border-zinc-800 mt-20">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Newspaper className="w-6 h-6 text-blue-400" />
              <span className="text-lg font-bold text-gray-100">Polaris News</span>
            </div>
            <p className="text-gray-400 text-sm">
              © 2025 Polaris News. Professional news intelligence platform.
            </p>
          </div>
        </div>
      </footer>
    </div>
    </NewsSidebar>
  );
}
