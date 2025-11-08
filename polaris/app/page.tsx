import SearchBar from "@/components/SearchBar";
import ArticleCard from "@/components/ArticleCard";
import StatsSection from "@/components/StatsSection";
import { ArticleDoc } from "@/models/articles";
import { Newspaper, TrendingUp, Shield, Sparkles } from "lucide-react";

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
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      {/* Hero Section */}
      <section className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
        <div className="max-w-7xl mx-auto px-6 py-16 lg:py-24">
          {/* Logo/Brand */}
          <div className="flex justify-center mb-12">
            <div className="flex items-center gap-3 px-6 py-3">
              <Newspaper className="w-10 h-10 text-blue-600 dark:text-blue-500" />
              <span className="text-4xl font-bold text-slate-900 dark:text-white">
                Polaris News
              </span>
            </div>
          </div>

          {/* Headline */}
          <div className="text-center mb-10">
            <h1 className="text-4xl md:text-6xl font-bold text-slate-900 dark:text-white mb-6 leading-tight">
              Professional News Intelligence
              <span className="block text-blue-600 dark:text-blue-500 mt-2">
                Powered by AI Analysis
              </span>
            </h1>
            <p className="text-lg md:text-xl text-slate-600 dark:text-slate-400 max-w-3xl mx-auto leading-relaxed">
              Access comprehensive news analysis with bias detection, sentiment analysis, and fact-checking capabilities.
            </p>
          </div>

          {/* Feature Pills */}
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            <div className="flex items-center gap-2 px-5 py-2.5 bg-slate-100 dark:bg-slate-700 rounded-lg border border-slate-200 dark:border-slate-600">
              <Shield className="w-5 h-5 text-green-600 dark:text-green-500" />
              <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">Bias Detection</span>
            </div>
            <div className="flex items-center gap-2 px-5 py-2.5 bg-slate-100 dark:bg-slate-700 rounded-lg border border-slate-200 dark:border-slate-600">
              <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-500" />
              <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">Sentiment Analysis</span>
            </div>
            <div className="flex items-center gap-2 px-5 py-2.5 bg-slate-100 dark:bg-slate-700 rounded-lg border border-slate-200 dark:border-slate-600">
              <Sparkles className="w-5 h-5 text-cyan-600 dark:text-cyan-500" />
              <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">Fact Checking</span>
            </div>
          </div>

          {/* Search Bar */}
          <SearchBar />
        </div>
      </section>

      {/* Stats Section */}
      <StatsSection totalArticles={articles.length > 0 ? articles.length * 167 : 1000} />

      {/* Featured Articles Section */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="flex items-center justify-between mb-10">
          <div>
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
              Latest Stories
            </h2>
            <p className="text-base text-slate-600 dark:text-slate-400">
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
          <div className="text-center py-20 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="w-24 h-24 mx-auto mb-6 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center">
              <Newspaper className="w-12 h-12 text-slate-400" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
              No articles yet
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              Start adding articles to your database to see them here
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {articles.map((article) => (
              <ArticleCard key={article.id || article._id?.toString()} article={article} />
            ))}
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 mt-20">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Newspaper className="w-6 h-6 text-blue-600 dark:text-blue-500" />
              <span className="text-lg font-bold text-slate-900 dark:text-white">Polaris News</span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-sm">
              © 2025 Polaris News. Professional news intelligence platform.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
