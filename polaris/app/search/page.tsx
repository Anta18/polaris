"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import SearchBar from "@/components/SearchBar";
import { ArticleDoc } from "@/models/articles";
import { Newspaper, Loader2, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function SearchPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get("q");
  const [articles, setArticles] = useState<ArticleDoc[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchArticles() {
      if (!query) return;
      
      setLoading(true);
      try {
        const response = await fetch(`/api/articles?q=${encodeURIComponent(query)}&limit=50`);
        const data = await response.json();
        setArticles(data.articles || []);
      } catch (error) {
        console.error("Error fetching articles:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchArticles();
  }, [query]);

  // Split articles into three columns based on bias
  const leftArticles = articles.filter((article) => {
    const bias = article.bias_classification_label?.toLowerCase() || "";
    return bias.includes("left") || bias.includes("liberal");
  });

  const centerArticles = articles.filter((article) => {
    const bias = article.bias_classification_label?.toLowerCase() || "";
    return bias.includes("center") || bias.includes("neutral") || !bias;
  });

  const rightArticles = articles.filter((article) => {
    const bias = article.bias_classification_label?.toLowerCase() || "";
    return bias.includes("right") || bias.includes("conservative");
  });

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center gap-6 mb-5">
            <Link href="/" className="flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-500 transition-colors">
              <ArrowLeft className="w-5 h-5" />
              <span className="font-semibold">Back to Home</span>
            </Link>
            <div className="flex items-center gap-3">
              <Newspaper className="w-7 h-7 text-blue-600 dark:text-blue-500" />
              <span className="text-2xl font-bold text-slate-900 dark:text-white">
                Polaris News
              </span>
            </div>
          </div>
          <SearchBar />
        </div>
      </header>

      {/* Search Results */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-10">
          <h1 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-4">
            Search Results
          </h1>
          {query && (
            <div className="flex items-center gap-3 flex-wrap">
              <p className="text-base text-slate-600 dark:text-slate-400">
                Showing results for
              </p>
              <span className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold">
                "{query}"
              </span>
              {!loading && (
                <span className="text-base text-slate-600 dark:text-slate-400">
                  • {articles.length} articles found
                </span>
              )}
            </div>
          )}
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-32">
            <Loader2 className="w-16 h-16 text-blue-600 animate-spin mb-4" />
            <p className="text-lg text-slate-600 dark:text-slate-400 font-semibold">
              Searching articles...
            </p>
          </div>
        ) : articles.length === 0 ? (
          <div className="text-center py-20 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="w-24 h-24 mx-auto mb-6 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center">
              <Newspaper className="w-12 h-12 text-slate-400" />
            </div>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
              No articles found
            </h2>
            <p className="text-base text-slate-600 dark:text-slate-400">
              Try a different search term or browse our latest articles
            </p>
            <Link
              href="/"
              className="inline-block mt-8 px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors shadow-sm"
            >
              Go to Home
            </Link>
          </div>
        ) : (
          <>
            {/* Column Headers */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
              <div className="bg-blue-50 dark:bg-slate-800 rounded-lg p-4 border-2 border-blue-600 dark:border-blue-500">
                <h3 className="text-lg font-bold text-blue-700 dark:text-blue-400 flex items-center justify-center gap-2">
                  <span className="text-xl">←</span>
                  Left Bias ({leftArticles.length})
                </h3>
              </div>
              <div className="bg-green-50 dark:bg-slate-800 rounded-lg p-4 border-2 border-green-600 dark:border-green-500">
                <h3 className="text-lg font-bold text-green-700 dark:text-green-400 flex items-center justify-center gap-2">
                  <span className="text-xl">⚖️</span>
                  Center/Neutral ({centerArticles.length})
                </h3>
              </div>
              <div className="bg-red-50 dark:bg-slate-800 rounded-lg p-4 border-2 border-red-600 dark:border-red-500">
                <h3 className="text-lg font-bold text-red-700 dark:text-red-400 flex items-center justify-center gap-2">
                  <span className="text-xl">→</span>
                  Right Bias ({rightArticles.length})
                </h3>
              </div>
            </div>

            {/* Three Column Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Left Column */}
              <div className="space-y-6">
                {leftArticles.length === 0 ? (
                  <div className="text-center py-12 bg-blue-50 dark:bg-slate-800 rounded-lg border border-blue-200 dark:border-slate-700">
                    <p className="text-blue-700 dark:text-blue-400 font-semibold text-sm">
                      No left-leaning articles found
                    </p>
                  </div>
                ) : (
                  leftArticles.map((article) => (
                    <ArticleCard
                      key={article.id || article._id?.toString()}
                      article={article}
                      variant="compact"
                    />
                  ))
                )}
              </div>

              {/* Center Column */}
              <div className="space-y-6">
                {centerArticles.length === 0 ? (
                  <div className="text-center py-12 bg-green-50 dark:bg-slate-800 rounded-lg border border-green-200 dark:border-slate-700">
                    <p className="text-green-700 dark:text-green-400 font-semibold text-sm">
                      No neutral articles found
                    </p>
                  </div>
                ) : (
                  centerArticles.map((article) => (
                    <ArticleCard
                      key={article.id || article._id?.toString()}
                      article={article}
                      variant="compact"
                    />
                  ))
                )}
              </div>

              {/* Right Column */}
              <div className="space-y-6">
                {rightArticles.length === 0 ? (
                  <div className="text-center py-12 bg-red-50 dark:bg-slate-800 rounded-lg border border-red-200 dark:border-slate-700">
                    <p className="text-red-700 dark:text-red-400 font-semibold text-sm">
                      No right-leaning articles found
                    </p>
                  </div>
                ) : (
                  rightArticles.map((article) => (
                    <ArticleCard
                      key={article.id || article._id?.toString()}
                      article={article}
                      variant="compact"
                    />
                  ))
                )}
              </div>
            </div>
          </>
        )}
      </main>

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

