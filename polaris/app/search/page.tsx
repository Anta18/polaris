"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ArticleCard from "@/components/ArticleCard";
import SearchBar from "@/components/SearchBar";
import { ArticleDoc } from "@/models/articles";
import { Newspaper, Loader2, ArrowLeft, TrendingUp, Scale, TrendingDown } from "lucide-react";
import Link from "next/link";
import { NewsSidebar } from "@/components/NewsSidebar";
import { motion } from "motion/react";

export default function SearchPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get("q");
  const [articles, setArticles] = useState<ArticleDoc[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchArticles() {
      if (!query) {
        setLoading(false);
        return;
      }
      
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
    <NewsSidebar>
    <div className="min-h-screen bg-black">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-zinc-900 border-b border-zinc-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center gap-6 mb-5">
            <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors">
              <ArrowLeft className="w-5 h-5" />
              <span className="font-semibold">Back to Home</span>
            </Link>
            <div className="flex items-center gap-3">
              <Newspaper className="w-7 h-7 text-blue-400" />
              <span className="text-2xl font-bold text-gray-100">
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
          <h1 className="text-3xl md:text-4xl font-bold text-gray-100 mb-4">
            Search Results
          </h1>
          {query && (
            <div className="flex items-center gap-3 flex-wrap">
              <p className="text-base text-gray-400">
                Showing results for
              </p>
              <span className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold">
                "{query}"
              </span>
              {!loading && (
                <span className="text-base text-gray-400">
                  • {articles.length} articles found
                </span>
              )}
            </div>
          )}
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-32">
            <Loader2 className="w-16 h-16 text-blue-400 animate-spin mb-4" />
            <p className="text-lg text-gray-400 font-semibold">
              Searching articles...
            </p>
          </div>
        ) : !query ? (
          <div className="text-center py-20 bg-zinc-900 rounded-lg border border-zinc-800">
            <div className="w-24 h-24 mx-auto mb-6 bg-zinc-800 rounded-full flex items-center justify-center">
              <Newspaper className="w-12 h-12 text-blue-400" />
            </div>
            <h2 className="text-2xl font-bold text-gray-100 mb-3">
              Start Your Search
            </h2>
            <p className="text-base text-gray-400 mb-2">
              Use the search bar above to find news articles by topic, author, or keyword.
            </p>
            <p className="text-sm text-gray-500">
              Discover articles from across the political spectrum
            </p>
          </div>
        ) : articles.length === 0 ? (
          <div className="text-center py-20 bg-zinc-900 rounded-lg border border-zinc-800">
            <div className="w-24 h-24 mx-auto mb-6 bg-zinc-800 rounded-full flex items-center justify-center">
              <Newspaper className="w-12 h-12 text-gray-500" />
            </div>
            <h2 className="text-2xl font-bold text-gray-100 mb-3">
              No articles found
            </h2>
            <p className="text-base text-gray-400">
              Try a different search term or browse our latest articles
            </p>
            <Link
              href="/"
              className="inline-block mt-8 px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-500 transition-colors shadow-sm"
            >
              Go to Home
            </Link>
          </div>
        ) : (
          <>
            {/* Column Headers */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              {/* Left Bias Column Header */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="group relative"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 rounded-xl opacity-20 group-hover:opacity-40 blur transition duration-300"></div>
                <div className="relative bg-zinc-900 rounded-xl p-6 border border-blue-500/30 group-hover:border-blue-400/50 transition-all duration-300">
                  <div className="flex flex-col items-center gap-4">
                    <div className="relative">
                      <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-xl group-hover:bg-blue-400/30 transition duration-300"></div>
                      <div className="relative bg-gradient-to-br from-blue-600 to-blue-500 p-4 rounded-xl shadow-lg group-hover:scale-110 transition-transform duration-300">
                        <TrendingUp className="w-6 h-6 text-white" />
                      </div>
                    </div>
                    <div className="text-center">
                      <h3 className="text-lg font-bold text-blue-400 mb-1 group-hover:text-blue-300 transition-colors">
                        Left Bias
                      </h3>
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-2xl font-bold text-gray-100">{leftArticles.length}</span>
                        <span className="text-sm text-gray-400">articles</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Center/Neutral Column Header */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="group relative"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-600 via-emerald-500 to-teal-500 rounded-xl opacity-20 group-hover:opacity-40 blur transition duration-300"></div>
                <div className="relative bg-zinc-900 rounded-xl p-6 border border-emerald-500/30 group-hover:border-emerald-400/50 transition-all duration-300">
                  <div className="flex flex-col items-center gap-4">
                    <div className="relative">
                      <div className="absolute inset-0 bg-emerald-500/20 rounded-full blur-xl group-hover:bg-emerald-400/30 transition duration-300"></div>
                      <div className="relative bg-gradient-to-br from-emerald-600 to-emerald-500 p-4 rounded-xl shadow-lg group-hover:scale-110 transition-transform duration-300">
                        <Scale className="w-6 h-6 text-white" />
                      </div>
                    </div>
                    <div className="text-center">
                      <h3 className="text-lg font-bold text-emerald-400 mb-1 group-hover:text-emerald-300 transition-colors">
                        Center/Neutral
                      </h3>
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-2xl font-bold text-gray-100">{centerArticles.length}</span>
                        <span className="text-sm text-gray-400">articles</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Right Bias Column Header */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="group relative"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-rose-600 via-rose-500 to-red-500 rounded-xl opacity-20 group-hover:opacity-40 blur transition duration-300"></div>
                <div className="relative bg-zinc-900 rounded-xl p-6 border border-rose-500/30 group-hover:border-rose-400/50 transition-all duration-300">
                  <div className="flex flex-col items-center gap-4">
                    <div className="relative">
                      <div className="absolute inset-0 bg-rose-500/20 rounded-full blur-xl group-hover:bg-rose-400/30 transition duration-300"></div>
                      <div className="relative bg-gradient-to-br from-rose-600 to-rose-500 p-4 rounded-xl shadow-lg group-hover:scale-110 transition-transform duration-300">
                        <TrendingDown className="w-6 h-6 text-white" />
                      </div>
                    </div>
                    <div className="text-center">
                      <h3 className="text-lg font-bold text-rose-400 mb-1 group-hover:text-rose-300 transition-colors">
                        Right Bias
                      </h3>
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-2xl font-bold text-gray-100">{rightArticles.length}</span>
                        <span className="text-sm text-gray-400">articles</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Three Column Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Left Column */}
              <div className="space-y-6">
                {leftArticles.length === 0 ? (
                  <div className="text-center py-12 bg-zinc-900 rounded-lg border border-zinc-800">
                    <p className="text-blue-400 font-semibold text-sm">
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
                  <div className="text-center py-12 bg-zinc-900 rounded-lg border border-zinc-800">
                    <p className="text-green-400 font-semibold text-sm">
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
                  <div className="text-center py-12 bg-zinc-900 rounded-lg border border-zinc-800">
                    <p className="text-red-400 font-semibold text-sm">
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

