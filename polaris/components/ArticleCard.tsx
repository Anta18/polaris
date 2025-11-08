"use client";

import Link from "next/link";
import Image from "next/image";
import { Calendar, User, TrendingUp, AlertCircle, Zap, Heart, MessageCircle } from "lucide-react";
import { ArticleDoc } from "@/models/articles";

interface ArticleCardProps {
  article: ArticleDoc;
  variant?: "default" | "compact";
}

export default function ArticleCard({ article, variant = "default" }: ArticleCardProps) {
  const formattedDate = new Date(article.publishedAt).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  // Analysis indicators
  const getBiasColor = (label: string | null) => {
    if (!label) return "bg-slate-500";
    const lower = label.toLowerCase();
    if (lower.includes("left")) return "bg-blue-600";
    if (lower.includes("right")) return "bg-red-600";
    return "bg-green-600";
  };

  const getSentimentColor = (label: string | null) => {
    if (!label) return "bg-slate-500";
    const lower = label.toLowerCase();
    if (lower.includes("positive")) return "bg-green-600";
    if (lower.includes("negative")) return "bg-red-600";
    return "bg-yellow-600";
  };

  const getSentimentTextColor = (label: string | null) => {
    if (!label) return "text-slate-600";
    const lower = label.toLowerCase();
    if (lower.includes("positive")) return "text-green-600 dark:text-green-500";
    if (lower.includes("negative")) return "text-red-600 dark:text-red-500";
    return "text-yellow-600 dark:text-yellow-500";
  };

  const getReliabilityColor = (score: number | null) => {
    if (!score) return "bg-slate-500";
    if (score >= 0.8) return "bg-green-600";
    if (score >= 0.5) return "bg-yellow-600";
    return "bg-red-600";
  };

  const getReliabilityTextColor = (score: number | null) => {
    if (!score) return "text-slate-600";
    if (score >= 0.8) return "text-green-600 dark:text-green-500";
    if (score >= 0.5) return "text-yellow-600 dark:text-yellow-500";
    return "text-red-600 dark:text-red-500";
  };

  if (variant === "compact") {
    return (
      <Link
        href={`/article/${article.id || article._id}`}
        className="block group"
      >
        <div className="bg-white dark:bg-slate-800 rounded-lg overflow-hidden shadow-md hover:shadow-xl transition-all duration-200 border border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:hover:border-blue-500">
          <div className="relative h-48 overflow-hidden bg-slate-200 dark:bg-slate-700">
            <Image
              src={article.imageUrl || "/placeholder.svg"}
              alt={article.title}
              fill
              className="object-cover group-hover:scale-105 transition-transform duration-300"
            />
            <div className="absolute top-3 right-3 flex gap-2">
              <span
                className={`${getBiasColor(article.bias_classification_label)} text-white text-xs px-2.5 py-1 rounded font-semibold shadow-sm`}
              >
                {article.bias_classification_label || "Neutral"}
              </span>
            </div>
            {article.clickbait_label === "clickbait" && (
              <div className="absolute top-3 left-3">
                <span className="bg-orange-600 text-white text-xs px-2.5 py-1 rounded font-semibold flex items-center gap-1 shadow-sm">
                  <Zap className="w-3 h-3" />
                  Clickbait
                </span>
              </div>
            )}
          </div>
          <div className="p-5">
            <h3 className="text-base font-bold text-slate-900 dark:text-white line-clamp-2 group-hover:text-blue-600 dark:group-hover:text-blue-500 transition-colors mb-2">
              {article.title}
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2 mb-3">
              {article.description}
            </p>
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 pt-3 border-t border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5" />
                {formattedDate}
              </div>
              <div className="flex items-center gap-1">
                <User className="w-3.5 h-3.5" />
                <span className="truncate max-w-[100px]">{article.author}</span>
              </div>
            </div>
          </div>
        </div>
      </Link>
    );
  }

  return (
    <Link
      href={`/article/${article.id || article._id}`}
      className="block group h-full"
    >
      <div className="bg-white dark:bg-slate-800 rounded-lg overflow-hidden shadow-md hover:shadow-xl transition-all duration-200 border border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:hover:border-blue-500 h-full flex flex-col">
        <div className="relative h-56 overflow-hidden bg-slate-200 dark:bg-slate-700">
          <Image
            src={article.imageUrl || "/placeholder.svg"}
            alt={article.title}
            fill
            className="object-cover group-hover:scale-105 transition-transform duration-300"
          />
          
          {/* Analysis Badges */}
          <div className="absolute top-3 right-3 flex flex-col gap-2">
            {article.sentiment_analysis_label && (
              <div className={`${getSentimentColor(article.sentiment_analysis_label)} text-white text-xs px-3 py-1.5 rounded font-semibold shadow-md flex items-center gap-1.5`}>
                <TrendingUp className="w-3.5 h-3.5" />
                {article.sentiment_analysis_label}
              </div>
            )}
            {article.source_reliability !== null && (
              <div className={`${getReliabilityColor(article.source_reliability)} text-white text-xs px-3 py-1.5 rounded font-semibold shadow-md flex items-center gap-1.5`}>
                <AlertCircle className="w-3.5 h-3.5" />
                {Math.round(article.source_reliability * 100)}%
              </div>
            )}
          </div>

          {/* Bias Label */}
          <div className="absolute bottom-3 left-3">
            <span className={`${getBiasColor(article.bias_classification_label)} text-white text-sm px-4 py-2 rounded font-bold shadow-md`}>
              {article.bias_classification_label || "Neutral"}
            </span>
          </div>

          {/* Clickbait Warning */}
          {article.clickbait_label === "clickbait" && (
            <div className="absolute top-3 left-3">
              <span className="bg-orange-600 text-white text-xs px-3 py-1.5 rounded font-bold shadow-md flex items-center gap-1.5">
                <Zap className="w-4 h-4" />
                Clickbait
              </span>
            </div>
          )}
        </div>

        <div className="p-5 flex-1 flex flex-col">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs font-semibold text-blue-600 dark:text-blue-500 bg-blue-50 dark:bg-blue-900/30 px-3 py-1 rounded">
              {article.category}
            </span>
            {article.topic && (
              <span className="text-xs text-slate-500 dark:text-slate-400">
                • {article.topic}
              </span>
            )}
          </div>

          <h3 className="text-lg font-bold text-slate-900 dark:text-white line-clamp-2 mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-500 transition-colors">
            {article.title}
          </h3>

          <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-3 mb-4 flex-1">
            {article.description}
          </p>

          <div className="flex items-center justify-between pt-4 border-t border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <Calendar className="w-4 h-4" />
              {formattedDate}
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <User className="w-4 h-4" />
              <span className="truncate max-w-[120px]">{article.author}</span>
            </div>
          </div>

          <div className="flex items-center gap-4 mt-3 text-sm">
            <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
              <Heart className="w-4 h-4" />
              <span>{article.likes || 0}</span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
              <MessageCircle className="w-4 h-4" />
              <span>{article.comments?.length || 0}</span>
            </div>
          </div>
        </div>
      </div>
    </Link>
  );
}
