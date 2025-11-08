"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { ArticleDoc } from "@/models/articles";
import {
  Newspaper,
  ArrowLeft,
  Calendar,
  User,
  Share2,
  Heart,
  MessageCircle,
  AlertTriangle,
  TrendingUp,
  Shield,
  Eye,
  Zap,
  CheckCircle,
  XCircle,
  FileText,
  Loader2,
  Send,
} from "lucide-react";

export default function ArticlePage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [article, setArticle] = useState<ArticleDoc | null>(null);
  const [loading, setLoading] = useState(true);
  const [liked, setLiked] = useState(false);
  const [localLikes, setLocalLikes] = useState(0);
  const [commentText, setCommentText] = useState("");
  const [comments, setComments] = useState<any[]>([]);
  const [relatedArticles, setRelatedArticles] = useState<ArticleDoc[]>([]);

  useEffect(() => {
    async function fetchArticle() {
      try {
        const response = await fetch(`/api/articles?id=${id}`);
        const data = await response.json();
        
        if (data.article) {
          setArticle(data.article);
          setLocalLikes(data.article.likes || 0);
          setComments(data.article.comments || []);

          // Fetch related articles
          if (data.article.category) {
            const relatedResponse = await fetch(
              `/api/articles?category=${encodeURIComponent(data.article.category)}&limit=3`
            );
            const relatedData = await relatedResponse.json();
            setRelatedArticles(
              (relatedData.articles || []).filter(
                (a: ArticleDoc) => a.id !== data.article.id
              ).slice(0, 3)
            );
          }
        }
      } catch (error) {
        console.error("Error fetching article:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchArticle();
  }, [id]);

  const handleLike = async () => {
    const newLikeCount = liked ? localLikes - 1 : localLikes + 1;
    setLiked(!liked);
    setLocalLikes(newLikeCount);

    // Persist to database
    try {
      await fetch(`/api/articles/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ likes: newLikeCount }),
      });
    } catch (error) {
      console.error("Error updating likes:", error);
    }
  };

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: article?.title,
        text: article?.description,
        url: window.location.href,
      });
    } else {
      navigator.clipboard.writeText(window.location.href);
      alert("Link copied to clipboard!");
    }
  };

  const handleCommentSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!commentText.trim()) return;

    const commentData = {
      userId: "anonymous",
      userName: "Anonymous User",
      content: commentText,
    };

    // Optimistically add to UI
    const tempComment = {
      id: Date.now().toString(),
      ...commentData,
      createdAt: new Date(),
      likes: 0,
    };

    setComments([tempComment, ...comments]);
    setCommentText("");

    // Persist to database
    try {
      await fetch(`/api/articles/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comment: commentData }),
      });
    } catch (error) {
      console.error("Error posting comment:", error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-zinc-50 via-white to-purple-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-purple-950 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-16 h-16 text-purple-600 animate-spin mx-auto mb-4" />
          <p className="text-xl text-zinc-600 dark:text-zinc-400 font-semibold">
            Loading article...
          </p>
        </div>
      </div>
    );
  }

  if (!article) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-zinc-50 via-white to-purple-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-purple-950 flex items-center justify-center">
        <div className="text-center">
          <XCircle className="w-24 h-24 text-red-500 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-white mb-4">
            Article not found
          </h1>
          <Link
            href="/"
            className="inline-block px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-bold hover:shadow-2xl transition-all duration-200 hover:scale-105"
          >
            Go to Home
          </Link>
        </div>
      </div>
    );
  }

  const formattedDate = new Date(article.publishedAt).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const getBiasColor = (label: string | null) => {
    if (!label) return "bg-zinc-500";
    const lower = label.toLowerCase();
    if (lower.includes("left")) return "bg-blue-500";
    if (lower.includes("right")) return "bg-red-500";
    return "bg-green-500";
  };

  const getSentimentColor = (label: string | null) => {
    if (!label) return "bg-zinc-500";
    const lower = label.toLowerCase();
    if (lower.includes("positive")) return "bg-green-500";
    if (lower.includes("negative")) return "bg-red-500";
    return "bg-yellow-500";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-50 via-white to-purple-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-purple-950">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 dark:bg-zinc-900/80 border-b border-zinc-200 dark:border-zinc-800 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link
                href="/"
                className="flex items-center gap-2 text-zinc-600 dark:text-zinc-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span className="font-semibold">Back</span>
              </Link>
              <div className="flex items-center gap-3">
                <Newspaper className="w-6 h-6 text-purple-600" />
                <span className="text-xl font-black bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  Polaris News
                </span>
              </div>
            </div>

            {/* Engagement Actions */}
            <div className="flex items-center gap-3">
              <button
                onClick={handleLike}
                className={`flex items-center gap-2 px-4 py-2 rounded-full font-semibold transition-all duration-200 ${
                  liked
                    ? "bg-red-500 text-white"
                    : "bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-red-50 dark:hover:bg-red-900/20"
                }`}
              >
                <Heart className={`w-5 h-5 ${liked ? "fill-current" : ""}`} />
                <span>{localLikes}</span>
              </button>
              <button
                onClick={handleShare}
                className="flex items-center gap-2 px-4 py-2 rounded-full font-semibold bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-all duration-200"
              >
                <Share2 className="w-5 h-5" />
                <span>Share</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Article Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Category & Topic */}
            <div className="flex items-center gap-3 flex-wrap">
              <span className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-bold">
                {article.category}
              </span>
              {article.topic && (
                <span className="px-4 py-2 bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 rounded-full font-semibold border border-zinc-200 dark:border-zinc-700">
                  {article.topic}
                </span>
              )}
            </div>

            {/* Title */}
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-black text-zinc-900 dark:text-white leading-tight">
              {article.title}
            </h1>

            {/* Meta Info */}
            <div className="flex items-center gap-6 text-zinc-600 dark:text-zinc-400">
              <div className="flex items-center gap-2">
                <User className="w-5 h-5" />
                <span className="font-semibold">{article.author}</span>
              </div>
              <div className="flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                <span>{formattedDate}</span>
              </div>
              <div className="flex items-center gap-2">
                <Eye className="w-5 h-5" />
                <span>{article.source}</span>
              </div>
            </div>

            {/* Featured Image */}
            <div className="relative h-96 md:h-[500px] rounded-2xl overflow-hidden shadow-2xl">
              <Image
                src={article.imageUrl || "/placeholder.svg"}
                alt={article.title}
                fill
                className="object-cover"
              />
            </div>

            {/* Description */}
            <div className="bg-white dark:bg-zinc-900 rounded-2xl p-8 shadow-lg border border-zinc-200 dark:border-zinc-800">
              <p className="text-xl text-zinc-700 dark:text-zinc-300 leading-relaxed italic">
                {article.description}
              </p>
            </div>

            {/* Content */}
            <div className="bg-white dark:bg-zinc-900 rounded-2xl p-8 shadow-lg border border-zinc-200 dark:border-zinc-800">
              <div className="prose prose-lg dark:prose-invert max-w-none">
                <p className="text-zinc-700 dark:text-zinc-300 leading-relaxed whitespace-pre-line">
                  {article.content}
                </p>
              </div>
            </div>

            {/* Summaries */}
            {(article.single_source_summary || article.muti_source_summary) && (
              <div className="space-y-6">
                {article.single_source_summary && (
                  <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-2xl p-8 shadow-lg border border-blue-200 dark:border-blue-800">
                    <div className="flex items-center gap-3 mb-4">
                      <FileText className="w-6 h-6 text-blue-600" />
                      <h3 className="text-2xl font-bold text-zinc-900 dark:text-white">
                        Single Source Summary
                      </h3>
                    </div>
                    <p className="text-zinc-700 dark:text-zinc-300 leading-relaxed">
                      {article.single_source_summary}
                    </p>
                  </div>
                )}

                {article.muti_source_summary && (
                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20 rounded-2xl p-8 shadow-lg border border-purple-200 dark:border-purple-800">
                    <div className="flex items-center gap-3 mb-4">
                      <FileText className="w-6 h-6 text-purple-600" />
                      <h3 className="text-2xl font-bold text-zinc-900 dark:text-white">
                        Multi-Source Summary
                      </h3>
                    </div>
                    <p className="text-zinc-700 dark:text-zinc-300 leading-relaxed">
                      {article.muti_source_summary}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Comments Section */}
            <div className="bg-white dark:bg-zinc-900 rounded-2xl p-8 shadow-lg border border-zinc-200 dark:border-zinc-800">
              <div className="flex items-center gap-3 mb-6">
                <MessageCircle className="w-6 h-6 text-purple-600" />
                <h3 className="text-2xl font-bold text-zinc-900 dark:text-white">
                  Comments ({comments.length})
                </h3>
              </div>

              {/* Comment Form */}
              <form onSubmit={handleCommentSubmit} className="mb-8">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder="Share your thoughts..."
                    className="flex-1 px-4 py-3 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-white placeholder:text-zinc-500 outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <button
                    type="submit"
                    className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold hover:shadow-xl transition-all duration-200 active:scale-95 flex items-center gap-2"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </form>

              {/* Comments List */}
              <div className="space-y-4">
                {comments.length === 0 ? (
                  <p className="text-center text-zinc-500 py-8">
                    No comments yet. Be the first to comment!
                  </p>
                ) : (
                  comments.map((comment) => (
                    <div
                      key={comment.id}
                      className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded-xl"
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center text-white font-bold">
                          {comment.userName.charAt(0)}
                        </div>
                        <div>
                          <p className="font-semibold text-zinc-900 dark:text-white">
                            {comment.userName}
                          </p>
                          <p className="text-xs text-zinc-500">
                            {new Date(comment.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <p className="text-zinc-700 dark:text-zinc-300 ml-13">
                        {comment.content}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Analysis Sidebar */}
          <div className="space-y-6">
            <div className="sticky top-24 space-y-6">
              {/* Quick Stats */}
              <div className="bg-white dark:bg-zinc-900 rounded-2xl p-6 shadow-lg border border-zinc-200 dark:border-zinc-800">
                <h3 className="text-xl font-bold text-zinc-900 dark:text-white mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-purple-600" />
                  AI Analysis
                </h3>

                {/* Bias */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
                      Bias
                    </span>
                    <span
                      className={`${getBiasColor(
                        article.bias_classification_label
                      )} text-white text-xs px-3 py-1 rounded-full font-bold`}
                    >
                      {article.bias_classification_label || "Neutral"}
                    </span>
                  </div>
                  {article.bias_classification_probs && Object.keys(article.bias_classification_probs).length > 0 && (
                    <div className="space-y-1">
                      {Object.entries(article.bias_classification_probs).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between text-xs">
                          <span className="text-zinc-600 dark:text-zinc-400 capitalize">{key}</span>
                          <span className="text-zinc-900 dark:text-white font-semibold">
                            {((value as number) * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Sentiment */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
                      Sentiment
                    </span>
                    <span
                      className={`${getSentimentColor(
                        article.sentiment_analysis_label
                      )} text-white text-xs px-3 py-1 rounded-full font-bold`}
                    >
                      {article.sentiment_analysis_label || "Neutral"}
                    </span>
                  </div>
                  {article.sentiment_analysis_probs && Object.keys(article.sentiment_analysis_probs).length > 0 && (
                    <div className="space-y-1">
                      {Object.entries(article.sentiment_analysis_probs).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between text-xs">
                          <span className="text-zinc-600 dark:text-zinc-400 capitalize">{key}</span>
                          <span className="text-zinc-900 dark:text-white font-semibold">
                            {((value as number) * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Clickbait */}
                {article.clickbait_label && (
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
                        Clickbait
                      </span>
                      <span
                        className={`${
                          article.clickbait_label === "clickbait"
                            ? "bg-orange-500"
                            : "bg-green-500"
                        } text-white text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1`}
                      >
                        {article.clickbait_label === "clickbait" ? (
                          <>
                            <Zap className="w-3 h-3" />
                            Yes
                          </>
                        ) : (
                          <>
                            <CheckCircle className="w-3 h-3" />
                            No
                          </>
                        )}
                      </span>
                    </div>
                    {article.clickbait_score !== null && (
                      <p className="text-xs text-zinc-600 dark:text-zinc-400">
                        Score: {(article.clickbait_score * 100).toFixed(1)}%
                      </p>
                    )}
                    {article.clickbait_explanation && (
                      <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2">
                        {article.clickbait_explanation}
                      </p>
                    )}
                  </div>
                )}

                {/* Reliability */}
                {article.source_reliability !== null && (
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
                        Source Reliability
                      </span>
                      <span className="text-zinc-900 dark:text-white font-bold">
                        {(article.source_reliability * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          article.source_reliability >= 0.8
                            ? "bg-green-500"
                            : article.source_reliability >= 0.5
                            ? "bg-yellow-500"
                            : "bg-red-500"
                        }`}
                        style={{ width: `${article.source_reliability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                {/* Fake News */}
                {article.fake_news_label && (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
                        Authenticity
                      </span>
                      <span
                        className={`${
                          article.fake_news_label.toLowerCase().includes("fake")
                            ? "bg-red-500"
                            : "bg-green-500"
                        } text-white text-xs px-3 py-1 rounded-full font-bold`}
                      >
                        {article.fake_news_label}
                      </span>
                    </div>
                    {article.fake_news_probs && Object.keys(article.fake_news_probs).length > 0 && (
                      <div className="space-y-1">
                        {Object.entries(article.fake_news_probs).map(([key, value]) => (
                          <div key={key} className="flex items-center justify-between text-xs">
                            <span className="text-zinc-600 dark:text-zinc-400 capitalize">{key}</span>
                            <span className="text-zinc-900 dark:text-white font-semibold">
                              {((value as number) * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Bias Explanation */}
              {article.bias_explain && article.bias_explain.length > 0 && (
                <div className="bg-white dark:bg-zinc-900 rounded-2xl p-6 shadow-lg border border-zinc-200 dark:border-zinc-800">
                  <h3 className="text-xl font-bold text-zinc-900 dark:text-white mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-orange-600" />
                    Bias Indicators
                  </h3>
                  <div className="space-y-3">
                    {article.bias_explain.slice(0, 5).map((item, index) => (
                      <div
                        key={index}
                        className="p-3 bg-orange-50 dark:bg-orange-950/20 rounded-lg border border-orange-200 dark:border-orange-800"
                      >
                        <p className="text-sm font-semibold text-zinc-900 dark:text-white mb-1">
                          "{item.phrase}"
                        </p>
                        <div className="flex items-center gap-3 text-xs text-zinc-600 dark:text-zinc-400">
                          <span>Score: {item.score.toFixed(2)}</span>
                          <span>Weight: {item.weight.toFixed(2)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Omitted Facts */}
              {article.omitted_facts_articles && article.omitted_facts_articles.length > 0 && (
                <div className="bg-white dark:bg-zinc-900 rounded-2xl p-6 shadow-lg border border-zinc-200 dark:border-zinc-800">
                  <h3 className="text-xl font-bold text-zinc-900 dark:text-white mb-4 flex items-center gap-2">
                    <Eye className="w-5 h-5 text-blue-600" />
                    Cross-Reference Check
                  </h3>
                  <div className="space-y-3">
                    {article.omitted_facts_articles.slice(0, 3).map((item, index) => (
                      <a
                        key={index}
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
                      >
                        <p className="text-sm font-semibold text-zinc-900 dark:text-white mb-2 line-clamp-2">
                          {item.title}
                        </p>
                        {item.omitted_segments && item.omitted_segments.length > 0 && (
                          <p className="text-xs text-zinc-600 dark:text-zinc-400">
                            {item.omitted_segments.length} omitted segment(s)
                          </p>
                        )}
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Original Source Link */}
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-bold text-center hover:shadow-2xl transition-all duration-200 hover:scale-105"
              >
                View Original Article →
              </a>
            </div>
          </div>
        </div>

        {/* Related Articles */}
        {relatedArticles.length > 0 && (
          <section className="mt-20">
            <div className="mb-10">
              <h2 className="text-4xl font-black text-zinc-900 dark:text-white mb-2">
                Related Articles
              </h2>
              <p className="text-lg text-zinc-600 dark:text-zinc-400">
                More stories from {article?.category}
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {relatedArticles.map((relatedArticle) => {
                const formattedDate = new Date(relatedArticle.publishedAt).toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                });

                const getBiasColor = (label: string | null) => {
                  if (!label) return "bg-zinc-500";
                  const lower = label.toLowerCase();
                  if (lower.includes("left")) return "bg-blue-500";
                  if (lower.includes("right")) return "bg-red-500";
                  return "bg-green-500";
                };

                return (
                  <Link
                    key={relatedArticle.id || relatedArticle._id?.toString()}
                    href={`/article/${relatedArticle.id || relatedArticle._id}`}
                    className="block group"
                  >
                    <div className="bg-white dark:bg-zinc-900 rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 border border-zinc-200 dark:border-zinc-800 hover:border-purple-500 dark:hover:border-purple-500 h-full">
                      <div className="relative h-48 overflow-hidden">
                        <Image
                          src={relatedArticle.imageUrl || "/placeholder.svg"}
                          alt={relatedArticle.title}
                          fill
                          className="object-cover group-hover:scale-110 transition-transform duration-300"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                        <div className="absolute bottom-3 left-3">
                          <span
                            className={`${getBiasColor(relatedArticle.bias_classification_label)} text-white text-xs px-3 py-1 rounded-full font-bold`}
                          >
                            {relatedArticle.bias_classification_label || "Neutral"}
                          </span>
                        </div>
                      </div>
                      <div className="p-5">
                        <h3 className="text-lg font-bold text-zinc-900 dark:text-white line-clamp-2 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors mb-2">
                          {relatedArticle.title}
                        </h3>
                        <p className="text-sm text-zinc-600 dark:text-zinc-400 line-clamp-2 mb-3">
                          {relatedArticle.description}
                        </p>
                        <div className="flex items-center gap-4 text-xs text-zinc-500">
                          <div className="flex items-center gap-1">
                            <Calendar className="w-3.5 h-3.5" />
                            {formattedDate}
                          </div>
                          <div className="flex items-center gap-1">
                            <User className="w-3.5 h-3.5" />
                            {relatedArticle.author}
                          </div>
                        </div>
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800 mt-20">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Newspaper className="w-6 h-6 text-purple-600" />
              <span className="text-xl font-bold text-zinc-900 dark:text-white">
                Polaris News
              </span>
            </div>
            <p className="text-zinc-600 dark:text-zinc-400">
              © 2025 Polaris News. Empowered by AI, driven by truth.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

