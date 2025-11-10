// "use client";

// import { useEffect, useState } from "react";
// import { useParams, useRouter } from "next/navigation";
// import Image from "next/image";
// import Link from "next/link";
// import { ArticleDoc } from "@/models/articles";
// import {
//   Newspaper,
//   ArrowLeft,
//   Calendar,
//   User,
//   Share2,
//   Heart,
//   MessageCircle,
//   AlertTriangle,
//   TrendingUp,
//   Shield,
//   Eye,
//   Zap,
//   CheckCircle,
//   XCircle,
//   FileText,
//   Loader2,
//   Send,
// } from "lucide-react";

// export default function ArticlePage() {
//   const params = useParams();
//   const router = useRouter();
//   const id = params.id as string;

//   const [article, setArticle] = useState<ArticleDoc | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [liked, setLiked] = useState(false);
//   const [localLikes, setLocalLikes] = useState(0);
//   const [commentText, setCommentText] = useState("");
//   const [comments, setComments] = useState<any[]>([]);
//   const [relatedArticles, setRelatedArticles] = useState<ArticleDoc[]>([]);

//   useEffect(() => {
//     async function fetchArticle() {
//       try {
//         const response = await fetch(`/api/articles?id=${id}`);
//         const data = await response.json();

//         if (data.article) {
//           setArticle(data.article);
//           setLocalLikes(data.article.likes || 0);
//           setComments(data.article.comments || []);

//           // Fetch related articles
//           if (data.article.category) {
//             const relatedResponse = await fetch(
//               `/api/articles?category=${encodeURIComponent(data.article.category)}&limit=3`
//             );
//             const relatedData = await relatedResponse.json();
//             setRelatedArticles(
//               (relatedData.articles || []).filter(
//                 (a: ArticleDoc) => a.id !== data.article.id
//               ).slice(0, 3)
//             );
//           }
//         }
//       } catch (error) {
//         console.error("Error fetching article:", error);
//       } finally {
//         setLoading(false);
//       }
//     }

//     fetchArticle();
//   }, [id]);

//   const handleLike = async () => {
//     const newLikeCount = liked ? localLikes - 1 : localLikes + 1;
//     setLiked(!liked);
//     setLocalLikes(newLikeCount);

//     // Persist to database
//     try {
//       await fetch(`/api/articles/${id}`, {
//         method: "PATCH",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ likes: newLikeCount }),
//       });
//     } catch (error) {
//       console.error("Error updating likes:", error);
//     }
//   };

//   const handleShare = () => {
//     if (navigator.share) {
//       navigator.share({
//         title: article?.title,
//         text: article?.description,
//         url: window.location.href,
//       });
//     } else {
//       navigator.clipboard.writeText(window.location.href);
//       alert("Link copied to clipboard!");
//     }
//   };

//   const handleCommentSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!commentText.trim()) return;

//     const commentData = {
//       userId: "anonymous",
//       userName: "Anonymous User",
//       content: commentText,
//     };

//     // Optimistically add to UI
//     const tempComment = {
//       id: Date.now().toString(),
//       ...commentData,
//       createdAt: new Date(),
//       likes: 0,
//     };

//     setComments([tempComment, ...comments]);
//     setCommentText("");

//     // Persist to database
//     try {
//       await fetch(`/api/articles/${id}`, {
//         method: "PATCH",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ comment: commentData }),
//       });
//     } catch (error) {
//       console.error("Error posting comment:", error);
//     }
//   };

//   if (loading) {
//     return (
//       <div className="min-h-screen bg-slate-100 dark:bg-slate-950 flex items-center justify-center">
//         <div className="text-center">
//           <Loader2 className="w-16 h-16 text-blue-600 animate-spin mx-auto mb-4" />
//           <p className="text-lg text-slate-600 dark:text-slate-400 font-semibold">
//             Loading article...
//           </p>
//         </div>
//       </div>
//     );
//   }

//   if (!article) {
//     return (
//       <div className="min-h-screen bg-slate-100 dark:bg-slate-950 flex items-center justify-center">
//         <div className="text-center">
//           <XCircle className="w-24 h-24 text-red-600 mx-auto mb-4" />
//           <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
//             Article not found
//           </h1>
//           <Link
//             href="/"
//             className="inline-block px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors shadow-sm"
//           >
//             Go to Home
//           </Link>
//         </div>
//       </div>
//     );
//   }

//   const formattedDate = new Date(article.publishedAt).toLocaleDateString("en-US", {
//     month: "long",
//     day: "numeric",
//     year: "numeric",
//     hour: "2-digit",
//     minute: "2-digit",
//   });

//   const getBiasColor = (label: string | null) => {
//     if (!label) return "bg-slate-600";
//     const lower = label.toLowerCase();
//     if (lower.includes("left")) return "bg-blue-600";
//     if (lower.includes("right")) return "bg-red-600";
//     return "bg-green-600";
//   };

//   const getSentimentColor = (label: string | null) => {
//     if (!label) return "bg-slate-600";
//     const lower = label.toLowerCase();
//     if (lower.includes("positive")) return "bg-green-600";
//     if (lower.includes("negative")) return "bg-red-600";
//     return "bg-yellow-600";
//   };

//   return (
//     <div className="min-h-screen bg-slate-100 dark:bg-slate-950">
//       {/* Header */}
//       <header className="sticky top-0 z-50 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shadow-sm">
//         <div className="max-w-7xl mx-auto px-6 py-4">
//           <div className="flex items-center justify-between">
//             <div className="flex items-center gap-6">
//               <Link
//                 href="/"
//                 className="flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-500 transition-colors"
//               >
//                 <ArrowLeft className="w-5 h-5" />
//                 <span className="font-semibold">Back</span>
//               </Link>
//               <div className="flex items-center gap-3">
//                 <Newspaper className="w-6 h-6 text-blue-600 dark:text-blue-500" />
//                 <span className="text-xl font-bold text-slate-900 dark:text-white">
//                   Polaris News
//                 </span>
//               </div>
//             </div>

//             {/* Engagement Actions */}
//             <div className="flex items-center gap-3">
//               <button
//                 onClick={handleLike}
//                 className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-colors shadow-sm ${
//                   liked
//                     ? "bg-red-600 text-white hover:bg-red-700"
//                     : "bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-red-50 dark:hover:bg-red-900/20 border border-slate-200 dark:border-slate-600"
//                 }`}
//               >
//                 <Heart className={`w-5 h-5 ${liked ? "fill-current" : ""}`} />
//                 <span>{localLikes}</span>
//               </button>
//               <button
//                 onClick={handleShare}
//                 className="flex items-center gap-2 px-4 py-2 rounded-lg font-semibold bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors shadow-sm border border-slate-200 dark:border-slate-600"
//               >
//                 <Share2 className="w-5 h-5" />
//                 <span>Share</span>
//               </button>
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Article Content */}
//       <main className="max-w-7xl mx-auto px-6 py-12">
//         <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
//           {/* Main Content */}
//           <div className="lg:col-span-2 space-y-6">
//             {/* Category & Topic */}
//             <div className="flex items-center gap-3 flex-wrap">
//               <span className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold shadow-sm">
//                 {article.category}
//               </span>
//               {article.topic && (
//                 <span className="px-4 py-2 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-lg font-semibold border border-slate-200 dark:border-slate-700">
//                   {article.topic}
//                 </span>
//               )}
//             </div>

//             {/* Title */}
//             <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-slate-900 dark:text-white leading-tight">
//               {article.title}
//             </h1>

//             {/* Meta Info */}
//             <div className="flex items-center gap-6 text-slate-600 dark:text-slate-400 text-sm">
//               <div className="flex items-center gap-2">
//                 <User className="w-4 h-4" />
//                 <span className="font-medium">{article.author}</span>
//               </div>
//               <div className="flex items-center gap-2">
//                 <Calendar className="w-4 h-4" />
//                 <span>{formattedDate}</span>
//               </div>
//               <div className="flex items-center gap-2">
//                 <Eye className="w-4 h-4" />
//                 <span>{article.source}</span>
//               </div>
//             </div>

//             {/* Featured Image */}
//             <div className="relative h-80 md:h-[450px] rounded-lg overflow-hidden shadow-lg border border-slate-200 dark:border-slate-700 bg-slate-200 dark:bg-slate-800">
//               <Image
//                 src={article.imageUrl || "/placeholder.svg"}
//                 alt={article.title}
//                 fill
//                 className="object-cover"
//               />
//             </div>

//             {/* Description */}
//             <div className="bg-white dark:bg-slate-800 rounded-lg p-6 shadow-md border border-slate-200 dark:border-slate-700">
//               <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed italic">
//                 {article.description}
//               </p>
//             </div>

//             {/* Content */}
//             <div className="bg-white dark:bg-slate-800 rounded-lg p-6 shadow-md border border-slate-200 dark:border-slate-700">
//               <div className="prose prose-lg dark:prose-invert max-w-none">
//                 <p className="text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-line">
//                   {article.content}
//                 </p>
//               </div>
//             </div>

//             {/* Summaries */}
//             {(article.single_source_summary || article.muti_source_summary) && (
//               <div className="space-y-4">
//                 {article.single_source_summary && (
//                   <div className="bg-blue-50 dark:bg-slate-800 rounded-lg p-6 shadow-md border-l-4 border-blue-600">
//                     <div className="flex items-center gap-3 mb-3">
//                       <FileText className="w-5 h-5 text-blue-600 dark:text-blue-500" />
//                       <h3 className="text-lg font-bold text-slate-900 dark:text-white">
//                         Single Source Summary
//                       </h3>
//                     </div>
//                     <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
//                       {article.single_source_summary}
//                     </p>
//                   </div>
//                 )}

//                 {article.muti_source_summary && (
//                   <div className="bg-purple-50 dark:bg-slate-800 rounded-lg p-6 shadow-md border-l-4 border-purple-600">
//                     <div className="flex items-center gap-3 mb-3">
//                       <FileText className="w-5 h-5 text-purple-600 dark:text-purple-500" />
//                       <h3 className="text-lg font-bold text-slate-900 dark:text-white">
//                         Multi-Source Summary
//                       </h3>
//                     </div>
//                     <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
//                       {article.muti_source_summary}
//                     </p>
//                   </div>
//                 )}
//               </div>
//             )}

//             {/* Comments Section */}
//             <div className="bg-white dark:bg-slate-800 rounded-lg p-6 shadow-md border border-slate-200 dark:border-slate-700">
//               <div className="flex items-center gap-3 mb-6">
//                 <MessageCircle className="w-5 h-5 text-blue-600 dark:text-blue-500" />
//                 <h3 className="text-xl font-bold text-slate-900 dark:text-white">
//                   Comments ({comments.length})
//                 </h3>
//               </div>

//               {/* Comment Form */}
//               <form onSubmit={handleCommentSubmit} className="mb-6">
//                 <div className="flex gap-3">
//                   <input
//                     type="text"
//                     value={commentText}
//                     onChange={(e) => setCommentText(e.target.value)}
//                     placeholder="Share your thoughts..."
//                     className="flex-1 px-4 py-3 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-white placeholder:text-slate-500 outline-none focus:ring-2 focus:ring-blue-500 border border-slate-200 dark:border-slate-600"
//                   />
//                   <button
//                     type="submit"
//                     className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors shadow-sm flex items-center gap-2"
//                   >
//                     <Send className="w-5 h-5" />
//                   </button>
//                 </div>
//               </form>

//               {/* Comments List */}
//               <div className="space-y-3">
//                 {comments.length === 0 ? (
//                   <p className="text-center text-slate-500 py-8 text-sm">
//                     No comments yet. Be the first to comment!
//                   </p>
//                 ) : (
//                   comments.map((comment) => (
//                     <div
//                       key={comment.id}
//                       className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border border-slate-200 dark:border-slate-600"
//                     >
//                       <div className="flex items-center gap-3 mb-2">
//                         <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
//                           {comment.userName.charAt(0)}
//                         </div>
//                         <div>
//                           <p className="font-semibold text-slate-900 dark:text-white text-sm">
//                             {comment.userName}
//                           </p>
//                           <p className="text-xs text-slate-500">
//                             {new Date(comment.createdAt).toLocaleDateString()}
//                           </p>
//                         </div>
//                       </div>
//                       <p className="text-slate-700 dark:text-slate-300 text-sm ml-13">
//                         {comment.content}
//                       </p>
//                     </div>
//                   ))
//                 )}
//               </div>
//             </div>
//           </div>

//           {/* Analysis Sidebar */}
//           <div className="space-y-6">
//             <div className="sticky top-24 space-y-6">
//               {/* Quick Stats */}
//               <div className="bg-white dark:bg-slate-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-slate-700">
//                 <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
//                   <Shield className="w-5 h-5 text-blue-600 dark:text-blue-500" />
//                   AI Analysis
//                 </h3>

//                 {/* Bias */}
//                 <div className="mb-5">
//                   <div className="flex items-center justify-between mb-2">
//                     <span className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide">
//                       Bias
//                     </span>
//                     <span
//                       className={`${getBiasColor(
//                         article.bias_classification_label
//                       )} text-white text-xs px-2.5 py-1 rounded font-bold`}
//                     >
//                       {article.bias_classification_label || "Neutral"}
//                     </span>
//                   </div>
//                   {article.bias_classification_probs && Object.keys(article.bias_classification_probs).length > 0 && (
//                     <div className="space-y-1.5 mt-3">
//                       {Object.entries(article.bias_classification_probs).map(([key, value]) => (
//                         <div key={key} className="flex items-center justify-between text-xs">
//                           <span className="text-slate-600 dark:text-slate-400 capitalize">{key}</span>
//                           <span className="text-slate-900 dark:text-white font-semibold">
//                             {((value as number) * 100).toFixed(1)}%
//                           </span>
//                         </div>
//                       ))}
//                     </div>
//                   )}
//                 </div>

//                 {/* Sentiment */}
//                 <div className="mb-5 pt-5 border-t border-slate-200 dark:border-slate-700">
//                   <div className="flex items-center justify-between mb-2">
//                     <span className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide">
//                       Sentiment
//                     </span>
//                     <span
//                       className={`${getSentimentColor(
//                         article.sentiment_analysis_label
//                       )} text-white text-xs px-2.5 py-1 rounded font-bold`}
//                     >
//                       {article.sentiment_analysis_label || "Neutral"}
//                     </span>
//                   </div>
//                   {article.sentiment_analysis_probs && Object.keys(article.sentiment_analysis_probs).length > 0 && (
//                     <div className="space-y-1.5 mt-3">
//                       {Object.entries(article.sentiment_analysis_probs).map(([key, value]) => (
//                         <div key={key} className="flex items-center justify-between text-xs">
//                           <span className="text-slate-600 dark:text-slate-400 capitalize">{key}</span>
//                           <span className="text-slate-900 dark:text-white font-semibold">
//                             {((value as number) * 100).toFixed(1)}%
//                           </span>
//                         </div>
//                       ))}
//                     </div>
//                   )}
//                 </div>

//                 {/* Clickbait */}
//                 {article.clickbait_label && (
//                   <div className="mb-6">
//                     <div className="flex items-center justify-between mb-2">
//                       <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
//                         Clickbait
//                       </span>
//                       <span
//                         className={`${
//                           article.clickbait_label === "clickbait"
//                             ? "bg-orange-500"
//                             : "bg-green-500"
//                         } text-white text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1`}
//                       >
//                         {article.clickbait_label === "clickbait" ? (
//                           <>
//                             <Zap className="w-3 h-3" />
//                             Yes
//                           </>
//                         ) : (
//                           <>
//                             <CheckCircle className="w-3 h-3" />
//                             No
//                           </>
//                         )}
//                       </span>
//                     </div>
//                     {article.clickbait_score !== null && (
//                       <p className="text-xs text-zinc-600 dark:text-zinc-400">
//                         Score: {(article.clickbait_score * 100).toFixed(1)}%
//                       </p>
//                     )}
//                     {article.clickbait_explanation && (
//                       <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2">
//                         {article.clickbait_explanation}
//                       </p>
//                     )}
//                   </div>
//                 )}

//                 {/* Reliability */}
//                 {article.source_reliability !== null && (
//                   <div className="mb-6">
//                     <div className="flex items-center justify-between mb-2">
//                       <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
//                         Source Reliability
//                       </span>
//                       <span className="text-zinc-900 dark:text-white font-bold">
//                         {(article.source_reliability * 100).toFixed(0)}%
//                       </span>
//                     </div>
//                     <div className="h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
//                       <div
//                         className={`h-full ${
//                           article.source_reliability >= 0.8
//                             ? "bg-green-500"
//                             : article.source_reliability >= 0.5
//                             ? "bg-yellow-500"
//                             : "bg-red-500"
//                         }`}
//                         style={{ width: `${article.source_reliability * 100}%` }}
//                       ></div>
//                     </div>
//                   </div>
//                 )}

//                 {/* Fake News */}
//                 {article.fake_news_label && (
//                   <div>
//                     <div className="flex items-center justify-between mb-2">
//                       <span className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
//                         Authenticity
//                       </span>
//                       <span
//                         className={`${
//                           article.fake_news_label.toLowerCase().includes("fake")
//                             ? "bg-red-500"
//                             : "bg-green-500"
//                         } text-white text-xs px-3 py-1 rounded-full font-bold`}
//                       >
//                         {article.fake_news_label}
//                       </span>
//                     </div>
//                     {article.fake_news_probs && Object.keys(article.fake_news_probs).length > 0 && (
//                       <div className="space-y-1">
//                         {Object.entries(article.fake_news_probs).map(([key, value]) => (
//                           <div key={key} className="flex items-center justify-between text-xs">
//                             <span className="text-zinc-600 dark:text-zinc-400 capitalize">{key}</span>
//                             <span className="text-zinc-900 dark:text-white font-semibold">
//                               {((value as number) * 100).toFixed(1)}%
//                             </span>
//                           </div>
//                         ))}
//                       </div>
//                     )}
//                   </div>
//                 )}
//               </div>

//               {/* Bias Explanation */}
//               {article.bias_explain && article.bias_explain.length > 0 && (
//                 <div className="bg-white dark:bg-slate-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-slate-700">
//                   <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
//                     <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-500" />
//                     Bias Indicators
//                   </h3>
//                   <div className="space-y-2.5">
//                     {article.bias_explain.slice(0, 5).map((item, index) => (
//                       <div
//                         key={index}
//                         className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800/50"
//                       >
//                         <p className="text-sm font-semibold text-slate-900 dark:text-white mb-1.5">
//                           "{item.phrase}"
//                         </p>
//                         <div className="flex items-center gap-3 text-xs text-slate-600 dark:text-slate-400">
//                           <span>Score: {item.score.toFixed(2)}</span>
//                           <span>Weight: {item.weight.toFixed(2)}</span>
//                         </div>
//                       </div>
//                     ))}
//                   </div>
//                 </div>
//               )}

//               {/* Omitted Facts */}
//               {article.omitted_chunks && article.omitted_chunks.length > 0 && (
//                 <div className="bg-white dark:bg-slate-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-slate-700">
//                   <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
//                     <Eye className="w-5 h-5 text-cyan-600 dark:text-cyan-500" />
//                     Cross-Reference Check
//                   </h3>
//                   <div className="space-y-2.5">
//                     {article.omitted_chunks.slice(0, 3).map((item, index) => (
//                       <a
//                         key={index}
//                         href={item.url}
//                         target="_blank"
//                         rel="noopener noreferrer"
//                         className="block p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border border-cyan-200 dark:border-cyan-800/50 hover:bg-cyan-100 dark:hover:bg-cyan-900/30 transition-colors"
//                       >
//                         <p className="text-sm font-semibold text-slate-900 dark:text-white mb-1.5 line-clamp-2">
//                           {item.title}
//                         </p>
//                         {item.omitted_segments && item.omitted_segments.length > 0 && (
//                           <p className="text-xs text-slate-600 dark:text-slate-400">
//                             {item.omitted_segments.length} omitted segment(s)
//                           </p>
//                         )}
//                       </a>
//                     ))}
//                   </div>
//                 </div>
//               )}

//               {/* Original Source Link */}
//               <a
//                 href={article.url}
//                 target="_blank"
//                 rel="noopener noreferrer"
//                 className="block w-full px-6 py-4 bg-blue-600 text-white rounded-lg font-semibold text-center hover:bg-blue-700 transition-colors shadow-sm"
//               >
//                 View Original Article →
//               </a>
//             </div>
//           </div>
//         </div>

//         {/* Related Articles */}
//         {relatedArticles.length > 0 && (
//           <section className="mt-20">
//             <div className="mb-10">
//               <h2 className="text-4xl font-black text-zinc-900 dark:text-white mb-2">
//                 Related Articles
//               </h2>
//               <p className="text-lg text-zinc-600 dark:text-zinc-400">
//                 More stories from {article?.category}
//               </p>
//             </div>
//             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
//               {relatedArticles.map((relatedArticle) => {
//                 const formattedDate = new Date(relatedArticle.publishedAt).toLocaleDateString("en-US", {
//                   month: "short",
//                   day: "numeric",
//                   year: "numeric",
//                 });

//                 const getBiasColor = (label: string | null) => {
//                   if (!label) return "bg-zinc-500";
//                   const lower = label.toLowerCase();
//                   if (lower.includes("left")) return "bg-blue-500";
//                   if (lower.includes("right")) return "bg-red-500";
//                   return "bg-green-500";
//                 };

//                 return (
//                   <Link
//                     key={relatedArticle.id || relatedArticle._id?.toString()}
//                     href={`/article/${relatedArticle.id || relatedArticle._id}`}
//                     className="block group"
//                   >
//                     <div className="bg-white dark:bg-zinc-900 rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 border border-zinc-200 dark:border-zinc-800 hover:border-purple-500 dark:hover:border-purple-500 h-full">
//                       <div className="relative h-48 overflow-hidden">
//                         <Image
//                           src={relatedArticle.imageUrl || "/placeholder.svg"}
//                           alt={relatedArticle.title}
//                           fill
//                           className="object-cover group-hover:scale-110 transition-transform duration-300"
//                         />
//                         <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
//                         <div className="absolute bottom-3 left-3">
//                           <span
//                             className={`${getBiasColor(relatedArticle.bias_classification_label)} text-white text-xs px-3 py-1 rounded-full font-bold`}
//                           >
//                             {relatedArticle.bias_classification_label || "Neutral"}
//                           </span>
//                         </div>
//                       </div>
//                       <div className="p-5">
//                         <h3 className="text-lg font-bold text-zinc-900 dark:text-white line-clamp-2 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors mb-2">
//                           {relatedArticle.title}
//                         </h3>
//                         <p className="text-sm text-zinc-600 dark:text-zinc-400 line-clamp-2 mb-3">
//                           {relatedArticle.description}
//                         </p>
//                         <div className="flex items-center gap-4 text-xs text-zinc-500">
//                           <div className="flex items-center gap-1">
//                             <Calendar className="w-3.5 h-3.5" />
//                             {formattedDate}
//                           </div>
//                           <div className="flex items-center gap-1">
//                             <User className="w-3.5 h-3.5" />
//                             {relatedArticle.author}
//                           </div>
//                         </div>
//                       </div>
//                     </div>
//                   </Link>
//                 );
//               })}
//             </div>
//           </section>
//         )}
//       </main>

//       {/* Footer */}
//       <footer className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 mt-20">
//         <div className="max-w-7xl mx-auto px-6 py-10">
//           <div className="flex flex-col md:flex-row items-center justify-between gap-4">
//             <div className="flex items-center gap-3">
//               <Newspaper className="w-6 h-6 text-blue-600 dark:text-blue-500" />
//               <span className="text-lg font-bold text-slate-900 dark:text-white">
//                 Polaris News
//               </span>
//             </div>
//             <p className="text-slate-600 dark:text-slate-400 text-sm">
//               © 2025 Polaris News. Professional news intelligence platform.
//             </p>
//           </div>
//         </div>
//       </footer>
//     </div>
//   );
// }

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
  ExternalLink,
  Clock,
  BarChart3,
  Sparkles,
} from "lucide-react";
import { NewsSidebar } from "@/components/NewsSidebar";
import { motion } from "motion/react";

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
  const [animateBars, setAnimateBars] = useState(false);

  useEffect(() => {
    async function fetchArticle() {
      try {
        // Reset animation state when fetching new article
        setAnimateBars(false);

        const response = await fetch(`/api/articles?id=${id}`);
        const data = await response.json();

        if (data.article) {
          setArticle(data.article);
          setLocalLikes(data.article.likes || 0);
          setComments(data.article.comments || []);

          if (data.article.category) {
            const relatedResponse = await fetch(
              `/api/articles?category=${encodeURIComponent(
                data.article.category
              )}&limit=3`
            );
            const relatedData = await relatedResponse.json();
            setRelatedArticles(
              (relatedData.articles || [])
                .filter((a: ArticleDoc) => a.id !== data.article.id)
                .slice(0, 3)
            );
          }

          // Trigger animation after a short delay
          setTimeout(() => {
            setAnimateBars(true);
          }, 300);
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

    const tempComment = {
      id: Date.now().toString(),
      ...commentData,
      createdAt: new Date(),
      likes: 0,
    };

    setComments([tempComment, ...comments]);
    setCommentText("");

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
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-400 font-medium tracking-wide">
            Loading article...
          </p>
        </div>
      </div>
    );
  }

  if (!article) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 rounded-2xl bg-[#e23e57]/10 flex items-center justify-center mx-auto mb-6 border border-[#e23e57]/20">
            <XCircle className="w-10 h-10 text-[#e23e57]" />
          </div>
          <h1 className="text-2xl font-bold text-[#e8eef7] mb-2">
            Article not found
          </h1>
          <p className="text-[#94a3b8] mb-6">
            The article you're looking for doesn't exist
          </p>
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-6 py-3 bg-[#3a7cf6] text-white rounded-xl font-medium hover:bg-[#2047b8] transition-all"
          >
            <ArrowLeft className="w-4 h-4" />
            Return Home
          </Link>
        </div>
      </div>
    );
  }

  const formattedDate = new Date(article.publishedAt).toLocaleDateString(
    "en-US",
    {
      month: "long",
      day: "numeric",
      year: "numeric",
    }
  );

  const formattedTime = new Date(article.publishedAt).toLocaleTimeString(
    "en-US",
    {
      hour: "2-digit",
      minute: "2-digit",
    }
  );

  const getBiasColor = (label: string | null) => {
    if (!label) return "bg-[#94a3b8]";
    const lower = label.toLowerCase();
    if (lower.includes("left")) return "bg-[#3a7cf6]";
    if (lower.includes("right")) return "bg-[#e23e57]";
    return "bg-[#16a34a]";
  };

  const getSentimentColor = (label: string | null) => {
    if (!label) return "bg-[#94a3b8]";
    const lower = label.toLowerCase();
    if (lower.includes("positive")) return "bg-[#16a34a]";
    if (lower.includes("negative")) return "bg-[#e23e57]";
    return "bg-[#f4b227]";
  };

  return (
    <NewsSidebar>
      <div className="min-h-screen bg-black">
        {/* Header */}
        <header className="sticky top-0 z-50 bg-black/95 backdrop-blur-xl border-b border-zinc-800">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-8">
                <Link
                  href="/"
                  className="flex items-center gap-2 text-[#94a3b8] hover:text-[#3a7cf6] transition-colors group"
                >
                  <div className="p-2 rounded-xl bg-white/5 group-hover:bg-[#3a7cf6]/10 transition-colors border border-white/5">
                    <ArrowLeft className="w-4 h-4" />
                  </div>
                  <span className="font-medium text-sm">Back</span>
                </Link>
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-[#3a7cf6] flex items-center justify-center shadow-lg shadow-[#3a7cf6]/20">
                    <Sparkles className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-lg font-bold text-[#e8eef7]">
                    Polaris News
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleLike}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
                    liked
                      ? "bg-[#e23e57] text-white shadow-lg shadow-[#e23e57]/30"
                      : "bg-white/5 text-[#94a3b8] hover:bg-[#e23e57]/10 hover:text-[#e23e57] border border-white/5"
                  }`}
                >
                  <Heart className={`w-4 h-4 ${liked ? "fill-current" : ""}`} />
                  <span className="text-sm">{localLikes}</span>
                </button>
                <button
                  onClick={handleShare}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl font-medium bg-white/5 text-[#94a3b8] hover:bg-[#3a7cf6]/10 hover:text-[#3a7cf6] transition-all border border-white/5"
                >
                  <Share2 className="w-4 h-4" />
                  <span className="text-sm">Share</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Article Content */}
        <main className="max-w-7xl mx-auto px-6 py-12">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
            {/* Main Content */}
            <div className="lg:col-span-8 space-y-8">
              {/* Category & Metadata */}
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-3">
                  <span className="px-4 py-1.5 bg-[#3a7cf6] text-white rounded-full text-sm font-semibold shadow-lg shadow-[#3a7cf6]/30">
                    {article.category}
                  </span>
                  {article.topic && (
                    <span className="px-4 py-1.5 bg-white/5 text-[#94a3b8] rounded-full text-sm font-medium border border-white/5">
                      {article.topic}
                    </span>
                  )}
                </div>
              </div>

              {/* Title */}
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-[#e8eef7] leading-tight tracking-tight">
                {article.title}
              </h1>

              {/* Meta Info */}
              <div className="flex items-center gap-6 text-[#94a3b8] flex-wrap">
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-full bg-[#3a7cf6] flex items-center justify-center text-white font-bold text-sm">
                    {article.author.charAt(0)}
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-[#e8eef7]">
                      {article.author}
                    </p>
                    <p className="text-xs text-[#94a3b8]">{article.source}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-1.5">
                    <Calendar className="w-4 h-4" />
                    <span>{formattedDate}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-4 h-4" />
                    <span>{formattedTime}</span>
                  </div>
                </div>
              </div>

              {/* Featured Image */}
              <div className="relative h-96 md:h-[500px] rounded-2xl overflow-hidden shadow-2xl border border-zinc-800 bg-zinc-900">
                <Image
                  src={article.imageUrl || "/placeholder.svg"}
                  alt={article.title}
                  fill
                  className="object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent"></div>
              </div>

              {/* Description */}
              <div className="relative">
                <div className="absolute -left-4 top-0 w-1 h-full bg-gradient-to-b from-[#3a7cf6] to-[#22b7e3] rounded-full"></div>
                <p className="text-xl text-[#94a3b8] leading-relaxed pl-8 font-medium italic">
                  {article.description}
                </p>
              </div>

            {/* Content */}
            <div className="prose prose-lg prose-invert max-w-none">
              <div className="text-[#94a3b8] leading-relaxed text-lg space-y-6 
                              first-letter:text-5xl first-letter:font-bold 
                              first-letter:text-[#3a7cf6] first-letter:float-left 
                              first-letter:mr-3 first-letter:leading-none first-letter:mt-1">
                {article.content.split('\n\n').map((paragraph, idx) => (
                  <p key={idx}>{paragraph}</p>
                ))}
              </div>
            </div>

              {/* Summaries */}
              {(article.single_source_summary ||
                article.muti_source_summary) && (
                <div className="space-y-6 pt-8 border-t border-white/5">
                  <h2 className="text-2xl font-bold text-[#e8eef7] flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-[#3a7cf6] flex items-center justify-center">
                      <BarChart3 className="w-4 h-4 text-white" />
                    </div>
                    AI-Generated Summaries
                  </h2>

                  {article.single_source_summary && (
                    <div className="relative p-6 rounded-2xl bg-[#3a7cf6]/5 border border-[#3a7cf6]/20">
                      <div className="flex items-center gap-2 mb-4">
                        <FileText className="w-5 h-5 text-[#3a7cf6]" />
                        <h3 className="text-lg font-bold text-[#e8eef7]">
                          Single Source Analysis
                        </h3>
                      </div>
                      <p className="text-[#94a3b8] leading-relaxed">
                        {article.single_source_summary}
                      </p>
                    </div>
                  )}

                  {article.muti_source_summary && (
                    <div className="relative p-6 rounded-2xl bg-[#22b7e3]/5 border border-[#22b7e3]/20">
                      <div className="flex items-center gap-2 mb-4">
                        <FileText className="w-5 h-5 text-[#22b7e3]" />
                        <h3 className="text-lg font-bold text-[#e8eef7]">
                          Cross-Source Analysis
                        </h3>
                      </div>
                      <p className="text-[#94a3b8] leading-relaxed">
                        {article.muti_source_summary}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Comments Section */}
              <div className="pt-8 border-t border-white/5">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-8 h-8 rounded-xl bg-[#16a34a] flex items-center justify-center">
                    <MessageCircle className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-[#e8eef7]">
                    Discussion
                  </h3>
                  <span className="px-3 py-1 bg-white/5 text-[#94a3b8] rounded-full text-sm font-medium">
                    {comments.length}
                  </span>
                </div>

                {/* Comment Form */}
                <form onSubmit={handleCommentSubmit} className="mb-8">
                  <div className="flex gap-3">
                    <input
                      type="text"
                      value={commentText}
                      onChange={(e) => setCommentText(e.target.value)}
                      placeholder="Share your perspective..."
                      className="flex-1 px-5 py-3 rounded-xl bg-white/5 text-[#e8eef7] placeholder:text-[#94a3b8] outline-none focus:ring-2 focus:ring-[#3a7cf6] border border-white/5 transition-all"
                    />
                    <button
                      type="submit"
                      className="px-6 py-3 bg-[#3a7cf6] text-white rounded-xl font-medium hover:bg-[#2563eb] hover:shadow-lg hover:shadow-[#3a7cf6]/30 transition-all flex items-center gap-2"
                    >
                      <Send className="w-4 h-4" />
                    </button>
                  </div>
                </form>

                {/* Comments List */}
                <div className="space-y-4">
                  {comments.length === 0 ? (
                    <div className="text-center py-12 bg-white/5 rounded-xl border border-dashed border-white/10">
                      <MessageCircle className="w-12 h-12 text-[#94a3b8]/30 mx-auto mb-3" />
                      <p className="text-[#94a3b8] text-sm">
                        Be the first to share your thoughts
                      </p>
                    </div>
                  ) : (
                    comments.map((comment) => (
                      <div
                        key={comment.id}
                        className="p-5 bg-white/5 rounded-xl border border-white/5 hover:border-[#3a7cf6]/30 transition-all"
                      >
                        <div className="flex items-start gap-4">
                          <div className="w-10 h-10 rounded-full bg-[#3a7cf6] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                            {comment.userName.charAt(0)}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <p className="font-semibold text-[#e8eef7] text-sm">
                                {comment.userName}
                              </p>
                              <span className="text-xs text-[#94a3b8]">•</span>
                              <p className="text-xs text-[#94a3b8]">
                                {new Date(
                                  comment.createdAt
                                ).toLocaleDateString()}
                              </p>
                            </div>
                            <p className="text-[#94a3b8] text-sm leading-relaxed">
                              {comment.content}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Sidebar */}
            <div className="lg:col-span-4 space-y-6">
              <div className="sticky top-24 space-y-6">
                {/* AI Analysis Card */}
                <div className="bg-white/5 rounded-2xl p-6 shadow-xl border border-white/10">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 rounded-xl bg-[#3a7cf6] flex items-center justify-center">
                      <Shield className="w-5 h-5 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-[#e8eef7]">
                      AI Analysis
                    </h3>
                  </div>

                  {/* Bias Analysis */}
                  <div className="mb-6 pb-6 border-b border-white/5">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-bold text-[#94a3b8] uppercase tracking-wider">
                        Political Bias
                      </span>
                      <span
                        className={`${getBiasColor(
                          article.bias_classification_label
                        )} text-white text-xs px-3 py-1 rounded-full font-bold shadow-lg`}
                      >
                        {article.bias_classification_label || "Neutral"}
                      </span>
                    </div>
                    {article.bias_classification_probs &&
                      Object.keys(article.bias_classification_probs).length >
                        0 && (
                        <div className="space-y-2 mt-4">
                          {Object.entries(
                            article.bias_classification_probs
                          ).map(([key, value], index) => (
                            <div key={key}>
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-[#94a3b8] capitalize font-medium">
                                  {key}
                                </span>
                                <span className="text-[#e8eef7] font-bold">
                                  {((value as number) * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                <motion.div
                                  className="h-full bg-gradient-to-r from-[#3a7cf6] to-[#22b7e3] rounded-full"
                                  initial={{ width: 0 }}
                                  animate={{
                                    width: animateBars
                                      ? `${(value as number) * 100}%`
                                      : 0,
                                  }}
                                  transition={{
                                    duration: 0.8,
                                    delay: index * 0.1,
                                    ease: "easeOut",
                                  }}
                                ></motion.div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                  </div>

                  {/* Sentiment Analysis */}
                  <div className="mb-6 pb-6 border-b border-white/5">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-bold text-[#94a3b8] uppercase tracking-wider">
                        Sentiment
                      </span>
                      <span
                        className={`${getSentimentColor(
                          article.sentiment_analysis_label
                        )} text-white text-xs px-3 py-1 rounded-full font-bold shadow-lg`}
                      >
                        {article.sentiment_analysis_label || "Neutral"}
                      </span>
                    </div>
                    {article.sentiment_analysis_probs &&
                      Object.keys(article.sentiment_analysis_probs).length >
                        0 && (
                        <div className="space-y-2 mt-4">
                          {Object.entries(article.sentiment_analysis_probs).map(
                            ([key, value], index) => (
                              <div key={key}>
                                <div className="flex items-center justify-between text-xs mb-1">
                                  <span className="text-[#94a3b8] capitalize font-medium">
                                    {key}
                                  </span>
                                  <span className="text-[#e8eef7] font-bold">
                                    {((value as number) * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                  <motion.div
                                    className="h-full bg-gradient-to-r from-[#16a34a] to-[#22b7e3] rounded-full"
                                    initial={{ width: 0 }}
                                    animate={{
                                      width: animateBars
                                        ? `${(value as number) * 100}%`
                                        : 0,
                                    }}
                                    transition={{
                                      duration: 0.8,
                                      delay: index * 0.1,
                                      ease: "easeOut",
                                    }}
                                  ></motion.div>
                                </div>
                              </div>
                            )
                          )}
                        </div>
                      )}
                  </div>

                  {/* Clickbait Detection */}
                  {article.clickbait_label && (
                    <div className="mb-6 pb-6 border-b border-white/5">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-bold text-[#94a3b8] uppercase tracking-wider">
                          Clickbait Score
                        </span>
                        <span
                          className={`${
                            article.clickbait_label === "clickbait"
                              ? "bg-gradient-to-r from-[#f4b227] to-[#e23e57]"
                              : "bg-gradient-to-r from-[#16a34a] to-[#22b7e3]"
                          } text-white text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1.5 shadow-lg`}
                        >
                          {article.clickbait_label === "clickbait" ? (
                            <>
                              <Zap className="w-3 h-3" />
                              High
                            </>
                          ) : (
                            <>
                              <CheckCircle className="w-3 h-3" />
                              Low
                            </>
                          )}
                        </span>
                      </div>
                      {article.clickbait_score !== null && (
                        <div className="mt-3">
                          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                            <motion.div
                              className={`h-full ${
                                article.clickbait_score > 0.7
                                  ? "bg-gradient-to-r from-[#e23e57] to-[#f4b227]"
                                  : article.clickbait_score > 0.4
                                  ? "bg-gradient-to-r from-[#f4b227] to-[#22b7e3]"
                                  : "bg-gradient-to-r from-[#16a34a] to-[#22b7e3]"
                              } rounded-full`}
                              initial={{ width: 0 }}
                              animate={{
                                width: animateBars
                                  ? `${article.clickbait_score * 100}%`
                                  : 0,
                              }}
                              transition={{
                                duration: 0.8,
                                delay: 0.3,
                                ease: "easeOut",
                              }}
                            ></motion.div>
                          </div>
                          <p className="text-xs text-[#94a3b8] mt-2 text-center">
                            {(article.clickbait_score * 100).toFixed(0)}%
                            confidence
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Source Reliability */}
                  {article.source_reliability !== null && (
                    <div className="mb-6 pb-6 border-b border-white/5">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs font-bold text-[#94a3b8] uppercase tracking-wider">
                          Source Reliability
                        </span>
                        <span className="text-[#e8eef7] font-bold text-lg">
                          {(article.source_reliability * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                          className={`h-full ${
                            article.source_reliability >= 0.8
                              ? "bg-gradient-to-r from-[#16a34a] to-[#22b7e3]"
                              : article.source_reliability >= 0.5
                              ? "bg-gradient-to-r from-[#f4b227] to-[#22b7e3]"
                              : "bg-gradient-to-r from-[#e23e57] to-[#f4b227]"
                          } rounded-full`}
                          initial={{ width: 0 }}
                          animate={{
                            width: animateBars
                              ? `${article.source_reliability * 100}%`
                              : 0,
                          }}
                          transition={{
                            duration: 0.8,
                            delay: 0.2,
                            ease: "easeOut",
                          }}
                        ></motion.div>
                      </div>
                    </div>
                  )}

                  {/* Authenticity Check */}
                  {article.fake_news_label && (
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs font-bold text-[#94a3b8] uppercase tracking-wider">
                          Authenticity
                        </span>
                        <span
                          className={`${
                            article.fake_news_label
                              .toLowerCase()
                              .includes("fake")
                              ? "bg-gradient-to-r from-[#e23e57] to-[#f4b227]"
                              : "bg-gradient-to-r from-[#16a34a] to-[#22b7e3]"
                          } text-white text-xs px-3 py-1 rounded-full font-bold shadow-lg`}
                        >
                          {article.fake_news_label}
                        </span>
                      </div>
                      {article.fake_news_probs &&
                        Object.keys(article.fake_news_probs).length > 0 && (
                          <div className="space-y-2 mt-3">
                            {Object.entries(article.fake_news_probs).map(
                              ([key, value], index) => (
                                <div key={key}>
                                  <div className="flex items-center justify-between text-xs mb-1">
                                    <span className="text-[#94a3b8] capitalize font-medium">
                                      {key}
                                    </span>
                                    <span className="text-[#e8eef7] font-bold">
                                      {((value as number) * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                    <motion.div
                                      className="h-full bg-gradient-to-r from-[#3a7cf6] to-[#22b7e3] rounded-full"
                                      initial={{ width: 0 }}
                                      animate={{
                                        width: animateBars
                                          ? `${(value as number) * 100}%`
                                          : 0,
                                      }}
                                      transition={{
                                        duration: 0.8,
                                        delay: index * 0.1,
                                        ease: "easeOut",
                                      }}
                                    ></motion.div>
                                  </div>
                                </div>
                              )
                            )}
                          </div>
                        )}
                    </div>
                  )}
                </div>

                {/* Bias Indicators */}
                {article.bias_explain && article.bias_explain.length > 0 && (
                  <div className="bg-[#f4b227]/5 rounded-2xl p-6 shadow-xl border border-[#f4b227]/20">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="w-10 h-10 rounded-xl bg-[#f4b227] flex items-center justify-center">
                        <AlertTriangle className="w-5 h-5 text-white" />
                      </div>
                      <h3 className="text-lg font-bold text-[#e8eef7]">
                        Bias Indicators
                      </h3>
                    </div>
                    <div className="space-y-3">
                      {article.bias_explain.slice(0, 5).map((item, index) => (
                        <div
                          key={index}
                          className="p-4 bg-white/5 rounded-xl border border-white/10"
                        >
                          <p className="text-sm font-semibold text-[#e8eef7] mb-2">
                            "{item.phrase}"
                          </p>
                          <div className="flex items-center gap-4 text-xs">
                            <div className="flex items-center gap-1.5">
                              <span className="text-[#94a3b8]">Score:</span>
                              <span className="text-[#f4b227] font-bold">
                                {item.score.toFixed(2)}
                              </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              <span className="text-[#94a3b8]">Weight:</span>
                              <span className="text-[#f4b227] font-bold">
                                {item.weight.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Cross-Reference - Omitted Chunks */}
                {article.omitted_chunks &&
                  article.omitted_chunks.length > 0 && (
                    <div className="bg-[#22b7e3]/5 rounded-2xl p-6 shadow-xl border border-[#22b7e3]/20">
                      <div className="flex items-center gap-3 mb-6">
                        <div className="w-10 h-10 rounded-xl bg-[#22b7e3] flex items-center justify-center">
                          <Eye className="w-5 h-5 text-white" />
                        </div>
                        <h3 className="text-lg font-bold text-[#e8eef7]">
                          Omitted Chunks
                        </h3>
                      </div>
                      <div className="space-y-3">
                        {article.omitted_chunks
                          .slice(0, 3)
                          .map((chunk, index) => (
                            <div
                              key={index}
                              className="block p-4 bg-white/5 rounded-xl border border-white/10 hover:border-[#22b7e3]/50 transition-all group"
                            >
                              <p className="text-sm text-[#e8eef7] line-clamp-3 group-hover:text-[#22b7e3] transition-colors">
                                {chunk}
                              </p>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                {/* Omitted Summary */}
                {article.omitted_summary && (
                  <div className="bg-[#f4b227]/5 rounded-2xl p-6 shadow-xl border border-[#f4b227]/20">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-10 h-10 rounded-xl bg-[#f4b227] flex items-center justify-center">
                        <FileText className="w-5 h-5 text-white" />
                      </div>
                      <h3 className="text-lg font-bold text-[#e8eef7]">
                        Omitted Facts Summary
                      </h3>
                    </div>
                    <p className="text-[#94a3b8] leading-relaxed">
                      {article.omitted_summary}
                    </p>
                  </div>
                )}

                {/* View Original */}
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full px-6 py-4 bg-[#3a7cf6] text-white rounded-xl font-semibold text-center hover:bg-[#2563eb] hover:shadow-xl hover:shadow-[#3a7cf6]/30 transition-all group"
                >
                  <span className="flex items-center justify-center gap-2">
                    View Original Article
                    <ExternalLink className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </span>
                </a>
              </div>
            </div>
          </div>

          {/* Related Articles */}
          {relatedArticles.length > 0 && (
            <section className="mt-24">
              <div className="mb-12">
                <div className="flex items-center gap-4 mb-3">
                  <div className="w-12 h-12 rounded-xl bg-[#3a7cf6] flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold text-[#e8eef7]">
                      Related Articles
                    </h2>
                    <p className="text-[#94a3b8]">
                      More from {article?.category}
                    </p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {relatedArticles.map((relatedArticle) => {
                  const formattedDate = new Date(
                    relatedArticle.publishedAt
                  ).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  });

                  const getBiasColor = (label: string | null) => {
                    if (!label) return "bg-[#94a3b8]";
                    const lower = label.toLowerCase();
                    if (lower.includes("left")) return "bg-[#3a7cf6]";
                    if (lower.includes("right")) return "bg-[#e23e57]";
                    return "bg-[#16a34a]";
                  };

                  return (
                    <Link
                      key={
                        relatedArticle.id ||
                        (relatedArticle as any)._id?.toString()
                      }
                      href={`/article/${
                        relatedArticle.id || (relatedArticle as any)._id
                      }`}
                      className="block group"
                    >
                      <div className="bg-zinc-900 rounded-2xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 border border-zinc-800 hover:border-blue-500/30 h-full">
                        <div className="relative h-52 overflow-hidden">
                          <Image
                            src={relatedArticle.imageUrl || "/placeholder.svg"}
                            alt={relatedArticle.title}
                            fill
                            className="object-cover group-hover:scale-110 transition-transform duration-500"
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-black via-black/20 to-transparent"></div>
                          <div className="absolute bottom-4 left-4">
                            <span
                              className={`${getBiasColor(
                                relatedArticle.bias_classification_label
                              )} text-white text-xs px-3 py-1.5 rounded-full font-bold shadow-lg`}
                            >
                              {relatedArticle.bias_classification_label ||
                                "Neutral"}
                            </span>
                          </div>
                        </div>
                        <div className="p-6">
                          <h3 className="text-lg font-bold text-[#e8eef7] line-clamp-2 group-hover:text-[#3a7cf6] transition-colors mb-3 leading-snug">
                            {relatedArticle.title}
                          </h3>
                          <p className="text-sm text-[#94a3b8] line-clamp-2 mb-4 leading-relaxed">
                            {relatedArticle.description}
                          </p>
                          <div className="flex items-center gap-4 text-xs text-[#94a3b8]">
                            <div className="flex items-center gap-1.5">
                              <Calendar className="w-3.5 h-3.5" />
                              {formattedDate}
                            </div>
                            <div className="flex items-center gap-1.5">
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
        <footer className="bg-zinc-900 border-t border-zinc-800 mt-24">
          <div className="max-w-7xl mx-auto px-6 py-10">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <Newspaper className="w-6 h-6 text-blue-400" />
                <span className="text-lg font-bold text-gray-100">
                  Polaris News
                </span>
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
