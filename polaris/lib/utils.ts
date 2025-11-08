import { NextResponse } from "next/server";

export function mapToObj(m: any): Record<string, number> | null {
  if (!m) return null;
  if (m instanceof Map) return Object.fromEntries(m);
  // Mongoose Map sometimes comes as plain object already
  if (typeof m === "object") return m as Record<string, number>;
  return null;
}

export function formatArticle(article: any, includeMongoId = false) {
  const base: any = {
    id: article.id,
    title: article.title,
    description: article.description,
    content: article.content,
    author: article.author,
    source: article.source,
    url: article.url,
    imageUrl: article.imageUrl,
    publishedAt: article.publishedAt ? new Date(article.publishedAt).toISOString() : null,
    category: article.category,
    topics: article.topics,
    likes: article.likes,
    comments: article.comments,

    bias_classification_label: article.bias_classification_label,
    bias_classification_probs: mapToObj(article.bias_classification_probs),
    bias_explain: article.bias_explain,

    sentiment_analysis_label: article.sentiment_analysis_label,
    sentiment_analysis_probs: mapToObj(article.sentiment_analysis_probs),

    clickbait_label: article.clickbait_label,
    clickbait_score: article.clickbait_score,
    clickbait_explanation: article.clickbait_explanation,

    topic: article.topic,
    omitted_facts_articles: article.omitted_facts_articles,

    fake_news_label: article.fake_news_label,
    fake_news_probs: mapToObj(article.fake_news_probs),
    source_reliability: article.source_reliability,

    muti_source_summary: article.muti_source_summary,
    single_source_summary: article.single_source_summary,
  };

  if (includeMongoId) base._id = article._id?.toString?.() ?? article._id;
  return base;
}

export function jsonOK(data: any, init?: ResponseInit) {
  return NextResponse.json(data, {
    ...init,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
      ...(init?.headers || {}),
    },
  });
}

export function jsonError(message: string, status = 500) {
  return jsonOK({ error: message }, { status });
}
