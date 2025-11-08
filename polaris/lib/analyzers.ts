// lib/analyzers.ts
import { postJSON } from "./http";
import { cleanText } from "./text";
import { Article, ArticleDoc } from "@/models/articles";

const SCORE_URL = process.env.SCORE_URL || "http://10.145.124.158:8001/v1/score_article";
const BIAS_PREDICT_URL = process.env.BIAS_PREDICT_URL || "http://10.145.89.82:9000/predict";
const BIAS_EXPLAIN_URL = process.env.BIAS_EXPLAIN_URL || "http://10.145.89.82:9000/explain";
const SENTIMENT_URL_PRIMARY = process.env.SENTIMENT_URL_PRIMARY || "http://10.145.89.82:8080/predict";
const SENTIMENT_URL_FALLBACK = process.env.SENTIMENT_URL_FALLBACK || "http://10.145.89.82:8080/predict";
const CLICKBAIT_URL = process.env.CLICKBAIT_URL || "http://10.145.35.209:8003/analyze";
const OMITTED_URL = process.env.OMITTED_URL || "http://10.145.35.209:8004/detect_omitted_facts";
const SUMMARISE_URL = process.env.SUMMARISE_URL || "http://10.145.124.158:8000/v1/summarise_topics";

type ScoreArticleResp = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  source?: { domain?: string; reliability_0_100?: number };
};
type BiasPredictResp = { probabilities: Record<string, number> };
type BiasExplainResp = {
  predicted_label: string;
  predicted_prob: number;
  top_explanations: { phrase: string; score: number; weight: number }[];
};
type SentimentResp = { probabilities: Record<"negative" | "neutral" | "positive", number> };
type ClickbaitResp = { score: number; label: string; explanation?: string };
type OmittedResp = {
  topic: string;
  articles: {
    title: string;
    url: string;
    omitted_segments: { chunk: string; max_similarity: number }[];
  }[];
};
type SummariseReqArticle = {
  title: string;
  text: string;
  subject?: string | null;
  date?: string | null;
  source_url?: string | null;
};
type SummariseResp = {
  groups: { cluster_id: number; indices: number[]; titles: string[]; digest: string }[];
  per_article_summary: string[];
  main_article_group?: number | null;
};

function argmax(prob: Record<string, number> | Map<string, number>): string | null {
  const entries = prob instanceof Map ? Array.from(prob.entries()) : Object.entries(prob);
  if (!entries.length) return null;
  return entries.reduce((best, cur) => (cur[1] > best[1] ? cur : best))[0];
}

export async function upsertBaseArticle(input: {
  title: string;
  content: string;
  url: string;
  author?: string | null;
  source?: string | null;
  imageUrl?: string | null;
  publishedAt?: string | Date | null;
  category?: string | null;
  topics?: string[];
  topic?: string | null;
  description?: string | null;
}) {
  const doc = await Article.findOneAndUpdate(
    { url: input.url },
    {
      $setOnInsert: {
        url: input.url,
      },
      $set: {
        title: cleanText(input.title) || "Untitled News",
        content: cleanText(input.content) || "No content available.",
        description: cleanText(input.description || "") || "No description available.",
        author: input.author || "Unknown Author",
        source: input.source || "Unknown Source",
        imageUrl: input.imageUrl || "https://via.placeholder.com/150",
        publishedAt: input.publishedAt ? new Date(input.publishedAt) : new Date(),
        category: input.category || "General",
        topics: input.topics?.length ? input.topics : ["news"],
        topic: input.topic ?? null,
      },
    },
    { upsert: true, new: true, setDefaultsOnInsert: true }
  );
  return doc;
}

export async function enrichSingleArticle(article: ArticleDoc) {
  const title = cleanText(article.title || "");
  const text = cleanText(article.content || "");
  const src = article.source || "Unknown Source";
  const subject = article.topic || article.topics?.[0] || article.category || "General";
  const dateISO = article.publishedAt ? new Date(article.publishedAt).toISOString().slice(0, 10) : null;

  // Fire independent services concurrently (settled -> partial writes allowed)
  const [
    scoreRes,
    biasProbRes,
    biasExplainRes,
    sentResPrimary,
    clickbaitRes,
    omittedRes,
  ] = await Promise.allSettled([
    postJSON<ScoreArticleResp>(SCORE_URL, {
      title,
      text,
      subject,
      date: dateISO,
      source_url: article.url,
    }, { retries: 1 }),
    postJSON<BiasPredictResp>(BIAS_PREDICT_URL, { Title: title, Text: text, Source: src }, { retries: 1 }),
    postJSON<BiasExplainResp>(BIAS_EXPLAIN_URL, { Title: title, Text: text, Source: src }, { retries: 1, query: { top_k: 10 } }),
    // sentiment: try primary then fallback if it throws
    (async () => {
      try {
        return await postJSON<SentimentResp>(SENTIMENT_URL_PRIMARY, { text }, { retries: 1 });
      } catch {
        return await postJSON<SentimentResp>(SENTIMENT_URL_FALLBACK, { text }, { retries: 1 });
      }
    })(),
    postJSON<ClickbaitResp>(CLICKBAIT_URL, { content: text, title }, { retries: 1 }),
    postJSON<OmittedResp>(OMITTED_URL, {
      topic: subject,
      articles: [{ title, url: article.url }],
    }, { retries: 1 }),
  ]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const updates: Record<string, any> = {};

  // --- :8000/v1/score_article → fake news / reliability
  if (scoreRes.status === "fulfilled") {
    const probs = scoreRes.value.probabilities || {};
    const label = scoreRes.value.label || argmax(probs);
    updates.fake_news_label = label ?? undefined;
    updates.fake_news_probs = new Map(Object.entries(probs));
    const rel = scoreRes.value.source?.reliability_0_100;
    updates.source_reliability = Number.isFinite(rel) ? rel! : undefined;
  }

  // --- :8000/predict + :8000/explain → bias classification
  if (biasProbRes.status === "fulfilled") {
    updates.bias_classification_probs = new Map(Object.entries(biasProbRes.value.probabilities || {}));
    updates.bias_classification_label = argmax(biasProbRes.value.probabilities) ?? undefined;
  }
  if (biasExplainRes.status === "fulfilled") {
    // If explain returns its own predicted label, prefer it for the final label
    const explainLabel = biasExplainRes.value.predicted_label;
    if (explainLabel) updates.bias_classification_label = explainLabel;
    updates.bias_explain = (biasExplainRes.value.top_explanations || []).map((e) => ({
      phrase: cleanText(e.phrase),
      score: e.score,
      weight: e.weight,
    }));
  }

  // --- :8000/sentiment → sentiment fields
  if (sentResPrimary.status === "fulfilled") {
    const probs = sentResPrimary.value.probabilities || {};
    updates.sentiment_analysis_probs = new Map(Object.entries(probs));
    updates.sentiment_analysis_label = argmax(probs as Record<string, number>) ?? undefined;
  }

  // --- :8002/analyze → clickbait
  if (clickbaitRes.status === "fulfilled") {
    updates.clickbait_label = clickbaitRes.value.label || undefined;
    updates.clickbait_score = clickbaitRes.value.score ?? undefined;
    updates.clickbait_explanation = cleanText(clickbaitRes.value.explanation || "") || undefined;
  }

  // --- :8002/detect_omitted_facts → omitted_facts_articles[]
  if (omittedRes.status === "fulfilled") {
    updates.omitted_facts_articles = (omittedRes.value.articles || []).map((a) => ({
      title: cleanText(a.title),
      url: a.url,
      omitted_segments: (a.omitted_segments || []).map((s) => ({
        chunk: cleanText(s.chunk),
        max_similarity: s.max_similarity,
      })),
    }));
  }

  if (Object.keys(updates).length) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await Article.updateOne({ _id: (article as any)._id }, { $set: updates });
  }
}

// ---------- NEW: Enrich only where fields are missing ----------
export async function enrichMissingOnly(article: ArticleDoc) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const articleId = (article as any)._id?.toString() || "unknown";
  const articleTitle = article.title?.substring(0, 50) || "no title";
  console.log(`[DEBUG enrichMissingOnly] Starting for article: ${articleTitle} (${articleId})`);

  const title = cleanText(article.title || "");
  const text = cleanText(article.content || "");
  const src = article.source || "Unknown Source";
  const subject = article.topic || article.topics?.[0] || article.category || "General";
  const dateISO = article.publishedAt ? new Date(article.publishedAt).toISOString().slice(0, 10) : null;

  console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Prepared data:`, {
    title_length: title.length,
    text_length: text.length,
    src,
    subject,
    dateISO,
  });

  const needsScore =
    !article.fake_news_label ||
    article.source_reliability == null;

  const needsBias =
    !article.bias_classification_label ||
    !article.bias_explain || (Array.isArray(article.bias_explain) && article.bias_explain.length === 0) ||
    !article.bias_classification_probs || (article.bias_classification_probs instanceof Map
      ? article.bias_classification_probs.size === 0
      : !Object.keys(article.bias_classification_probs || {}).length);

  const needsSentiment =
    !article.sentiment_analysis_label ||
    !article.sentiment_analysis_probs || (article.sentiment_analysis_probs instanceof Map
      ? article.sentiment_analysis_probs.size === 0
      : !Object.keys(article.sentiment_analysis_probs || {}).length);

  const needsClickbait =
    article.clickbait_label == null || article.clickbait_score == null;

  const needsOmitted =
    !article.omitted_facts_articles || (Array.isArray(article.omitted_facts_articles) && article.omitted_facts_articles.length === 0);

  console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Field status:`, {
    needsScore,
    current_fake_news_label: article.fake_news_label,
    current_source_reliability: article.source_reliability,
    needsBias,
    current_bias_classification_label: article.bias_classification_label,
    bias_explain_type: Array.isArray(article.bias_explain) ? `array[${article.bias_explain.length}]` : typeof article.bias_explain,
    bias_probs_type: article.bias_classification_probs instanceof Map ? `Map[${article.bias_classification_probs.size}]` : typeof article.bias_classification_probs,
    needsSentiment,
    current_sentiment_analysis_label: article.sentiment_analysis_label,
    sentiment_probs_type: article.sentiment_analysis_probs instanceof Map ? `Map[${article.sentiment_analysis_probs.size}]` : typeof article.sentiment_analysis_probs,
    needsClickbait,
    current_clickbait_label: article.clickbait_label,
    current_clickbait_score: article.clickbait_score,
    needsOmitted,
    omitted_facts_type: Array.isArray(article.omitted_facts_articles) ? `array[${article.omitted_facts_articles.length}]` : typeof article.omitted_facts_articles,
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const updates: Record<string, any> = {};

  const tasks: Promise<void>[] = [];

  if (needsScore) {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Calling SCORE_URL: ${SCORE_URL}`);
    tasks.push((async () => {
      try {
        const r = await postJSON<ScoreArticleResp>(SCORE_URL, {
          title, text, subject, date: dateISO, source_url: article.url,
        }, { retries: 1 });
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - SCORE_URL response received`);
        const probs = r.probabilities || {};
        const label = r.label || argmax(probs);
        updates.fake_news_label = label ?? undefined;
        updates.fake_news_probs = new Map(Object.entries(probs));
        const rel = r.source?.reliability_0_100;
        updates.source_reliability = Number.isFinite(rel) ? rel! : undefined;
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Score updates:`, {
          fake_news_label: updates.fake_news_label,
          source_reliability: updates.source_reliability,
        });
      } catch (error) {
        console.error(`[DEBUG enrichMissingOnly] Article ${articleId} - SCORE_URL error:`, error);
        throw error;
      }
    })());
  }

  if (needsBias) {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Calling BIAS_PREDICT_URL: ${BIAS_PREDICT_URL}`);
    tasks.push((async () => {
      try {
        const probsRes = await postJSON<BiasPredictResp>(BIAS_PREDICT_URL, { Title: title, Text: text, Source: src }, { retries: 1 });
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - BIAS_PREDICT_URL response received`);
        updates.bias_classification_probs = new Map(Object.entries(probsRes.probabilities || {}));
        updates.bias_classification_label = argmax(probsRes.probabilities) ?? undefined;

        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Calling BIAS_EXPLAIN_URL: ${BIAS_EXPLAIN_URL}`);
        const explainRes = await postJSON<BiasExplainResp>(BIAS_EXPLAIN_URL, { Title: title, Text: text, Source: src }, { retries: 1, query: { top_k: 10 } });
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - BIAS_EXPLAIN_URL response received`);
        if (explainRes.predicted_label) updates.bias_classification_label = explainRes.predicted_label;
        updates.bias_explain = (explainRes.top_explanations || []).map(e => ({
          phrase: cleanText(e.phrase),
          score: e.score,
          weight: e.weight,
        }));
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Bias updates:`, {
          bias_classification_label: updates.bias_classification_label,
          bias_explain_count: updates.bias_explain?.length || 0,
        });
      } catch (error) {
        console.error(`[DEBUG enrichMissingOnly] Article ${articleId} - BIAS API error:`, error);
        throw error;
      }
    })());
  }

  if (needsSentiment) {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Calling SENTIMENT_URL_PRIMARY: ${SENTIMENT_URL_PRIMARY}`);
    tasks.push((async () => {
      try {
        let r: SentimentResp;
        try {
          r = await postJSON<SentimentResp>(SENTIMENT_URL_PRIMARY, { text }, { retries: 1 });
          console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - SENTIMENT_URL_PRIMARY response received`);
        } catch (error) {
          console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - SENTIMENT_URL_PRIMARY failed, trying fallback:`, error);
          r = await postJSON<SentimentResp>(SENTIMENT_URL_FALLBACK, { text }, { retries: 1 });
          console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - SENTIMENT_URL_FALLBACK response received`);
        }
        const probs = r.probabilities || {};
        updates.sentiment_analysis_probs = new Map(Object.entries(probs));
        updates.sentiment_analysis_label = argmax(probs as Record<string, number>);
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Sentiment updates:`, {
          sentiment_analysis_label: updates.sentiment_analysis_label,
        });
      } catch (error) {
        console.error(`[DEBUG enrichMissingOnly] Article ${articleId} - SENTIMENT API error:`, error);
        throw error;
      }
    })());
  }

  if (needsClickbait) {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Calling CLICKBAIT_URL: ${CLICKBAIT_URL}`);
    tasks.push((async () => {
      try {
        const r = await postJSON<ClickbaitResp>(CLICKBAIT_URL, { content: text, title }, { retries: 1 });
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - CLICKBAIT_URL response received`);
        updates.clickbait_label = r.label || undefined;
        updates.clickbait_score = r.score ?? undefined;
        updates.clickbait_explanation = cleanText(r.explanation || "") || undefined;
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Clickbait updates:`, {
          clickbait_label: updates.clickbait_label,
          clickbait_score: updates.clickbait_score,
        });
      } catch (error) {
        console.error(`[DEBUG enrichMissingOnly] Article ${articleId} - CLICKBAIT_URL error:`, error);
        throw error;
      }
    })());
  }

  if (needsOmitted) {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Calling OMITTED_URL: ${OMITTED_URL}`);
    tasks.push((async () => {
      try {
        const r = await postJSON<OmittedResp>(OMITTED_URL, {
          topic: subject,
          articles: [{ title, url: article.url }],
        }, { retries: 1 });
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - OMITTED_URL response received`);
        updates.omitted_facts_articles = (r.articles || []).map(a => ({
          title: cleanText(a.title),
          url: a.url,
          omitted_segments: (a.omitted_segments || []).map(s => ({
            chunk: cleanText(s.chunk),
            max_similarity: s.max_similarity,
          })),
        }));
        console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Omitted facts updates:`, {
          omitted_facts_articles_count: updates.omitted_facts_articles?.length || 0,
        });
      } catch (error) {
        console.error(`[DEBUG enrichMissingOnly] Article ${articleId} - OMITTED_URL error:`, error);
        throw error;
      }
    })());
  }

  if (tasks.length === 0) {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - No tasks needed, all fields already present`);
    return; // nothing to do
  }

  console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Executing ${tasks.length} tasks`);
  const results = await Promise.allSettled(tasks);
  console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Tasks completed:`, results.map((r, idx) => ({
    task: idx,
    status: r.status,
    reason: r.status === "rejected" ? String(r.reason) : undefined,
  })));

  if (Object.keys(updates).length) {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Updating database with:`, Object.keys(updates));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await Article.updateOne({ _id: (article as any)._id }, { $set: updates });
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - Database update completed`);
  } else {
    console.log(`[DEBUG enrichMissingOnly] Article ${articleId} - No updates to apply`);
  }
}

export async function summariseBatchAndWrite(articles: ArticleDoc[]) {
  console.log(`[DEBUG summariseBatchAndWrite] Starting for ${articles.length} articles`);
  if (!articles.length) {
    console.log(`[DEBUG summariseBatchAndWrite] No articles to process`);
    return;
  }
  
  // TypeScript limitation: Mongoose document types are too complex for type inference
  // Use for loop to avoid complex type inference issues with map
  const payload: SummariseReqArticle[] = [];
  for (const a of articles) {
    payload.push({
      title: cleanText(a.title || ""),
      text: cleanText(a.content || ""),
      subject: a.topic || a.topics?.[0] || a.category || "General",
      date: a.publishedAt ? new Date(a.publishedAt).toISOString().slice(0, 10) : null,
      source_url: a.url || null,
    });
  }

  console.log(`[DEBUG summariseBatchAndWrite] Prepared payload for ${payload.length} articles`);
  console.log(`[DEBUG summariseBatchAndWrite] Calling SUMMARISE_URL: ${SUMMARISE_URL}`);

  try {
    const resp = await postJSON<SummariseResp>(SUMMARISE_URL, {
      articles: payload,
      distance_threshold: 0.45,
      map_words: 80,
      reduce_words: 180,
    });

    console.log(`[DEBUG summariseBatchAndWrite] SUMMARISE_URL response received`);
    console.log(`[DEBUG summariseBatchAndWrite] Response has ${resp.per_article_summary?.length || 0} summaries`);

    const per = resp.per_article_summary || [];
    // Write the digest for each article into `muti_source_summary`
    console.log(`[DEBUG summariseBatchAndWrite] Updating ${articles.length} articles with summaries`);
    const updatePromises: Promise<unknown>[] = [];
    for (let i = 0; i < articles.length; i++) {
      const a = articles[i];
      const summary = cleanText(per[i] || "") || undefined;
      console.log(`[DEBUG summariseBatchAndWrite] Updating article ${i + 1}/${articles.length} with summary length: ${summary?.length || 0}`);
      updatePromises.push(Article.updateOne(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        { _id: (a as any)._id },
        { $set: { muti_source_summary: summary } }
      ));
    }
    const updateResults = await Promise.all(updatePromises);
    console.log(`[DEBUG summariseBatchAndWrite] Updated ${updateResults.length} articles`);
  } catch (error) {
    console.error(`[DEBUG summariseBatchAndWrite] Error:`, error);
    throw error;
  }
}
