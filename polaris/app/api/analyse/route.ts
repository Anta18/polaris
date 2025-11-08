// app/api/analyze/route.ts
import { NextRequest, NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { Article, ArticleDoc } from "@/models/articles";
import { enrichMissingOnly, summariseBatchAndWrite } from "@/lib/analyzers";

/**
 * POST /api/analyze
 * Scans DB for articles with missing analytics and fills only the empty fields.
 * Optional query params:
 *   - limit (default 100)
 *
 * This will:
 *   1) For each article needing any single-output fields, call only the required microservices.
 *   2) Batch-call :8001/v1/summarise_topics for articles missing `muti_source_summary`.
 */
export async function POST(req: NextRequest) {
  console.log("[DEBUG] Starting POST /api/analyse");
  await connectDB();
  console.log("[DEBUG] Database connected");

  const { searchParams } = new URL(req.url);
  const limit = Math.max(1, Math.min(Number(searchParams.get("limit") || 100), 2000));
  console.log("[DEBUG] Query params:", { limit, url: req.url });

  // First, check total articles in database
  const totalArticles = await Article.countDocuments({});
  console.log("[DEBUG] Total articles in database:", totalArticles);

  if (totalArticles === 0) {
    console.log("[DEBUG] No articles found in database at all");
    return NextResponse.json({ 
      message: "No articles require analysis.",
      debug: {
        totalArticles: 0,
        queryFilter: "N/A - no articles in database"
      }
    });
  }

  // Sample a few articles to see their field values
  const sampleArticles = await Article.find({}).limit(3).lean();
  console.log("[DEBUG] Sample articles (first 3):", JSON.stringify(sampleArticles.map(a => ({
    _id: a._id,
    title: a.title?.substring(0, 50),
    fake_news_label: a.fake_news_label,
    source_reliability: a.source_reliability,
    bias_classification_label: a.bias_classification_label,
    bias_explain_length: Array.isArray(a.bias_explain) ? a.bias_explain.length : typeof a.bias_explain,
    sentiment_analysis_label: a.sentiment_analysis_label,
    clickbait_label: a.clickbait_label,
    clickbait_score: a.clickbait_score,
    omitted_facts_articles_length: Array.isArray(a.omitted_facts_articles) ? a.omitted_facts_articles.length : typeof a.omitted_facts_articles,
    muti_source_summary: a.muti_source_summary ? "exists" : "null/undefined",
  })), null, 2));

  // Build the query filter
  const queryFilter = {
    $or: [
      { fake_news_label: null },
      { source_reliability: null },
      { bias_classification_label: null },
      { bias_explain: { $size: 0 } },
      { sentiment_analysis_label: null },
      { clickbait_label: null },
      { clickbait_score: null },
      { omitted_facts_articles: { $size: 0 } },
      { muti_source_summary: null },
    ],
  };
  console.log("[DEBUG] Query filter:", JSON.stringify(queryFilter, null, 2));

  // Check counts for each condition individually
  const counts = {
    fake_news_label_null: await Article.countDocuments({ fake_news_label: null }),
    source_reliability_null: await Article.countDocuments({ source_reliability: null }),
    bias_classification_label_null: await Article.countDocuments({ bias_classification_label: null }),
    bias_explain_empty: await Article.countDocuments({ bias_explain: { $size: 0 } }),
    sentiment_analysis_label_null: await Article.countDocuments({ sentiment_analysis_label: null }),
    clickbait_label_null: await Article.countDocuments({ clickbait_label: null }),
    clickbait_score_null: await Article.countDocuments({ clickbait_score: null }),
    omitted_facts_articles_empty: await Article.countDocuments({ omitted_facts_articles: { $size: 0 } }),
    muti_source_summary_null: await Article.countDocuments({ muti_source_summary: null }),
  };
  console.log("[DEBUG] Individual condition counts:", JSON.stringify(counts, null, 2));

  // Pull candidates that are missing at least one target field OR summary
  // (We still double-check inside enrichMissingOnly for finer conditions.)
  console.log("[DEBUG] Executing query with filter...");
  const candidatesRaw = await Article.find(queryFilter)
    .sort({ updatedAt: 1 }) // oldest first
    .limit(limit)
    .exec();

  console.log("[DEBUG] Query returned", candidatesRaw.length, "candidates");

  // Type assertion to simplify complex Mongoose document types
  const candidates = candidatesRaw as unknown as ArticleDoc[];

  if (!candidates.length) {
    console.log("[DEBUG] No candidates found. Checking what fields articles actually have...");
    
    // Get a sample article with all its fields
    const sampleFull = await Article.findOne({}).lean();
    if (sampleFull) {
      console.log("[DEBUG] Sample article fields:", JSON.stringify({
        fake_news_label: sampleFull.fake_news_label,
        fake_news_label_type: typeof sampleFull.fake_news_label,
        source_reliability: sampleFull.source_reliability,
        source_reliability_type: typeof sampleFull.source_reliability,
        bias_classification_label: sampleFull.bias_classification_label,
        bias_classification_label_type: typeof sampleFull.bias_classification_label,
        bias_explain: Array.isArray(sampleFull.bias_explain) ? `array[${sampleFull.bias_explain.length}]` : sampleFull.bias_explain,
        sentiment_analysis_label: sampleFull.sentiment_analysis_label,
        sentiment_analysis_label_type: typeof sampleFull.sentiment_analysis_label,
        clickbait_label: sampleFull.clickbait_label,
        clickbait_label_type: typeof sampleFull.clickbait_label,
        clickbait_score: sampleFull.clickbait_score,
        clickbait_score_type: typeof sampleFull.clickbait_score,
        omitted_facts_articles: Array.isArray(sampleFull.omitted_facts_articles) ? `array[${sampleFull.omitted_facts_articles.length}]` : sampleFull.omitted_facts_articles,
        muti_source_summary: sampleFull.muti_source_summary,
        muti_source_summary_type: typeof sampleFull.muti_source_summary,
      }, null, 2));
    }

    return NextResponse.json({ 
      message: "No articles require analysis.",
      debug: {
        totalArticles,
        queryFilter,
        individualCounts: counts,
        sampleArticle: sampleFull ? {
          fake_news_label: sampleFull.fake_news_label,
          source_reliability: sampleFull.source_reliability,
          bias_classification_label: sampleFull.bias_classification_label,
          sentiment_analysis_label: sampleFull.sentiment_analysis_label,
          clickbait_label: sampleFull.clickbait_label,
          clickbait_score: sampleFull.clickbait_score,
          muti_source_summary: sampleFull.muti_source_summary,
        } : null,
      }
    });
  }

  console.log("[DEBUG] Found", candidates.length, "candidates requiring analysis");

  // Split into two sets: per-article enrichment vs batch-summarization
  console.log("[DEBUG] Analyzing candidates for missing fields...");
  const toSummarise: ArticleDoc[] = [];
  for (const a of candidates) {
    if (!a.muti_source_summary) {
      toSummarise.push(a);
    }
  }
  console.log("[DEBUG] Articles needing summarization:", toSummarise.length);
  const toEnrich: ArticleDoc[] = candidates; // all may have some missing single-output fields

  // Log what each candidate is missing
  console.log("[DEBUG] Missing fields per candidate:");
  for (let i = 0; i < Math.min(candidates.length, 5); i++) {
    const a = candidates[i];
    const missing = {
      fake_news_label: !a.fake_news_label,
      source_reliability: a.source_reliability == null,
      bias_classification_label: !a.bias_classification_label,
      bias_explain: !a.bias_explain || (Array.isArray(a.bias_explain) && a.bias_explain.length === 0),
      sentiment_analysis_label: !a.sentiment_analysis_label,
      clickbait_label: a.clickbait_label == null,
      clickbait_score: a.clickbait_score == null,
      omitted_facts_articles: !a.omitted_facts_articles || (Array.isArray(a.omitted_facts_articles) && a.omitted_facts_articles.length === 0),
      muti_source_summary: !a.muti_source_summary,
    };
    console.log(`[DEBUG] Candidate ${i + 1} (${a.title?.substring(0, 30)}...):`, JSON.stringify(missing, null, 2));
  }

  // Limit concurrency to avoid hammering services
  const CONCURRENCY = Number(process.env.ANALYZE_CONCURRENCY || 6);
  console.log("[DEBUG] Concurrency limit:", CONCURRENCY);
  const chunks: ArticleDoc[][] = [];
  for (let i = 0; i < toEnrich.length; i += CONCURRENCY) {
    chunks.push(toEnrich.slice(i, i + CONCURRENCY));
  }
  console.log("[DEBUG] Split into", chunks.length, "chunks");

  let enriched = 0;
  for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
    const batch = chunks[chunkIdx];
    console.log(`[DEBUG] Processing chunk ${chunkIdx + 1}/${chunks.length} with ${batch.length} articles`);
    // TypeScript limitation: Mongoose document types are too complex for type inference
    // Use for loop to avoid complex type inference issues with map
    const promises: Promise<void>[] = [];
    for (const doc of batch) {
      console.log(`[DEBUG] Enriching article: ${doc.title?.substring(0, 50)}...`);
      promises.push(enrichMissingOnly(doc));
    }
    const results = await Promise.allSettled(promises);
    console.log(`[DEBUG] Chunk ${chunkIdx + 1} results:`, results.map((r, idx) => ({
      index: idx,
      status: r.status,
      reason: r.status === "rejected" ? String(r.reason) : undefined,
    })));
    enriched += batch.length;
  }
  console.log("[DEBUG] Total enriched:", enriched);

  // Batch summarization for those missing muti_source_summary
  let summarised = 0;
  if (toSummarise.length > 0) {
    console.log("[DEBUG] Starting batch summarization for", toSummarise.length, "articles");
    await summariseBatchAndWrite(toSummarise);
    summarised = toSummarise.length;
    console.log("[DEBUG] Batch summarization completed");
  } else {
    console.log("[DEBUG] No articles need summarization");
  }

  // Return a light snapshot of updated docs
  const updatedIds: unknown[] = [];
  for (const d of candidates) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    updatedIds.push((d as any)._id);
  }
  const latest = await Article.find({ _id: { $in: updatedIds } })
    .select({
      title: 1,
      url: 1,
      bias_classification_label: 1,
      sentiment_analysis_label: 1,
      clickbait_label: 1,
      fake_news_label: 1,
      source_reliability: 1,
      muti_source_summary: 1,
      updatedAt: 1,
    })
    .lean();

  console.log("[DEBUG] Final results:", {
    scanned: candidates.length,
    enriched_single_output: enriched,
    summarised_topics: summarised,
    updated_count: latest.length,
  });

  return NextResponse.json({
    scanned: candidates.length,
    enriched_single_output: enriched,
    summarised_topics: summarised,
    updated: latest,
    debug: {
      totalArticles,
      individualCounts: counts,
    },
  });
}
