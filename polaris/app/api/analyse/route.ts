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
  await connectDB();

  const { searchParams } = new URL(req.url);
  const limit = Math.max(1, Math.min(Number(searchParams.get("limit") || 100), 2000));

  // Pull candidates that are missing at least one target field OR summary
  // (We still double-check inside enrichMissingOnly for finer conditions.)
  const candidatesRaw = await Article.find({
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
  })
    .sort({ updatedAt: 1 }) // oldest first
    .limit(limit)
    .exec();

  // Type assertion to simplify complex Mongoose document types
  const candidates = candidatesRaw as unknown as ArticleDoc[];

  if (!candidates.length) {
    return NextResponse.json({ message: "No articles require analysis." });
  }

  // Split into two sets: per-article enrichment vs batch-summarization
  const toSummarise: ArticleDoc[] = [];
  for (const a of candidates) {
    if (!a.muti_source_summary) {
      toSummarise.push(a);
    }
  }
  const toEnrich: ArticleDoc[] = candidates; // all may have some missing single-output fields

  // Limit concurrency to avoid hammering services
  const CONCURRENCY = Number(process.env.ANALYZE_CONCURRENCY || 6);
  const chunks: ArticleDoc[][] = [];
  for (let i = 0; i < toEnrich.length; i += CONCURRENCY) {
    chunks.push(toEnrich.slice(i, i + CONCURRENCY));
  }

  let enriched = 0;
  for (const batch of chunks) {
    // TypeScript limitation: Mongoose document types are too complex for type inference
    // Use for loop to avoid complex type inference issues with map
    const promises: Promise<void>[] = [];
    for (const doc of batch) {
      promises.push(enrichMissingOnly(doc));
    }
    await Promise.allSettled(promises);
    enriched += batch.length;
  }

  // Batch summarization for those missing muti_source_summary
  let summarised = 0;
  if (toSummarise.length > 0) {
    await summariseBatchAndWrite(toSummarise);
    summarised = toSummarise.length;
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

  return NextResponse.json({
    scanned: candidates.length,
    enriched_single_output: enriched,
    summarised_topics: summarised,
    updated: latest,
  });
}
