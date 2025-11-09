import { NextRequest, NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { Article } from "@/models/articles";
import axios from "axios";
import { JSDOM } from "jsdom";
import { Readability } from "@mozilla/readability";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface ProcessResult {
  articleId: string;
  title: string;
  url: string;
  success: boolean;
  error?: string;
  contentLength?: number;
}

export async function POST(request: NextRequest) {
  try {
    await connectDB();
    console.log("✅ Connected to MongoDB");

    const body = await request.json().catch(() => ({}));
    const limit = body.limit || 1000; // Number of articles to process
    const skip = body.skip || 0; // Offset
    const onlyEmpty = false; // Only process articles with default/empty content (default: true)

    // Build filter: if onlyEmpty is true, only get articles with default content
    let filter: any = {};
    if (onlyEmpty) {
      filter.$or = [
        { content: { $exists: false } },
        { content: "No content available." },
        { content: "" },
      ];
    }

    // Fetch articles from database
    const articles = await Article.find(filter)
      .sort({ publishedAt: -1 })
      .skip(skip)
      .limit(limit)
      .lean();

    console.log(`📰 Found ${articles.length} articles to process`);

    if (articles.length === 0) {
      return NextResponse.json({
        message: "No articles found to process",
        processed: 0,
        results: [],
      });
    }

    const results: ProcessResult[] = [];

    // Process articles one by one
    for (const article of articles) {
      const result: ProcessResult = {
        articleId: article._id?.toString() || article.id || "unknown",
        title: article.title || "Untitled",
        url: article.url || "",
        success: false,
      };

      try {
        // Skip if no URL
        if (!article.url || article.url === "https://bbc.com/news") {
          result.error = "Invalid or default URL";
          results.push(result);
          continue;
        }

        console.log(`📥 Fetching content from: ${article.url}`);

        // Download the HTML
        const response = await axios.get(article.url, {
          timeout: 30000, // 30 second timeout
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
          },
        });

        // Convert HTML to DOM using jsdom
        const dom = new JSDOM(response.data, {
          url: article.url,
        });

        // Extract article content using Readability
        const reader = new Readability(dom.window.document);
        const parsedArticle = reader.parse();

        if (!parsedArticle) {
          result.error = "Readability could not parse the article";
          results.push(result);
          continue;
        }

        // Get the text content
        const content = parsedArticle.textContent || parsedArticle.content || "";

        if (!content || content.trim().length === 0) {
          result.error = "No content extracted";
          results.push(result);
          continue;
        }

        // Update the article in the database
        await Article.updateOne(
          { _id: article._id },
          { $set: { content: content.trim() } }
        );

        result.success = true;
        result.contentLength = content.length;
        console.log(`✅ Successfully updated article: ${article.title} (${content.length} chars)`);

        results.push(result);

        // Small delay to avoid overwhelming the server
        await new Promise((resolve) => setTimeout(resolve, 500));
      } catch (error: any) {
        result.error =
          error.message || error.toString() || "Unknown error occurred";
        console.error(`❌ Error processing article ${article.title}:`, error.message);
        results.push(result);
      }
    }

    const successCount = results.filter((r) => r.success).length;
    const failureCount = results.filter((r) => !r.success).length;

    return NextResponse.json({
      message: `Processed ${articles.length} articles`,
      processed: articles.length,
      successful: successCount,
      failed: failureCount,
      results,
    });
  } catch (error: any) {
    console.error("Error in /api/articles/fetch-content:", error);
    return NextResponse.json(
      {
        error: "Failed to fetch and update article content",
        details: error.message || "Unknown error",
      },
      { status: 500 }
    );
  }
}

// GET endpoint for status/info
export async function GET(request: NextRequest) {
  try {
    await connectDB();

    const searchParams = request.nextUrl.searchParams;
    const onlyEmpty = searchParams.get("onlyEmpty") !== "false";

    let filter: any = {};
    if (onlyEmpty) {
      filter.$or = [
        { content: { $exists: false } },
        { content: "No content available." },
        { content: "" },
      ];
    }

    const totalArticles = await Article.countDocuments();
    const articlesNeedingContent = await Article.countDocuments(filter);
    const articlesWithContent = totalArticles - articlesNeedingContent;

    return NextResponse.json({
      totalArticles,
      articlesNeedingContent,
      articlesWithContent,
      message: "Use POST to process articles",
    });
  } catch (error: any) {
    console.error("Error in GET /api/articles/fetch-content:", error);
    return NextResponse.json(
      {
        error: "Failed to get status",
        details: error.message || "Unknown error",
      },
      { status: 500 }
    );
  }
}

