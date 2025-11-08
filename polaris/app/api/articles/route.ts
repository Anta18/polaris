import { NextRequest, NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { Article } from "@/models/articles";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    await connectDB();
    console.log("✅ Connected to MongoDB");

    const searchParams = request.nextUrl.searchParams;
    const query = searchParams.get("q");
    const limit = parseInt(searchParams.get("limit") || "10");
    const id = searchParams.get("id");
    const category = searchParams.get("category");

    // Fetch a single article by ID
    if (id) {
      const article = await Article.findOne({ id }).lean();
      if (!article) {
        return NextResponse.json({ error: "Article not found" }, { status: 404 });
      }
      return NextResponse.json({ article });
    }

    // Build search filter
    let filter: any = {};

    // Search query: search in title, description, content, author
    if (query && query.trim()) {
      filter.$or = [
        { title: { $regex: query, $options: "i" } },
        { description: { $regex: query, $options: "i" } },
        { content: { $regex: query, $options: "i" } },
        { author: { $regex: query, $options: "i" } },
        { topics: { $in: [new RegExp(query, "i")] } },
      ];
    }

    // Filter by category
    if (category) {
      filter.category = category;
    }

    // Check total count in collection
    const totalCount = await Article.countDocuments();
    console.log("📊 Total articles in database:", totalCount);
    console.log("🔍 Using filter:", JSON.stringify(filter));

    // Fetch articles
    const articles = await Article.find(filter)
      .sort({ publishedAt: -1 })
      .limit(limit)
      .lean();

    console.log("✅ Articles found:", articles.length);
    if (articles.length > 0) {
      console.log("📰 First article:", {
        id: articles[0].id,
        title: articles[0].title,
        _id: articles[0]._id
      });
    }

    return NextResponse.json({ articles, count: articles.length });
  } catch (error) {
    console.error("Error fetching articles:", error);
    return NextResponse.json(
      { error: "Failed to fetch articles", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

