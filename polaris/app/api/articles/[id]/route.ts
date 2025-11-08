import { NextRequest, NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { Article } from "@/models/articles";

export const dynamic = "force-dynamic";

// GET a single article by ID
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    await connectDB();
    
    const { id } = await params;

    const article = await Article.findOne({ id }).lean();
    
    if (!article) {
      return NextResponse.json(
        { error: "Article not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({ article });
  } catch (error) {
    console.error("Error fetching article:", error);
    return NextResponse.json(
      { error: "Failed to fetch article", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// PATCH to update likes or add comments
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    await connectDB();
    
    const { id } = await params;
    const body = await request.json();

    const updateFields: any = {};

    // Update likes
    if (typeof body.likes === "number") {
      updateFields.likes = body.likes;
    }

    // Add comment
    if (body.comment) {
      const newComment = {
        id: Date.now().toString(),
        userId: body.comment.userId || "anonymous",
        userName: body.comment.userName || "Anonymous User",
        content: body.comment.content,
        createdAt: new Date(),
        likes: 0,
      };

      const article = await Article.findOne({ id });
      if (!article) {
        return NextResponse.json(
          { error: "Article not found" },
          { status: 404 }
        );
      }

      article.comments = article.comments || [];
      article.comments.push(newComment);
      await article.save();

      return NextResponse.json({ success: true, comment: newComment });
    }

    // Update other fields
    if (Object.keys(updateFields).length > 0) {
      const article = await Article.findOneAndUpdate(
        { id },
        { $set: updateFields },
        { new: true }
      ).lean();

      if (!article) {
        return NextResponse.json(
          { error: "Article not found" },
          { status: 404 }
        );
      }

      return NextResponse.json({ success: true, article });
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error updating article:", error);
    return NextResponse.json(
      { error: "Failed to update article", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

