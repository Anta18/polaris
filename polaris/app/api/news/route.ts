import { connectDB } from "@/lib/db";
import { Article } from "@/models/articles";
import { formatArticle, jsonError, jsonOK } from "@/lib/utils";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    await connectDB();
    const articles = await Article.find({}).sort({ _id: -1 });
    const out = articles.map((a) => formatArticle(a, false));
    return jsonOK(out);
  } catch (e) {
    console.error("Error in /api/news:", e);
    return jsonError("Internal Server Error", 500);
  }
}

export async function OPTIONS() {
  return jsonOK({});
}
