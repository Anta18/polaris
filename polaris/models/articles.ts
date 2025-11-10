import mongoose, { Schema, InferSchemaType, models, model } from "mongoose";

/* ---------- Subschemas for analytics ---------- */
const biasExplainSchema = new Schema(
  {
    phrase: { type: String, required: true },
    score: { type: Number, required: true },
    weight: { type: Number, required: true },
  },
  { _id: false }
);

/* ---------- Comment subdoc ---------- */
const commentSchema = new Schema(
  {
    id: String,
    userId: String,
    userName: String,
    content: String,
    createdAt: { type: Date, default: Date.now },
    likes: { type: Number, default: 0 },
  },
  { _id: false }
);

/* ---------- Article Schema ---------- */
const articleSchema = new Schema(
  {
    id: { type: String, default: null, index: true },
    title: { type: String, default: "Untitled News" },
    description: { type: String, default: "No description available." },
    content: { type: String, default: "No content available." },
    author: { type: String, default: "Unknown Author" },
    source: { type: String, default: "Unknown Source" },
    url: { type: String, default: "https://bbc.com/news" },
    imageUrl: { type: String, default: "https://via.placeholder.com/150" },
    publishedAt: { type: Date, default: Date.now },
    category: { type: String, default: "General" },
    topics: { type: [String], default: ["news"] },

    // Bias classification
    bias_classification_label: { type: String, default: null },
    bias_classification_probs: { type: Map, of: Number, default: {} },
    bias_explain: { type: [biasExplainSchema], default: [] },

    // Sentiment
    sentiment_analysis_label: { type: String, default: null },
    sentiment_analysis_probs: { type: Map, of: Number, default: {} },

    // Clickbait
    clickbait_label: { type: String, default: null },
    clickbait_score: { type: Number, default: null },
    clickbait_explanation: { type: String, default: null },

    // Topic (singular)
    topic: { type: String, default: null },

    // Omitted facts cross-checks - array of string chunks
    omitted_chunks: { type: [String], default: [] },
    omitted_summary: { type: String, default: null },

    // Fake news / reliability
    fake_news_label: { type: String, default: null },
    fake_news_probs: { type: Map, of: Number, default: {} },
    source_reliability: { type: Number, default: null },

    // Summaries (keep original key)
    muti_source_summary: { type: String, default: null },
    single_source_summary: { type: String, default: null },

    // Engagement
    likes: { type: Number, default: 0 },
    comments: { type: [commentSchema], default: [] },
  },
  { timestamps: true }
);

export type ArticleDoc = InferSchemaType<typeof articleSchema>;
export const Article = (models.Article as mongoose.Model<ArticleDoc>) || model<ArticleDoc>("Article", articleSchema);
