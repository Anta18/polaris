import express from "express";
import mongoose from "mongoose";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB Connection
const MONGO_URI = "mongodb://127.0.0.1:27017/polaris_news_db";

mongoose
  .connect(MONGO_URI, { authSource: "admin" })
  .then(() => console.log("Connected to MongoDB!"))
  .catch((err) => console.error("MongoDB Connection Error:", err));

/* ---------- Subschemas for analytics ---------- */
const biasExplainSchema = new mongoose.Schema(
  {
    phrase: { type: String, required: true },
    score: { type: Number, required: true },
    weight: { type: Number, required: true },
  },
  { _id: false }
);

const omittedSegmentSchema = new mongoose.Schema(
  {
    chunk: { type: String, required: true },
    max_similarity: { type: Number, required: true },
  },
  { _id: false }
);

const omittedFactsArticleSchema = new mongoose.Schema(
  {
    title: { type: String, required: true },
    url: { type: String, required: true },
    omitted_segments: { type: [omittedSegmentSchema], default: [] },
  },
  { _id: false }
);

/* ---------- Article Schema & Model ---------- */
const articleSchema = new mongoose.Schema({
  id: { type: String, default: null },
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

  // --- NEW: Bias classification ---
  bias_classification_label: { type: String, default: null },
  bias_classification_probs: { type: Map, of: Number, default: {} }, // e.g. {"lean left": 0.98, ...}
  bias_explain: { type: [biasExplainSchema], default: [] },

  // --- NEW: Sentiment analysis ---
  sentiment_analysis_label: { type: String, default: null },
  sentiment_analysis_probs: { type: Map, of: Number, default: {} }, // e.g. {"negative": 0.03, ...}

  // --- NEW: Clickbait detection ---
  clickbait_label: { type: String, default: null }, // e.g. "clickbait" / "not_clickbait"
  clickbait_score: { type: Number, default: null }, // probability/score
  clickbait_explanation: { type: String, default: null },

  // --- NEW: Topic (singular) ---
  topic: { type: String, default: null },

  // --- NEW: Omitted-facts cross-checks ---
  omitted_facts_articles: { type: [omittedFactsArticleSchema], default: [] },

  // --- NEW: Fake news / reliability ---
  fake_news_label: { type: String, default: null }, // e.g. "misleading" / "trustworthy"
  fake_news_probs: { type: Map, of: Number, default: {} }, // {"misleading": x, "trustworthy": y}
  source_reliability: { type: Number, default: null }, // keep numeric; adjust if your API returns strings

  // --- NEW: Summaries ---
  muti_source_summary: { type: String, default: null }, // (spelled per your API)
  single_source_summary: { type: String, default: null },

  // --- Existing engagement fields ---
  likes: { type: Number, default: 0 },
  comments: {
    type: [
      {
        id: String,
        userId: String,
        userName: String,
        content: String,
        createdAt: { type: Date, default: Date.now },
        likes: { type: Number, default: 0 },
      },
    ],
    default: [],
  },
});

const Article = mongoose.model("Article", articleSchema);

/* ---------- Helpers ---------- */
const formatArticle = (article, includeMongoId = false) => {
  const base = {
    // _id: includeMongoId ? article._id : undefined,
    id: article.id,
    title: article.title,
    description: article.description,
    content: article.content,
    author: article.author,
    source: article.source,
    url: article.url,
    imageUrl: article.imageUrl,
    publishedAt: article.publishedAt ? article.publishedAt.toISOString() : null,
    category: article.category,
    topics: article.topics,
    likes: article.likes,
    comments: article.comments,

    // --- analytics fields in responses ---
    bias_classification_label: article.bias_classification_label,
    bias_classification_probs: article.bias_classification_probs,
    bias_explain: article.bias_explain,

    sentiment_analysis_label: article.sentiment_analysis_label,
    sentiment_analysis_probs: article.sentiment_analysis_probs,

    clickbait_label: article.clickbait_label,
    clickbait_score: article.clickbait_score,
    clickbait_explanation: article.clickbait_explanation,

    topic: article.topic,
    omitted_facts_articles: article.omitted_facts_articles,

    fake_news_label: article.fake_news_label,
    fake_news_probs: article.fake_news_probs,
    source_reliability: article.source_reliability,

    muti_source_summary: article.muti_source_summary,
    single_source_summary: article.single_source_summary,
  };

  if (includeMongoId) base._id = article._id;
  return base;
};

/* ---------- Routes ---------- */

// GET News API
app.get("/news", async (req, res) => {
  try {
    const articles = await Article.find({}).sort({ _id: -1 });
    const formattedArticles = [];
    for (let i = 0; i < articles.length; i++) {
      try {
        formattedArticles.push(formatArticle(articles[i], false));
      } catch (e) {
        console.error(`Error formatting article at index ${i}:`, e);
      }
    }
    res.json(formattedArticles);
  } catch (error) {
    console.error("Error in /news route:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// GET /for_you_news - optional filters
app.get("/for_you_news", async (req, res) => {
  try {
    let filter = {};
    if (req.query.filter) {
      try {
        const parsedFilter = JSON.parse(req.query.filter);
        const orConditions = [];
        if (parsedFilter.category) {
          orConditions.push({ category: { $in: parsedFilter.category.$in } });
        }
        if (parsedFilter.source) {
          orConditions.push({ source: { $in: parsedFilter.source.$in } });
        }
        if (orConditions.length > 0) {
          filter = { $or: orConditions };
        }
      } catch (parseError) {
        return res.status(400).json({ error: "Invalid filter format" });
      }
    }
    const articles = await Article.find(filter).sort({ _id: -1 });
    const formattedArticles = articles.map((a) => formatArticle(a, true));
    res.json(formattedArticles);
  } catch (error) {
    console.error("Error fetching news:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// User_liked_articles
app.get("/user_liked_articles", async (req, res) => {
  try {
    const { articleIds } = req.query;
    if (!articleIds) {
      return res.status(400).json({ error: "Article IDs are required" });
    }
    const likedArticleIds = Array.isArray(articleIds)
      ? articleIds
      : articleIds.split(",");
    const articles = await Article.find({ id: { $in: likedArticleIds } }).sort({
      _id: -1,
    });
    if (!articles.length) return res.json([]);
    const formattedArticles = articles.map((a) => formatArticle(a, true));
    res.json(formattedArticles);
  } catch (error) {
    console.error("Error fetching articles:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// GET single article
app.get("/api/articles/:id", async (req, res) => {
  try {
    const article = await Article.findOne({ id: req.params.id });
    if (!article) return res.status(404).json({ error: "Article not found" });
    res.json(formatArticle(article, true));
  } catch (error) {
    console.error("Error in /api/articles/:id route:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// POST: Like an Article
app.post("/api/articles/:id/like", async (req, res) => {
  try {
    const article = await Article.findOne({ id: req.params.id });
    if (!article) return res.status(404).json({ error: "Article not found" });
    article.likes = (article.likes || 0) + 1;
    await article.save();
    res.json(formatArticle(article, true));
  } catch (error) {
    console.error("Error liking article:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// POST: Add a comment
app.post("/api/articles/:id/comment", async (req, res) => {
  try {
    const { userId, userName, text } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "Comment text is required" });
    }
    const article = await Article.findOne({ id: req.params.id });
    if (!article) return res.status(404).json({ error: "Article not found" });

    const comment = {
      id: new mongoose.Types.ObjectId().toString(),
      userId: userId || "anonymous",
      userName: userName || "Anonymous",
      content: text,
      createdAt: new Date(),
      likes: 0,
    };

    article.comments.push(comment);
    await article.save();
    res.json(formatArticle(article, true));
  } catch (error) {
    console.error("Error adding comment:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// Start Server
const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));