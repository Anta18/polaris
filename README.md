# Polaris

**A news-analysis workspace that brings several perspectives into one interface.** Polaris combines a Next.js application with independent Python services for bias classification, sentiment, clickbait analysis, source-reliability scoring, cross-article comparison, and summarization.

Search a topic, compare articles grouped by predicted political leaning, and open the analysis behind an individual story. The repository includes both the application and model-training or serving code.

## Explore the implementation

| Layer | What lives here |
| --- | --- |
| [Web application](polaris/) | Next.js, React, TypeScript, MongoDB-backed articles, search, and article detail views |
| [Analysis orchestration](polaris/lib/analyzers.ts) | Calls independent analysis services and combines the returned outputs |
| [Bias classification](models/bias_classification/) | RoBERTa adapter training and serving |
| [Sentiment analysis](models/sentiment_analysis/) | BERT adapter training and serving |
| [Clickbait analysis](models/clickbait/) | Model preparation, training, explanations, and an API application |
| [Cross-article comparison](models/llmproject/) | Embeddings, comparison, topic summaries, and service documentation |
| [Reliability](models/reliability/) · [Summarization](models/summarizer/) | Separate model experiments and API applications |

The analysis services expose model outputs. Their presence in the interface does not establish an independently measured accuracy rate.

## Run the web application

From a clone of this repository:

```bash
cd polaris
npm ci
```

Create `polaris/.env.local` relative to the repository root:

```dotenv
MONGO_URI=mongodb://127.0.0.1:27017/polaris_news_db
MONGO_AUTH_SOURCE=admin
NEXT_PUBLIC_BASE_URL=http://localhost:3000
```

Then start the development server from the `polaris/` application directory:

```bash
npm run dev
```

Open [localhost:3000](http://localhost:3000). Use a Node.js version compatible with the pinned Next.js version in [package.json](polaris/package.json).

The frontend and MongoDB setup are one part of the system. To generate new analysis, configure and run the model services separately. Their URL overrides are declared in [analyzers.ts](polaris/lib/analyzers.ts): `SCORE_URL`, `BIAS_PREDICT_URL`, `BIAS_EXPLAIN_URL`, `SENTIMENT_URL_PRIMARY`, `SENTIMENT_URL_FALLBACK`, `CLICKBAIT_URL`, `OMITTED_URL`, and `SUMMARISE_URL`. Existing defaults point to development hosts; set them for your own environment.

## Read further

- [Application guide](polaris/README.md)
- [Feature map](polaris/FEATURES.md)
- [Cross-article service API](models/llmproject/API_DOCS.md)
- [Reliability service API](models/reliability/api_documentation.md)

Polaris is a multi-service research prototype. Deployment, model artifacts, and training dependencies are managed within the individual service directories.
