# 🌟 POLARIS: Political Leaning Analysis and Reliable Information System

A stunning, AI-powered news aggregation platform with advanced analytics including bias detection, sentiment analysis, clickbait detection, and fact-checking capabilities.

## ✨ Features

### 🏠 Home Page
- Beautiful hero section with animated background
- Search bar with gradient effects
- Top 6 featured articles with AI analysis indicators
- Responsive grid layout
- Dark mode support

### 🔍 Search Functionality
- Real-time article search
- **3-Column Bias View**: Articles are automatically categorized by political bias:
  - **Left Column**: Left-leaning articles
  - **Center Column**: Neutral/Balanced articles
  - **Right Column**: Right-leaning articles
- Visual bias indicators with color coding
- Empty state handling for each bias category

### 📰 Article Detail Page
- Full article content with featured image
- Comprehensive AI Analysis sidebar:
  - **Bias Classification** with probability scores
  - **Sentiment Analysis** (Positive/Negative/Neutral)
  - **Clickbait Detection** with explanation
  - **Source Reliability Score** (0-100%)
  - **Fake News Detection** with probabilities
  - **Bias Indicators** highlighting biased phrases
  - **Cross-Reference Check** for omitted facts
- Real-time engagement features:
  - ❤️ Like system (persisted to database)
  - 💬 Comments section (persisted to database)
  - 📤 Share functionality
- Single and multi-source summaries
- Related article cross-checks
- Link to original source

### 🎨 Design Features
- Modern gradient backgrounds
- Smooth animations and transitions
- Custom scrollbar with gradient
- Hover effects on cards
- Responsive design (mobile, tablet, desktop)
- Beautiful typography with Geist fonts
- Loading states with spinners
- Empty states with helpful messages

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- MongoDB database
- npm or yarn

### Installation

1. Clone the repository:
```bash
cd polaris
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment variables:
Create a `.env.local` file in the `polaris` directory:
```env
MONGO_URI=mongodb://127.0.0.1:27017/polaris_news_db
MONGO_AUTH_SOURCE=admin
NEXT_PUBLIC_BASE_URL=http://localhost:3000
```

4. Run the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## 📡 API Endpoints

### Get Articles
**GET** `/api/articles`

Query parameters:
- `q` (optional): Search query (searches title, description, content, author, topics)
- `limit` (optional): Number of articles to return (default: 10)
- `category` (optional): Filter by category
- `id` (optional): Get a specific article by ID

**Response:**
```json
{
  "articles": [...],
  "count": 10
}
```

### Get Single Article
**GET** `/api/articles/[id]`

**Response:**
```json
{
  "article": {
    "id": "article-123",
    "title": "Article Title",
    "description": "Article description...",
    "content": "Full article content...",
    "author": "John Doe",
    "source": "News Source",
    "imageUrl": "https://...",
    "publishedAt": "2025-01-01T00:00:00.000Z",
    "category": "Technology",
    "bias_classification_label": "Center",
    "sentiment_analysis_label": "Neutral",
    "clickbait_label": "not-clickbait",
    "source_reliability": 0.85,
    "likes": 42,
    "comments": [...]
  }
}
```

### Update Article (Likes/Comments)
**PATCH** `/api/articles/[id]`

**Request Body (for likes):**
```json
{
  "likes": 43
}
```

**Request Body (for comments):**
```json
{
  "comment": {
    "userId": "user-123",
    "userName": "John Doe",
    "content": "Great article!"
  }
}
```

**Response:**
```json
{
  "success": true,
  "article": {...}
}
```

## 📊 Database Schema

### Article Model

```typescript
{
  id: string;                           // Unique identifier
  title: string;                        // Article title
  description: string;                  // Short description
  content: string;                      // Full article content
  author: string;                       // Author name
  source: string;                       // News source
  url: string;                          // Original article URL
  imageUrl: string;                     // Featured image URL
  publishedAt: Date;                    // Publication date
  category: string;                     // Category (e.g., "Technology")
  topics: string[];                     // Related topics
  
  // AI Analysis Fields
  bias_classification_label: string;    // e.g., "Left", "Center", "Right"
  bias_classification_probs: Map<string, number>;
  bias_explain: Array<{
    phrase: string;
    score: number;
    weight: number;
  }>;
  
  sentiment_analysis_label: string;     // e.g., "Positive", "Negative", "Neutral"
  sentiment_analysis_probs: Map<string, number>;
  
  clickbait_label: string;              // "clickbait" or "not-clickbait"
  clickbait_score: number;              // 0-1 score
  clickbait_explanation: string;
  
  topic: string;                        // Primary topic
  
  omitted_chunks: string[];             // Array of omitted fact chunks (strings)
  omitted_summary: string;              // Summary of omitted facts
  
  fake_news_label: string;              // Authenticity label
  fake_news_probs: Map<string, number>;
  source_reliability: number;           // 0-1 score
  
  muti_source_summary: string;          // Multi-source summary
  single_source_summary: string;        // Single-source summary
  
  // Engagement
  likes: number;                        // Number of likes
  comments: Array<{
    id: string;
    userId: string;
    userName: string;
    content: string;
    createdAt: Date;
    likes: number;
  }>;
}
```

## 🎨 Tech Stack

- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Database**: MongoDB with Mongoose
- **Icons**: Lucide React
- **Fonts**: Geist Sans & Geist Mono

## 🗂️ Project Structure

```
polaris/
├── app/
│   ├── api/
│   │   ├── articles/
│   │   │   ├── route.ts          # Main articles API
│   │   │   └── [id]/
│   │   │       └── route.ts      # Single article API
│   │   ├── analyse/
│   │   │   └── route.ts
│   │   └── news/
│   │       └── route.ts
│   ├── article/
│   │   └── [id]/
│   │       └── page.tsx          # Article detail page
│   ├── search/
│   │   └── page.tsx              # Search results page
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Home page
│   └── globals.css               # Global styles
├── components/
│   ├── ArticleCard.tsx           # Article card component
│   └── SearchBar.tsx             # Search bar component
├── lib/
│   ├── db.ts                     # Database connection
│   ├── analyzers.ts
│   ├── http.ts
│   ├── text.ts
│   └── utils.ts
├── models/
│   └── articles.ts               # Article Mongoose model
└── public/
    └── placeholder.svg           # Placeholder image
```

## 🎯 Key Features Explained

### Bias Detection
Articles are analyzed and classified into political bias categories (Left, Center, Right). The search results page displays articles in three columns based on their bias, making it easy to see different perspectives on the same topic.

### Sentiment Analysis
Each article is analyzed for emotional tone (Positive, Negative, Neutral) to help readers understand the article's framing.

### Clickbait Detection
Advanced ML models detect if an article's headline is clickbait, helping readers identify sensationalized content.

### Source Reliability
Articles are scored (0-100%) based on the reliability of their source, helping readers identify trustworthy content.

### Fact Cross-Checking
The system cross-references articles with other sources to identify potentially omitted information, promoting comprehensive news coverage.

### Real-time Engagement
Users can like articles and leave comments, with all interactions immediately persisted to MongoDB.

## 🌈 Customization

### Colors
The primary color scheme uses purple and pink gradients. To customize:
- Edit `app/globals.css` for global color variables
- Modify gradient classes in components (e.g., `from-purple-600 to-pink-600`)

### Animations
Custom animations are defined in `globals.css`:
- `animate-float`: Floating effect
- `animate-shimmer`: Shimmer effect
- Custom scrollbar styles

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb://127.0.0.1:27017/polaris_news_db` |
| `MONGO_AUTH_SOURCE` | MongoDB auth source | `admin` |
| `NEXT_PUBLIC_BASE_URL` | Base URL for API calls | `http://localhost:3000` |

## 🚀 Deployment

### Build for Production
```bash
npm run build
npm start
```

### Deploy to Vercel
The easiest way to deploy is using [Vercel](https://vercel.com):

1. Push your code to GitHub
2. Import the project in Vercel
3. Configure environment variables
4. Deploy!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Next.js team for the amazing framework
- Tailwind CSS for the styling system
- Lucide for the beautiful icons
- MongoDB for the database solution

---

Built with ❤️ by the Polaris News team

