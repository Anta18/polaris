# 🚀 Quick Start Guide - Polaris News

Get your news platform up and running in minutes!

## Prerequisites

- **Node.js** 18 or higher
- **MongoDB** running locally or remotely
- **npm** or **yarn** package manager

## Step-by-Step Setup

### 1. Install Dependencies

```bash
cd polaris
npm install
```

### 2. Set Up MongoDB

If you don't have MongoDB installed, you can:

**Option A: Install MongoDB locally**
- Download from [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
- Start MongoDB service

**Option B: Use MongoDB Atlas (Cloud)**
- Create a free account at [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
- Get your connection string

### 3. Configure Environment Variables

Create a `.env.local` file in the `polaris` directory:

```env
# For local MongoDB
MONGO_URI=mongodb://127.0.0.1:27017/polaris_news_db
MONGO_AUTH_SOURCE=admin

# Or for MongoDB Atlas
# MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/polaris_news_db

# Application URL
NEXT_PUBLIC_BASE_URL=http://localhost:3000
```

### 4. Seed Database (Optional)

To test the application, you'll want to add some articles to your database. You can:

1. Use MongoDB Compass or MongoDB Shell to insert sample data
2. Use your existing article collection
3. Create articles programmatically through the API

**Sample Article Structure:**

```javascript
{
  "id": "article-1",
  "title": "Breaking: New Technology Revolutionizes AI",
  "description": "Scientists announce groundbreaking discovery in artificial intelligence.",
  "content": "Full article content goes here...",
  "author": "Jane Doe",
  "source": "Tech News",
  "url": "https://example.com/article",
  "imageUrl": "https://example.com/image.jpg",
  "publishedAt": new Date(),
  "category": "Technology",
  "topics": ["AI", "Science"],
  "bias_classification_label": "Center",
  "sentiment_analysis_label": "Positive",
  "clickbait_label": "not-clickbait",
  "source_reliability": 0.85,
  "likes": 0,
  "comments": []
}
```

### 5. Start the Development Server

```bash
npm run dev
```

The application will start at [http://localhost:3000](http://localhost:3000)

## 🎉 You're Ready!

Visit the application in your browser:

### Home Page
- Browse top 6 articles
- Use the search bar

### Search
- Enter a query and see results organized by bias (Left/Center/Right)

### Article Detail
- Click any article card to view full details
- See AI analysis
- Like and comment on articles

## 📚 Next Steps

1. **Add Real Data**: Populate your MongoDB with actual news articles
2. **Customize Styling**: Modify colors and themes in `app/globals.css`
3. **Connect AI Models**: Integrate with your ML models for analysis
4. **Deploy**: Deploy to Vercel, AWS, or your preferred platform

## 🐛 Troubleshooting

### MongoDB Connection Error

**Problem:** "Failed to connect to MongoDB"

**Solution:**
- Ensure MongoDB is running
- Check your `MONGO_URI` in `.env.local`
- Verify firewall settings

### No Articles Showing

**Problem:** Home page shows "No articles yet"

**Solution:**
- Add articles to your MongoDB collection
- Check database name matches `MONGO_URI`
- Verify collection name is "articles"

### Images Not Loading

**Problem:** Article images show placeholder

**Solution:**
- Ensure `imageUrl` field contains valid URLs
- Check Next.js image configuration in `next.config.ts`
- Verify external image domains are allowed

### API Errors

**Problem:** API returns 500 errors

**Solution:**
- Check MongoDB connection
- Review server logs in terminal
- Verify article schema matches database

## 📞 Need Help?

- Check the main [README.md](./README.md) for detailed documentation
- Review API endpoints in [README.md](./README.md#-api-endpoints)
- Examine the database schema

## 🎨 Customization Tips

### Change Color Scheme

Edit `app/globals.css`:
```css
/* Change primary gradient colors */
.bg-gradient-to-r.from-purple-600.to-pink-600 {
  /* Your custom colors */
}
```

### Modify Logo/Brand Name

Edit `app/page.tsx`, `app/layout.tsx`, and other pages to change "Polaris News" to your brand name.

### Add More Features

- Implement user authentication
- Add article bookmarking
- Create category filters
- Build RSS feed integration
- Add email notifications

---

Happy coding! 🚀

