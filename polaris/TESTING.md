# 🧪 Testing Guide - Polaris News

Complete guide to test all features of your news platform.

## Prerequisites

Before testing, ensure:
1. MongoDB is running
2. Development server is started (`npm run dev`)
3. Database has some articles (see sample data below)

## 📋 Test Checklist

### Home Page Testing

#### Visual Elements
- [ ] Hero section loads with animated background
- [ ] "Polaris News" logo displays correctly
- [ ] Feature pills (Bias Detection, Sentiment Analysis, Fact Checking) are visible
- [ ] Search bar renders with gradient border
- [ ] Statistics section shows 4 metrics
- [ ] Latest Stories section displays articles

#### Interactions
- [ ] Search bar accepts text input
- [ ] Search bar submits on Enter key
- [ ] Search button click navigates to search page
- [ ] Article cards are clickable
- [ ] Article cards show hover effects (scale, shadow)
- [ ] Stats cards show hover effects

#### Data Display
- [ ] Top 6 articles load correctly
- [ ] Article images display (or placeholder shown)
- [ ] Article titles and descriptions truncate properly
- [ ] Publication dates format correctly
- [ ] Author names display
- [ ] Like and comment counts show

#### Analysis Badges
- [ ] Bias labels display with correct colors
  - Left = Blue
  - Center/Neutral = Green
  - Right = Red
- [ ] Sentiment badges show
- [ ] Reliability scores display as percentage
- [ ] Clickbait warning appears (if applicable)

---

### Search Results Testing

#### Search Functionality
- [ ] Search query displays in URL
- [ ] Results load after submitting search
- [ ] Loading spinner shows while fetching
- [ ] Result count displays correctly

#### Column Layout
- [ ] Three columns visible on desktop
- [ ] Left column shows left-biased articles
- [ ] Center column shows neutral articles
- [ ] Right column shows right-biased articles
- [ ] Each column header shows count
- [ ] Empty state shows if column has no articles

#### Navigation
- [ ] "Back to Home" link works
- [ ] Search bar in header is functional
- [ ] Logo links back to home
- [ ] Article cards link to detail page

#### Responsive Design
- [ ] Mobile: Single column view
- [ ] Tablet: Two columns
- [ ] Desktop: Three columns

---

### Article Detail Page Testing

#### Content Display
- [ ] Article title displays correctly
- [ ] Featured image loads (full width)
- [ ] Author name shows
- [ ] Publication date formatted properly
- [ ] Source name displays
- [ ] Category badge visible
- [ ] Topic tag shows (if available)
- [ ] Full article content readable
- [ ] Description in styled box

#### AI Analysis Sidebar
- [ ] Bias classification displays
- [ ] Bias probability percentages show
- [ ] Bias explanation phrases listed (top 5)
- [ ] Sentiment label visible
- [ ] Sentiment probabilities display
- [ ] Clickbait detection shows
- [ ] Clickbait explanation (if detected)
- [ ] Source reliability score displays
- [ ] Reliability progress bar renders
- [ ] Fake news label shows
- [ ] Authenticity probabilities display

#### Engagement Features
- [ ] Like button functional
- [ ] Like count updates immediately
- [ ] Like persists to database
- [ ] Unlike works (toggle)
- [ ] Comment input accepts text
- [ ] Submit button posts comment
- [ ] New comment appears instantly
- [ ] Comment persists to database
- [ ] Comment shows user avatar
- [ ] Comment shows timestamp
- [ ] Share button opens share dialog (mobile)
- [ ] Share copies link (desktop)

#### Additional Sections
- [ ] Single source summary displays (if available)
- [ ] Multi-source summary shows (if available)
- [ ] Cross-reference articles listed (if available)
- [ ] Cross-reference links work
- [ ] Related articles section appears
- [ ] Related articles link correctly
- [ ] "View Original Article" button works
- [ ] Original article opens in new tab

#### Navigation
- [ ] Back button returns to previous page
- [ ] Logo returns to home
- [ ] Related article cards clickable

---

## 🗄️ Sample Test Data

### Minimal Test Article

Insert this into your MongoDB `articles` collection:

```javascript
db.articles.insertOne({
  id: "test-1",
  title: "Breaking: Revolutionary AI Technology Unveiled",
  description: "Scientists announce groundbreaking discovery in artificial intelligence that could change computing forever.",
  content: "In a stunning development today, researchers at TechLab have unveiled a new AI system that demonstrates unprecedented capabilities in natural language understanding and generation. The system, dubbed 'NeoMind,' represents a significant leap forward in machine learning technology.\n\nThe breakthrough comes after years of research into neural network architectures and training methodologies. According to lead researcher Dr. Jane Smith, 'This technology has the potential to revolutionize how we interact with computers and process information.'\n\nKey features of NeoMind include advanced reasoning capabilities, improved contextual understanding, and the ability to perform complex tasks with minimal training data. Industry experts are calling it a game-changer for the AI field.",
  author: "John Anderson",
  source: "Tech Daily News",
  url: "https://example.com/ai-breakthrough",
  imageUrl: "https://images.unsplash.com/photo-1677442136019-21780ecad995",
  publishedAt: new Date("2025-11-08"),
  category: "Technology",
  topics: ["AI", "Innovation", "Science"],
  
  // Analysis fields
  bias_classification_label: "Center",
  bias_classification_probs: {
    "Left": 0.2,
    "Center": 0.6,
    "Right": 0.2
  },
  bias_explain: [
    {
      phrase: "groundbreaking discovery",
      score: 0.75,
      weight: 0.8
    },
    {
      phrase: "game-changer",
      score: 0.65,
      weight: 0.7
    }
  ],
  
  sentiment_analysis_label: "Positive",
  sentiment_analysis_probs: {
    "Positive": 0.8,
    "Neutral": 0.15,
    "Negative": 0.05
  },
  
  clickbait_label: "not-clickbait",
  clickbait_score: 0.3,
  clickbait_explanation: "The headline is informative and accurately represents the content without sensationalism.",
  
  topic: "Artificial Intelligence",
  
  fake_news_label: "Real",
  fake_news_probs: {
    "Real": 0.95,
    "Fake": 0.05
  },
  source_reliability: 0.85,
  
  single_source_summary: "Scientists unveil NeoMind, a revolutionary AI system with advanced reasoning and contextual understanding capabilities.",
  muti_source_summary: "Multiple tech outlets confirm TechLab's announcement of NeoMind AI system, marking significant advancement in machine learning.",
  
  omitted_facts_articles: [
    {
      title: "AI Ethics Concerns Raised by Experts",
      url: "https://example.com/ai-ethics",
      omitted_segments: [
        {
          chunk: "Privacy concerns regarding data collection",
          max_similarity: 0.75
        }
      ]
    }
  ],
  
  likes: 42,
  comments: [
    {
      id: "comment-1",
      userId: "user-1",
      userName: "Tech Enthusiast",
      content: "This is incredibly exciting! Can't wait to see how this develops.",
      createdAt: new Date("2025-11-08T10:30:00"),
      likes: 5
    }
  ]
});
```

### Articles with Different Biases

```javascript
// Left-leaning article
db.articles.insertOne({
  id: "test-left-1",
  title: "Government Initiative Brings Hope to Communities",
  description: "New social program promises support for underserved populations.",
  content: "Progressive policies continue to shape positive change...",
  bias_classification_label: "Left",
  sentiment_analysis_label: "Positive",
  category: "Politics",
  // ... other required fields
});

// Right-leaning article  
db.articles.insertOne({
  id: "test-right-1",
  title: "Market Freedom Drives Economic Growth",
  description: "Deregulation leads to business expansion and job creation.",
  content: "Free market principles demonstrate effectiveness...",
  bias_classification_label: "Right",
  sentiment_analysis_label: "Positive",
  category: "Economy",
  // ... other required fields
});
```

---

## 🔍 API Testing

### Test GET /api/articles

```bash
# Get all articles
curl http://localhost:3000/api/articles

# Get limited articles
curl http://localhost:3000/api/articles?limit=3

# Search articles
curl http://localhost:3000/api/articles?q=technology

# Get by ID
curl http://localhost:3000/api/articles?id=test-1

# Filter by category
curl http://localhost:3000/api/articles?category=Technology
```

### Test GET /api/articles/[id]

```bash
curl http://localhost:3000/api/articles/test-1
```

### Test PATCH /api/articles/[id]

```bash
# Update likes
curl -X PATCH http://localhost:3000/api/articles/test-1 \
  -H "Content-Type: application/json" \
  -d '{"likes": 50}'

# Add comment
curl -X PATCH http://localhost:3000/api/articles/test-1 \
  -H "Content-Type: application/json" \
  -d '{
    "comment": {
      "userId": "test-user",
      "userName": "Test User",
      "content": "Great article!"
    }
  }'
```

---

## 🎨 Visual Testing

### Colors to Verify

- **Bias Colors**:
  - Left: Blue (#3b82f6)
  - Center: Green (#22c55e)
  - Right: Red (#ef4444)

- **Sentiment Colors**:
  - Positive: Green
  - Neutral: Yellow
  - Negative: Red

- **Reliability Colors**:
  - High (≥80%): Green
  - Medium (50-79%): Yellow
  - Low (<50%): Red

### Animations to Check

- Background orbs pulse
- Cards scale on hover
- Images zoom on hover
- Loading spinner rotates
- Transitions are smooth (150ms)
- No jank or lag

---

## 🐛 Common Issues & Solutions

### Issue: No articles showing
**Solution**: Check MongoDB connection and ensure articles collection has data

### Issue: Images not loading
**Solution**: Verify Next.js image configuration allows external domains

### Issue: Search returns no results
**Solution**: Ensure search query matches article fields (case-insensitive)

### Issue: Like/comment not persisting
**Solution**: Check API endpoint is reachable and MongoDB is writable

### Issue: TypeScript errors in terminal
**Solution**: These are often type definition issues; check if app runs despite errors

---

## ✅ Acceptance Criteria

The application is working correctly if:

1. ✅ Home page loads and displays articles
2. ✅ Search functionality returns filtered results
3. ✅ Search results organized by bias columns
4. ✅ Article detail page shows full content and analysis
5. ✅ Like and comment features work
6. ✅ All badges and indicators display correctly
7. ✅ Navigation works across all pages
8. ✅ Responsive design works on mobile
9. ✅ Dark mode is supported (system preference)
10. ✅ No console errors in browser

---

## 📸 Screenshots to Take

Document your testing with screenshots of:

1. Home page (full page)
2. Search results (3-column layout)
3. Article detail page (with sidebar)
4. Mobile view of home page
5. Comment section with posted comment
6. Analysis badges on article cards
7. Related articles section

---

## 🚀 Performance Testing

### Load Time Checks

- [ ] Home page loads in < 2 seconds
- [ ] Search results appear in < 1 second
- [ ] Article detail loads in < 1.5 seconds
- [ ] No layout shift during loading

### Network Testing

- [ ] Test with slow 3G throttling
- [ ] Images lazy load correctly
- [ ] API calls complete successfully

---

Happy Testing! 🎉

Report any issues or unexpected behavior for quick resolution.

