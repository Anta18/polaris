from pydantic import BaseModel, Field

class ArticleInput(BaseModel):
    title: str = Field(..., description="Article title", min_length=1)
    content: str = Field(..., description="Article content", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "You Won't Believe What Happened Next!",
                "content": "This is the article content..."
            }
        }

class AnalysisOutput(BaseModel):
    score: float = Field(..., description="Clickbait probability score (0 to 1)", ge=0.0, le=1.0)
    label: str = Field(..., description="Classification label: 'clickbait' or 'clean'")
    explanation: str = Field(..., description="Explanation of the prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.85,
                "label": "clickbait",
                "explanation": "The title uses emotional or sensational wording like 'you won't believe'."
            }
        }
