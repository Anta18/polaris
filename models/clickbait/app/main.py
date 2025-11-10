from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.schemas import ArticleInput, AnalysisOutput
from app.pipeline import DetectorPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clickbait Detection API",
    description="FastAPI service for detecting clickbait and sensationalism in news articles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = None
try:
    pipeline = DetectorPipeline()
    if not hasattr(pipeline, 'initialized') or not pipeline.initialized:
        raise RuntimeError("Pipeline initialization incomplete")
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {str(e)}")
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Clickbait Detection API",
        "pipeline_loaded": pipeline is not None
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalysisOutput)
def analyze_article(data: ArticleInput):
    """
    Analyze an article for clickbait detection.
    
    Args:
        data: ArticleInput containing title and content
        
    Returns:
        AnalysisOutput with score, label, and explanation
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Please check model files.")
    
    try:
        score, label, explanation = pipeline.analyze(data.title, data.content)
        
        payload = {
            "score": score,  # Round to 4 decimal places
            "label": label,
            "explanation": explanation,
        }
        return JSONResponse(content=payload, status_code=200)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during analysis")
