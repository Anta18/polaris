import torch
from app.model_loader import ModelLoader
from app.explainer import ExplanationEngine
from fastapi.responses import JSONResponse
class DetectorPipeline:
    def __init__(self):
        
        self.loader = ModelLoader()
        self.explainer = ExplanationEngine()
        self.initialized = True  # Flag to track initialization

    def analyze(self, title, content):
        """
        Analyze an article for clickbait detection.
        
        Args:
            title: Article title (str)
            content: Article content (str)
            
        Returns:
            tuple: (score, label, explanation)
                - score: float between 0 and 1
                - label: "clickbait" or "clean"
                - explanation: str explaining the prediction
        """
        
        if not title or not isinstance(title, str):
            raise ValueError("Title must be a non-empty string")
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        
    
        inputs = self.loader.encode(title, content)

        
        with torch.no_grad():
            outputs = self.loader.model(**inputs)
            
            logits = outputs.logits
            if logits.dim() > 1:
                logits = logits.squeeze()
            score = torch.sigmoid(logits).item()
            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, score))

        # Determine label
        label = "clickbait" if score > 0.7 else "clean"


        explanation = self.explainer.generate(title, score, content=content, loader=self.loader)

        
        return score, label, explanation
