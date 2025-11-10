import re

class ExplanationEngine:
    """
    Explanation engine that prefers an LLM-based explanation but falls back to
    a lightweight heuristic when the model can't produce an explanation.

    Usage:
      - Preferred: call with loader provided (a ModelLoader instance). The
        loader will be asked to generate a brief explanation using the LLM and
        will take both title and content into account.
      - Fallback: if loader-based explanation fails, the heuristic below will
        produce a short, rule-based explanation derived from the title.
    """

    CLICKBAIT_WORDS = [
        "shocking", "unbelievable", "you won't believe", "you'll never believe",
        "crazy", "insane", "jaw dropping", "mind blowing", "mind-blowing",
        "surprising", "revealed", "exposed", "secret", "secrets",
        "amazing", "incredible", "unreal", "this will blow your mind",
        "doctors hate", "this one trick", "this simple trick",
        "what happened next", "the reason why", "the truth about"
    ]

    CLICKBAIT_PATTERNS = [
        (r'\d+\s+(things|ways|reasons|facts|secrets)', "Uses numbered list pattern"),
        (r'you won\'?t believe|you\'?ll never', "Uses incredulity phrases"),
        (r'\.{3,}', "Uses ellipsis to create suspense"),
        (r'!{2,}', "Uses excessive exclamation marks"),
        (r'^this\s+', "Starts with vague reference"),
    ]

    def generate(self, title: str, score: float, content: str = None, loader=None) -> str:
        """
        Generate an explanation for the clickbait prediction.

        If a `loader` is provided, it will be asked to generate the explanation
        with the LLM (so the explanation comes from the same model that
        produced or evaluated the score). If the loader cannot produce an
        explanation (missing generate support, runtime error, etc.), the
        method falls back to the original heuristic using only the `title`.

        Args:
            title: Article title (str)
            score: Clickbait probability score (float, 0-1)
            content: Article content (str) — passed to loader for context when available
            loader: Optional[ModelLoader] — used to drive LLM-based explanation

        Returns:
            str: Explanation text
        """
       
        if loader is not None:
            try:
                llm_expl = loader.explain_with_model(title=title, content=content, score=score)
                if llm_expl:
                    print("yes here")
                    print(title)
                    print(content)
                    print(score)
                    print(llm_expl.strip())
                    return llm_expl.strip()
            except Exception:
               
                pass

       
        if score < 0.5:
            return "The title does not contain strong emotional or manipulative indicators typical of clickbait."

        t = title.lower() if title else ""
        explanations = []

       
        for word in self.CLICKBAIT_WORDS:
            if word in t:
                explanations.append(f"uses emotional or sensational wording like '{word}'")
                break

        
        for pattern, description in self.CLICKBAIT_PATTERNS:
            if re.search(pattern, t, re.IGNORECASE):
                explanations.append(description.lower())
                break

        
        if title and "?" in title and score > 0.6:
            explanations.append("uses a question format that withholds key information")

        # Check for vague references
        if title and re.match(r'^(this|these|that|those)\s+', t):
            explanations.append("withholds specific details through vague references")

       
        if title and len(title) > 10 and sum(1 for c in title if c.isupper()) / len(title) > 0.5:
            explanations.append("uses excessive capitalization")

       
        if explanations:
            reason = explanations[0] if len(explanations) == 1 else f"{', '.join(explanations[:-1])}, and {explanations[-1]}"
            return f"The title {reason}, which are common clickbait patterns."

        
        if score > 0.7:
            return "The title exhibits multiple patterns commonly associated with sensational or manipulative framing designed to attract clicks."
        else:
            return "The title has some patterns that may indicate clickbait, though to a moderate degree."
