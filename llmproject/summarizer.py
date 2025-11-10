from typing import List, Dict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")


def format_segments_for_prompt(segments: List[Dict]) -> str:
    formatted = []
    print("here")
    print(segments)
    for segment in segments:
        chunk = segment.get("chunk", "").strip()
        if not chunk:
            continue

        max_sim = segment.get("max_similarity", 0.0)
        matches = segment.get("nearest_matches", []) or []

        lines = [
            f"Candidate detail: {chunk}",
            f"Highest similarity vs peers: {max_sim:.2f}",
        ]

        for match in matches[:2]:
            peer_text = match.get("text", "").strip()
            if peer_text:
                lines.append(f"Peer coverage ({match.get('similarity', 0.0):.2f}): {peer_text}")

        formatted.append("\n".join(lines))

    return "\n\n".join(formatted)


def create_comparative_prompt(formatted_segments: str) -> str:
    return f"""Context: Comparative review of news articles about the same topic.
Goal: Surface facts or perspectives that appear in some sources but are missing or under-emphasised in others.

Evidence:
{formatted_segments}

Instructions:
- Highlight concrete facts, actors, timelines, and consequences that look unique.
- Mention when a detail is absent or muted in peer coverage.
- Avoid repeating identical wording from the evidence; paraphrase.
- Present findings as concise bullet-style statements.

Output: List the most compelling potentially omitted facts and why they matter."""


def summarize_unique_chunks(segments: List[str], max_length: int = 220) -> str:
    if not segments:
        return "No significant omitted facts detected."


    segments = [{"chunk": text, "summary": ""} for text in segments]

    formatted_evidence = format_segments_for_prompt(segments)
    if not formatted_evidence:
        return "No significant omitted facts detected."

    prompt = create_comparative_prompt(formatted_evidence)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=60,
            temperature=0.25,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()
