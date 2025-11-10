import os, json
from typing import List, Dict, Optional

import torch
from torch import nn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
import string
from fastapi import Query

try:
    from ftfy import fix_text as _fix_text
    def _clean(s: Optional[str]) -> str:
        return _fix_text("" if s is None else str(s))
except Exception:
    def _clean(s: Optional[str]) -> str:
        if s is None:
            return ""
        s = str(s)
        try:
            s2 = s.encode("latin1").decode("utf-8")
            if len(s2) >= 0.6 * len(s):
                s = s2
        except Exception:
            pass
        repl = {
            "â€™": "’", "â€˜": "‘",
            "â€œ": "“", "â€": "”",
            "â€“": "–", "â€”": "—",
            "â€¦": "…", "Â": "",
            "âĢĻ": "’",
        }
        for a, b in repl.items():
            s = s.replace(a, b)
        return s

MODEL_DIR = "./roberta_adapter_bias"

class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        self.act = nn.ReLU()
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

class RobertaAdapterClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, bottleneck: int = 64):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(base_model_name)
        hidden = self.roberta.config.hidden_size
        self.adapter = ResidualAdapter(hidden, bottleneck)
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        for p in self.roberta.parameters():
            p.requires_grad = False
    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
    def forward(self, input_ids=None, attention_mask=None):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(out.last_hidden_state, attention_mask)
        adapted = self.adapter(pooled)
        x = self.dropout(adapted)
        logits = self.classifier(x)
        return logits

class PredictBody(BaseModel):
    Title: Optional[str] = ""
    Text: Optional[str] = ""
    Source: Optional[str] = ""

class RawBody(BaseModel):
    input_text: str

app = FastAPI(title="Bias Adapter Classifier", version="1.0")

# Load artifacts
MODEL_DIR = os.environ.get("MODEL_DIR", "./roberta_adapter_bias")
with open(os.path.join(MODEL_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

labels: List[str] = cfg["labels"]
tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
model = RobertaAdapterClassifier(
    base_model_name=cfg["base_model"],
    num_labels=len(labels),
    bottleneck=cfg.get("bottleneck", 64),
)

state = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# def _fix_encoding(s: Optional[str]) -> str:
#     if s is None:
#         return ""
#     s = str(s)
#     # try Latin-1 -> UTF-8 roundtrip (common Windows mojibake fix)
#     try:
#         s2 = s.encode("latin1").decode("utf-8")
#         # accept if it didn't shrink dramatically (heuristic)
#         if len(s2) >= 0.6 * len(s):
#             s = s2
#     except Exception:
#         pass
#     # common replacements
#     repl = {
#         "â€™": "’", "â€˜": "‘",
#         "â€œ": "“", "â€": "”",
#         "â€“": "–", "â€”": "—",
#         "â€¦": "…", "Â": "",
#         "âĢĻ": "’", "âĢ£": "“", "âĢ¥": "”",
#     }
#     for a, b in repl.items():
#         s = s.replace(a, b)
#     return s

def _pack_input(title: str, text: str, source: str) -> str:
    return f"{_clean(title)} [SEP] {_clean(text)} [SEP] {_clean(source)}"

def _predict_str(input_text: str) -> Dict[str, float]:
    enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten().tolist()
    return {lbl: float(p) for lbl, p in zip(labels, probs)}

@app.post("/predict")
def predict(body: PredictBody):
    input_text = _pack_input(body.Title, body.Text, body.Source)
    return {"input_text": input_text, "probabilities": _predict_str(input_text)}

@app.post("/predict_raw")
def predict_raw(body: RawBody):
    return {"probabilities": _predict_str(body.input_text)}

def _token_ids_text_scores(
    input_text: str,
    target_idx: Optional[int] = None,
) -> Tuple[List[int], List[str], List[float], int, float]:
    """
    Returns (token_ids, tokens, token_scores, pred_idx, pred_prob)
    - Zeros out scores for tokens overlapping the literal " [SEP] " markers
    - Zeros out scores for tokens in the final Source segment (after 2nd [SEP])
    """
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        logits = model(**enc)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax().item())
        pred_prob = float(probs[pred_idx].item())
    if target_idx is None:
        target_idx = pred_idx

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state.detach().requires_grad_(True)

        pooled  = model.mean_pooling(hs, attention_mask)
        adapted = model.adapter(pooled)
        x       = model.dropout(adapted)
        logits2 = model.classifier(x)

        selected = logits2[0, target_idx]
        selected.backward()

        grads = hs.grad.abs()[0]
        token_scores = grads.sum(dim=-1)
        token_scores *= attention_mask[0]

        sep = " [SEP] "
        seps = []
        i = -1
        while True:
            i = input_text.find(sep, i + 1)
            if i == -1:
                break
            seps.append((i, i + len(sep)))
        source_start = seps[1][1] if len(seps) >= 2 else None

        keep = torch.ones_like(token_scores)
        for t, (a, b) in enumerate(offsets):
            for s, e in seps:
                if not (b <= s or a >= e):
                    keep[t] = 0
                    break
            if source_start is not None and a >= source_start:
                keep[t] = 0

        token_scores = (token_scores * keep).cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    return input_ids[0].cpu().tolist(), tokens, token_scores, pred_idx, pred_prob

def _merge_roberta_bpe_to_words(tokens: List[str], scores: List[float], token_ids: List[int]) -> List[Tuple[str, float]]:
    """
    Merge byte-level BPE tokens into words; treat '[', ']', 'sep' as boundaries.
    """
    special_ids = set(tokenizer.all_special_ids)
    words: List[str] = []
    w_scores: List[float] = []
    cur_w, cur_s = "", 0.0

    def flush():
        nonlocal cur_w, cur_s
        w = cur_w.strip()
        if w:
            words.append(w)
            w_scores.append(cur_s)
        cur_w, cur_s = "", 0.0

    for tid, tok, sc in zip(token_ids, tokens, scores):
        if tid in special_ids:
            flush()
            continue

        starts_new = tok.startswith("Ġ")
        piece = tok.lstrip("Ġ").replace("Ċ", "").replace("▁", "")

        low = piece.lower()
        if low in {"[", "]", "sep"} or (piece and all(ch in string.punctuation for ch in piece)):
            flush()
            continue

        if starts_new:
            flush()
            cur_w = piece
            cur_s = float(sc)
        else:
            cur_w += piece
            cur_s += float(sc)

    flush()
    return [(w.strip(), s) for w, s in zip(words, w_scores) if w.strip()]

def _explain_text(input_text: str, top_k: int = 10):
    token_ids, toks, t_scores, pred_idx, pred_prob = _token_ids_text_scores(input_text)
    word_scores = _merge_roberta_bpe_to_words(toks, t_scores, token_ids)

    word_scores.sort(key=lambda x: x[1], reverse=True)
    top = word_scores[:max(1, top_k)]

    total = sum(s for _, s in top) or 1.0
    top_payload = [{"phrase": _clean(w), "score": float(s), "weight": float(s / total)} for w, s in top]

    return {
        "predicted_label": labels[pred_idx],
        "predicted_prob": pred_prob,
        "top_explanations": top_payload,
    }

@app.post("/explain")
def explain(body: PredictBody, top_k: int = Query(10, ge=1, le=50)):
    """
    Returns the predicted label and the top-k words/phrases (by saliency)
    that most contributed to that prediction.
    """
    input_text = _pack_input(body.Title, body.Text, body.Source)
    return {"input_text": input_text, **_explain_text(input_text, top_k=top_k)}