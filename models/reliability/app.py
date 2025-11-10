import os, json, math, re
from typing import Optional, List, Dict, Union
from datetime import datetime
from urllib.parse import urlsplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel, Field

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_DIR = os.getenv("MODEL_DIR", "./trained_model")
BASE_MODEL_FALLBACK = os.getenv("BASE_MODEL", "roberta-base")
MAX_LEN = int(os.getenv("MAX_LEN", "384"))

ADAPTER_TYPE = os.getenv("ADAPTER_TYPE", "residual")
N_ADAPTER_LAYERS = int(os.getenv("N_ADAPTER_LAYERS", "4"))
ADAPTER_BOTTLENECK = int(os.getenv("ADAPTER_BOTTLENECK", "64"))
SPARSE_TOPK = float(os.getenv("SPARSE_TOPK", "128"))
SPARSE_USE_FRACTION = os.getenv("SPARSE_USE_FRACTION", "false").lower() == "true"

SOURCES_JSON = os.getenv("SOURCE_RELIABILITY_JSON", "")
DEFAULT_SOURCE_MAP = {
    "apnews.com": 88,
    "reuters.com": 90,
    "bbc.com": 86,
    "nytimes.com": 85,
    "guardian.com": 82,
    "snopes.com": 92,
    "politifact.com": 92,
}


def device_setup():
    has_cuda = torch.cuda.is_available()
    bf16_ok = has_cuda and torch.cuda.is_bf16_supported()
    fp16_ok = has_cuda and (not bf16_ok)
    device = torch.device("cuda" if has_cuda else "cpu")
    amp_dtype = torch.bfloat16 if bf16_ok else (torch.float16 if fp16_ok else None)
    return device, amp_dtype


def make_meta_prefix(subject: Optional[str], date_str: Optional[str]) -> str:
    subj = (subject or "unknown").strip().replace(" ", "_")[:30]
    yr = "UNK"
    try:
        yr_val = pd_to_datetime_safe(date_str)
        yr = "UNK" if yr_val is None else str(int(yr_val.year))
    except Exception:
        pass
    return f"[SUBJ_{subj}] [YEAR_{yr}] "


def pd_to_datetime_safe(s: Optional[str]):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None


def domain_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlsplit(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or None
    except Exception:
        return None


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


SENSATIONAL = [
    "shocking",
    "unbelievable",
    "exposed",
    "secret",
    "scandal",
    "busted",
    "you won't believe",
    "BREAKING",
    "urgent",
    "must see",
    "truth about",
    "viral",
    "destroyed",
    "obliterates",
    "crushed",
    "nightmare",
    "hoax",
    "fake",
    "cover-up",
]
HEDGES = ["allegedly", "reportedly", "it seems", "it appears", "suggests", "rumor"]


def style_signals(text: str) -> Dict:
    raw = text or ""
    lower = raw.lower()
    tokens = re.findall(r"[A-Za-z]+", raw)
    n = max(1, len(tokens))
    exclam = raw.count("!")
    qmarks = raw.count("?")
    all_caps_tokens = sum(1 for t in tokens if len(t) >= 3 and t.isupper())
    all_caps_ratio = all_caps_tokens / n
    sensational_hits = sum(lower.count(w.lower()) for w in SENSATIONAL)
    hedge_hits = sum(lower.count(w.lower()) for w in HEDGES)

    sensational_score = min(
        1.0, (0.2 * exclam + 0.1 * qmarks + 5 * sensational_hits + 3 * all_caps_ratio)
    )
    return {
        "exclamation_count": exclam,
        "question_count": qmarks,
        "all_caps_ratio": round(all_caps_ratio, 4),
        "sensational_terms": int(sensational_hits),
        "hedge_terms": int(hedge_hits),
        "sensational_score": round(float(sensational_score), 4),
    }


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=64):
        super().__init__()
        self.down_layer = nn.Linear(hidden_size, bottleneck)
        self.up_layer = nn.Linear(bottleneck, hidden_size)
        nn.init.constant_(self.up_layer.weight, 0)
        nn.init.constant_(self.up_layer.bias, 0)

    def forward(self, x):
        return x + self.up_layer(F.relu(self.down_layer(x)))


class SparseAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=64, topk=128, use_fraction=False):
        super().__init__()
        self.down_layer = nn.Linear(hidden_size, bottleneck)
        self.up_layer = nn.Linear(bottleneck, hidden_size)
        nn.init.constant_(self.up_layer.weight, 0)
        nn.init.constant_(self.up_layer.bias, 0)
        self.gate = nn.Linear(hidden_size, 1)
        self.topk = topk
        self.use_fraction = use_fraction

    def sample_gumbel(self, shape, device="cpu", eps=1e-20):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def relaxed_topk_mask(self, logits: torch.Tensor, k: int, tau: float = 1.0):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        g = self.sample_gumbel(logits.shape, device=logits.device)
        probs = F.softmax((logits + g) / tau, dim=-1)
        topk_idx = torch.topk(logits, k=k, dim=-1).indices
        hard = torch.zeros_like(logits)
        hard[0, topk_idx[0]] = 1.0
        return ((hard - probs).detach() + probs).squeeze(0)

    def forward(self, x):
        # x: [B,T,H]
        B, T, H = x.shape
        k = (
            max(1, min(T, int(self.topk * T)))
            if self.use_fraction
            else max(1, min(T, int(self.topk)))
        )
        gate_logits = self.gate(x).squeeze(-1)  # [B,T]
        masks = [self.relaxed_topk_mask(gate_logits[i], k) for i in range(B)]
        mask = torch.stack(masks, dim=0).unsqueeze(-1)  # [B,T,1]
        delta = self.up_layer(F.relu(self.down_layer(x)))  # [B,T,H]
        return x + delta * mask


import types


def _wrap_roberta_layer_with_adapter(layer, adapter: nn.Module):
    original_forward = layer.forward

    def forward_with_adapter(self, hidden_states, *args, **kwargs):
        outputs = original_forward(hidden_states, *args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return adapter(outputs)
        elif isinstance(outputs, tuple):
            return (adapter(outputs[0]),) + outputs[1:]
        else:
            return outputs

    layer.forward = types.MethodType(forward_with_adapter, layer)


def inject_adapters(
    model: nn.Module,
    adapter_type: str,
    n_layers: int,
    bottleneck: int,
    sparse_topk: float,
    sparse_use_fraction: bool,
):
    # freeze backbone
    for p in model.base_model.parameters():
        p.requires_grad = False
    # keep classifier trainable
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    enc_layers = model.base_model.encoder.layer
    L = len(enc_layers)
    target = list(range(max(0, L - n_layers), L))
    hidden = model.config.hidden_size
    adapters = nn.ModuleList()
    for i in target:
        adp = (
            SparseAdapter(
                hidden, bottleneck, topk=sparse_topk, use_fraction=sparse_use_fraction
            )
            if adapter_type == "sparse"
            else ResidualAdapter(hidden, bottleneck)
        )
        adapters.append(adp)
        _wrap_roberta_layer_with_adapter(enc_layers[i], adp)
    model.adapter_modules = adapters  # register for (state_dict) saving/loading
    return model


def try_load_peft_model(adapter_path: str):
    try:
        from peft import PeftModel, PeftConfig

        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        base = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path or BASE_MODEL_FALLBACK, num_labels=2
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        tok = AutoTokenizer.from_pretrained(
            (
                adapter_path
                if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json"))
                else (peft_cfg.base_model_name_or_path or BASE_MODEL_FALLBACK)
            ),
            use_fast=True,
        )
        return tok, model
    except Exception:
        return None, None


def load_tokenizer(model_dir: str):
    try:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(BASE_MODEL_FALLBACK, use_fast=True)


def load_with_adapters(model_dir: str):
    """
    Handles both:
    1) PEFT-style LoRA/DyLoRA adapters saved in `model_dir`
    2) HF model saved with custom adapters embedded via state_dict
    """
    tok, model = try_load_peft_model(model_dir)
    if tok is not None and model is not None:
        return tok, model

    tok = load_tokenizer(model_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        (
            model_dir
            if os.path.exists(os.path.join(model_dir, "config.json"))
            else BASE_MODEL_FALLBACK
        ),
        num_labels=2,
    )

    base = inject_adapters(
        base,
        adapter_type=ADAPTER_TYPE,
        n_layers=N_ADAPTER_LAYERS,
        bottleneck=ADAPTER_BOTTLENECK,
        sparse_topk=SPARSE_TOPK,
        sparse_use_fraction=SPARSE_USE_FRACTION,
    )

    pt_bin = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(pt_bin):
        state = torch.load(pt_bin, map_location="cpu")
        base.load_state_dict(state, strict=False)

    return tok, base


class ArticleInput(BaseModel):
    title: str = Field(..., description="Headline/title of the article")
    text: str = Field(..., description="Full article text or body")
    subject: Optional[str] = Field(None, description="Topic/category string")
    date: Optional[str] = Field(
        None, description="Article date (YYYY-MM-DD or ISO8601)"
    )
    source_url: Optional[str] = Field(None, description="URL of the article")


class BatchInput(BaseModel):
    items: List[ArticleInput]


class ScoreOutput(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    style: Dict[str, float]
    source: Dict[str, Optional[Union[str, int, float]]]
    meta: Dict[str, Optional[str]]


import pandas as pd

app = FastAPI(title="Trustworthiness Scorer (RoBERTa + Adapters + LoRA/DyLoRA)")
tokenizer, model = load_with_adapters(MODEL_DIR)
device, amp_dtype = device_setup()
model.to(device).eval()

SOURCE_MAP = DEFAULT_SOURCE_MAP.copy()
if SOURCES_JSON and os.path.exists(SOURCES_JSON):
    try:
        with open(SOURCES_JSON, "r", encoding="utf-8") as f:
            SOURCE_MAP.update(json.load(f))
    except Exception:
        pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "bf16_or_fp16": (str(amp_dtype) if amp_dtype else None),
        "model_dir": MODEL_DIR,
    }


@app.post("/reload_sources")
def reload_sources():
    global SOURCE_MAP
    if SOURCES_JSON and os.path.exists(SOURCES_JSON):
        with open(SOURCES_JSON, "r", encoding="utf-8") as f:
            SOURCE_MAP = json.load(f)
    return {"loaded": len(SOURCE_MAP)}


def _score_one(x: ArticleInput) -> ScoreOutput:
    prefix = make_meta_prefix(x.subject, x.date)
    full_text = f"{prefix}{x.title} {x.text}"

    enc = tokenizer(
        full_text,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**enc)
        else:
            out = model(**enc)

    logits = out.logits[0].float().cpu()
    probs = torch.softmax(logits, dim=-1).tolist()
    p_misleading, p_trust = probs[0], probs[1]

    dom = domain_from_url(x.source_url)
    rel = SOURCE_MAP.get(dom, None)
    rel_norm = None if rel is None else (rel / 100.0)

    if rel_norm is None:
        final_conf = p_trust
    else:
        final_conf = 0.8 * p_trust + 0.2 * rel_norm

    style = style_signals(x.text or "")

    label = "trustworthy" if final_conf >= 0.5 else "misleading"

    return ScoreOutput(
        label=label,
        confidence=round(float(final_conf), 6),
        probabilities={
            "misleading": round(float(p_misleading), 6),
            "trustworthy": round(float(p_trust), 6),
        },
        style=style,
        source={
            "domain": dom,
            "reliability_0_100": (None if rel is None else int(rel)),
        },
        meta={
            "adapter_type": ADAPTER_TYPE,
            "n_adapter_layers": str(N_ADAPTER_LAYERS),
            "adapter_bottleneck": str(ADAPTER_BOTTLENECK),
            "sparse_topk": str(SPARSE_TOPK),
            "sparse_use_fraction": str(SPARSE_USE_FRACTION),
            "max_len": str(MAX_LEN),
        },
    )


@app.post("/v1/score_article", response_model=ScoreOutput)
def score_article(inp: ArticleInput):
    return _score_one(inp)


@app.post("/v1/score_batch")
def score_batch(inp: BatchInput):
    return {"results": [_score_one(x).dict() for x in inp.items]}
