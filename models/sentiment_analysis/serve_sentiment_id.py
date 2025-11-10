import os, json
from typing import List, Dict

import torch
from torch import nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

MODEL_DIR= "./bert_adapter_sentiment"

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

class BertAdapterClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, bottleneck: int = 64, pooling: str = "mean"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden = self.bert.config.hidden_size
        self.pooling = pooling
        self.adapter = ResidualAdapter(hidden, bottleneck)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        for p in self.bert.parameters():
            p.requires_grad = False

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :] if self.pooling == "cls" else self.mean_pool(out.last_hidden_state, attention_mask)
        x = self.adapter(pooled)
        x = self.dropout(x)
        return self.classifier(x)

class PredictBody(BaseModel):
    text: str

class PredictBatch(BaseModel):
    items: List[str]

app = FastAPI(title="Sentiment Adapter Classifier (IDs)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.environ.get("MODEL_DIR", "./bert_adapter_sentiment")
with open(os.path.join(MODEL_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

labels: List[str] = cfg["labels"]
ids: List[int] = cfg["ids"]
tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)

model = BertAdapterClassifier(
    base_model_name=cfg["base_model"],
    num_labels=cfg["num_labels"],
    bottleneck=cfg.get("bottleneck", 64),
    pooling=cfg.get("pooling", "mean"),
)
state = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/")
def root():
    return {"status": "ok", "labels": labels, "ids": ids}

def _probs(texts: List[str]) -> List[Dict[str, float]]:
    enc = tokenizer(texts, truncation=True, padding=True, max_length=cfg.get("max_length", 256), return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
    return [{name: float(p[i]) for i, name in enumerate(labels)} for p in probs]

@app.post("/predict")
def predict(body: PredictBody):
    return {"probabilities": _probs([body.text])[0]}

@app.post("/predict_batch")
def predict_batch(body: PredictBatch):
    return {"probabilities": _probs(body.items)}