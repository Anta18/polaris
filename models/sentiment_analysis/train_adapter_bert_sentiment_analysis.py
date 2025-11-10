import os, json, argparse, random, inspect
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, Trainer, TrainingArguments

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
    def __init__(self, base_model_name: str, num_labels: int, bottleneck: int = 64, class_weights: Optional[torch.Tensor] = None, pooling: str = "mean"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden = self.bert.config.hidden_size
        self.pooling = pooling
        self.adapter = ResidualAdapter(hidden, bottleneck)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)

        for p in self.bert.parameters():
            p.requires_grad = False

        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0), persistent=False)

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :] if self.pooling == "cls" else self.mean_pool(out.last_hidden_state, attention_mask)
        x = self.adapter(pooled)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            ce = nn.CrossEntropyLoss(weight=self.class_weights if self.class_weights.numel() > 0 else None)
            loss = ce(logits, labels)
        return {"loss": loss, "logits": logits}

class SentimentIdsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1_score(labels, preds, average="macro")}

# def make_training_args(**kw):
#     sig = inspect.signature(TrainingArguments.__init__)
#     supported = set(sig.parameters.keys())
#     out = {k: v for k, v in kw.items() if k in supported}
#     if "evaluation_strategy" not in supported and kw.get("evaluation_strategy"):
#         if "evaluate_during_training" in supported:
#             out["evaluate_during_training"] = True
#     return TrainingArguments(**out)

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--csv", required=True)
    ap.add_argument("--base_model", default="bert-base-uncased")
    ap.add_argument("--output_dir", default="./bert_adapter_sentiment")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pooling", choices=["mean","cls"], default="mean")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    splits = {'train': 'train_df.csv', 'validation': 'val_df.csv', 'test': 'test_df.csv'}
    df = pd.read_csv("hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/" + splits["train"])
    for col in ["text", "label"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)

    ids_sorted = sorted(df["label"].unique().tolist())
    if "sentiment" in df.columns:
        name_for = (df[["label", "sentiment"]].dropna()
                    .groupby("label")["sentiment"]
                    .agg(lambda s: Counter(s.astype(str)).most_common(1)[0][0]))
        id2label = {int(k): str(v) for k, v in name_for.to_dict().items()}
    else:
        id2label = {i: str(i) for i in ids_sorted}
    labels = [id2label[i] for i in ids_sorted]
    num_labels = len(ids_sorted)

    train_df, val_df = train_test_split(df, test_size=args.val_ratio, random_state=args.seed, stratify=df["label"])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    train_ds = SentimentIdsDataset(train_df, tokenizer, max_length=args.max_length)
    val_ds   = SentimentIdsDataset(val_df, tokenizer, max_length=args.max_length)

    counts = train_df["label"].value_counts().reindex(ids_sorted).fillna(0).values.astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0)) / len(counts)
    class_weights = torch.tensor(weights, dtype=torch.float)

    model = BertAdapterClassifier(
        base_model_name=args.base_model,
        num_labels=num_labels,
        bottleneck=args.bottleneck,
        class_weights=class_weights,
        pooling=args.pooling
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("Eval:", trainer.evaluate())

    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model,
            "labels": labels,
            "ids": ids_sorted,
            "id2label": id2label,
            "num_labels": num_labels,
            "bottleneck": args.bottleneck,
            "pooling": args.pooling,
            "max_length": args.max_length
        }, f, ensure_ascii=False, indent=2)

    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()