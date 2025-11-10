# train_adapter_roberta.py
import os, json, argparse, random, math
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer, AutoModel,
    DataCollatorWithPadding, Trainer, TrainingArguments
)

# -------------------------
# Adapter module (residual)
# -------------------------
class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        # Zero-init the up-projection (weights AND bias)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H) or (B, T, H)
        if x.dim() == 2:
            delta = self.up(self.act(self.down(x)))
        elif x.dim() == 3:
            B, T, H = x.size()
            delta = self.up(self.act(self.down(x)))
        else:
            raise ValueError("Adapter expects 2D or 3D tensor")
        return x + delta

# -------------------------
# Model: RoBERTa + Adapter
# -------------------------
class RobertaAdapterClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, bottleneck: int = 64, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(base_model_name)
        hidden = self.roberta.config.hidden_size
        self.adapter = ResidualAdapter(hidden_size=hidden, bottleneck=bottleneck)
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

        # Freeze all RoBERTa parameters
        for p in self.roberta.parameters():
            p.requires_grad = False

        # Optionally set weighted CE for class imbalance
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def mean_pooling(self, last_hidden_state, attention_mask):
        # (B, T, H), (B, T)
        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1)  # avoid div by zero
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)  # (B, H)
        adapted = self.adapter(pooled)  # residual
        x = self.dropout(adapted)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

# -------------------------
# Dataset
# -------------------------
class BiasDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: Dict[str, int], max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.label2id = label2id

        # Build input_text per spec: Title + " [SEP] " + Text + " [SEP] " + Source
        def safe_text(x):
            if pd.isna(x): return ""
            s = str(x)
            # quick de-mojibake for common cases
            s = s.replace("â€™", "’").replace("â€œ", "“").replace("â€", "”").replace("â€“", "–").replace("â€”", "—")
            return s

        titles = df["Title"].apply(safe_text).fillna("")
        texts  = df["Text"].apply(safe_text).fillna("")
        sources = df["Source"].apply(safe_text).fillna("")
        self.samples = (titles + " [SEP] " + texts + " [SEP] " + sources).tolist()

        biases = df["Bias"].str.strip().str.lower()
        self.labels = [label2id[b] for b in biases]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with Title, Text, Source, Bias columns")
    parser.add_argument("--base_model", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="./roberta_adapter_bias")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)  # adapters often like a slightly higher LR
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load & normalize labels
    df = pd.read_csv(args.csv)
    needed_cols = {"Title", "Text", "Source", "Bias"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # canonical label order
    labels = ["lean left", "left", "center", "right", "lean right"]
    label2id = {lbl: i for i, lbl in enumerate(labels)}

    # Clean Bias column to canonical set
    df["Bias"] = df["Bias"].str.strip().str.lower()
    bad = set(df["Bias"].unique()) - set(labels)
    if bad:
        raise ValueError(f"Found unexpected labels: {bad}. Expected one of {labels}")

    # Split
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=args.seed, stratify=df["Bias"].values
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Datasets
    train_ds = BiasDataset(train_df, tokenizer, label2id)
    val_ds = BiasDataset(val_df, tokenizer, label2id)

    # Class weights (balanced)
    counts = train_df["Bias"].value_counts().reindex(labels).fillna(0).values.astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0)) / len(counts)
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Model
    model = RobertaAdapterClassifier(
        base_model_name=args.base_model,
        num_labels=len(labels),
        bottleneck=args.bottleneck,
        class_weights=class_weights
    )

    # Data collator (pad to longest in batch)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # Save weights + tokenizer + small config for the server
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model,
            "num_labels": len(labels),
            "labels": labels,
            "bottleneck": args.bottleneck
        }, f, ensure_ascii=False, indent=2)

    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()
