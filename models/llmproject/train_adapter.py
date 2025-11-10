# ============================================================
# Imports
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import types


# ============================================================
# Residual Adapter (same as before)
# ============================================================
class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = F.relu(self.down(x))
        return x + self.up(z)


# ============================================================
# Injection helper
# ============================================================
def inject_adapter(layer, adapter):
    old_forward = layer.forward

    def new_forward(self, hidden_states, *args, **kwargs):
        out = old_forward(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            hidden = adapter(out[0])
            return (hidden,) + out[1:]
        return adapter(out)

    layer.forward = types.MethodType(new_forward, layer)


# ============================================================
# Model wrapper with adapters
# ============================================================
class MiniLMWithAdapters(nn.Module):
    def __init__(self, model_name, bottleneck=64, use_last_n_layers=2, device="cpu"):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False

        # Inject adapters
        layers = self.model.encoder.layer
        hidden_size = self.model.config.hidden_size

        target_ids = range(len(layers) - use_last_n_layers, len(layers))
        self.adapters = nn.ModuleList()

        for i in target_ids:
            adapter = ResidualAdapter(hidden_size, bottleneck)
            self.adapters.append(adapter)
            inject_adapter(layers[i], adapter)

        self.to(device)

    def encode(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(self.device)

        out = self.model(**enc)
        hidden = out.last_hidden_state

        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(1e-9)

        return F.normalize(emb, p=2, dim=1)


# ============================================================
# Dataset for article → summary
# ============================================================
class ArticleSummaryDataset(Dataset):
    def __init__(self, articles, summaries):
        self.articles = articles
        self.summaries = summaries

    def __getitem__(self, idx):
        return self.articles[idx], self.summaries[idx]

    def __len__(self):
        return len(self.articles)


# ============================================================
# Training function
# ============================================================
def train_adapter(
    model,
    articles,
    summaries,
    batch_size=8,
    epochs=3,
    lr=3e-4,
    print_every=20,
    show_gradient_norms=False
):

    ds = ArticleSummaryDataset(articles, summaries)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.adapters.parameters(), lr=lr)

    print("\n================ TRAINING ADAPTERS ================")
    print(f"Total training samples: {len(ds)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {model.device}")
    print("---------------------------------------------------")

    # Count parameters
    train_params = sum(p.numel() for p in model.adapters.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.model.parameters())

    print(f"Trainable adapter params: {train_params:,}")
    print(f"Frozen backbone params: {frozen_params:,}")
    print("===================================================\n")

    for epoch in range(1, epochs+1):
        epoch_loss = 0
        print(f"\n************* Epoch {epoch}/{epochs} *************")

        for step, (batch_articles, batch_summaries) in enumerate(dl):

            # Debug: show example
            if step == 0:
                print("\n▶ Example Article:", batch_articles[0][:120], "...")
                print("▶ Example Summary:", batch_summaries[0][:120], "...\n")

            emb_a = model.encode(batch_articles)
            emb_s = model.encode(batch_summaries)

            # Cosine similarity (maximize)
            # Full similarity matrix
            sims = emb_a @ emb_s.t()  # [batch, batch]

            labels = torch.arange(sims.size(0), device=model.device)

            temperature = 0.05
            loss = F.cross_entropy(sims / temperature, labels)

            optimizer.zero_grad()
            loss.backward()

            # Optional: print gradient norms
            if show_gradient_norms and step % print_every == 0:
                total_norm = 0
                for p in model.adapters.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item()
                print(f"[step {step}] Gradient norm: {total_norm:.4f}")

            optimizer.step()

            epoch_loss += loss.item()

            # Print detailed batch info
            if step % print_every == 0:
                print(f"\nStep {step}/{len(dl)}:")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Avg cos-sim (batch): {sims.mean().item():.4f}")
                print(f"  Cos-sim min/max: {sims.min().item():.4f} / {sims.max().item():.4f}")
                print(f"  Embedding norms: article={emb_a.norm(dim=1).mean():.4f}, summary={emb_s.norm(dim=1).mean():.4f}")

        print(f"\nEpoch {epoch} summary:")
        print(f"  Mean loss: {epoch_loss/len(dl):.4f}")
        print("***************************************************")

    print("\nSaving adapters → residual_adapters.pt")
    torch.save(model.adapters.state_dict(), "residual_adapters.pt")
    print("✅ Training complete!")

# ============================================================
# Main
# ============================================================
def main():

    # Load summarization dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
    articles = ds["article"]
    summaries = ds["highlights"]

    # Split
    split = int(len(articles)*0.9)
    train_articles = articles[:split]
    train_summaries = summaries[:split]

    model = MiniLMWithAdapters(
        "sentence-transformers/all-MiniLM-L6-v2",
        bottleneck=64,
        use_last_n_layers=2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    train_adapter(model, train_articles, train_summaries)


if __name__ == "__main__":
    main()
