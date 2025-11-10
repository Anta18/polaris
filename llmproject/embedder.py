import os
import torch

from train_adapter import MiniLMWithAdapters


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ADAPTER_PATH = "residual_adapters.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MiniLMWithAdapters(
    model_name=MODEL_NAME,
    bottleneck=64,
    use_last_n_layers=2,
    device=device
)

if os.path.exists(ADAPTER_PATH):
    state = torch.load(ADAPTER_PATH, map_location=device)
    missing, unexpected = model.adapters.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"⚠ Adapter state mismatch. Missing: {missing}, Unexpected: {unexpected}")
else:
    print("⚠ No adapter weights found; using base MiniLM.")

model.eval()


def get_embedding(texts, batch_size: int = 32):
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            prefixed = [f"passage: {text}" for text in batch]
            batch_embs = model.encode(prefixed)
            embeddings.append(batch_embs.cpu())

    if not embeddings:
        return torch.empty((0, model.model.config.hidden_size))

    if len(embeddings) == 1:
        return embeddings[0]

    return torch.cat(embeddings, dim=0)
