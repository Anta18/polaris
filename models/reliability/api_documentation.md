# Trustworthiness Scoring API – Usage Guide

This service scores a news article as **trustworthy** or **misleading** using your trained **RoBERTa + adapters (+ LoRA/DyLoRA if present)** model. It reproduces your training pre-processing (`[SUBJ_*][YEAR_*]` prefix), runs on GPU (bf16/fp16 when available), and can blend model scores with optional source-reliability metadata.

---

## Base URL

```
http://<host>:8000
```

Interactive docs:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- Health: `/health`

Content type: `application/json`

---

## Environment variables

Set these before starting `uvicorn`:

| Var                   | Default                       | Purpose                                                                                                         |
| --------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `MODEL_DIR`           | `./authcred_roberta_adapters` | Folder with your trained checkpoint (`model.safetensors` and tokenizer files) or a `checkpoint-XXXX` subfolder. |
| `BASE_MODEL`          | `roberta-base`                | Fallback base model if needed (also used by PEFT loaders).                                                      |
| `MAX_LEN`             | `384`                         | Tokenization max length.                                                                                        |
| `ADAPTER_TYPE`        | `residual`                    | `residual` or `sparse` — must match training.                                                                   |
| `N_ADAPTER_LAYERS`    | `4`                           | Number of last encoder layers that have adapters.                                                               |
| `ADAPTER_BOTTLENECK`  | `64`                          | Adapter bottleneck size.                                                                                        |
| `SPARSE_TOPK`         | `128`                         | If `ADAPTER_TYPE=sparse`: absolute k (or fraction when `SPARSE_USE_FRACTION=true`).                             |
| `SPARSE_USE_FRACTION` | `false`                       | Treat `SPARSE_TOPK` as fraction in (0,1].                                                                       |

> **PEFT (LoRA/DyLoRA) support:** If `MODEL_DIR` is a PEFT adapter folder, the loader detects it automatically and attaches it to the base model.

---

## Endpoints

### 1) Health

`GET /health`

**200 Response**

```json
{
  "status": "ok",
  "device": "cuda",
  "amp_dtype": "torch.bfloat16",
  "model_dir": "/path/to/checkpoint",
  "adapter_type": "residual",
  "n_adapter_layers": 4
}
```

---

### 2) Score a single article

`POST /v1/score_article`

**Request body**

```json
{
  "title": "Government announces new climate pact",
  "text": "Officials said the agreement was reached after weeks of negotiation...",
  "subject": "climate",
  "date": "2024-11-05",
  "source_url": "https://www.reuters.com/world/europe/..."
}
```

**Response (200)**

```json
{
  "label": "trustworthy",
  "confidence": 0.871234,
  "probabilities": {
    "misleading": 0.128766,
    "trustworthy": 0.871234
  },
  "style": {
    "exclamation_count": 0,
    "question_count": 0,
    "all_caps_ratio": 0.0,
    "sensational_terms": 0,
    "hedge_terms": 0,
    "sensational_score": 0.0
  },
  "source": {
    "domain": "reuters.com",
    "reliability_0_100": 90
  },
  "meta": {
    "max_len": "384",
    "adapter_type": "residual",
    "n_adapter_layers": "4",
    "adapter_bottleneck": "64",
    "sparse_topk": "128",
    "sparse_use_fraction": "False"
  }
}
```

**Semantics**

- `probabilities.trustworthy` and `probabilities.misleading` are the model softmax outputs (label 1 = trustworthy, label 0 = misleading).
- `confidence` = blended score:

  - If `source_url` domain has a reliability score `r∈[0,100]`: `0.8 * p_trustworthy + 0.2 * (r/100)`.
  - Otherwise: `p_trustworthy`.

- `style` contains quick linguistic heuristics (caps, sensational terms, etc.).

**Errors**

- `422 Unprocessable Entity` – invalid payload (e.g., missing required fields or wrong types).
- `500 Internal Server Error` – unexpected exception (see server logs).

---

### 3) Score a batch

`POST /v1/score_batch`

**Request body**

```json
{
  "items": [
    {
      "title": "Headline A",
      "text": "Body A ...",
      "subject": "politics",
      "date": "2024-08-01",
      "source_url": "https://apnews.com/..."
    },
    {
      "title": "Headline B",
      "text": "Body B ..."
    }
  ]
}
```

**Response (200)**

```json
{
  "results": [
    { "... single-article response as above ..." },
    { "... single-article response as above ..." }
  ]
}
```

---

## Request/Response Schemas

### ArticleInput

```json
{
  "title": "string",
  "text": "string",
  "subject": "string|null",
  "date": "YYYY-MM-DD or ISO8601|null",
  "source_url": "string|null"
}
```

### ScoreOutput

```json
{
  "label": "trustworthy|misleading",
  "confidence": 0.0,
  "probabilities": { "misleading": 0.0, "trustworthy": 0.0 },
  "style": {
    "exclamation_count": 0,
    "question_count": 0,
    "all_caps_ratio": 0.0,
    "sensational_terms": 0,
    "hedge_terms": 0,
    "sensational_score": 0.0
  },
  "source": {
    "domain": "string|null",
    "reliability_0_100": 0..100|null
  },
  "meta": {
    "max_len": "string",
    "adapter_type": "string",
    "n_adapter_layers": "string",
    "adapter_bottleneck": "string",
    "sparse_topk": "string",
    "sparse_use_fraction": "string"
  }
}
```

---

## Examples

### cURL

```bash
curl -s -X POST http://localhost:8000/v1/score_article \
  -H "Content-Type: application/json" \
  -d '{"title":"Govt announces new climate pact","text":"Officials said...","subject":"climate","date":"2024-11-05","source_url":"https://www.reuters.com/..."}'
```

### Python

```python
import requests

payload = {
  "title": "Govt announces new climate pact",
  "text": "Officials said the agreement...",
  "subject": "climate",
  "date": "2024-11-05",
  "source_url": "https://www.reuters.com/..."
}
r = requests.post("http://localhost:8000/v1/score_article", json=payload, timeout=30)
print(r.status_code, r.json())
```

### Node (fetch)

```js
const resp = await fetch("http://localhost:8000/v1/score_article", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    title: "Govt announces new climate pact",
    text: "Officials said...",
    subject: "climate",
    date: "2024-11-05",
    source_url: "https://www.reuters.com/...",
  }),
});
const data = await resp.json();
console.log(data);
```

---

## Running the server

```bash
pip install -r requirements.txt
export MODEL_DIR=/path/to/your/trained/folder         # or a checkpoint-XXXX
export ADAPTER_TYPE=residual                           # or sparse (match training)
export N_ADAPTER_LAYERS=4
export ADAPTER_BOTTLENECK=64
uvicorn app:app --host 0.0.0.0 --port 8000
```

GPU/bf16/fp16 are chosen automatically when available.

---
