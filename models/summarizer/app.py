import os, sys, math, time, uuid, warnings, logging, contextvars
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-8B")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "Qwen_Finetuned_8B")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

MAX_NEW_TOKENS_MAP = int(os.getenv("MAX_NEW_TOKENS_MAP", "192"))
MAX_NEW_TOKENS_REDUCE = int(os.getenv("MAX_NEW_TOKENS_REDUCE", "256"))
CTX_LEN = int(os.getenv("CTX_LEN", "4096"))
CLUSTER_DISTANCE = float(os.getenv("CLUSTER_DISTANCE", "0.45"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "/no_think You are POLARIS, a precise, fair news summariser. Be neutral; include key numbers/dates/actors; "
    "highlight agreements and clearly mark conflicts without taking sides. Just give me the summary, dont include what you are thinking.",
)

request_id_var = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


logger = logging.getLogger("polaris")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | req=%(request_id)s | %(message)s"
)
_handler.setFormatter(_formatter)
_handler.addFilter(_RequestIdFilter())
logger.handlers.clear()
logger.addHandler(_handler)
logger.propagate = False

try:
    _lvl = LOG_LEVEL.upper()
    if _lvl == "DEBUG":
        hf_logging.set_verbosity_debug()
    elif _lvl == "INFO":
        hf_logging.set_verbosity_info()
    elif _lvl == "WARNING":
        hf_logging.set_verbosity_warning()
    elif _lvl == "ERROR":
        hf_logging.set_verbosity_error()
    else:
        hf_logging.set_verbosity_warning()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
except Exception as e:
    logger.debug(f"hf logging setup failed: {e}")

for _name in ("sentence_transformers", "urllib3", "httpx", "huggingface_hub"):
    try:
        logging.getLogger(_name).setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    except Exception:
        pass


def _hms(ms: float) -> str:
    s = int(ms / 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _first_real_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for _, dev in sorted(model.hf_device_map.items()):
            if isinstance(dev, str) and dev not in ("cpu", "disk", "meta"):
                return torch.device(dev)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_cuda_snapshot(tag: str = "gpu"):
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_memory
            alloc = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            logger.info(
                f"{tag}: device={torch.cuda.get_device_name(idx)} "
                f"total={total/1e9:.2f}GB alloc={alloc/1e9:.2f}GB reserved={reserved/1e9:.2f}GB"
            )
        except Exception as e:
            logger.debug(f"{tag}: cuda snapshot failed: {e}")


def _param_stats(m):
    """Return (total_params, trainable_params) for diagnostics."""
    try:
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable
    except Exception:
        return None, None


def _load_llm():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"loading base model: {BASE_MODEL} on device={device}")

    quant_kwargs = {}
    used_4bit = False
    try:
        from transformers import BitsAndBytesConfig

        compute_dtype = (
            torch.bfloat16
            if (device == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            device_map="auto",
        )
        used_4bit = True
        logger.info("quantization: using 4-bit NF4")
    except Exception as e:
        quant_kwargs["torch_dtype"] = (
            torch.bfloat16
            if (device == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if device == "cuda" else torch.float32)
        )
        logger.info(
            f"quantization: fallback precision={quant_kwargs['torch_dtype']} (bitsandbytes unavailable: {e})"
        )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    logger.debug(
        f"tokenizer: vocab={len(tok)} eos={tok.eos_token_id} "
        f"pad={tok.pad_token_id} model_max_len={getattr(tok,'model_max_length', None)}"
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        low_cpu_mem_usage=True,
        **quant_kwargs,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.config.pad_token_id = tok.pad_token_id
    mdl.config.max_position_embeddings = getattr(
        mdl.config, "max_position_embeddings", CTX_LEN
    )

    if ADAPTER_DIR and os.path.isdir(ADAPTER_DIR):
        try:
            from peft import PeftModel

            logger.info(f"attaching adapter: {ADAPTER_DIR}")
            mdl = PeftModel.from_pretrained(mdl, ADAPTER_DIR)
            logger.info("adapter: attached successfully")
        except Exception as e:
            logger.warning(f"adapter: failed to load from {ADAPTER_DIR}: {e}")

    mdl.eval()
    mdl._first_device = _first_real_device(mdl)

    dm = getattr(mdl, "hf_device_map", None)
    if isinstance(dm, dict):
        _preview = ", ".join(f"{k}->{v}" for k, v in list(dm.items())[:12])
        if len(dm) > 12:
            _preview += ", ..."
        logger.debug(f"device_map: {len(dm)} shards | {_preview}")
    total_p, train_p = _param_stats(mdl)
    if total_p:
        logger.info(
            f"model params: total={total_p/1e6:.1f}M trainable={(train_p or 0)/1e6:.1f}M"
        )

    _log_cuda_snapshot("after-load")
    logger.info(
        f"model ready | device={mdl._first_device} | 4bit={used_4bit} | "
        f"startup={(time.time()-t0)*1000:.0f}ms"
    )
    return tok, mdl


def _apply_chat_template(tokenizer, user_text: str, system_text: str = SYSTEM_PROMPT):
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )


@dataclass
class Summariser:
    tok: AutoTokenizer
    mdl: AutoModelForCausalLM

    def _gen(
        self,
        user_prompt: str,
        max_new: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        t0 = time.time()

        t_enc0 = time.time()
        ids = _apply_chat_template(self.tok, user_prompt)
        t_enc_ms = (time.time() - t_enc0) * 1000

        t_move0 = time.time()
        input_ids = ids.to(self.mdl._first_device)
        t_move_ms = (time.time() - t_move0) * 1000

        prompt_tokens = input_ids.shape[-1]
        _preview = user_prompt.replace("\n", " ")[:160]
        logger.debug(f"prompt preview: {_preview!r}")
        logger.info(
            "generate: start | prompt_tokens=%d max_new=%d temp=%.3f top_p=%.3f dev=%s | tokenize=%.1fms to_device=%.1fms",
            prompt_tokens,
            max_new,
            temperature,
            top_p,
            self.mdl._first_device,
            t_enc_ms,
            t_move_ms,
        )
        _log_cuda_snapshot("before-gen")

        try:
            with torch.no_grad():
                out = self.mdl.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                )
            new_tokens = out[:, input_ids.shape[-1] :]
            t_dec0 = time.time()
            text = self.tok.decode(new_tokens[0], skip_special_tokens=True).strip()
            t_dec_ms = (time.time() - t_dec0) * 1000

            dt = time.time() - t0
            gen_tokens = new_tokens.shape[-1]
            tps = (gen_tokens / dt) if dt > 0 else float("inf")
            _log_cuda_snapshot("after-gen")
            logger.info(
                "generate: done | prompt=%d new=%d total=%.0fms decode=%.1fms speed=%.1f tok/s",
                prompt_tokens,
                gen_tokens,
                dt * 1000,
                t_dec_ms,
                tps,
            )
            logger.debug("generated head: %r", text[:160])
            logger.debug("generated tail: %r", text[-160:])
            return text
        except Exception as e:
            logger.exception(f"generate: failed: {e}")
            raise

    def summarise_single(self, title: str, body: str, target_words: int = 140) -> str:
        logger.info(
            f"single-summary: target_words={target_words} title_len={len(title)} body_len={len(body)}"
        )
        logger.debug("single-summary title preview: %r", title[:120])
        prompt = f"""Summarise the following single news article in about {target_words} words.
Focus on key arguments and facts. Avoid sensational language. Preserve attributions if present.

TITLE:
{title}

ARTICLE:
{body}
"""
        return self._gen(prompt, max_new=MAX_NEW_TOKENS_MAP)

    def map_note(self, title: str, body: str, target_words: int = 80) -> str:
        logger.debug(
            f"map-note: target_words={target_words} title_len={len(title)} body_len={len(body)}"
        )
        prompt = f"""Write a concise factual NOTE (~{target_words} words) capturing the article's key claims, actors, numbers, and evidence.
Avoid opinions and adjectives; quote numbers/dates; keep it neutral.

TITLE:
{title}

ARTICLE:
{body}
"""
        return self._gen(prompt, max_new=MAX_NEW_TOKENS_MAP)

    def reduce_digest(
        self, topic_hint: str, notes: List[str], target_words: int = 180
    ) -> str:
        logger.info(
            f"reduce-digest: notes={len(notes)} target_words={target_words} topic='{topic_hint[:64]}'"
        )
        joined = "\n\n".join([f"- NOTE {i+1}: {n}" for i, n in enumerate(notes)])
        prompt = f"""You are creating a balanced topic digest from multiple sources.
SYNTHESISE the NOTES into ~{target_words} words:
- Merge overlaps into concise statements.
- Highlight AGREEMENTS across outlets.
- Explicitly list any CONFLICTS or disputed facts.
- Include crucial numbers/dates/actors.
- If sources contradict, present both sides neutrally.

TOPIC: {topic_hint}

NOTES:
{joined}

DIGEST:"""
        return self._gen(prompt, max_new=MAX_NEW_TOKENS_REDUCE)


class TopicGrouper:
    def __init__(self, model_name: str = EMB_MODEL):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"embedding model: {model_name} on {device}")
        self.embedder = SentenceTransformer(model_name, device=device)
        try:
            dim = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"embedding dimension={dim}")
        except Exception:
            pass

    def cluster(
        self, articles: List[str], distance_threshold: float = CLUSTER_DISTANCE
    ):
        t0 = time.time()
        logger.info(
            f"cluster: n_articles={len(articles)} threshold={distance_threshold}"
        )
        embs = self.embedder.encode(
            articles,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        t_emb = (time.time() - t0) * 1000
        logger.debug(
            f"embeddings: shape={getattr(embs,'shape',None)} dtype={getattr(embs,'dtype',None)}"
        )
        D = cosine_distances(embs)
        dvals = D[np.triu_indices_from(D, k=1)] if hasattr(D, "__array__") else []
        if len(dvals) > 0:
            logger.debug(
                f"cluster: dist min={dvals.min():.3f} median={np.median(dvals):.3f} max={dvals.max():.3f}"
            )
        t1 = time.time()
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        clustering.fit(D)
        t_clu = (time.time() - t1) * 1000
        groups: Dict[int, List[int]] = {}
        for i, lab in enumerate(clustering.labels_):
            groups.setdefault(int(lab), []).append(i)
        sizes = sorted(len(v) for v in groups.values())
        logger.info(
            f"cluster: clusters={len(groups)} sizes={sizes} "
            f"| embed={t_emb:.0f}ms cluster={t_clu:.0f}ms total={(time.time()-t0)*1000:.0f}ms"
        )
        logger.debug(
            f"cluster labels: {{ {', '.join(f'{int(k)}:{v}' for k,v in groups.items())} }}"
        )
        return groups, embs


class Article(BaseModel):
    title: str = Field("(untitled)")
    text: str
    subject: Optional[str] = None
    date: Optional[str] = None
    source_url: Optional[str] = None


class SingleRequest(BaseModel):
    article: Article
    target_words: int = 140


class SingleResponse(BaseModel):
    summary: str


class TopicsRequest(BaseModel):
    articles: List[Article]
    main_index: Optional[int] = None
    distance_threshold: float = CLUSTER_DISTANCE
    map_words: int = 80
    reduce_words: int = 180


class TopicGroup(BaseModel):
    cluster_id: int
    indices: List[int]
    titles: List[str]
    digest: str


class TopicsResponse(BaseModel):
    groups: List[TopicGroup]
    per_article_summary: List[str]
    main_article_group: Optional[int] = None


tok, mdl = _load_llm()
summariser = Summariser(tok, mdl)
grouper = TopicGrouper()

app = FastAPI(title="POLARIS Topic Digest API", version="1.0.0")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(rid)
        t0 = time.time()
        try:
            logger.info(f"{request.method} {request.url.path} start")
            response = await call_next(request)
            ms = (time.time() - t0) * 1000
            logger.info(
                f"{request.method} {request.url.path} done {ms:.1f}ms status={response.status_code}"
            )
            response.headers["x-request-id"] = rid
            return response
        except Exception as e:
            ms = (time.time() - t0) * 1000
            logger.exception(
                f"{request.method} {request.url.path} error after {ms:.1f}ms: {e}"
            )
            return JSONResponse(
                {"detail": "internal error", "request_id": rid}, status_code=500
            )
        finally:
            request_id_var.reset(token)


app.add_middleware(RequestIdMiddleware)


@app.get("/health")
def health():
    dev = str(mdl._first_device)
    msg = {
        "ok": True,
        "device": dev,
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR or None,
    }
    logger.info(f"health: {msg}")
    return msg


@app.post("/v1/summarise_single", response_model=SingleResponse)
def summarise_single(req: SingleRequest):
    logger.info("route: summarise_single")
    logger.debug(
        "req.article: title_len=%d text_len=%d subject=%r date=%r url=%r",
        len(req.article.title or ""),
        len(req.article.text or ""),
        req.article.subject,
        req.article.date,
        req.article.source_url,
    )
    s = summariser.summarise_single(
        req.article.title, req.article.text, target_words=req.target_words
    )
    return SingleResponse(summary=s)


@app.post("/v1/summarise_topics", response_model=TopicsResponse)
def summarise_topics(req: TopicsRequest):
    logger.info(
        f"route: summarise_topics | n_articles={len(req.articles)} "
        f"main_index={req.main_index} threshold={req.distance_threshold} "
        f"map_words={req.map_words} reduce_words={req.reduce_words}"
    )
    try:
        lens = [(len(a.title or ""), len(a.text or "")) for a in req.articles]
        logger.debug(f"articles lens (title,text): {lens}")
    except Exception:
        pass

    if not req.articles:
        return TopicsResponse(
            groups=[], per_article_summary=[], main_article_group=None
        )

    texts = [a.text for a in req.articles]
    titles = [a.title for a in req.articles]
    groups, _ = grouper.cluster(texts, distance_threshold=req.distance_threshold)

    # summarise per group
    per_article_summary = [""] * len(req.articles)
    out_groups: List[TopicGroup] = []

    for gid, idxs in groups.items():
        logger.info(
            f"group {gid}: indices={idxs} titles={[titles[i][:40] for i in idxs]}"
        )
        logger.debug(f"group {gid}: sizes={{'n_articles': {len(idxs)}}}")

        if len(idxs) == 1:
            i = idxs[0]
            logger.info(f"group {gid}: singleton=True -> using summarise_single()")
            digest = summariser.summarise_single(
                titles[i],
                texts[i],
                target_words=req.reduce_words,
            )
            per_article_summary[i] = digest
            out_groups.append(
                TopicGroup(
                    cluster_id=int(gid),
                    indices=idxs,
                    titles=[titles[i] for i in idxs],
                    digest=digest,
                )
            )
            continue

        notes = [
            summariser.map_note(titles[i], texts[i], target_words=req.map_words)
            for i in idxs
        ]
        topic_hint = titles[idxs[0]] if titles[idxs[0]] else "Topic"
        digest = summariser.reduce_digest(
            topic_hint, notes, target_words=req.reduce_words
        )

        for i in idxs:
            per_article_summary[i] = digest

        out_groups.append(
            TopicGroup(
                cluster_id=int(gid),
                indices=idxs,
                titles=[titles[i] for i in idxs],
                digest=digest,
            )
        )

    main_group = None
    if req.main_index is not None:
        for g in out_groups:
            if req.main_index in g.indices:
                main_group = g.cluster_id
                break
        logger.info(f"main_index={req.main_index} -> main_group={main_group}")

    return TopicsResponse(
        groups=sorted(out_groups, key=lambda g: min(g.indices)),
        per_article_summary=per_article_summary,
        main_article_group=main_group,
    )
