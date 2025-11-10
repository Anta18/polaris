import os, re, math, hashlib, time, logging, warnings
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ServerSelectionTimeoutError

# Silence harmless sklearn rename warning
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------- Logging ----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("topics")

# ---------------------- Config -----------------------
# I/O toggles
READ_FROM_DB   = 1   
WRITE_TO_DB    = 1  

# Mongo
MONGO_URI="mongodb+srv://siddhantsingh15032005_db_user:NL8faLkskcXUa6Nb@polaris.jkw8jwe.mongodb.net/polaris_news_db?retryWrites=true&w=majority"
DB_NAME = "polaris_news_db"
COLL_NAME = "articles"

# Field names in your collection
FIELD_ID = "id"
FIELD_TITLE   = "title"
FIELD_CONTENT = "content"

# Field to write
DB_TOPIC_FIELD = "topic"  # 


# EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
USE_GPU     = 0   # set to 1 if you have CUDA


ATTACH_SIM_THRESHOLD = float(os.getenv("ATTACH_SIM_THRESHOLD", "0.62"))
LABEL_KEYWORDS = 6   
MAX_LABEL_LEN  = int(os.getenv("MAX_LABEL_LEN", "96"))   # cap label length
PHRASE_NGRAMS  = (1, 3)                                  

TOY_ARTICLES: List[Dict[str, Any]] = [
    # Elections / Politics
    {"id": 1, "title": "Tight race in swing states as early votes surge",
     "content": "Election officials reported high volumes of early ballots. Late-deciding independents could tilt outcomes. Campaigns refocus on volatile suburban counties."},
    {"id": 2, "title": "Exit polls show economy outweighs foreign policy in voter priorities",
     "content": "Inflation and job security dominated concerns; healthcare ranked third. Analysts expect close margins in the industrial Midwest to decide results."},
    {"id": 3, "title": "Counting delays in battlegrounds prompt legal challenges",
     "content": "Some counties delayed verification of late mail ballots. Both campaigns filed motions on provisional handling; recount thresholds likely in two states."},

    # Football / Domestic League
    {"id": 4, "title": "City edge United in dramatic derby decided by late header",
     "content": "An 88th-minute set-piece delivered the winner. United led early but retreated into a low block. City’s bench pressed high and created flank overloads."},
    {"id": 5, "title": "Manager defends high line after United concede on counter",
     "content": "Coach insisted the aggressive shape was necessary to disrupt build-up. Spacing between midfield and defense invited transitions down the right."},
    {"id": 6, "title": "Derby analysis: midfield rotations and pressing traps",
     "content": "Rotations opened half-spaces and pulled markers. The winning move began with a pressing trap and a switch to a 3-2-5 in possession."},

    # Tech Earnings
    {"id": 7, "title": "Acme beats Q3 expectations as cloud margin expands",
     "content": "Record revenue and wider cloud margins. Guidance raised on modernization deals. FCF improved as capex tapered after data center build-out."},
    {"id": 8, "title": "Record quarter for Acme’s AI services offsets hardware softness",
     "content": "AI APIs and managed inference grew triple digits. Multimodal models for call centers shone. Channel partners saw higher renewal ACVs."},
    {"id": 9, "title": "BetaSoft warns on currency headwinds despite strong pipeline",
     "content": "Robust European pipeline, but FX pressures. Deferred revenue climbed; hiring disciplined with tighter expense controls."},

    # Climate / Weather
    {"id": 10, "title": "Heatwave grips southern Europe as authorities issue red alerts",
     "content": "Temperatures far above seasonal norms strain grids and transit. Cities opened cooling centers; grid operators urged conservation."},
    {"id": 11, "title": "Wildfires prompt evacuations amid prolonged drought",
     "content": "Dry winds accelerated spread near rural towns. Evacuations covered thousands; lightning could ignite new hotspots."},
    {"id": 12, "title": "Flash floods disrupt transport after record overnight rainfall",
     "content": "Rail suspended as water overtopped embankments. Crews cleared debris; meteorologists called it a once-in-a-decade deluge."},

    # Health / Public Health
    {"id": 13, "title": "Hospitals expand ICU capacity as respiratory cases climb",
     "content": "Surge protocols activated; pharmacies saw higher antiviral demand. Agencies urged high-risk groups to update vaccinations."},
    {"id": 14, "title": "New booster targets circulating variants in fall campaign",
     "content": "Reformulated booster authorized; can be coadministered with flu shots. Early data shows improved neutralization."},
    {"id": 15, "title": "Wastewater trends hint at plateau in community transmission",
     "content": "Indicators stabilized after weeks of ascent. Holiday travel could shift trends; schools keep ventilation upgrades."},

    # Markets / Economy
    {"id": 16, "title": "Inflation cools more than expected as services decelerate",
     "content": "Headline eased with declines in travel and used vehicles. Core moderated; traders trimmed hike odds and pulled forward cut expectations."},
    {"id": 17, "title": "Central bank signals patience despite easing price pressures",
      "content": "Officials want more evidence toward target. Labor resilient; consumption solid. Yields fell after press conference."},
    {"id": 18, "title": "Earnings beats lift equities while dollar retreats",
     "content": "Indices advanced on positive surprises. Dollar softened, easing commodity pressure; rotation into cyclicals and small caps."},

    # Entertainment / Film & OTT
    {"id": 19, "title": "Bollywood biopic sets opening weekend record nationwide",
     "content": "Lead performance and soundtrack praised. Multiplexes added late-night screens. Overseas collections strong."},
    {"id": 20, "title": "Streaming thriller tops charts as word of mouth spreads",
     "content": "Limited series jumped to #1 on buzz around a twist ending. Critics cited taut pacing and standout support cast."},
    {"id": 21, "title": "Festival darling lands wide release after awards buzz",
     "content": "Post-festival wins, drama secured wide distribution. Strong audience scores; expansion to smaller towns planned."},

    # Space / Science
    {"id": 22, "title": "Lunar lander transmits crisp images from near-polar site",
     "content": "Stable thermal conditions and nominal power. Micro-rover to sample regolith within 48 hours; ice signatures targeted."},
    {"id": 23, "title": "Telescope detects atmosphere around warm sub-Neptune",
     "content": "Spectroscopy revealed water vapor and cloud hints. Six-day orbit around K-type star; follow-up refines composition."},
    {"id": 24, "title": "Lab achieves stable fusion gain in repeat experiments",
     "content": "Unity gain surpassed in three shots with improved targets. Better confinement; dataset to be published for review."},

    # Startups / Funding
    {"id": 25, "title": "Startup unveils biodegradable packaging to cut e-commerce waste",
     "content": "Material decomposes in municipal facilities within twelve weeks. Pilots focus on mailers; investors watch cost parity."},
    {"id": 26, "title": "Agritech firm raises Series B to expand precision irrigation",
     "content": "Funds sensors and localized weather modeling. Trials cut water use without yield losses; cooperative partnerships help."},
    {"id": 27, "title": "Fintech secures license for cross-border remittances",
     "content": "Real-time FX and compliance checks integrated. Corridors open in phases; tiered fees encourage digital adoption."},

    # Education / Policy
    {"id": 28, "title": "Universities pilot holistic admissions with skills portfolios",
     "content": "Applicants submit project evidence like code repos. Approach complements tests; outcomes to be shared with regulators."},
    {"id": 29, "title": "Curriculum reform emphasizes data literacy across grades",
     "content": "Standards add visualization and basic statistics. Teacher modules use local datasets; progress tracked three years."},
    {"id": 30, "title": "Public schools invest in ventilation and outdoor spaces",
     "content": "Grants retrofit HVAC and build shaded courtyards. Multipurpose layouts support classes and clubs; parents welcome changes."},

    # AI / ML
    {"id": 31, "title": "Research lab releases lightweight multimodal model for edge devices",
     "content": "On-device speech, image, text via QAT. Strong performance relative to size; permissive license and detailed docs."},
    {"id": 32, "title": "Industry group proposes safety evals for high-risk AI systems",
     "content": "Framework covers incident reporting, red-teaming, monitoring. Standardized data disclosures; regulators interested."},
    {"id": 33, "title": "Startups race to build domain-specific copilots for SMEs",
     "content": "RAG tailored for accounting and legal. Customers value audit trails and controllability; pricing bundles API + SLA."},

    # Geopolitics / Trade
    {"id": 34, "title": "Trade talks resume as negotiators narrow differences on tariffs",
     "content": "Revised schedules for phased tariff reductions. Quotas and digital taxes remain; limited accord could unlock investment."},
    {"id": 35, "title": "Sanctions tightened on firms accused of evading export controls",
     "content": "More entities listed; banks warned about layered intermediaries. Shipping routes via free zones complicate screening."},
    {"id": 36, "title": "Maritime insurers adjust premiums amid rising strait tensions",
     "content": "Underwriters lifted rates for choke points. Shippers weigh longer routes vs fuel costs; ports increase patrols."},
]

def l2norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def connect_mongo(uri: str):
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        log.info("Mongo connected: %s", uri)
        return client
    except Exception as e:
        log.warning("Mongo connect failed: %s", e)
        return None

def fetch_from_db(client: MongoClient) -> List[Dict[str, Any]]:
    coll = client[DB_NAME][COLL_NAME]
    cursor = coll.find({}, {FIELD_ID: 1, FIELD_TITLE: 1, FIELD_CONTENT: 1})
    recs = []
    for d in cursor:
        recs.append({
            FIELD_ID: d.get(FIELD_ID),
            FIELD_TITLE: d.get(FIELD_TITLE, "") or "",
            FIELD_CONTENT: d.get(FIELD_CONTENT, "") or "",
        })
    return recs

def write_topics_to_db(client: MongoClient, results: List[Dict[str, Any]]) -> None:
    coll = client[DB_NAME][COLL_NAME]
    ops = []
    skipped = 0

    for r in results:
        doc_id = r.get("id")
        topic  = r.get("topic")

        
        if not doc_id or topic is None:
            skipped += 1
            continue

        
        if not isinstance(topic, str):
            topic = str(topic)

        
        if MAX_LABEL_LEN:
            topic = topic[:MAX_LABEL_LEN].rstrip()

        ops.append(UpdateOne(
            {FIELD_ID: doc_id},                 
            {"$set": {DB_TOPIC_FIELD: topic}},  
            upsert=False                        
        ))

    if not ops:
        log.info("No DB updates prepared. skipped=%d", skipped)
        return

    t0 = time.time()
    res = coll.bulk_write(ops, ordered=False)
    log.info(
        "Mongo bulk_write: matched=%d modified=%d in %.3fs (skipped=%d)",
        res.matched_count, res.modified_count, time.time() - t0, skipped
    )



def embed(records: List[Dict[str, Any]]) -> Tuple[List[Any], List[str], List[str], np.ndarray]:
    """Return (ids, titles_display, texts_for_labeling, embeddings[L2])."""
    ids = [r.get(FIELD_ID) for r in records]
    titles_display = [r.get(FIELD_TITLE, "") or "" for r in records]
   
    label_texts = [f"{(r.get(FIELD_TITLE) or '').strip()}. {(r.get(FIELD_CONTENT) or '')[:300].strip()}"
                   for r in records]
    device = "cuda" if USE_GPU else "cpu"
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL, device=device)
    Z = model.encode(label_texts, batch_size=256 if device=="cuda" else 64, show_progress_bar=False)
    Z = l2norm_rows(np.asarray(Z))
    log.info("Embeddings: %d docs in %.3fs (device=%s, model=%s)", len(ids), time.time()-t0, device, EMBED_MODEL)
    return ids, titles_display, label_texts, Z

# ---------------------- Clustering -------------------
def cluster_hdbscan_only(Z: np.ndarray) -> np.ndarray:
    """HDBSCAN with euclidean on L2-normalized embeddings (≈ cosine), with lenient fallbacks."""
    n = Z.shape[0]
    tries = [
        (max(3, int(round(3 * math.log(max(n, 2), 2)))), max(1, int(round((max(n, 2) ** 0.5) / 5)))),
        (3, 1),
        (2, 1),
    ]
    for (mcs, ms) in tries:
        log.info("HDBSCAN params: min_cluster_size=%d, min_samples=%d (metric=euclidean)", mcs, ms)
        t0 = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",          
            cluster_selection_method="eom",
            prediction_data=True
        ).fit(Z.astype("float32"))
        labels = np.asarray(clusterer.labels_, dtype=int)
        n_clusters = len({c for c in labels if c >= 0})
        n_noise = int(np.sum(labels == -1))
        log.info("HDBSCAN: %.3fs | clusters=%d | noise=%d", time.time() - t0, n_clusters, n_noise)
        if n_clusters > 0:
            return labels
    return labels  


def attach_noise_to_nearest(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # Build centroids
    cluster_to_idx = defaultdict(list)
    for i, c in enumerate(labels):
        if c >= 0:
            cluster_to_idx[c].append(i)
    if not cluster_to_idx:
        return labels
    centroids = {c: l2norm_rows(np.mean(Z[idxs], axis=0, keepdims=True)).ravel()
                 for c, idxs in cluster_to_idx.items()}

    noise = [i for i, c in enumerate(labels) if c == -1]
    if not noise:
        return labels
    C = l2norm_rows(np.vstack(list(centroids.values())))
    ordered_cids = list(centroids.keys())
    sims = Z[noise] @ C.T
    best_pos = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(len(noise)), best_pos]
    attached = 0
    for i_noise, sim, pos in zip(noise, best_sim, best_pos):
        if sim >= ATTACH_SIM_THRESHOLD:
            labels[i_noise] = ordered_cids[pos]
            attached += 1
    log.info("Noise attach: attached=%d / %d (threshold=%.2f)", attached, len(noise), ATTACH_SIM_THRESHOLD)
    return labels

# ---------------------- Labeling ---------------------
def _ctfidf_top_terms_per_cluster(cluster_texts: List[List[str]], top_k: int) -> List[List[str]]:
    """Return top_k keywords per cluster using a class-based TF-IDF approximation."""
    if top_k <= 0 or not cluster_texts:
        return [[] for _ in cluster_texts]
    docs = [" ".join(clean(t) for t in texts) for texts in cluster_texts]
    if not any(docs):
        return [[] for _ in cluster_texts]
    vec = TfidfVectorizer(ngram_range=PHRASE_NGRAMS, stop_words="english", max_features=6000)
    X = vec.fit_transform(docs)  
    if X.shape[1] == 0:
        return [[] for _ in cluster_texts]
    terms = np.array(vec.get_feature_names_out())
    out = []
    for r in range(X.shape[0]):
        row = X.getrow(r)
        if row.nnz == 0:
            out.append([])
            continue
        idx = np.asarray(row.toarray()).ravel().argsort()[::-1][:top_k]
        out.append([terms[i] for i in idx if terms[i]])
    return out

def _medoid_title_for_cluster(idxs: List[int], titles_display: List[str], Z: np.ndarray) -> str:
    """Pick the title closest to the cluster centroid (very descriptive)."""
    centroid = l2norm_rows(np.mean(Z[idxs], axis=0, keepdims=True)).ravel()
    sims = (Z[idxs] @ centroid)
    medoid_local = int(np.argmax(sims))
    return titles_display[idxs[medoid_local]]

def build_results_descriptive(ids: List[Any],
                              titles_display: List[str],
                              label_texts: List[str],
                              Z: np.ndarray,
                              labels: np.ndarray) -> List[Dict[str, Any]]:
    # Group indices
    cluster_to_idx = defaultdict(list)
    for i, c in enumerate(labels):
        cluster_to_idx[c].append(i)

    
    nonneg = sorted([c for c in cluster_to_idx if c >= 0], key=int)
    if not nonneg:
        log.warning("No clusters formed; returning singleton topics from titles.")
        results = []
        for i in range(len(ids)):
            lab = titles_display[i]
            if len(lab) > MAX_LABEL_LEN:
                lab = lab[:MAX_LABEL_LEN].rstrip()
            results.append({"id": ids[i], "title": titles_display[i], "topic": lab})
        return results

    # Prepare cluster-wise texts for keywords
    cluster_title_lists_for_kw = [[label_texts[i] for i in cluster_to_idx[c]] for c in nonneg]
    top_terms = _ctfidf_top_terms_per_cluster(cluster_title_lists_for_kw, LABEL_KEYWORDS)


    cid_to_label = {}
    for c, terms in zip(nonneg, top_terms):
        medoid_title = _medoid_title_for_cluster(cluster_to_idx[c], titles_display, Z)
        label = medoid_title
        if terms:
            label = f"{medoid_title} — {', '.join(terms)}"
        if len(label) > MAX_LABEL_LEN:
            label = label[:MAX_LABEL_LEN].rstrip()
        cid_to_label[c] = label

    
    singleton_label = {}
    for i in cluster_to_idx.get(-1, []):
        lab = titles_display[i]
        if len(lab) > MAX_LABEL_LEN:
            lab = lab[:MAX_LABEL_LEN].rstrip()
        singleton_label[i] = lab

    # Ensure label uniqueness (suffix if duplicates)
    all_labels = [cid_to_label[c] for c in nonneg] + [singleton_label[i] for i in cluster_to_idx.get(-1, [])]
    seen = Counter()
    uniq_map: Dict[str, str] = {}
    for lab in all_labels:
        seen[lab] += 1
        uniq_map.setdefault(lab, lab if seen[lab] == 1 else f"{lab} #{seen[lab]}")

    # Build final results (no topic_id; just 'topic')
    results: List[Dict[str, Any]] = []
    for i, c in enumerate(labels):
        lab = singleton_label[i] if c == -1 else cid_to_label[c]
        lab = uniq_map.get(lab, lab)
        results.append({"id": ids[i], "title": titles_display[i], "topic": lab})
    return results


def main():
    log.info("Starting topic assignment (HDBSCAN + descriptive labels)…")
    client = connect_mongo(MONGO_URI) if (READ_FROM_DB or WRITE_TO_DB) else None
    do_read  = bool(READ_FROM_DB and client)
    do_write = bool(WRITE_TO_DB and client)

    # 1) Load records
    if do_read:
        records = fetch_from_db(client)
        log.info("Loaded %d docs from MongoDB %s/%s", len(records), DB_NAME, COLL_NAME)
    else:
        if READ_FROM_DB and not client:
            log.warning("READ_FROM_DB=1 but Mongo unreachable; using toy dataset.")
        records = TOY_ARTICLES
        log.info("Using toy dataset: %d docs", len(records))

    if not records:
        log.warning("No records to process. Exiting.")
        return

    # 2) Embed
    ids, titles_display, label_texts, Z = embed(records)

    # 3) Cluster (HDBSCAN only) + attach noise
    labels = cluster_hdbscan_only(Z)
    labels = attach_noise_to_nearest(Z, labels)

    # 4) Build descriptive topics
    results = build_results_descriptive(ids, titles_display, label_texts, Z, labels)

    # 5) Write back to DB by `id` (only 'topic')
    if do_write:
        try:
            write_topics_to_db(client, results)
        except ServerSelectionTimeoutError as e:
            log.error("Bulk write failed (Mongo unreachable): %s", e)
            log.warning("Continuing in dry-run; no DB updates were made.")
    else:
        log.info("Dry-run: skipping DB writes.")

    # 6) Console summary
    print("\n=== Topic assignments (flat) ===")
    id_w = max(len(str(r["id"])) for r in results)
    for r in results:
        _id   = str(r["id"]).rjust(id_w)
        print(f"{_id} | {r['title']}  ->  [{r['topic']}]")

    # Grouped view
    groups = defaultdict(list)
    for r in results:
        groups[r["topic"]].append((r["id"], r["title"]))
    print("\n=== Topics (grouped) ===")
    for topic, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n[{topic}]  (articles={len(items)})")
        for _id, title in items:
            print(f" - {_id}: {title}")

if __name__ == "__main__":
    main()
