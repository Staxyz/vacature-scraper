"""SBERT pipeline for job vacancies: clustering + relevancy + classification into AI / Tech / Data / Other.

One file, one command:

    pip install pandas numpy scikit-learn matplotlib sentence-transformers
    python sbert_pipeline.py --input ey_vacancies.csv

Input:  comma-separated CSV with the columns  title, location, link, description
Output: <input>_sbert.csv with the columns
        job_category, category_confidence_percent, esco_occupation, esco_code, isco_code,
        cluster, relevancy_percent, title, location, link, description
        + a folder <input>_sbert/ with a chart, an overview per cluster and a cache.

What happens:
    1. EMBEDDINGS  Each vacancy (title + description) is converted by a multilingual SBERT model
                   (Dutch + English) into an embedding: 768 numbers that represent its meaning.
                   The model reads at most 128 tokens; longer texts are split into chunks and the
                   chunk embeddings are averaged, so the whole vacancy counts.
    2. CLUSTERS    KMeans groups vacancies with a similar meaning. The number of clusters (6-15)
                   is chosen with the silhouette score.
    3. RELEVANCY   How close a vacancy is to the centre of its cluster (cosine), scaled over the
                   dataset: 100% = most typical vacancy, 0% = least typical.
    4. NAMES       Per cluster the 3 most distinctive words (class-based TF-IDF). Readable names can
                   be stored in <input>_sbert/cluster_names.json.
    5. ESCO        Each vacancy is matched to the closest of the ~2,900 occupations in ESCO, the European
                   classification of occupations: the title against all Dutch/English occupation names
                   (60%), the full text against the ESCO description (40%). The category follows from
                   that occupation's code: Tech = Eurostat/OECD ICT specialists (ISCO-08), Data and AI =
                   the data- and AI-occupations within ESCO (see ESCO classification below).
                   ESCO has no product names, so ~50 market names (SAP consultant, ServiceNow developer,
                   AI engineer, ...) are added to the ESCO occupation that describes that work.
                   On KPMG data: ~90% agreement with a manual check; the differences are mostly
                   definitional (e.g. ESCO counts digital sourcing as purchasing, not ICT).
                   The ESCO list is downloaded once (~20 min) to esco_occupations.json next to this file;
                   put the shared esco_occupations.json next to this script to skip that.

Options:  --k 9   fixed number of clusters      --output <path>   different output file name
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("sbert")

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # SBERT, multilingual (NL + EN)
CHUNK_WORDS = 70          # ~110 tokens: fits within the model's 128-token limit
RANDOM_STATE = 42         # fixed, so every run gives the same clusters
DEFAULT_K = range(6, 16)


# ---------------------------------------------------------------- embeddings


def chunks(text: str, n: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)] or [""]


def embed(df: pd.DataFrame, model_name: str, cache: Path) -> np.ndarray:
    """One normalised embedding per vacancy: the average of the title and all text chunks."""
    key = f"{model_name}|{CHUNK_WORDS}|" + "|".join(df["link"])
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if str(stored["key"]) == key:
            log.info("Embeddings loaded from cache")
            return stored["emb"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts, owner = [], []
    for i, r in df.iterrows():
        parts = [r["title"]] + chunks(r["description"])
        texts += parts
        owner += [i] * len(parts)
    log.info("Embedding %d text chunks for %d vacancies with %s", len(texts), len(df), model_name)
    vec = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    owner = np.array(owner)
    emb = np.vstack([vec[owner == i].mean(axis=0) for i in range(len(df))])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, key=np.array(key))
    return emb


# ---------------------------------------------------------------- clustering


def choose_k(emb: np.ndarray, candidates) -> tuple[int, pd.DataFrame]:
    """Silhouette per number of clusters (cosine). Higher = clusters better separated."""
    rows = []
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(emb)
        rows.append({"k": k, "silhouette": silhouette_score(emb, labels, metric="cosine")})
    scores = pd.DataFrame(rows)
    best = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return best, scores


def cluster(emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(emb)
    centres = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    similarity = np.einsum("ij,ij->i", emb, centres[km.labels_])
    # All KPMG vacancies are basically similar (cosine 0.82-0.98), so raw percentages say little.
    # We scale over the range that actually occurs:
    # 100% = most typical vacancy for its cluster, 0% = least typical in the dataset.
    relevancy = (similarity - similarity.min()) / (similarity.max() - similarity.min()) * 100
    return km.labels_, relevancy, similarity


# ---------------------------------------------------------------- names


def stopwords() -> set[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    try:
        from spacy.lang.nl.stop_words import STOP_WORDS as NL
    except ImportError:
        NL = set()
    # Extra (partly Dutch) words that appear in almost every vacancy and say nothing about the cluster
    extra = {"kpmg", "je", "jij", "jouw", "wij", "we", "onze", "bent", "hebt", "ervaring", "experience",
             "work", "team", "klanten", "clients", "goed", "nieuwe", "binnen", "samen", "the", "and"}
    return set(ENGLISH_STOP_WORDS) | set(NL) | extra


def top_words(df: pd.DataFrame, labels: np.ndarray, n: int = 10) -> dict[int, list[str]]:
    """Distinctive words per cluster (class-based TF-IDF): frequent in this cluster, rare in the rest."""
    cv = CountVectorizer(stop_words=list(stopwords()), token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
                         min_df=3, max_df=0.5, lowercase=True)
    X = cv.fit_transform(df["title"] + " " + df["description"])
    vocab = cv.get_feature_names_out()
    tf = np.vstack([np.asarray(X[labels == c].sum(axis=0)).ravel() for c in sorted(set(labels))])
    ctfidf = (tf / tf.sum(axis=1, keepdims=True)) * np.log(1 + tf.sum(axis=1).mean() / tf.sum(axis=0))
    return {c: list(vocab[np.argsort(ctfidf[c])[::-1][:n]]) for c in sorted(set(labels))}


def cluster_names(top: dict[int, list[str]], file: Path) -> dict[int, str]:
    """Readable names from cluster_names.json, only if the cluster still has the same top words
    (Jaccard >= 0.5); otherwise the top 3 words."""
    fixed = json.loads(file.read_text(encoding="utf-8")) if file.exists() else []
    names = {}
    for c, words in top.items():
        best, score = None, 0.0
        for item in fixed:
            a, b = set(words), set(item["words"])
            j = len(a & b) / len(a | b)
            if j > score:
                best, score = item["name"], j
        names[c] = best if best and score >= 0.5 else " / ".join(words[:3])
    return names


# ---------------------------------------------------------------- output


def plot_silhouette(scores: pd.DataFrame, k: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(scores["k"], scores["silhouette"], marker="o", color="#2e9e5b", linewidth=2)
    best = scores[scores["k"] == k].iloc[0]
    ax.scatter([k], [best["silhouette"]], s=160, facecolors="none", edgecolors="#2e9e5b", linewidths=2)
    ax.annotate(f"chosen: {k}", (k, best["silhouette"]), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette (cosine)")
    ax.set_title("SBERT + KMeans: how well are the clusters separated?")
    ax.set_xticks(list(scores["k"]))
    ax.margins(y=0.2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def overview(df: pd.DataFrame, top: dict, names: dict, scores: pd.DataFrame, k: int, model: str,
             path: Path) -> None:
    lines = ["# SBERT clustering (generated automatically)", "",
             f"- Model: `{model}` (max. 128 tokens; text in chunks of {CHUNK_WORDS} words, averaged)",
             f"- Vacancies: {len(df)}; clusters: {k} (highest silhouette); random_state = {RANDOM_STATE}",
             f"- relevancy_percent = cosine similarity to the centre of the own cluster, scaled "
             f"between the lowest ({df.cosine.min():.3f} = 0%) and highest ({df.cosine.max():.3f} = 100%) value "
             f"in this dataset; average {df.relevancy_percent.mean():.0f}%", "",
             "## Silhouette per number of clusters", "", scores.round(4).to_markdown(index=False), ""]
    for c in sorted(top):
        g = df[df["cluster_id"] == c].sort_values("relevancy_percent", ascending=False)
        lines += [f"## {names[c]} ({len(g)} vacancies, avg. relevancy {g.relevancy_percent.mean():.1f}%)", "",
                  f"Distinctive words: {', '.join(top[c])}", "",
                  "Most typical: " + "; ".join(f"{t} ({p:.0f}%)" for t, p in
                                               zip(g.title.head(3), g.relevancy_percent.head(3))),
                  "", "Least typical: " + "; ".join(f"{t} ({p:.0f}%)" for t, p in
                                                    zip(g.title.tail(2), g.relevancy_percent.tail(2))), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- ESCO classification
#
# ESCO = the European classification of occupations (European Commission). Every ESCO occupation
# has an ISCO-08 code. The categories below follow European definitions:
#
#   Tech  = ICT specialists as defined by Eurostat/OECD: ISCO-08 groups 133, 25, 35, 2152, 2153,
#           2166, 2356, 2434, 3114 and 742.
#   Data  = ESCO occupations that revolve around data (listed below by ESCO code). Most are ICT
#           occupations; 'business intelligence manager' and 'statistician' are data jobs outside the
#           Eurostat ICT codes and are included on purpose.
#   AI    = ESCO occupations that revolve around artificial intelligence (listed below).
#   Other = all other ESCO occupations.
#
# ESCO has no separate AI or Data group: these occupations sit inside ICT, so AI and Data are
# subgroups of Tech. A vacancy gets the category of the ESCO occupation it matches best.

ESCO_API = "https://ec.europa.eu/esco/api"
ICT_ISCO = ("133", "25", "35", "2152", "2153", "2166", "2356", "2434", "3114", "742")
AI_ESCO = {
    "2511.10": "ICT intelligent systems designer",
    "2529.8": "knowledge engineer",
    "2152.1.4": "language engineer (natural language processing)",
}
DATA_ESCO = {
    "2511.3": "data scientist", "2511.2": "data analyst", "2519.1": "data quality specialist",
    "1330.1": "chief data officer", "2521.1": "database administrator", "2521.2": "database designer",
    "2521.3": "database developer", "2521.4": "database integrator", "2521.5": "data warehouse designer",
    "2421.3": "business intelligence manager", "2120.6": "statistician",
}
# ESCO has no product names (SAP, ServiceNow, Dynamics, ...), so vacancies for those roles would match on
# generic words ("finance", "sales") and land in Other. These current market names are added as extra
# names to the EXISTING ESCO occupation that describes the work. The category still comes from that
# occupation's ESCO code, so the definitions stay European.
ESCO_ALIASES = {
    "2514.1": ["SAP consultant", "SAP functional consultant", "SAP S/4HANA consultant", "ERP consultant",
               "Microsoft Dynamics 365 consultant", "Dynamics 365 functional consultant", "ServiceNow consultant",
               "ServiceNow developer", "ServiceNow manager", "Dynamics 365 specialist", "Salesforce consultant", "Power Platform consultant", "low-code developer",
               "Mendix developer"],                                         # ICT application configurator
    "2511.16": ["SAP integration consultant", "SAP BTP integration architect", "integration consultant"],
    "2511.5": ["solution architect", "SAP solution architect", "cloud architect", "digital architect"],
    "1330.7": ["SAP project manager", "ERP project manager", "IT project manager"],  # ICT project manager
    "2434.1": ["SAP sales manager", "software sales manager"],                  # ICT account manager
    "2529.3": ["red team operator", "red teamer", "penetration tester", "offensive security specialist"],
    "2529.6": ["cyber security consultant", "incident response consultant", "identity and access management consultant"],
    "2511.17": ["IT risk consultant", "IT risk officer", "technology assurance consultant",
                "IT risk and compliance consultant"],                       # IT auditor
    "2511.9": ["technology consultant", "technology transformation consultant", "cloud consultant",
               "IT sourcing consultant", "digital transformation consultant"],  # ICT consultant
    "2519.4": ["quality assurance engineer", "software test engineer"],     # ICT quality assurance manager
    "2511.2": ["process mining analyst", "BI consultant", "Power BI developer", "data and insights consultant"],
    "2519.1": ["data management consultant", "data governance consultant", "master data management consultant"],
    "2521.4": ["data migration consultant"],                                # database integrator
    "2511.3": ["data engineer", "data platform engineer"],                  # data scientist
    "2511.10": ["AI engineer", "machine learning engineer", "AI architect", "AI consultant", "GenAI consultant",
                "AI governance consultant"],                                # ICT intelligent systems designer
}
# Big Four vacancies are always for managers, professionals or technicians (ISCO major groups 1-3).
# Restricting the candidates to those groups prevents nonsense matches such as 'life guard'.
PROFESSIONAL_GROUPS = ("1", "2", "3")
TITLE_WEIGHT = 0.6   # vacancy title vs ESCO occupation names; the rest: full text vs ESCO description
MAX_ALT_LABELS = 15  # alternative names per language per occupation


def esco_category(code: str) -> str:
    if code in AI_ESCO:
        return "AI"
    if code in DATA_ESCO:
        return "Data"
    if code.split(".")[0].startswith(ICT_ISCO):
        return "Tech"
    return "Other"


def download_esco(path: Path) -> None:
    """Download all ESCO occupations (~20 min, once). The API's 'offset' is a page number; full records
    are ~50 KB each, so we use small pages, cache them and fall back to single records on server errors."""
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    cache = path.with_name(".esco_cache")
    cache.mkdir(exist_ok=True)

    def get(url):
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 500:
                    raise
            except Exception:
                pass
            time.sleep(min(60, 3 * 2 ** attempt))
        raise RuntimeError(f"ESCO API not reachable ({url}); run again to resume")

    def page(n, size=10):
        cached = cache / f"page_{size}_{n:04d}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        params = {"type": "occupation", "language": "en", "limit": size, "offset": n}
        try:
            data = get(f"{ESCO_API}/search?{urllib.parse.urlencode({**params, 'full': 'true'})}")
        except urllib.error.HTTPError:
            data = get(f"{ESCO_API}/search?{urllib.parse.urlencode({**params, 'full': 'false'})}")
            records = []
            for light in data["_embedded"]["results"]:
                try:
                    records.append(get(f"{ESCO_API}/resource/occupation?"
                                       f"{urllib.parse.urlencode({'uri': light['uri'], 'language': 'en'})}"))
                except urllib.error.HTTPError:
                    records.append(light)
            data["_embedded"]["results"] = records
        cached.write_text(json.dumps(data), encoding="utf-8")
        return data

    log.info("Downloading ESCO occupations (one-off, takes a while)...")
    first = page(0)
    results = first["_embedded"]["results"]
    for n in range(1, -(-first["total"] // 10)):
        results += page(n)["_embedded"]["results"]
    occupations = []
    for r in results:
        code = r.get("code", "") or ""
        occupations.append({
            "uri": r["uri"], "code": code, "isco": code.split(".")[0][:4],
            "label_en": (r.get("preferredLabel") or {}).get("en", ""),
            "label_nl": (r.get("preferredLabel") or {}).get("nl", ""),
            "alt_en": (r.get("alternativeLabel") or {}).get("en", []),
            "alt_nl": (r.get("alternativeLabel") or {}).get("nl", []),
            "description_en": ((r.get("description") or {}).get("en") or {}).get("literal", ""),
        })
    path.write_text(json.dumps(occupations, ensure_ascii=False, indent=1), encoding="utf-8")


def load_esco(path: Path) -> list[dict]:
    if not path.exists():
        download_esco(path)
    return json.loads(path.read_text(encoding="utf-8"))


def embed_esco(occupations: list[dict], model, model_name: str, cache: Path):
    """Embeddings of all ESCO occupation names (EN + NL, incl. alternatives) and descriptions."""
    key = f"{model_name}|{len(occupations)}|{MAX_ALT_LABELS}"
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if str(stored["key"]) == key:
            return stored["names"], stored["owner"], stored["desc"]
    names, owner = [], []
    for j, o in enumerate(occupations):
        labels = [o["label_en"], o["label_nl"]] + o["alt_en"][:MAX_ALT_LABELS] + o["alt_nl"][:MAX_ALT_LABELS]
        labels = list(dict.fromkeys(l for l in labels if l))
        names += labels
        owner += [j] * len(labels)
    log.info("Embedding %d ESCO occupation names (one-off, cached)", len(names))
    name_emb = model.encode(names, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    desc_emb = model.encode([o["description_en"] or o["label_en"] for o in occupations], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=False)
    np.savez(cache, names=name_emb, owner=np.array(owner), desc=desc_emb, key=np.array(key))
    return name_emb, np.array(owner), desc_emb


def add_aliases(occupations: list[dict], model, name_emb, owner):
    """Append the market names from ESCO_ALIASES to the names of their ESCO occupation."""
    index = {o["code"]: j for j, o in enumerate(occupations)}
    names, extra_owner = [], []
    for code, aliases in ESCO_ALIASES.items():
        if code in index:
            names += aliases
            extra_owner += [index[code]] * len(aliases)
    alias_emb = model.encode(names, normalize_embeddings=True, show_progress_bar=False)
    return np.vstack([name_emb, alias_emb]), np.concatenate([owner, extra_owner])


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - x.max(axis=1, keepdims=True)) / temperature
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- main program


def categorise(df: pd.DataFrame, emb: np.ndarray, model_name: str, esco_path: Path, temperature: float = 0.02):
    """Match every vacancy to its closest ESCO occupation and derive AI / Data / Tech / Other from it.

    score(vacancy, occupation) = 0.6 x best match of the vacancy title with any of the occupation's names
                               + 0.4 x match of the full vacancy text with the occupation's description
    category_confidence_percent = softmax over the best-scoring occupation per category: how clearly the
    winning category beats the others."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    occupations = load_esco(esco_path)
    name_emb, owner, desc_emb = embed_esco(occupations, model, model_name,
                                           esco_path.with_name("esco_embeddings.npz"))
    name_emb, owner = add_aliases(occupations, model, name_emb, owner)
    titles = model.encode(df["title"].tolist(), normalize_embeddings=True)

    name_sim = titles @ name_emb.T                                   # vacancies x all names
    title_score = np.full((len(df), len(occupations)), -1.0)
    np.maximum.at(title_score.T, owner, name_sim.T)                  # best name per occupation
    score = TITLE_WEIGHT * title_score + (1 - TITLE_WEIGHT) * (emb @ desc_emb.T)
    professional = np.array([o["isco"][:1] in PROFESSIONAL_GROUPS for o in occupations])
    score[:, ~professional] = -np.inf

    best = score.argmax(axis=1)
    categories = ["AI", "Data", "Tech", "Other"]
    occ_cat = np.array([esco_category(o["code"]) for o in occupations])
    per_cat = np.column_stack([np.where(occ_cat == c, score, -np.inf).max(axis=1) for c in categories])
    prob = softmax(per_cat, temperature)
    winner = [categories[i] for i in per_cat.argmax(axis=1)]
    return pd.DataFrame({
        "job_category": winner,
        "category_confidence_percent": (prob.max(axis=1) * 100).round(1),
        "esco_occupation": [occupations[j]["label_en"] for j in best],
        "esco_code": [occupations[j]["code"] for j in best],
        "isco_code": [occupations[j]["isco"] for j in best],
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cluster vacancies and classify them into AI/Tech/Data with SBERT.")
    ap.add_argument("--input", type=Path, required=True, help="CSV with title, location, link, description")
    ap.add_argument("--output", type=Path, help="default: <input>_sbert.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, help="fixed number of clusters (default: chosen by silhouette, 6-15)")
    ap.add_argument("--esco", type=Path, default=Path(__file__).resolve().parent / "esco_occupations.json",
                    help="ESCO occupations file (downloaded automatically if missing)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("sentence_transformers", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    output = args.output or args.input.with_name(args.input.stem + "_sbert.csv")
    extra = args.input.with_name(args.input.stem + "_sbert")
    extra.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    missing = {"title", "location", "link", "description"} - set(df.columns)
    if missing:
        raise SystemExit(f"Columns missing in {args.input}: {sorted(missing)}")
    df = df[df["description"].str.strip() != ""].reset_index(drop=True)

    # 1. embeddings
    emb = embed(df, args.model, extra / "embeddings.npz")

    # 2-4. clusters, relevancy and names
    if args.k:
        k = args.k
        scores = pd.DataFrame([{"k": k, "silhouette": silhouette_score(
            emb, KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(emb), metric="cosine")}])
    else:
        k, scores = choose_k(emb, DEFAULT_K)
    labels, relevancy, similarity = cluster(emb, k)
    top = top_words(df, labels)
    names = cluster_names(top, extra / "cluster_names.json")
    df["cluster_id"] = labels
    df["cluster"] = [names[c] for c in labels]
    df["relevancy_percent"] = relevancy.round(1)
    df["cosine"] = similarity.round(3)

    # 5. AI / Data / Tech / Other via ESCO
    esco = categorise(df, emb, args.model, args.esco)
    df = pd.concat([df, esco], axis=1)

    columns = ["job_category", "category_confidence_percent", "esco_occupation", "esco_code", "isco_code",
               "cluster", "relevancy_percent", "title", "location", "link", "description"]
    order = {"AI": 0, "Data": 1, "Tech": 2, "Other": 3}
    df.sort_values(["job_category", "category_confidence_percent"], ascending=[True, False],
                   key=lambda s: s.map(order) if s.name == "job_category" else s
                   )[columns].to_csv(output, index=False, encoding="utf-8")

    # extra: overview, cluster table and silhouette chart
    pd.DataFrame([{"cluster_id": c, "cluster": names[c], "vacancies": int((labels == c).sum()),
                   "avg_relevancy": round(float(relevancy[labels == c].mean()), 1),
                   "top_10_words": ", ".join(top[c])} for c in sorted(top)]
                 ).to_csv(extra / "clusters.csv", index=False, encoding="utf-8")
    scores.to_csv(extra / "silhouette.csv", index=False)
    if len(scores) > 1:
        plot_silhouette(scores, k, extra / "silhouette.png")
    overview(df, top, names, scores, k, args.model, extra / "overview.md")

    print(f"\n{len(df)} vacancies -> {output}")
    print(f"\n{k} clusters:")
    for c in sorted(top):
        print(f"  {names[c][:45]:45} {int((labels == c).sum()):3} vac.  avg. relevancy {relevancy[labels == c].mean():5.1f}%")
    print("\nCategories:")
    for c in order:
        g = df[df.job_category == c]
        print(f"  {c:6} {len(g):3} vacancies, avg. confidence {g.category_confidence_percent.mean():.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
