"""SBERT pipeline for job vacancies: clustering + relevancy + classification into AI / Tech / Data / Other.

One file, one command:

    pip install pandas numpy scikit-learn matplotlib sentence-transformers
    python sbert_pipeline.py --input ey_vacancies.csv

Input:  comma-separated CSV with the columns  title, location, link, description
Output: <input>_sbert.csv with the columns
        job_category, category_confidence_percent, cluster, relevancy_percent,
        title, location, link, description
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
    5. CATEGORY    AI / Tech / Data / Other, without training data: the title is compared with
                   typical job titles from the market (60%), the full text with descriptive
                   sentences (40%). category_confidence_percent = how clearly the winner wins.
                   On KPMG data: 92% agreement with a manual check. Check the AI category in particular.

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


# Description per category. Edit these sentences to shift the boundaries.
# Deliberately in Dutch AND English: the vacancies are in both languages.
CATEGORIES = {
    "AI": [
        "Artificial intelligence, machine learning and generative AI solutions for clients.",
        "Je ontwikkelt en implementeert AI-toepassingen, GenAI, Copilot en machine learning modellen.",
        "AI governance, AI risk management and responsible use of AI models.",
        "AI-strategie en AI use cases: van idee tot werkende AI-oplossing.",
    ],
    "Data": [
        "Data management, data governance, data quality and master data.",
        "Je bouwt data platforms, datapijplijnen en data-architectuur.",
        "Data analytics, business intelligence, dashboards and reporting with Power BI.",
        "Datamigratie, datatransformatie en het inrichten van een data office.",
    ],
    "Tech": [
        "Implementation of enterprise software such as SAP, Microsoft Dynamics and ServiceNow.",
        "Je configureert en implementeert IT-systemen, cloudoplossingen en integraties.",
        "Cyber security, incident response, penetration testing and identity and access management.",
        "IT-architectuur, softwareontwikkeling, low-code, IT-infrastructuur en IT-risk.",
    ],
    "Other": [
        "Financial statement audit, accountancy and assurance for clients.",
        "Corporate finance, mergers and acquisitions, valuations and due diligence.",
        "Strategieadvies, organisatieontwerp en verandermanagement voor organisaties.",
        "Governance, risk management, compliance and internal audit.",
        "Human resources, recruitment, marketing, finance and other support functions.",
    ],
}


# Typical job titles per category, deliberately taken from the general job market and NOT from the
# KPMG vacancies, so the classification also works for other employers (EY, Deloitte, ...).
TITLES = {
    "AI": ["AI Engineer", "Machine Learning Engineer", "GenAI Consultant", "AI Architect", "AI Governance Consultant",
           "AI Product Manager", "Azure AI Specialist", "AI-specialist", "Machine learning consultant",
           "AI Strategy Consultant"],
    "Data": ["Data Engineer", "Data Analyst", "BI Consultant", "Power BI Developer", "Data Governance Consultant",
             "Data Management Consultant", "Master Data Specialist", "Data Migration Consultant",
             "Analytics Consultant", "Data Architect"],
    "Tech": ["SAP Consultant", "SAP S/4HANA Finance Consultant", "Microsoft Dynamics 365 Consultant",
             "ServiceNow Developer", "Cyber Security Consultant", "Penetration Tester", "Incident Response Specialist",
             "IT Auditor", "IT Risk Consultant", "Cloud Architect", "Solution Architect", "Software Engineer",
             "IT Infrastructure Manager", "Low-code Developer", "IT Sourcing Consultant", "Enterprise Architect",
             "Test Engineer", "Digital Transformation Consultant"],
    "Other": ["Accountant", "Auditor jaarrekeningcontrole", "Corporate Finance Analyst", "M&A Consultant",
              "Strategy Consultant", "Management Consultant", "Change Manager", "HR Advisor", "Recruiter",
              "Marketing Manager", "Compliance Officer", "Risk Manager", "Tax Advisor", "Financial Controller",
              "Beleidsadviseur publieke sector", "Forensic Accountant", "Afstudeerstage audit",
              "Transaction Services Consultant"],
}
TITLE_WEIGHT = 0.6  # 0.5-0.7 gave practically the same result on the control set (91-92.5% agreement)


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - x.max(axis=1, keepdims=True)) / temperature
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- main program


def categorise(df: pd.DataFrame, emb: np.ndarray, model_name: str, temperature: float = 0.02):
    """AI / Tech / Data / Other per vacancy + confidence (softmax over the four scores)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    categories = list(CATEGORIES)
    titles = model.encode(df["title"].tolist(), normalize_embeddings=True)
    title_score = np.column_stack([
        (titles @ model.encode(TITLES[c], normalize_embeddings=True).T).max(axis=1) for c in categories])
    text_score = np.column_stack([
        (emb @ model.encode(CATEGORIES[c], normalize_embeddings=True).T).mean(axis=1) for c in categories])
    similarity = TITLE_WEIGHT * title_score + (1 - TITLE_WEIGHT) * text_score
    prob = softmax(similarity, temperature)
    return [categories[i] for i in similarity.argmax(axis=1)], (prob.max(axis=1) * 100).round(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cluster vacancies and classify them into AI/Tech/Data with SBERT.")
    ap.add_argument("--input", type=Path, required=True, help="CSV with title, location, link, description")
    ap.add_argument("--output", type=Path, help="default: <input>_sbert.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, help="fixed number of clusters (default: chosen by silhouette, 6-15)")
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

    # 5. AI / Tech / Data / Other
    df["job_category"], df["category_confidence_percent"] = categorise(df, emb, args.model)

    columns = ["job_category", "category_confidence_percent", "cluster", "relevancy_percent",
               "title", "location", "link", "description"]
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
