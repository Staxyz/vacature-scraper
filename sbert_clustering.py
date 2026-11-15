"""Vacatures clusteren met SBERT (Sentence-BERT) en per vacature een relevancy-percentage geven.

Werkwijze:
    1. Elke vacature (titel + description) wordt met een meertalig SBERT-model omgezet naar een
       embedding: een rij getallen die de betekenis van de tekst weergeeft.
       Het model leest maximaal 128 tokens; langere teksten worden in stukken geknipt en de
       embeddings van de stukken worden gemiddeld, zodat de hele vacature meetelt.
    2. KMeans groepeert de embeddings. Het aantal clusters wordt gekozen met de silhouette-score
       (hoe goed vacatures bij hun eigen cluster passen tegenover het dichtstbijzijnde andere).
    3. relevancy_percent = hoe dicht de vacature bij het midden van zijn cluster ligt (cosine-gelijkenis),
       geschaald over de dataset: 100% = meest typische vacature, 0% = minst typische.
    4. Elk cluster krijgt een naam: uit cluster_namen.json als die past, anders de top 3
       kenmerkende woorden (class-based TF-IDF).

Gebruik:
    python sbert_clustering.py                                   # output/kpmg_vacancies.csv
    python sbert_clustering.py --input output/ey_vacancies.csv --output output/ey_vacancies_clustered.csv
    python sbert_clustering.py --k 10                            # vast aantal clusters

Invoer: CSV met de kolommen title, location, link, description (zie export_voor_nlp.py).
Uitvoer: dezelfde CSV met vooraan de kolommen cluster en relevancy_percent, plus een
overzicht en grafiek in output/sbert/.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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

ROOT = Path(__file__).resolve().parent
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # SBERT, meertalig (NL + EN)
CHUNK_WOORDEN = 70        # ±110 tokens: past binnen de 128-token-grens van het model
RANDOM_STATE = 42         # vast, zodat elke run dezelfde clusters geeft
STANDAARD_K = range(6, 16)


# ---------------------------------------------------------------- embeddings


def stukken(tekst: str, n: int = CHUNK_WOORDEN) -> list[str]:
    woorden = tekst.split()
    return [" ".join(woorden[i:i + n]) for i in range(0, len(woorden), n)] or [""]


def embed(df: pd.DataFrame, model_naam: str, cache: Path) -> np.ndarray:
    """Eén genormaliseerde embedding per vacature: gemiddelde van de titel en alle tekststukken."""
    sleutel = f"{model_naam}|{CHUNK_WOORDEN}|" + "|".join(df["link"])
    if cache.exists():
        opgeslagen = np.load(cache, allow_pickle=True)
        if str(opgeslagen["sleutel"]) == sleutel:
            log.info("Embeddings uit cache")
            return opgeslagen["emb"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_naam)
    alle, eigenaar = [], []
    for i, r in df.iterrows():
        delen = [r["title"]] + stukken(r["description"])
        alle += delen
        eigenaar += [i] * len(delen)
    log.info("%d tekststukken voor %d vacatures embedden met %s", len(alle), len(df), model_naam)
    vec = model.encode(alle, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    eigenaar = np.array(eigenaar)
    emb = np.vstack([vec[eigenaar == i].mean(axis=0) for i in range(len(df))])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, sleutel=np.array(sleutel))
    return emb


# ---------------------------------------------------------------- clusteren


def kies_k(emb: np.ndarray, kandidaten) -> tuple[int, pd.DataFrame]:
    """Silhouette per aantal clusters (cosine). Hoger = clusters beter van elkaar gescheiden."""
    rijen = []
    for k in kandidaten:
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(emb)
        rijen.append({"k": k, "silhouette": silhouette_score(emb, labels, metric="cosine")})
    scores = pd.DataFrame(rijen)
    beste = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return beste, scores


def cluster(emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(emb)
    centra = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    gelijkenis = np.einsum("ij,ij->i", emb, centra[km.labels_])
    # Alle KPMG-vacatures lijken in de basis op elkaar (cosine 0,82-0,98), dus ruwe percentages
    # zeggen weinig. We schalen over het bereik dat echt voorkomt:
    # 100% = meest typische vacature voor zijn cluster, 0% = minst typische in de dataset.
    relevancy = (gelijkenis - gelijkenis.min()) / (gelijkenis.max() - gelijkenis.min()) * 100
    return km.labels_, relevancy, gelijkenis


# ---------------------------------------------------------------- namen


def stopwoorden() -> set[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    try:
        from spacy.lang.nl.stop_words import STOP_WORDS as NL
    except ImportError:
        NL = set()
    extra = {"kpmg", "je", "jij", "jouw", "wij", "we", "onze", "bent", "hebt", "ervaring", "experience",
             "work", "team", "klanten", "clients", "goed", "nieuwe", "binnen", "samen", "the", "and"}
    return set(ENGLISH_STOP_WORDS) | set(NL) | extra


def topwoorden(df: pd.DataFrame, labels: np.ndarray, n: int = 10) -> dict[int, list[str]]:
    """Kenmerkende woorden per cluster (class-based TF-IDF): vaak in dit cluster, weinig in de rest."""
    cv = CountVectorizer(stop_words=list(stopwoorden()), token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
                         min_df=3, max_df=0.5, lowercase=True)
    X = cv.fit_transform(df["title"] + " " + df["description"])
    vocab = cv.get_feature_names_out()
    tf = np.vstack([np.asarray(X[labels == c].sum(axis=0)).ravel() for c in sorted(set(labels))])
    ctfidf = (tf / tf.sum(axis=1, keepdims=True)) * np.log(1 + tf.sum(axis=1).mean() / tf.sum(axis=0))
    return {c: list(vocab[np.argsort(ctfidf[c])[::-1][:n]]) for c in sorted(set(labels))}


def clusternamen(top: dict[int, list[str]], bestand: Path) -> dict[int, str]:
    """Leesbare namen uit cluster_namen.json, alleen als het cluster nog dezelfde topwoorden heeft
    (Jaccard >= 0.5); anders de top 3 woorden."""
    vast = json.loads(bestand.read_text(encoding="utf-8")) if bestand.exists() else []
    namen = {}
    for c, woorden in top.items():
        beste, score = None, 0.0
        for item in vast:
            a, b = set(woorden), set(item["woorden"])
            j = len(a & b) / len(a | b)
            if j > score:
                beste, score = item["naam"], j
        namen[c] = beste if beste and score >= 0.5 else " / ".join(woorden[:3])
    return namen


# ---------------------------------------------------------------- output


def plot_silhouette(scores: pd.DataFrame, k: int, pad: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(scores["k"], scores["silhouette"], marker="o", color="#2e9e5b", linewidth=2)
    beste = scores[scores["k"] == k].iloc[0]
    ax.scatter([k], [beste["silhouette"]], s=160, facecolors="none", edgecolors="#2e9e5b", linewidths=2)
    ax.annotate(f"gekozen: {k}", (k, beste["silhouette"]), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9)
    ax.set_xlabel("Aantal clusters")
    ax.set_ylabel("Silhouette (cosine)")
    ax.set_title("SBERT + KMeans: hoe goed zijn de clusters gescheiden?")
    ax.set_xticks(list(scores["k"]))
    ax.margins(y=0.2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pad, dpi=150)
    plt.close(fig)


def overzicht(df: pd.DataFrame, top: dict, namen: dict, scores: pd.DataFrame, k: int, model: str,
              pad: Path) -> None:
    regels = ["# SBERT-clustering (automatisch gegenereerd)", "",
              f"- Model: `{model}` (max. 128 tokens; tekst in stukken van {CHUNK_WOORDEN} woorden, gemiddeld)",
              f"- Vacatures: {len(df)}; clusters: {k} (hoogste silhouette); random_state = {RANDOM_STATE}",
              f"- relevancy_percent = cosine-gelijkenis met het midden van het eigen cluster, geschaald "
              f"tussen de laagste ({df.cosine.min():.3f} = 0%) en hoogste ({df.cosine.max():.3f} = 100%) waarde "
              f"in deze dataset; gemiddeld {df.relevancy_percent.mean():.0f}%", "",
              "## Silhouette per aantal clusters", "", scores.round(4).to_markdown(index=False), ""]
    for c in sorted(top):
        g = df[df["cluster_id"] == c].sort_values("relevancy_percent", ascending=False)
        regels += [f"## {namen[c]} ({len(g)} vacatures, gem. relevancy {g.relevancy_percent.mean():.1f}%)", "",
                   f"Kenmerkende woorden: {', '.join(top[c])}", "",
                   "Meest typisch: " + "; ".join(f"{t} ({p:.0f}%)" for t, p in
                                                 zip(g.title.head(3), g.relevancy_percent.head(3))),
                   "", "Minst typisch: " + "; ".join(f"{t} ({p:.0f}%)" for t, p in
                                                    zip(g.title.tail(2), g.relevancy_percent.tail(2))), ""]
    pad.write_text("\n".join(regels), encoding="utf-8")


# ---------------------------------------------------------------- hoofdprogramma


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vacatures clusteren met SBERT.")
    ap.add_argument("--input", type=Path, default=ROOT / "output" / "kpmg_vacancies.csv")
    ap.add_argument("--output", type=Path, help="standaard: <input>_clustered.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, help="vast aantal clusters (standaard: kiezen op silhouette, 6-15)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for luid in ("sentence_transformers", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(luid).setLevel(logging.WARNING)

    uitvoer = args.output or args.input.with_name(args.input.stem + "_clustered.csv")
    extra = ROOT / "output" / "sbert"
    extra.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False).reset_index(drop=True)
    df = df[df["description"].str.strip() != ""].reset_index(drop=True)
    emb = embed(df, args.model, extra / f"{args.input.stem}_embeddings.npz")

    k, scores = (args.k, pd.DataFrame()) if args.k else kies_k(emb, STANDAARD_K)
    if args.k:
        scores = pd.DataFrame([{"k": k, "silhouette": silhouette_score(
            emb, KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(emb), metric="cosine")}])
    labels, relevancy, gelijkenis = cluster(emb, k)
    top = topwoorden(df, labels)
    namen = clusternamen(top, extra / "cluster_namen.json")

    df["cluster_id"] = labels
    df["cluster"] = [namen[c] for c in labels]
    df["relevancy_percent"] = relevancy.round(1)
    df["cosine"] = gelijkenis.round(3)

    # Kolomvolgorde zoals afgesproken: cluster, relevancy_percent, title, location, link, description
    resultaat = df.sort_values(["cluster", "relevancy_percent"], ascending=[True, False])
    resultaat[["cluster", "relevancy_percent", "title", "location", "link", "description"]].to_csv(
        uitvoer, index=False, encoding="utf-8")

    pd.DataFrame([{"cluster_id": c, "cluster": namen[c], "vacatures": int((labels == c).sum()),
                   "gem_relevancy": round(float(relevancy[labels == c].mean()), 1),
                   "top_10_woorden": ", ".join(top[c])} for c in sorted(top)]
                 ).to_csv(extra / "clusters.csv", index=False, encoding="utf-8")
    scores.to_csv(extra / "silhouette.csv", index=False)
    if len(scores) > 1:
        plot_silhouette(scores, k, extra / "silhouette.png")
    overzicht(df, top, namen, scores, k, args.model, extra / "overzicht.md")

    print(f"\n{len(df)} vacatures in {k} clusters -> {uitvoer}")
    for c in sorted(top):
        print(f"  {c:2} {namen[c][:45]:45} {int((labels == c).sum()):3} vac.  "
              f"gem {relevancy[labels == c].mean():5.1f}%  | {', '.join(top[c][:6])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
