"""SBERT-pipeline voor vacatures: clusteren + relevancy + indeling in AI / Tech / Data / Other.

Eén bestand, één commando:

    pip install pandas numpy scikit-learn matplotlib sentence-transformers
    python sbert_pipeline.py --input ey_vacancies.csv

Invoer: komma-gescheiden CSV met de kolommen  title, location, link, description
Uitvoer: <invoer>_sbert.csv met de kolommen
         job_category, category_confidence_percent, cluster, relevancy_percent,
         title, location, link, description
         + een map <invoer>_sbert/ met grafiek, overzicht per cluster en cache.

Wat er gebeurt:
    1. EMBEDDINGS  Elke vacature (titel + description) wordt met een meertalig SBERT-model
                   (NL + EN) omgezet naar een embedding: 768 getallen die de betekenis weergeven.
                   Het model leest max. 128 tokens; langere teksten worden in stukken geknipt en
                   de embeddings gemiddeld, zodat de hele vacature meetelt.
    2. CLUSTERS    KMeans groepeert vacatures die qua betekenis op elkaar lijken. Het aantal
                   clusters (6-15) wordt gekozen met de silhouette-score.
    3. RELEVANCY   Hoe dicht een vacature bij het midden van zijn cluster ligt (cosine), geschaald
                   over de dataset: 100% = meest typische vacature, 0% = minst typische.
    4. NAMEN       Per cluster de 3 kenmerkendste woorden (class-based TF-IDF). Leesbare namen kun
                   je vastleggen in <invoer>_sbert/cluster_namen.json.
    5. CATEGORIE   AI / Tech / Data / Other, zonder trainingsdata: de titel wordt vergeleken met
                   typische functietitels uit de markt (60%), de hele tekst met beschrijvende
                   zinnen (40%). category_confidence_percent = hoe duidelijk de winnaar wint.
                   Op KPMG-data: 92% gelijk aan een handmatige controle. Controleer vooral AI.

Opties:  --k 9   vast aantal clusters      --output <pad>   andere uitvoernaam
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


# Beschrijving per categorie. Pas deze zinnen aan om de grenzen te verschuiven.
CATEGORIEEN = {
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


# Typische functietitels per categorie, bewust uit de algemene arbeidsmarkt en NIET uit de
# KPMG-vacatures, zodat de indeling ook op andere werkgevers (EY, Deloitte, ...) werkt.
TITELS = {
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
TITEL_GEWICHT = 0.6  # 0,5-0,7 gaf vrijwel dezelfde uitkomst op de controleset (91-92,5% gelijk)


def softmax(x: np.ndarray, temperatuur: float) -> np.ndarray:
    z = (x - x.max(axis=1, keepdims=True)) / temperatuur
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)




# ---------------------------------------------------------------- hoofdprogramma


def categoriseer(df: pd.DataFrame, emb: np.ndarray, model_naam: str, temperatuur: float = 0.02):
    """AI / Tech / Data / Other per vacature + zekerheid (softmax over de vier scores)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_naam)
    namen = list(CATEGORIEEN)
    titels = model.encode(df["title"].tolist(), normalize_embeddings=True)
    titel_score = np.column_stack([
        (titels @ model.encode(TITELS[c], normalize_embeddings=True).T).max(axis=1) for c in namen])
    tekst_score = np.column_stack([
        (emb @ model.encode(CATEGORIEEN[c], normalize_embeddings=True).T).mean(axis=1) for c in namen])
    gelijkenis = TITEL_GEWICHT * titel_score + (1 - TITEL_GEWICHT) * tekst_score
    kans = softmax(gelijkenis, temperatuur)
    return [namen[i] for i in gelijkenis.argmax(axis=1)], (kans.max(axis=1) * 100).round(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vacatures clusteren en indelen in AI/Tech/Data met SBERT.")
    ap.add_argument("--input", type=Path, required=True, help="CSV met title, location, link, description")
    ap.add_argument("--output", type=Path, help="standaard: <input>_sbert.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, help="vast aantal clusters (standaard: kiezen op silhouette, 6-15)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for luid in ("sentence_transformers", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(luid).setLevel(logging.WARNING)

    uitvoer = args.output or args.input.with_name(args.input.stem + "_sbert.csv")
    extra = args.input.with_name(args.input.stem + "_sbert")
    extra.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    ontbrekend = {"title", "location", "link", "description"} - set(df.columns)
    if ontbrekend:
        raise SystemExit(f"Kolommen ontbreken in {args.input}: {sorted(ontbrekend)}")
    df = df[df["description"].str.strip() != ""].reset_index(drop=True)

    # 1. embeddings
    emb = embed(df, args.model, extra / "embeddings.npz")

    # 2-4. clusters, relevancy en namen
    if args.k:
        k = args.k
        scores = pd.DataFrame([{"k": k, "silhouette": silhouette_score(
            emb, KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(emb), metric="cosine")}])
    else:
        k, scores = kies_k(emb, STANDAARD_K)
    labels, relevancy, gelijkenis = cluster(emb, k)
    top = topwoorden(df, labels)
    namen = clusternamen(top, extra / "cluster_namen.json")
    df["cluster_id"] = labels
    df["cluster"] = [namen[c] for c in labels]
    df["relevancy_percent"] = relevancy.round(1)
    df["cosine"] = gelijkenis.round(3)

    # 5. AI / Tech / Data / Other
    df["job_category"], df["category_confidence_percent"] = categoriseer(df, emb, args.model)

    kolommen = ["job_category", "category_confidence_percent", "cluster", "relevancy_percent",
                "title", "location", "link", "description"]
    volgorde = {"AI": 0, "Data": 1, "Tech": 2, "Other": 3}
    df.sort_values(["job_category", "category_confidence_percent"], ascending=[True, False],
                   key=lambda s: s.map(volgorde) if s.name == "job_category" else s
                   )[kolommen].to_csv(uitvoer, index=False, encoding="utf-8")

    # extra: overzicht, clustertabel en silhouette-grafiek
    pd.DataFrame([{"cluster_id": c, "cluster": namen[c], "vacatures": int((labels == c).sum()),
                   "gem_relevancy": round(float(relevancy[labels == c].mean()), 1),
                   "top_10_woorden": ", ".join(top[c])} for c in sorted(top)]
                 ).to_csv(extra / "clusters.csv", index=False, encoding="utf-8")
    scores.to_csv(extra / "silhouette.csv", index=False)
    if len(scores) > 1:
        plot_silhouette(scores, k, extra / "silhouette.png")
    overzicht(df, top, namen, scores, k, args.model, extra / "overzicht.md")

    print(f"\n{len(df)} vacatures -> {uitvoer}")
    print(f"\n{k} clusters:")
    for c in sorted(top):
        print(f"  {namen[c][:45]:45} {int((labels == c).sum()):3} vac.  gem. relevancy {relevancy[labels == c].mean():5.1f}%")
    print("\nCategorieën:")
    for c in volgorde:
        g = df[df.job_category == c]
        print(f"  {c:6} {len(g):3} vacatures, gem. zekerheid {g.category_confidence_percent.mean():.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
