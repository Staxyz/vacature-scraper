"""Topic modeling op vacatureteksten: LDA versus NMF.

Gebruik (vanuit de projectmap):
    python topic_modeling.py                                  # nieuwste KPMG-CSV, standaardinstellingen
    python topic_modeling.py --input output/kpmg/kpmg_2026-11-01_1304.csv --topics 5 8 10 12 15

Stappen (zie de functies hieronder):
    1. Data verkennen        -> verken()
    2. Preprocessing         -> preprocess()
    3. Modellen trainen      -> train_modellen()
    4. Evalueren (c_v)       -> coherence(), plot_coherence()
    5. Output                -> schrijf_topics(), schrijf_documenten(), schrijf_voorbeelden(), pyldavis_html()
    6. Vergelijking          -> schrijf_resultaten()

Alle output komt in --output (standaard output/topic_models).
Leesbare topicnamen kun je vastleggen in <output>/topic_namen.json, bijvoorbeeld
    {"nmf": {"0": {"naam": "Internal audit & compliance", "woorden": ["internal", "audit", ...]}}}
Een naam wordt alleen gebruikt als het topic nog dezelfde topwoorden heeft; anders krijgt
het topic een automatische naam uit zijn top 3 woorden.

Herbruikbaar voor andere modellen/datasets: pas --text-cols, --lang-col en --topics aan,
of importeer de functies (preprocess, train_modellen, coherence) in een eigen script.
"""

from __future__ import annotations

import argparse
import glob
import warnings
import html
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # geen scherm nodig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

log = logging.getLogger("topic_modeling")
warnings.filterwarnings("ignore", category=FutureWarning)  # aankondigingen van sklearn/pyLDAvis

# ---------------------------------------------------------------- instellingen

ROOT = Path(__file__).resolve().parent
STANDAARD_TEKSTKOLOMMEN = ["sectie_jouw_uitdagingen", "sectie_wat_ga_je_doen", "sectie_dit_neem_jij_mee"]
SPACY_MODELLEN = {"NL": "nl_core_news_sm", "EN": "en_core_web_sm"}
# Woordsoorten die betekenis dragen; lidwoorden, voornaamwoorden enz. vallen weg
STANDAARD_POS = ["NOUN", "PROPN", "ADJ", "VERB"]
# Domeinwoorden die in vrijwel elke vacature staan en dus niets onderscheiden.
# Bewust kort gehouden; vul aan als ze een topic domineren.
EXTRA_STOPWOORDEN = {
    "kpmg", "kpmg'er", "kpmger", "jij", "jouw", "je", "wij", "we", "onze", "ons", "you", "your", "our",
    "zullen", "kunnen", "gaan", "worden", "hebben", "zijn", "will", "can", "be", "have",
    # algemene vacaturewoorden zonder onderscheidend vermogen (na de eerste run toegevoegd)
    "goed", "krijgen", "kans", "zorgen", "helpen", "nieuw", "gebied", "ver", "meet",
    "work", "role", "experience", "help", "grow", "make", "new",
}
# Fouten van de lemmatizer corrigeren: spaCy (NL) maakt van 'data' het meervoud van 'datum'.
LEMMA_CORRECTIES = {"data": "data", "change": "change", "changes": "change"}
TOP_N = 10            # woorden per topic (voor weergave én coherence)
RANDOM_STATE = 42     # vast, zodat elke run hetzelfde resultaat geeft


# ---------------------------------------------------------------- stap 1: verkennen


def laad_csv(pad: Path) -> pd.DataFrame:
    """De scraper schrijft CSV's met ';' en UTF-8 met BOM."""
    return pd.read_csv(pad, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def nieuwste_scraper_csv() -> Path:
    kandidaten = sorted(p for p in glob.glob(str(ROOT / "output" / "kpmg" / "kpmg_*.csv")) if "wijzigingen" not in p)
    if not kandidaten:
        raise SystemExit("Geen scraper-CSV gevonden in output/kpmg/. Geef --input op.")
    return Path(kandidaten[-1])


def verken(df: pd.DataFrame, tekstkolommen: list[str], taalkolom: str) -> dict:
    """Print een overzicht en geeft de kerncijfers terug voor het resultatenbestand."""
    woorden = df["tekst_origineel"].str.split().str.len()
    info = {
        "rijen": len(df),
        "kolommen": list(df.columns),
        "tekstkolommen": tekstkolommen,
        "talen": df[taalkolom].value_counts().to_dict() if taalkolom in df else {"onbekend": len(df)},
        "woorden_gem": round(float(woorden.mean()), 1),
        "woorden_mediaan": float(woorden.median()),
        "woorden_min": int(woorden.min()),
        "woorden_max": int(woorden.max()),
    }
    log.info("Rijen: %d, kolommen: %d", info["rijen"], len(info["kolommen"]))
    log.info("Tekst = %s", " + ".join(tekstkolommen))
    log.info("Talen: %s", info["talen"])
    log.info("Woorden per document: gem %.0f, mediaan %.0f (min %d, max %d)",
             info["woorden_gem"], info["woorden_mediaan"], info["woorden_min"], info["woorden_max"])
    return info


# ---------------------------------------------------------------- stap 2: preprocessing

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_HTML = re.compile(r"<[^>]+>")
_CIJFERS = re.compile(r"\d+")
_LEESTEKENS = re.compile(r"[^\w\s]|_")  # alles wat geen letter/cijfer/spatie is (incl. bullets •)


def schoon(tekst: str) -> str:
    """Kleine letters, zonder URL's, HTML, cijfers en leestekens."""
    tekst = html.unescape(tekst)
    tekst = _URL.sub(" ", tekst)
    tekst = _HTML.sub(" ", tekst)
    tekst = tekst.lower()
    tekst = _CIJFERS.sub(" ", tekst)
    tekst = _LEESTEKENS.sub(" ", tekst)
    return re.sub(r"\s+", " ", tekst).strip()


def preprocess(df: pd.DataFrame, taalkolom: str, pos: list[str]) -> pd.DataFrame:
    """Lege rijen/duplicaten eruit, opschonen, stopwoorden weg, lemmatiseren met spaCy (per taal)."""
    voor = len(df)
    df = df[df["tekst_origineel"].str.strip() != ""]
    df = df.drop_duplicates(subset="tekst_origineel").reset_index(drop=True)
    log.info("Preprocessing: %d lege/dubbele rijen verwijderd", voor - len(df))

    df["tekst_schoon"] = df["tekst_origineel"].map(schoon)
    if taalkolom not in df:
        df[taalkolom] = "NL"
    df[taalkolom] = df[taalkolom].where(df[taalkolom].isin(SPACY_MODELLEN), "NL")

    # Eén stopwoordenlijst voor beide talen: NL-vacatures bevatten ook Engelse vaktermen en omgekeerd.
    modellen = {taal: spacy.load(naam, disable=["parser", "ner"]) for taal, naam in SPACY_MODELLEN.items()
                if taal in set(df[taalkolom])}
    stop = set(EXTRA_STOPWOORDEN)
    for m in modellen.values():
        stop |= {w.lower() for w in m.Defaults.stop_words}

    lemmas = [""] * len(df)
    for taal, nlp in modellen.items():
        idx = df.index[df[taalkolom] == taal]
        for i, doc in zip(idx, nlp.pipe(df.loc[idx, "tekst_schoon"], batch_size=32)):
            tokens = [LEMMA_CORRECTIES.get(t.text, t.lemma_.lower()) for t in doc
                      if t.pos_ in pos and t.is_alpha and len(t) > 2]
            lemmas[i] = " ".join(w for w in tokens if w not in stop and len(w) > 2)
    df["tekst_lemma"] = lemmas
    leeg = (df["tekst_lemma"].str.len() == 0).sum()
    if leeg:
        log.warning("%d documenten zijn leeg na preprocessing en worden overgeslagen", leeg)
        df = df[df["tekst_lemma"].str.len() > 0].reset_index(drop=True)
    log.info("Lemma's per document: gem %.0f", df["tekst_lemma"].str.split().str.len().mean())
    return df


# ---------------------------------------------------------------- stap 3: modellen


def vectorizers(min_df, max_df):
    """Zelfde vocabulaire-instellingen voor beide modellen = eerlijke vergelijking.
    De tekst is al getokeniseerd (spaties), dus we splitsen alleen op witruimte."""
    gedeeld = dict(min_df=min_df, max_df=max_df, token_pattern=r"\S+", lowercase=False)
    return CountVectorizer(**gedeeld), TfidfVectorizer(**gedeeld)


def train_modellen(teksten: list[str], aantallen: list[int], min_df, max_df, seeds: list[int]) -> dict:
    """Traint LDA (op woordaantallen) en NMF (op TF-IDF) voor elk aantal topics en elke seed.

    De modellen met RANDOM_STATE worden bewaard voor de output; de andere seeds dienen alleen om
    te meten hoe stabiel de coherence is (LDA is gevoelig voor de startwaarde, NMF met nndsvda niet).
    """
    count_vec, tfidf_vec = vectorizers(min_df, max_df)
    X_count = count_vec.fit_transform(teksten)
    X_tfidf = tfidf_vec.fit_transform(teksten)
    log.info("Vocabulaire: %d woorden (CountVectorizer) / %d (TF-IDF)", X_count.shape[1], X_tfidf.shape[1])
    vocab_lda = count_vec.get_feature_names_out()
    vocab_nmf = tfidf_vec.get_feature_names_out()

    resultaat = {"count_vec": count_vec, "tfidf_vec": tfidf_vec, "X_count": X_count, "X_tfidf": X_tfidf,
                 "lda": {}, "nmf": {}, "topics_per_seed": []}
    for seed in seeds:
        for k in aantallen:
            lda = LatentDirichletAllocation(n_components=k, random_state=seed, learning_method="batch",
                                            max_iter=100)
            lda_doc = lda.fit_transform(X_count)          # rijen tellen op tot 1 = kansen
            nmf = NMF(n_components=k, random_state=seed, init="nndsvda", max_iter=1000)
            nmf_doc = nmf.fit_transform(X_tfidf)          # niet-genormaliseerde gewichten
            resultaat["topics_per_seed"] += [("LDA", k, seed, top_woorden(lda, vocab_lda)),
                                             ("NMF", k, seed, top_woorden(nmf, vocab_nmf))]
            if seed == RANDOM_STATE:
                resultaat["lda"][k] = {"model": lda, "doc_topic": lda_doc, "topics": top_woorden(lda, vocab_lda)}
                resultaat["nmf"][k] = {"model": nmf, "doc_topic": nmf_doc, "topics": top_woorden(nmf, vocab_nmf)}
        log.info("Getraind: seed %d (k = %s)", seed, ", ".join(map(str, aantallen)))
    return resultaat


def top_woorden(model, vocab: np.ndarray, n: int = TOP_N) -> list[list[str]]:
    return [list(vocab[np.argsort(comp)[::-1][:n]]) for comp in model.components_]


# ---------------------------------------------------------------- stap 4: evalueren


def coherence(topics: list[list[str]], tokens: list[list[str]], dictionary: Dictionary) -> float:
    """c_v-coherence (gensim): hoe vaak de topwoorden van een topic samen voorkomen in de teksten."""
    cm = CoherenceModel(topics=topics, texts=tokens, dictionary=dictionary, coherence="c_v", topn=TOP_N,
                        processes=1)
    return float(cm.get_coherence())


def overlap(topics: list[list[str]]) -> float:
    """Gemiddelde Jaccard-overlap van de topwoorden tussen alle paren topics.
    Lager = beter onderscheidbare topics."""
    sets = [set(t) for t in topics]
    paren = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean([len(a & b) / len(a | b) for a, b in paren])) if paren else 0.0


def kies_aantal(agg: pd.DataFrame, model: str, tolerantie: float) -> int:
    """Hoogste gemiddelde coherence; liggen meerdere aantallen binnen `tolerantie` daarvan
    (praktisch gelijk), dan het kleinste aantal: minder topics = beter leesbaar."""
    s = agg[agg["model"] == model]
    drempel = s["coherence_cv"].max() - tolerantie
    return int(s[s["coherence_cv"] >= drempel]["topics"].min())


def plot_coherence(agg: pd.DataFrame, beste: dict, pad: Path) -> None:
    """Gemiddelde c_v per aantal topics, met de spreiding over de seeds als foutbalk."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, kleur in (("LDA", "#2a6fdb"), ("NMF", "#e07a1f")):
        s = agg[agg["model"] == model]
        ax.errorbar(s["topics"], s["coherence_cv"], yerr=s["coherence_std"], marker="o", color=kleur,
                    label=f"{model} (gekozen: {beste[model]})", linewidth=2, capsize=4)
        k = s[s["topics"] == beste[model]].iloc[0]
        ax.scatter([k["topics"]], [k["coherence_cv"]], s=160, facecolors="none", edgecolors=kleur, linewidths=2)
        ax.annotate(f"{k['coherence_cv']:.3f}", (k["topics"], k["coherence_cv"]), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=kleur, fontsize=9)
    ax.set_xlabel("Aantal topics")
    ax.set_ylabel("Coherence (c_v), gemiddelde over seeds")
    ax.set_title("Coherence per aantal topics: LDA vs NMF")
    ax.set_xticks(sorted(agg["topics"].unique()))
    ax.margins(y=0.15)  # ruimte voor de labels boven de lijnen
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(pad, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- stap 5: output


def topic_namen(topics: list[list[str]], model: str, namen_bestand: Path) -> list[str]:
    """Leesbare namen uit topic_namen.json, anders de top 3 woorden.

    Een naam wordt alleen gebruikt als het topic nog grotendeels dezelfde topwoorden heeft als toen
    de naam werd gegeven (Jaccard >= 0.5). Zo blijft een oude naam niet aan een ander topic hangen
    na nieuwe data of andere instellingen.
    """
    namen = {}
    if namen_bestand.exists():
        namen = json.loads(namen_bestand.read_text(encoding="utf-8")).get(model, {})
    uit = []
    for woorden in topics:
        beste, score = None, 0.0
        for item in namen.values():
            a, b = set(woorden), set(item.get("woorden", []))
            j = len(a & b) / len(a | b) if a | b else 0
            if j > score:
                beste, score = item["naam"], j
        uit.append(beste if beste and score >= 0.5 else " / ".join(woorden[:3]))
    return uit


def schrijf_topics(lda_topics, nmf_topics, lda_namen, nmf_namen, out: Path) -> None:
    rijen = []
    for model, topics, namen in (("LDA", lda_topics, lda_namen), ("NMF", nmf_topics, nmf_namen)):
        for i, (woorden, naam) in enumerate(zip(topics, namen)):
            rijen.append({"model": model, "topic": i, "naam": naam, "top_10_woorden": ", ".join(woorden)})
    pd.DataFrame(rijen).to_csv(out / "topics_top10.csv", sep=";", index=False, encoding="utf-8-sig")


def schrijf_documenten(df, lda_doc, nmf_doc, lda_namen, nmf_namen, taalkolom, out: Path) -> pd.DataFrame:
    nmf_aandeel = nmf_doc / np.clip(nmf_doc.sum(axis=1, keepdims=True), 1e-12, None)
    res = pd.DataFrame({
        "vacature_id": df.get("vacature_id", pd.Series(range(len(df)))),
        "titel": df.get("titel", ""),
        "taal": df[taalkolom],
        "interessegebied": df.get("interessegebied", ""),
        "lda_topic": lda_doc.argmax(axis=1),
        "lda_topic_naam": [lda_namen[i] for i in lda_doc.argmax(axis=1)],
        "lda_kans": lda_doc.max(axis=1).round(4),
        "nmf_topic": nmf_doc.argmax(axis=1),
        "nmf_topic_naam": [nmf_namen[i] for i in nmf_doc.argmax(axis=1)],
        "nmf_gewicht": nmf_doc.max(axis=1).round(4),
        "nmf_aandeel": nmf_aandeel.max(axis=1).round(4),   # gewicht als deel van het totaal (0-1)
        "tekst_origineel": df["tekst_origineel"],
    })
    res.to_csv(out / "documenten_topics.csv", sep=";", index=False, encoding="utf-8-sig")
    return res


def schrijf_voorbeelden(res: pd.DataFrame, lda_topics, nmf_topics, lda_namen, nmf_namen, out: Path) -> None:
    """Per topic de 3 documenten die het sterkst bij het topic horen."""
    regels = ["# Voorbeelddocumenten per topic", "",
              "Per topic de 3 vacatures met de hoogste kans (LDA) of het hoogste gewicht (NMF).", ""]
    for model, topics, namen, kol, score in (("LDA", lda_topics, lda_namen, "lda_topic", "lda_kans"),
                                              ("NMF", nmf_topics, nmf_namen, "nmf_topic", "nmf_aandeel")):
        regels += [f"## {model}", ""]
        for i, woorden in enumerate(topics):
            groep = res[res[kol] == i].sort_values(score, ascending=False)
            regels += [f"### {model} {i}: {namen[i]} ({len(groep)} documenten)", "",
                       f"Topwoorden: {', '.join(woorden)}", ""]
            if groep.empty:
                regels += ["_Geen document heeft dit als hoofdtopic._", ""]
            for _, r in groep.head(3).iterrows():
                fragment = re.sub(r"\s+", " ", r["tekst_origineel"])[:220]
                regels.append(f"- **{r['titel']}** ({r['taal']}, {score} {r[score]:.2f}): {fragment}…")
            regels.append("")
    (out / "voorbeelden_per_topic.md").write_text("\n".join(regels), encoding="utf-8")


def pyldavis_html(lda, X_count, count_vec, pad: Path) -> None:
    """Interactieve LDA-visualisatie. Gebouwd uit de matrices, zodat het met elke pyLDAvis-versie werkt.
    Let op: de HTML laadt de JavaScript-bibliotheken van internet (cdn), dus open hem met internet."""
    import pyLDAvis

    topic_term = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    doc_topic = lda.transform(X_count)
    vis = pyLDAvis.prepare(
        topic_term_dists=topic_term,
        doc_topic_dists=doc_topic,
        doc_lengths=np.asarray(X_count.sum(axis=1)).ravel(),
        vocab=count_vec.get_feature_names_out().tolist(),
        term_frequency=np.asarray(X_count.sum(axis=0)).ravel(),
        sort_topics=False,  # topicnummers gelijk houden aan de rest van de output (pyLDAvis telt vanaf 1)
        mds="mmds",
    )
    pyLDAvis.save_html(vis, str(pad))


# ---------------------------------------------------------------- stap 6: resultaten


def schrijf_resultaten(info, instellingen, scores, beste, overlaps, kruistabellen, out: Path) -> None:
    """Automatisch gegenereerd overzicht met de cijfers. De inhoudelijke interpretatie staat in samenvatting.md."""
    tabel = scores.pivot(index="topics", columns="model", values="coherence_cv").round(4)
    spreiding = scores.pivot(index="topics", columns="model", values="coherence_std").round(4)
    ov = scores.pivot(index="topics", columns="model", values="overlap_top10").round(3)
    regels = [
        "# Resultaten topic modeling (automatisch gegenereerd)", "",
        f"- Documenten: {info['rijen']} (talen: {info['talen']})",
        f"- Tekst: {' + '.join(info['tekstkolommen'])}",
        f"- Woorden per document (origineel): gemiddeld {info['woorden_gem']:.0f}, "
        f"mediaan {info['woorden_mediaan']:.0f}, min {info['woorden_min']}, max {info['woorden_max']}",
        f"- Lemma's per document na preprocessing: gemiddeld {info['lemmas_gem']:.0f}",
        f"- Vocabulaire na min_df={instellingen['min_df']}, max_df={instellingen['max_df']}: "
        f"{info['vocab']} woorden",
        f"- random_state eindmodellen = {RANDOM_STATE}", "",
        f"## Coherence (c_v) per aantal topics, gemiddelde over seeds {instellingen['seeds']}", "",
        tabel.to_markdown(), "", "Standaardafwijking over de seeds:", "", spreiding.to_markdown(), "",
        f"Keuzeregel: hoogste gemiddelde; binnen {instellingen['tolerantie']} daarvan telt als gelijk "
        "en dan wint het kleinste aantal topics.", "",
        "## Overlap van top-10 woorden tussen topics (Jaccard, lager = beter onderscheidbaar)", "",
        ov.to_markdown(), "",
        "## Beste aantal topics", "",
        f"- LDA: {beste['LDA']} topics (c_v {tabel.loc[beste['LDA'], 'LDA']:.4f}, overlap {overlaps['LDA']:.3f})",
        f"- NMF: {beste['NMF']} topics (c_v {tabel.loc[beste['NMF'], 'NMF']:.4f}, overlap {overlaps['NMF']:.3f})", "",
    ]
    for naam, kt in kruistabellen.items():
        regels += [f"## {naam}", "", kt.to_markdown(), ""]
    (out / "resultaten.md").write_text("\n".join(regels), encoding="utf-8")


# ---------------------------------------------------------------- hoofdprogramma


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LDA vs NMF topic modeling op vacatureteksten.")
    ap.add_argument("--input", type=Path, help="scraper-CSV (standaard: nieuwste in output/kpmg/)")
    ap.add_argument("--output", type=Path, default=ROOT / "output" / "topic_models")
    ap.add_argument("--text-cols", nargs="+", default=STANDAARD_TEKSTKOLOMMEN,
                    help="kolommen die samen de te analyseren tekst vormen")
    ap.add_argument("--lang-col", default="taal", help="kolom met de taal per document (NL/EN)")
    ap.add_argument("--topics", nargs="+", type=int, default=[5, 8, 10, 12, 15])
    ap.add_argument("--min-df", type=float, default=3, help="woord moet in minstens zoveel documenten staan")
    ap.add_argument("--max-df", type=float, default=0.5, help="woord mag in hooguit dit deel van de documenten staan")
    ap.add_argument("--pos", nargs="+", default=STANDAARD_POS, help="woordsoorten die blijven staan (spaCy POS)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 123, 2024],
                    help="seeds om de stabiliteit van de coherence te meten (moet %d bevatten)" % RANDOM_STATE)
    ap.add_argument("--tolerantie", type=float, default=0.005,
                    help="verschil in c_v dat als 'praktisch gelijk' telt bij het kiezen van het aantal topics")
    args = ap.parse_args(argv)
    min_df = int(args.min_df) if args.min_df >= 1 else args.min_df
    seeds = sorted(set(args.seeds) | {RANDOM_STATE})

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for luid in ("gensim", "smart_open"):
        logging.getLogger(luid).setLevel(logging.WARNING)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    # Stap 1 - verkennen
    pad = args.input or nieuwste_scraper_csv()
    log.info("Bestand: %s", pad)
    df = laad_csv(pad)
    ontbrekend = [c for c in args.text_cols if c not in df]
    if ontbrekend:
        raise SystemExit(f"Kolommen niet gevonden: {ontbrekend}")
    df["tekst_origineel"] = df[args.text_cols].apply(
        lambda r: "\n\n".join(v for v in r if v.strip()), axis=1)
    info = verken(df, args.text_cols, args.lang_col)

    # Stap 2 - preprocessing (opgeschoonde tekst bewaard ter controle)
    df = preprocess(df, args.lang_col, args.pos)
    info["rijen"] = len(df)
    info["lemmas_gem"] = float(df["tekst_lemma"].str.split().str.len().mean())
    df[["vacature_id", "titel", args.lang_col, "tekst_origineel", "tekst_schoon", "tekst_lemma"]].to_csv(
        out / "documenten_preprocessed.csv", sep=";", index=False, encoding="utf-8-sig")

    # Stap 3 - trainen
    res = train_modellen(df["tekst_lemma"].tolist(), args.topics, min_df, args.max_df, seeds)
    info["vocab"] = res["X_count"].shape[1]

    # Stap 4 - evalueren: c_v per model, aantal topics en seed
    tokens = [t.split() for t in df["tekst_lemma"]]
    dictionary = Dictionary(tokens)
    per_seed = pd.DataFrame([
        {"model": m, "topics": k, "seed": seed, "coherence_cv": coherence(t, tokens, dictionary),
         "overlap_top10": overlap(t)}
        for m, k, seed, t in res["topics_per_seed"]])
    per_seed.to_csv(out / "coherence_per_seed.csv", sep=";", index=False, encoding="utf-8-sig")
    scores = (per_seed.groupby(["model", "topics"])
              .agg(coherence_cv=("coherence_cv", "mean"), coherence_std=("coherence_cv", "std"),
                   coherence_seed42=("coherence_cv", lambda x: x[per_seed.loc[x.index, "seed"] == RANDOM_STATE].iloc[0]),
                   overlap_top10=("overlap_top10", "mean"))
              .reset_index())
    scores["coherence_std"] = scores["coherence_std"].fillna(0)
    scores.round(4).to_csv(out / "coherence_scores.csv", sep=";", index=False, encoding="utf-8-sig")
    for _, r in scores.iterrows():
        log.info("%s k=%2d: c_v gem %.4f (sd %.4f), overlap %.3f", r["model"], r["topics"], r["coherence_cv"],
                 r["coherence_std"], r["overlap_top10"])
    beste = {m: kies_aantal(scores, m, args.tolerantie) for m in ("LDA", "NMF")}
    plot_coherence(scores, beste, out / "coherence_lda_vs_nmf.png")
    log.info("Beste aantal topics: LDA=%d, NMF=%d", beste["LDA"], beste["NMF"])

    # Stap 5 - output voor de beste modellen
    lda_best, nmf_best = res["lda"][beste["LDA"]], res["nmf"][beste["NMF"]]
    namen_bestand = out / "topic_namen.json"
    lda_namen = topic_namen(lda_best["topics"], "lda", namen_bestand)
    nmf_namen = topic_namen(nmf_best["topics"], "nmf", namen_bestand)
    schrijf_topics(lda_best["topics"], nmf_best["topics"], lda_namen, nmf_namen, out)
    docs = schrijf_documenten(df, lda_best["doc_topic"], nmf_best["doc_topic"], lda_namen, nmf_namen,
                              args.lang_col, out)
    schrijf_voorbeelden(docs, lda_best["topics"], nmf_best["topics"], lda_namen, nmf_namen, out)
    pyldavis_html(lda_best["model"], res["X_count"], res["count_vec"], out / "lda_pyldavis.html")

    # Stap 6 - cijfers voor de vergelijking
    kruis = {
        "Documenten per LDA-topic, per taal": pd.crosstab(docs["lda_topic"], docs["taal"]),
        "Documenten per NMF-topic, per taal": pd.crosstab(docs["nmf_topic"], docs["taal"]),
        "LDA-topic vs NMF-topic (aantal documenten)": pd.crosstab(docs["lda_topic"], docs["nmf_topic"]),
    }
    overlaps = {"LDA": overlap(lda_best["topics"]), "NMF": overlap(nmf_best["topics"])}
    schrijf_resultaten(info, {"min_df": min_df, "max_df": args.max_df, "seeds": seeds,
                              "tolerantie": args.tolerantie}, scores, beste, overlaps, kruis, out)

    print("\nTop 10 woorden per topic")
    for model, topics, namen in (("LDA", lda_best["topics"], lda_namen), ("NMF", nmf_best["topics"], nmf_namen)):
        print(f"\n{model} ({len(topics)} topics)")
        for i, (w, n) in enumerate(zip(topics, namen)):
            print(f"  {i:2} [{n}]  {', '.join(w)}")
    print(f"\nOutput in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
