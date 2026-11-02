"""Vacatures indelen met een LLM (Claude) en vergelijken met de topic models (LDA/NMF).

Drie stappen:
    python llm_categorisatie.py prepare                 # invoer + volledige prompt opbouwen
    python llm_categorisatie.py classify --run api1     # indelen via de Claude API (API-sleutel nodig)
    python llm_categorisatie.py evaluate                # output + vergelijking met LDA/NMF

De indeling kan ook buiten de API om gemaakt worden (bijvoorbeeld in een Claude-chat); zet het
resultaat dan als output/llm_categorisatie/llm_labels_<run>.json neer in dit formaat:
    [{"nr": 0, "categorie": "<id uit llm/categorieen.json>", "zekerheid": "hoog|middel|laag",
      "stage": false, "reden": "..."}, ...]
`evaluate` gebruikt run1 als hoofdresultaat en vergelijkt alle andere runs ermee (reproduceerbaarheid).

Vereist: eerst `python topic_modeling.py` (voor de opgeschoonde teksten en de LDA/NMF-uitkomsten).
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
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

import topic_modeling as tm  # zelfde coherence, overlap en instellingen als de topic models

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("llm_categorisatie")

ROOT = Path(__file__).resolve().parent
LLM_DIR = ROOT / "llm"                                   # prompt + categorieën (onder versiebeheer)
OUT = ROOT / "output" / "llm_categorisatie"              # resultaten (niet onder versiebeheer)
TOPIC_OUT = ROOT / "output" / "topic_models"
MODEL = "claude-opus-5-5"
FRAGMENT_WOORDEN = 60      # zoveel woorden van de functietekst gaan mee naar het model
BATCH = 40                 # vacatures per API-verzoek

# Prijzen per miljoen tokens (Claude API, okt 2026), voor de kostenschatting
PRIJZEN = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}


# ---------------------------------------------------------------- stap 1: voorbereiden


def laad_documenten() -> pd.DataFrame:
    """Dezelfde 182 documenten als de topic models, met hun LDA/NMF-topic en lemma's."""
    pad = TOPIC_OUT / "documenten_topics.csv"
    if not pad.exists():
        raise SystemExit("Draai eerst: python topic_modeling.py")
    docs = pd.read_csv(pad, sep=";", encoding="utf-8-sig", keep_default_na=False)
    pre = pd.read_csv(TOPIC_OUT / "documenten_preprocessed.csv", sep=";", encoding="utf-8-sig",
                      keep_default_na=False)[["vacature_id", "tekst_lemma"]]
    docs = docs.merge(pre, on="vacature_id", how="left").reset_index(drop=True)
    docs["nr"] = docs.index
    return docs


def fragment(tekst: str, n: int = FRAGMENT_WOORDEN) -> str:
    tekst = re.sub(r"\s+", " ", tekst.replace("•", " ")).strip()
    return " ".join(tekst.split()[:n])


def prepare(docs: pd.DataFrame) -> str:
    """Schrijft llm_input.jsonl en de volledige prompt (prompt_volledig.md) en geeft de prompt terug."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "llm_input.jsonl").open("w", encoding="utf-8") as f:
        for _, r in docs.iterrows():
            f.write(json.dumps({"nr": int(r.nr), "vacature_id": r.vacature_id, "titel": r.titel,
                                "fragment": fragment(r.tekst_origineel)}, ensure_ascii=False) + "\n")
    prompt = bouw_prompt(docs)
    (OUT / "prompt_volledig.md").write_text(prompt, encoding="utf-8")
    log.info("Invoer: %d vacatures, prompt %d tekens (~%d tokens)", len(docs), len(prompt), len(prompt) // 4)
    return prompt


def bouw_prompt(docs: pd.DataFrame) -> str:
    cats = json.loads((LLM_DIR / "categorieen.json").read_text(encoding="utf-8"))
    cat_tekst = "\n".join(f"- {c['id']}: {c['naam']} - {c['beschrijving']}" for c in cats)
    vac = "\n".join(f"{int(r.nr)} | {r.titel} | {fragment(r.tekst_origineel)}" for _, r in docs.iterrows())
    sjabloon = (LLM_DIR / "prompt.md").read_text(encoding="utf-8")
    return sjabloon.replace("{categorieen}", cat_tekst).replace("{vacatures}", vac)


# ---------------------------------------------------------------- stap 2: indelen via de API


def classify(docs: pd.DataFrame, run: str, model: str) -> None:
    """Deelt alle vacatures in via de Claude API, in batches, met gevalideerde JSON-output.
    Kosten worden per run gelogd in usage_<run>.json."""
    import anthropic
    from pydantic import BaseModel

    class Label(BaseModel):
        nr: int
        categorie: str
        zekerheid: str
        stage: bool
        reden: str

    class Antwoord(BaseModel):
        labels: list[Label]

    client = anthropic.Anthropic()  # leest ANTHROPIC_API_KEY of een `ant auth login`-profiel
    labels, gebruik = [], {"input_tokens": 0, "output_tokens": 0}
    for start in range(0, len(docs), BATCH):
        deel = docs.iloc[start:start + BATCH]
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            output_config={"effort": "low"},  # eenvoudige classificatie: lage effort is goedkoper en volstaat
            messages=[{"role": "user", "content": bouw_prompt(deel)}],
            output_format=Antwoord,
        )
        if response.stop_reason == "refusal":
            raise SystemExit(f"Model weigerde batch {start}: {response.stop_details}")
        labels += [lab.model_dump() for lab in response.parsed_output.labels]
        gebruik["input_tokens"] += response.usage.input_tokens
        gebruik["output_tokens"] += response.usage.output_tokens
        log.info("Batch %d-%d ingedeeld", start, start + len(deel) - 1)
    pin, pout = PRIJZEN.get(model, (np.nan, np.nan))
    gebruik["kosten_usd"] = round(gebruik["input_tokens"] / 1e6 * pin + gebruik["output_tokens"] / 1e6 * pout, 4)
    gebruik["model"] = model
    (OUT / f"llm_labels_{run}.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"usage_{run}.json").write_text(json.dumps(gebruik, indent=1), encoding="utf-8")
    log.info("Klaar: %s", gebruik)


# ---------------------------------------------------------------- stap 3: evalueren


def laad_labels(run: str, n: int) -> pd.DataFrame:
    lab = pd.DataFrame(json.loads((OUT / f"llm_labels_{run}.json").read_text(encoding="utf-8")))
    cats = {c["id"] for c in json.loads((LLM_DIR / "categorieen.json").read_text(encoding="utf-8"))}
    fout = set(lab["categorie"]) - cats
    if fout or sorted(lab["nr"]) != list(range(n)):
        raise SystemExit(f"Run {run} is ongeldig (onbekende categorieën {fout} of ontbrekende nummers)")
    return lab.sort_values("nr").reset_index(drop=True)


def topwoorden_per_categorie(docs: pd.DataFrame, kolom: str, n: int = tm.TOP_N) -> dict[str, list[str]]:
    """Kenmerkende woorden per categorie met class-based TF-IDF (zoals BERTopic):
    woorden die vaak in deze categorie staan en weinig in de andere."""
    cv = CountVectorizer(min_df=3, max_df=0.5, token_pattern=r"\S+", lowercase=False)
    X = cv.fit_transform(docs["tekst_lemma"])
    vocab = cv.get_feature_names_out()
    cats = sorted(docs[kolom].unique())
    tf = np.vstack([np.asarray(X[(docs[kolom] == c).to_numpy()].sum(axis=0)).ravel() for c in cats])
    tf_norm = tf / tf.sum(axis=1, keepdims=True)
    idf = np.log(1 + tf.sum(axis=1).mean() / tf.sum(axis=0))
    ctfidf = tf_norm * idf
    return {c: list(vocab[np.argsort(ctfidf[i])[::-1][:n]]) for i, c in enumerate(cats)}


def zuiverheid(groep: pd.Series, referentie: pd.Series) -> float:
    """Deel van de documenten dat het meest voorkomende interessegebied van hun groep heeft."""
    totaal = 0
    for g in groep.unique():
        ref = referentie[groep == g].str.split(" | ", regex=False).explode()
        top = ref.value_counts().index[0]
        totaal += referentie[groep == g].str.contains(top, regex=False).sum()
    return totaal / len(groep)


def schat_kosten(docs: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """Kostenschatting per run (tekens/4 ≈ tokens; exact tellen kan met de count_tokens-API)."""
    in_fragment = len(prompt) / 4
    vaste_deel = len((LLM_DIR / "prompt.md").read_text(encoding="utf-8")) / 4 + 400
    in_volledig = vaste_deel + docs["tekst_origineel"].str.len().sum() / 4
    uit = len(docs) * 35  # ±35 tokens JSON per vacature
    rijen = []
    for model, (pin, pout) in PRIJZEN.items():
        for variant, tin in (("titel + 60 woorden", in_fragment), ("volledige functietekst", in_volledig)):
            rijen.append({"model": model, "invoer": variant, "input_tokens": int(tin), "output_tokens": uit,
                          "kosten_usd_per_run": round(tin / 1e6 * pin + uit / 1e6 * pout, 3)})
    return pd.DataFrame(rijen)


def evaluate(docs: pd.DataFrame) -> None:
    cats = json.loads((LLM_DIR / "categorieen.json").read_text(encoding="utf-8"))
    naam = {c["id"]: c["naam"] for c in cats}
    runs = sorted(p.stem.removeprefix("llm_labels_") for p in OUT.glob("llm_labels_*.json"))
    if "run1" not in runs:
        raise SystemExit("Geen llm_labels_run1.json gevonden")
    hoofd = laad_labels("run1", len(docs))
    docs = docs.assign(llm_categorie=hoofd["categorie"], llm_naam=hoofd["categorie"].map(naam),
                       llm_zekerheid=hoofd["zekerheid"], llm_stage=hoofd["stage"], llm_reden=hoofd["reden"])

    # Reproduceerbaarheid: andere runs vergelijken met run1
    repro, verschillen = [], []
    for run in [r for r in runs if r != "run1"]:
        ander = laad_labels(run, len(docs))
        gelijk = ander["categorie"] == hoofd["categorie"]
        docs[f"llm_categorie_{run}"] = ander["categorie"]
        repro.append({"run": run, "zelfde_categorie": round(gelijk.mean(), 3),
                      "cohen_kappa": round(cohen_kappa_score(hoofd["categorie"], ander["categorie"]), 3),
                      "zelfde_stage": round((ander["stage"] == hoofd["stage"]).mean(), 3),
                      "verschillen_bij_hoge_zekerheid": int((~gelijk & (hoofd["zekerheid"] == "hoog")).sum())})
        for i in np.flatnonzero(~gelijk):
            verschillen.append({"run": run, "titel": docs.at[i, "titel"], "run1": naam[hoofd.at[i, "categorie"]],
                                run: naam[ander.at[i, "categorie"]], "zekerheid_run1": hoofd.at[i, "zekerheid"]})
    repro = pd.DataFrame(repro)

    # Kenmerkende woorden + c_v per categorie: dezelfde maat als bij LDA/NMF
    top = topwoorden_per_categorie(docs, "llm_categorie")
    tokens = [t.split() for t in docs["tekst_lemma"]]
    dictionary = Dictionary(tokens)
    llm_topics = [top[c] for c in sorted(top)]
    cv_llm = tm.coherence(llm_topics, tokens, dictionary)
    ov_llm = tm.overlap(llm_topics)

    # Vergelijking met LDA/NMF (beste modellen) en met KPMG's eigen interessegebied
    scores = pd.read_csv(TOPIC_OUT / "coherence_scores.csv", sep=";", encoding="utf-8-sig")
    k = {m: int(docs[f"{m.lower()}_topic"].nunique()) for m in ("LDA", "NMF")}
    rij = lambda m: scores[(scores.model == m) & (scores.topics == k[m])].iloc[0]
    # overlap van de eindmodellen zelf (niet het gemiddelde over seeds)
    t10 = pd.read_csv(TOPIC_OUT / "topics_top10.csv", sep=";", encoding="utf-8-sig")
    ov = {m: tm.overlap([w.split(", ") for w in t10[t10.model == m].top_10_woorden]) for m in ("LDA", "NMF")}
    eerste_ig = docs["interessegebied"].str.split(" | ", regex=False).str[0]
    vergelijking = pd.DataFrame([
        {"methode": "LDA", "groepen": k["LDA"], "coherence_cv": rij("LDA").coherence_seed42,
         "overlap_top10": ov["LDA"], "grootste_groep": docs.lda_topic.value_counts().max(),
         "zuiverheid_interessegebied": zuiverheid(docs.lda_topic, docs.interessegebied),
         "NMI_met_interessegebied": normalized_mutual_info_score(eerste_ig, docs.lda_topic)},
        {"methode": "NMF", "groepen": k["NMF"], "coherence_cv": rij("NMF").coherence_seed42,
         "overlap_top10": ov["NMF"], "grootste_groep": docs.nmf_topic.value_counts().max(),
         "zuiverheid_interessegebied": zuiverheid(docs.nmf_topic, docs.interessegebied),
         "NMI_met_interessegebied": normalized_mutual_info_score(eerste_ig, docs.nmf_topic)},
        {"methode": "LLM (Claude)", "groepen": docs.llm_categorie.nunique(), "coherence_cv": cv_llm,
         "overlap_top10": ov_llm, "grootste_groep": docs.llm_categorie.value_counts().max(),
         "zuiverheid_interessegebied": zuiverheid(docs.llm_categorie, docs.interessegebied),
         "NMI_met_interessegebied": normalized_mutual_info_score(eerste_ig, docs.llm_categorie)},
    ]).round(3)
    overeenkomst = pd.DataFrame(
        [[1.0 if a == b else round(adjusted_rand_score(docs[a], docs[b]), 3) for b in
          ("lda_topic", "nmf_topic", "llm_categorie")] for a in ("lda_topic", "nmf_topic", "llm_categorie")],
        index=["LDA", "NMF", "LLM"], columns=["LDA", "NMF", "LLM"])

    # ---- output (zelfde opbouw als output/topic_models)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"categorie": c["id"], "naam": c["naam"], "beschrijving": c["beschrijving"],
                   "documenten": int((docs.llm_categorie == c["id"]).sum()),
                   "top_10_woorden": ", ".join(top.get(c["id"], []))} for c in cats]
                 ).to_csv(OUT / "categorieen_top10.csv", sep=";", index=False, encoding="utf-8-sig")

    kolommen = ["vacature_id", "titel", "taal", "interessegebied", "llm_categorie", "llm_naam", "llm_zekerheid",
                "llm_stage", "llm_reden"] + [c for c in docs if c.startswith("llm_categorie_")] + \
               ["nmf_topic_naam", "lda_topic_naam", "tekst_origineel"]
    docs[kolommen].to_csv(OUT / "documenten_categorieen.csv", sep=";", index=False, encoding="utf-8-sig")

    regels = ["# Voorbeelddocumenten per LLM-categorie", "",
              "Per categorie 3 vacatures, bij voorkeur met hoge zekerheid.", ""]
    volgorde = {"hoog": 0, "middel": 1, "laag": 2}
    for c in cats:
        groep = docs[docs.llm_categorie == c["id"]].sort_values("llm_zekerheid", key=lambda s: s.map(volgorde))
        regels += [f"## {c['naam']} ({len(groep)} vacatures)", "", c["beschrijving"], "",
                   f"Kenmerkende woorden: {', '.join(top.get(c['id'], []))}", ""]
        for _, r in groep.head(3).iterrows():
            regels.append(f"- **{r.titel}** ({r.taal}, zekerheid {r.llm_zekerheid}; NMF: {r.nmf_topic_naam}): "
                          f"{fragment(r.tekst_origineel, 35)}…")
        regels.append("")
    (OUT / "voorbeelden_per_categorie.md").write_text("\n".join(regels), encoding="utf-8")

    plot_vergelijking(vergelijking, OUT / "coherence_lda_nmf_llm.png")
    kosten = schat_kosten(docs, bouw_prompt(docs))
    kosten.to_csv(OUT / "kosten_schatting.csv", sep=";", index=False, encoding="utf-8-sig")

    md = ["# Resultaten LLM-categorisering (automatisch gegenereerd)", "",
          f"- Documenten: {len(docs)} (dezelfde als de topic models)",
          f"- Invoer per vacature: titel + eerste {FRAGMENT_WOORDEN} woorden van de functietekst",
          f"- Categorieën: {len(cats)} (llm/categorieen.json), runs: {', '.join(runs)}", "",
          "## Verdeling (run1)", "",
          docs.groupby("llm_naam").agg(vacatures=("nr", "size"), stage=("llm_stage", "sum"),
                                        hoge_zekerheid=("llm_zekerheid", lambda s: (s == "hoog").sum()))
          .sort_values("vacatures", ascending=False).to_markdown(), "",
          f"Zekerheid: {hoofd.zekerheid.value_counts().to_dict()}; stage: {int(hoofd.stage.sum())}", "",
          "## Vergelijking LDA / NMF / LLM", "", vergelijking.to_markdown(index=False), "",
          "Coherence en overlap van de LLM zijn berekend op de 10 kenmerkendste woorden per categorie "
          "(class-based TF-IDF), met dezelfde gensim-c_v als de topic models.", "",
          "## Overeenkomst tussen de indelingen (Adjusted Rand Index; 0 = toeval, 1 = identiek)", "",
          overeenkomst.to_markdown(), "",
          "## LLM-categorie vs NMF-topic (aantal vacatures)", "",
          pd.crosstab(docs.llm_naam, docs.nmf_topic_naam).to_markdown(), ""]
    if not repro.empty:
        md += ["## Reproduceerbaarheid (zelfde prompt, onafhankelijke run)", "", repro.to_markdown(index=False), "",
               "Vacatures die anders zijn ingedeeld:", "", pd.DataFrame(verschillen).to_markdown(index=False), ""]
    md += ["## Kostenschatting per run (tekens/4 ≈ tokens)", "", kosten.to_markdown(index=False), ""]
    (OUT / "resultaten.md").write_text("\n".join(md), encoding="utf-8")

    print(vergelijking.to_string(index=False))
    print(overeenkomst)
    if not repro.empty:
        print(repro.to_string(index=False))
    print(f"\nOutput in: {OUT}")


def plot_vergelijking(v: pd.DataFrame, pad: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    kleuren = ["#2a6fdb", "#e07a1f", "#2e9e5b"]
    staven = ax.bar(v["methode"], v["coherence_cv"], color=kleuren, width=0.55)
    for s, w in zip(staven, v["coherence_cv"]):
        ax.text(s.get_x() + s.get_width() / 2, w + 0.01, f"{w:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Coherence (c_v)")
    ax.set_ylim(0, max(v["coherence_cv"]) * 1.2)
    ax.set_title("Hoe goed horen de woorden per groep bij elkaar?")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pad, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- hoofdprogramma


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vacatures indelen met een LLM en vergelijken met LDA/NMF.")
    ap.add_argument("stap", choices=["prepare", "classify", "evaluate"])
    ap.add_argument("--run", default="api1", help="naam van de run bij classify (bestand llm_labels_<run>.json)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("gensim").setLevel(logging.WARNING)
    docs = laad_documenten()
    if args.stap == "prepare":
        prepare(docs)
    elif args.stap == "classify":
        prepare(docs)
        classify(docs, args.run, args.model)
    else:
        evaluate(docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
