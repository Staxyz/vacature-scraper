"""Vacatures indelen in AI / Tech / Data / Other met SBERT (zero-shot, zonder trainingsdata).

Werkwijze:
    1. Elke categorie wordt beschreven met (a) typische functietitels uit de markt (TITELS) en
       (b) een paar beschrijvende zinnen (CATEGORIEEN), in het Nederlands en Engels.
    2. Alles wordt met hetzelfde SBERT-model omgezet naar embeddings.
    3. Score per categorie = 0,6 x hoogste gelijkenis van de vacaturetitel met een voorbeeldtitel
                           + 0,4 x gemiddelde gelijkenis van de hele vacature met de zinnen.
       De titel weegt zwaarder: die zegt het duidelijkst wat voor functie het is.
       De categorie met de hoogste score wint.
    4. category_confidence_percent = hoe duidelijk de winnaar wint: de kans volgens een softmax over
       de vier gelijkenissen. Laag (< 40%) = twijfelgeval tussen categorieën.

Gebruik:
    python sbert_categorie.py                         # op output/kpmg_vacancies_clustered.csv
    python sbert_categorie.py --input output/ey_vacancies_clustered.csv

Vereist: eerst sbert_clustering.py (hergebruikt de embeddings van de vacatures).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sbert_clustering as sc

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vacatures indelen in AI / Tech / Data / Other met SBERT.")
    ap.add_argument("--input", type=Path, default=ROOT / "output" / "kpmg_vacancies_clustered.csv")
    ap.add_argument("--output", type=Path, help="standaard: <input>_ai_tech_data.csv")
    ap.add_argument("--model", default=sc.MODEL)
    ap.add_argument("--controleset", type=Path, default=ROOT / "output" / "sbert" / "controleset_ai_tech_data.csv",
                    help="CSV met title + gold (handmatige indeling) om de nauwkeurigheid te meten")
    ap.add_argument("--temperatuur", type=float, default=0.02,
                    help="scherpte van de softmax voor de zekerheid (lager = uitgesprokener)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False).reset_index(drop=True)
    cache = ROOT / "output" / "sbert" / f"{args.input.stem}_embeddings.npz"
    emb = sc.embed(df, args.model, cache)

    model = SentenceTransformer(args.model)
    namen = list(CATEGORIEEN)
    titels = model.encode(df["title"].tolist(), normalize_embeddings=True)
    titel_score = np.column_stack([
        (titels @ model.encode(TITELS[c], normalize_embeddings=True).T).max(axis=1) for c in namen])
    tekst_score = np.column_stack([
        (emb @ model.encode(CATEGORIEEN[c], normalize_embeddings=True).T).mean(axis=1) for c in namen])
    gelijkenis = TITEL_GEWICHT * titel_score + (1 - TITEL_GEWICHT) * tekst_score
    kans = softmax(gelijkenis, args.temperatuur)
    winnaar = gelijkenis.argmax(axis=1)

    df.insert(0, "job_category", [namen[i] for i in winnaar])
    df.insert(1, "category_confidence_percent", (kans.max(axis=1) * 100).round(1))
    uitvoer = args.output or args.input.with_name(args.input.stem + "_ai_tech_data.csv")
    volgorde = {"AI": 0, "Data": 1, "Tech": 2, "Other": 3}
    df.sort_values(["job_category", "category_confidence_percent"], ascending=[True, False],
                   key=lambda s: s.map(volgorde) if s.name == "job_category" else s
                   ).to_csv(uitvoer, index=False, encoding="utf-8")

    print(f"{len(df)} vacatures -> {uitvoer}\n")
    if args.controleset and args.controleset.exists():
        ctrl = pd.read_csv(args.controleset).drop_duplicates("title").set_index("title")["gold"]
        echt = df["title"].map(ctrl)
        geldig = echt.notna()
        print(f"Overeenstemming met controleset: {(echt[geldig] == df.job_category[geldig]).mean():.1%} "
              f"({int(geldig.sum())} vacatures)\n")
    for c in namen:
        g = df[df.job_category == c]
        print(f"{c:6} {len(g):3} vacatures, gem. zekerheid {g.category_confidence_percent.astype(float).mean():.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
