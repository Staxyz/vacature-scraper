"""Zet de nieuwste scraper-CSV om naar het formaat voor de gezamenlijke NLP-code.

Kolommen (in deze volgorde): title, location, link, description
De NLP-code voegt daar zelf 'cluster' en 'relevancy_percent' aan toe.

Formaat: komma als scheidingsteken en UTF-8, zodat pandas.read_csv() het zonder extra opties leest.

Gebruik:
    python export_voor_nlp.py                       # nieuwste run van KPMG -> output/kpmg_vacancies.csv
    python export_voor_nlp.py --input <csv> --output <csv>
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# 'description' = de inhoud van de functie: wat je gaat doen en wat er gevraagd wordt.
# De vaste standaardtekst (arbeidsvoorwaarden, sollicitatieprocedure) laten we weg: die staat in
# bijna elke vacature en zou de clusters overheersen.
DESCRIPTION_KOLOMMEN = ["sectie_jouw_uitdagingen", "sectie_wat_ga_je_doen", "sectie_dit_neem_jij_mee"]


def nieuwste_run(site: str = "kpmg") -> Path:
    runs = sorted(p for p in glob.glob(str(ROOT / "output" / site / f"{site}_*.csv")) if "wijzigingen" not in p)
    if not runs:
        raise SystemExit(f"Geen scraper-output gevonden in output/{site}/")
    return Path(runs[-1])


def opschonen(tekst: str) -> str:
    """Opsommingstekens en overbodige witruimte weg; alinea's blijven gescheiden door een spatie."""
    tekst = tekst.replace("•", " ")
    return re.sub(r"\s+", " ", tekst).strip()


def exporteer(invoer: Path, uitvoer: Path) -> pd.DataFrame:
    df = pd.read_csv(invoer, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    description = df[DESCRIPTION_KOLOMMEN].apply(lambda r: " ".join(v for v in r if v.strip()), axis=1)
    out = pd.DataFrame({
        "title": df["titel"],
        "location": df["locaties"].str.replace(" | ", ", ", regex=False),
        "link": df["url"],
        "description": description.map(opschonen),
    })
    out = out[out["description"] != ""]
    uitvoer.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(uitvoer, index=False, encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scraper-CSV omzetten naar het NLP-formaat.")
    ap.add_argument("--input", type=Path, help="scraper-CSV (standaard: nieuwste KPMG-run)")
    ap.add_argument("--output", type=Path, default=ROOT / "output" / "kpmg_vacancies.csv")
    args = ap.parse_args(argv)
    invoer = args.input or nieuwste_run()
    out = exporteer(invoer, args.output)
    print(f"{len(out)} vacatures van {invoer.name} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
