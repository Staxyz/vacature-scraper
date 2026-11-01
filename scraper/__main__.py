"""Gebruik:  python -m scraper kpmg [--debug] [--limit 5] [--zichtbaar]"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from .sites import get_crawler_class
from .storage import Store

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scraper", description="Scrape vacatures naar CSV.")
    ap.add_argument("site", help="site uit config.toml, bv. 'kpmg'")
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    ap.add_argument("--limit", type=int, help="alleen de eerste N vacatures (om te testen; telt niet mee voor 'verdwenen')")
    ap.add_argument("--debug", action="store_true", help="bewaar HTML, screenshots en API-responses in output/<site>/debug")
    ap.add_argument("--zichtbaar", action="store_true", help="toon de browser")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("scraper")

    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    general = cfg.get("algemeen", {})
    sites = cfg.get("sites", {})
    if args.site not in sites:
        log.error("Onbekende site '%s'. Beschikbaar: %s", args.site, ", ".join(sites))
        return 2
    if args.zichtbaar:
        general["headless"] = False

    output_dir = Path(general.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    run_time = datetime.now()
    debug_dir = output_dir / args.site / "debug" / run_time.strftime("%Y-%m-%d_%H%M") if args.debug else None

    crawler = get_crawler_class(args.site)(args.site, sites[args.site], general, debug_dir)
    records, complete = crawler.run(limit=args.limit)
    if not records:
        log.error("Geen vacatures opgehaald; er is niets weggeschreven.")
        return 1

    store = Store(output_dir, args.site, general.get("csv_delimiter", ";"))
    if args.limit:
        # Testruns niet tussen de echte runs zetten, anders klopt de vergelijking niet
        store = Store(output_dir / "testruns", args.site, general.get("csv_delimiter", ";"))
    summary = store.finalize_run(records, run_time, complete=complete)

    fouten = sum(1 for r in records if r.get("scrape_fout"))
    print()
    print(f"Klaar: {summary['totaal']} vacatures  ->  {summary['csv']}")
    if summary["vorige_run"]:
        print(f"Vergeleken met: {summary['vorige_run'].name}")
        print(f"  nieuw: {summary['nieuw']}   verdwenen: {summary['verdwenen']}   ->  {summary['wijzigingen']}")
    else:
        print("Eerste run: volgende keer zie je welke vacatures nieuw zijn en welke verdwenen zijn.")
    if fouten:
        print(f"Let op: {fouten} vacature(s) konden niet worden uitgelezen (kolom 'scrape_fout').")
    if debug_dir:
        print(f"Debugbestanden: {debug_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
