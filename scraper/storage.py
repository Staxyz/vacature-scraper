"""CSV-output, geschiedenis en vergelijking met de vorige run (nieuw / verdwenen)."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

# Vaste volgorde van de belangrijkste kolommen; overige kolommen komen erachter (alfabetisch)
CORE_COLUMNS = [
    "vacature_id",
    "status",
    "titel",
    "organisatie",
    "plaats",
    "provincie_regio",
    "land",
    "dienstverband",
    "werktijden",
    "salaris_min",
    "salaris_max",
    "salaris_eenheid",
    "salaris_valuta",
    "opleidingseisen",
    "ervaringseisen",
    "functiecategorie",
    "branche",
    "publicatiedatum",
    "geldig_tot",
    "eerst_gezien",
    "laatst_gezien",
    "dagen_online",
    "url",
    "korte_omschrijving",
    "omschrijving_volledig",
]
TAIL_COLUMNS = ["aantal_woorden_omschrijving", "bron_jsonld", "verwijderde_contactsecties", "gescraped_op"]


class Store:
    def __init__(self, output_dir: str | Path, site: str, delimiter: str = ";"):
        self.dir = Path(output_dir) / site
        self.dir.mkdir(parents=True, exist_ok=True)
        self.site = site
        self.delimiter = delimiter
        self.history_path = self.dir / "geschiedenis.json"

    # ---------------------------------------------------------- geschiedenis

    def load_history(self) -> dict[str, dict]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        return {}

    def save_history(self, history: dict[str, dict]) -> None:
        self.history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------- CSV

    def previous_run(self, exclude: Path | None = None) -> Path | None:
        runs = sorted(p for p in self.dir.glob(f"{self.site}_*.csv") if "_wijzigingen" not in p.name)
        runs = [p for p in runs if p != exclude]
        return runs[-1] if runs else None

    def read_csv(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter=self.delimiter))

    def write_csv(self, path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
        if columns is None:
            columns = order_columns(rows)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, delimiter=self.delimiter, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------------------------- run afronden

    def finalize_run(self, records: list[dict], run_time: datetime, complete: bool = True) -> dict:
        """Verrijkt records met status/eerst_gezien, schrijft CSV + wijzigingen. Geeft samenvatting terug."""
        stamp = run_time.strftime("%Y-%m-%d_%H%M")
        today = run_time.date().isoformat()
        history = self.load_history()
        prev_path = self.previous_run()
        prev_rows = {r["vacature_id"]: r for r in self.read_csv(prev_path)} if prev_path else {}

        current_ids = set()
        for rec in records:
            vid = rec["vacature_id"]
            current_ids.add(vid)
            h = history.setdefault(vid, {"eerst_gezien": today})
            h["laatst_gezien"] = today
            h["titel"] = rec.get("titel", "")
            h["url"] = rec.get("url", "")
            h.pop("verdwenen_op", None)
            rec["eerst_gezien"] = h["eerst_gezien"]
            rec["laatst_gezien"] = today
            if not prev_path:
                rec["status"] = "eerste_run"
            else:
                rec["status"] = "bestaand" if vid in prev_rows else "nieuw"
            rec["dagen_online"] = str(_days_online(rec, today))
            rec["gescraped_op"] = run_time.isoformat(timespec="seconds")

        changes: list[dict] = []
        for rec in records:
            if rec["status"] == "nieuw":
                changes.append(_change_row("nieuw", rec, today))
        removed = []
        if complete:
            for vid, old in prev_rows.items():
                if vid not in current_ids:
                    h = history.get(vid, {})
                    h.setdefault("verdwenen_op", today)
                    old = {**old, "eerst_gezien": h.get("eerst_gezien", old.get("eerst_gezien", "")),
                           "laatst_gezien": h.get("laatst_gezien", old.get("laatst_gezien", ""))}
                    removed.append(old)
                    changes.append(_change_row("verdwenen", old, today))

        out_path = self.dir / f"{self.site}_{stamp}.csv"
        self.write_csv(out_path, records)
        changes_path = None
        if prev_path:
            changes_path = self.dir / f"{self.site}_{stamp}_wijzigingen.csv"
            self.write_csv(changes_path, changes, CHANGE_COLUMNS)
        self.save_history(history)
        return {
            "csv": out_path,
            "wijzigingen": changes_path,
            "vorige_run": prev_path,
            "totaal": len(records),
            "nieuw": sum(1 for r in records if r["status"] == "nieuw"),
            "verdwenen": len(removed),
        }


CHANGE_COLUMNS = ["wijziging", "vacature_id", "titel", "plaats", "dienstverband", "functiecategorie",
                  "eerst_gezien", "laatst_gezien", "dagen_online", "datum_wijziging", "url"]


def _change_row(kind: str, rec: dict, today: str) -> dict:
    row = {k: rec.get(k, "") for k in CHANGE_COLUMNS}
    row["wijziging"] = kind
    row["datum_wijziging"] = today
    if kind == "verdwenen":
        row["dagen_online"] = str(_days_between(rec.get("publicatiedatum") or rec.get("eerst_gezien"),
                                                rec.get("laatst_gezien")) or rec.get("dagen_online", ""))
    return row


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _days_between(a: str | None, b: str | None):
    da, db = _parse_date(a), _parse_date(b)
    return (db - da).days if da and db else None


def _days_online(rec: dict, today: str) -> int:
    """Op basis van publicatiedatum (site) of anders de eerste keer dat de scraper hem zag."""
    d = _days_between(rec.get("publicatiedatum"), today)
    if d is None or d < 0:
        d = _days_between(rec.get("eerst_gezien"), today) or 0
    return d


def order_columns(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    for r in rows:
        seen.update(r.keys())
    core = [c for c in CORE_COLUMNS if c in seen]
    tail = [c for c in TAIL_COLUMNS if c in seen]
    kenmerken = sorted(c for c in seen if c.startswith("kenmerk_"))
    secties = sorted(c for c in seen if c.startswith("sectie_"))
    rest = sorted(seen - set(core) - set(tail) - set(kenmerken) - set(secties))
    return core + rest + kenmerken + secties + tail
