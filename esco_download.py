"""Download all ESCO occupations (European Skills, Competences, Qualifications and Occupations).

For each occupation it stores: ESCO code, ISCO-08 unit group (4 digits), English and Dutch
preferred/alternative labels and the English description.

    python esco_download.py            # -> esco_occupations.json (next to this script)

Source: the public ESCO API of the European Commission (https://ec.europa.eu/esco/api).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://ec.europa.eu/esco/api/search"
OUTPUT = Path(__file__).resolve().parent / "esco_occupations.json"
PAGE = 100


def fetch(page: int) -> dict:
    """Note: the ESCO API's 'offset' is a page number, not a record number."""
    query = urllib.parse.urlencode({"type": "occupation", "language": "en", "full": "true",
                                    "limit": PAGE, "offset": page})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=240) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ESCO API not reachable (page {page})")


def label(d: dict | None, lang: str) -> str:
    return (d or {}).get(lang, "") or ""


def main() -> int:
    first = fetch(0)
    total = first["total"]
    results = first["_embedded"]["results"]
    for page in range(1, -(-total // PAGE)):
        results += fetch(page)["_embedded"]["results"]
        print(f"{len(results)}/{total}", flush=True)
    if len(results) != total:
        raise RuntimeError(f"Expected {total} occupations, got {len(results)}")
    occupations = []
    for r in results:
        code = r.get("code", "") or ""
        occupations.append({
            "uri": r["uri"],
            "code": code,
            "isco": code.split(".")[0][:4],
            "label_en": label(r.get("preferredLabel"), "en"),
            "label_nl": label(r.get("preferredLabel"), "nl"),
            "alt_en": (r.get("alternativeLabel") or {}).get("en", []),
            "alt_nl": (r.get("alternativeLabel") or {}).get("nl", []),
            "description_en": ((r.get("description") or {}).get("en") or {}).get("literal", ""),
        })
    OUTPUT.write_text(json.dumps(occupations, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(occupations)} ESCO occupations -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
