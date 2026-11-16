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
import urllib.error
import urllib.request
from pathlib import Path

API = "https://ec.europa.eu/esco/api/search"
RESOURCE = "https://ec.europa.eu/esco/api/resource/occupation"
OUTPUT = Path(__file__).resolve().parent / "esco_occupations.json"
CACHE = Path(__file__).resolve().parent / ".esco_cache"   # one file per page: an interrupted run resumes
PAGE = 10  # full records are ~50 KB each (all EU languages); small pages avoid timeouts


def fetch(page: int) -> dict:
    """One page of full occupation records. Note: the ESCO API's 'offset' is a page number,
    not a record number. Pages are cached, so a rerun only fetches what is still missing."""
    cached = CACHE / f"page_{PAGE}_{page:04d}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    params = {"type": "occupation", "language": "en", "limit": PAGE, "offset": page}
    try:
        data = get(f"{API}?{urllib.parse.urlencode({**params, 'full': 'true'})}")
    except urllib.error.HTTPError as e:
        if e.code != 500:
            raise
        # Some ESCO records make the server fail (HTTP 500). Then fetch this page record by record.
        print(f"  page {page}: server error, fetching records one by one", flush=True)
        data = get(f"{API}?{urllib.parse.urlencode({**params, 'full': 'false'})}")
        full = []
        for light in data["_embedded"]["results"]:
            try:
                full.append(get(f"{RESOURCE}?{urllib.parse.urlencode({'uri': light['uri'], 'language': 'en'})}"))
            except urllib.error.HTTPError:
                print(f"    '{light['title']}' unavailable in ESCO; using name and code only", flush=True)
                full.append(light)
        data["_embedded"]["results"] = full
    CACHE.mkdir(exist_ok=True)
    cached.write_text(json.dumps(data), encoding="utf-8")
    return data


def get(url: str) -> dict:
    """GET with retries for timeouts and connection errors; a server error (HTTP 500) is raised at once."""
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 500:
                raise
            last = e
        except Exception as e:
            last = e
        print(f"    attempt {attempt + 1} failed ({type(last).__name__}), retrying", flush=True)
        time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"ESCO API not reachable ({url}); run again to resume")


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
