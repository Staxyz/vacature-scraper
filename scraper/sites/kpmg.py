"""KPMG Nederland (werkenbijkpmg.nl).

De site (Drupal) laadt de vacaturelijst client-side uit een zoekindex op
`/_lunr/vacancies`: één JSON met álle vacatures plus hun filterwaarden
(ervaringsniveau, opleiding, markt, interessegebied, locaties, taal).
Die index gebruiken we om de vacatures te vinden en om de analysevelden te vullen;
de detailpagina's leveren de rest. Werkt de index niet, dan valt de crawler terug
op de generieke aanpak (overzichtspagina + sitemap).
"""

from __future__ import annotations

import html
import json
import logging
import re

from playwright.sync_api import Page

from ..crawler import CareerSiteCrawler

log = logging.getLogger("scraper")

# Kolomnaam in de CSV  <-  veld in de zoekindex
FACETS = {
    "ervaringsniveau": "term_experience",
    "opleidingsniveau": "term_education",
    "markt": "term_markets",
    "interessegebied": "term_interest",
    "locaties": "term_location",
}
_HREF = re.compile(r'href="([^"]+)"')
_DATETIME = re.compile(r'datetime="([^"]+)"')


def _facet_values(v) -> str:
    if not isinstance(v, dict):
        return ""
    vals = [html.unescape(str(x)).strip() for x in v.values()]
    return " | ".join(x for x in vals if x)


class KPMGCrawler(CareerSiteCrawler):
    def discover(self, page: Page) -> tuple[set[str], dict[str, dict]]:
        index_url = self.base + self.cfg.get("index_pad", "/_lunr/vacancies")
        raw = self.fetch_text(page, index_url)
        try:
            items = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            items = []
        if not items:
            log.warning("Zoekindex %s leeg of onleesbaar; val terug op overzichtspagina + sitemap", index_url)
            return super().discover(page)

        links: set[str] = set()
        meta: dict[str, dict] = {}
        skipped = 0
        for item in items:
            m = _HREF.search(item.get("card") or "")
            if not m:  # bv. testitems zonder kaartje
                skipped += 1
                continue
            url = self.normalize_url(m.group(1))
            links.add(url)
            fields = {col: _facet_values(item.get(key)) for col, key in FACETS.items()}
            fields["taal"] = {"Dutch": "NL", "Engels": "EN", "English": "EN"}.get(item.get("langcode"), item.get("langcode") or "")
            created = _DATETIME.search(item.get("created") or "")
            fields["aangemaakt_op"] = created.group(1)[:10] if created else ""
            fields["index_id"] = item.get("_itemsjs_id", "")
            meta[url] = fields
        log.info("Zoekindex: %d vacatures (%d items zonder link overgeslagen)", len(links), skipped)
        if self.debug_dir:
            self._debug_write("zoekindex.json", raw or "")
        return links, meta

    def postprocess(self, rec: dict, page: Page) -> dict:
        rec.setdefault("organisatie", "KPMG")
        return rec
