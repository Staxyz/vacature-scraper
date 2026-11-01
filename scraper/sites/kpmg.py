"""KPMG Nederland (werkenbijkpmg.nl).

Gebruikt de generieke crawler. Site-specifieke aanpassingen komen hier zodra de
structuur van werkenbijkpmg.nl met een --debug run is bevestigd.
"""

from __future__ import annotations

from playwright.sync_api import Page

from ..crawler import CareerSiteCrawler


class KPMGCrawler(CareerSiteCrawler):
    def postprocess(self, rec: dict, page: Page) -> dict:
        rec.setdefault("organisatie", "KPMG Nederland")
        return rec
