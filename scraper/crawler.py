"""Generieke Playwright-crawler voor carrièresites.

Stap 1: verzamel alle vacature-URL's (overzichtspagina + 'meer laden'/paginering + sitemap).
Stap 2: open elke vacaturepagina en laat extract.py de data eruit halen.

Site-specifieke afwijkingen horen in scraper/sites/<site>.py (subclass van CareerSiteCrawler).
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from playwright.sync_api import Page, TimeoutError as PWTimeout, sync_playwright

from .extract import build_alias_map, parse_vacancy_page

log = logging.getLogger("scraper")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36"
)
_COOKIE_BUTTONS = [
    "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "#CybotCookiebotDialogBodyButtonAccept",
    "button:has-text('Alles accepteren')",
    "button:has-text('Alle cookies accepteren')",
    "button:has-text('Accepteren')",
    "button:has-text('Akkoord')",
    "button:has-text('Accept all')",
    "button:has-text('Accept')",
]


class CareerSiteCrawler:
    def __init__(self, site_key: str, site_cfg: dict, general_cfg: dict, debug_dir: Path | None = None):
        self.site_key = site_key
        self.cfg = site_cfg
        self.general = general_cfg
        self.start_url: str = site_cfg["start_url"]
        self.base = "{0.scheme}://{0.netloc}".format(urlparse(self.start_url))
        self.link_re = re.compile(site_cfg["vacature_link_pattern"], re.IGNORECASE)
        excl = site_cfg.get("uitsluit_pattern")
        self.exclude_re = re.compile(excl, re.IGNORECASE) if excl else None
        self.delay = float(general_cfg.get("delay_seconds", 1.5))
        self.retries = int(general_cfg.get("max_retries", 3))
        self.debug_dir = debug_dir
        self.robots: RobotFileParser | None = None
        self.json_responses: list[dict] = []
        self.alias_map = build_alias_map(site_cfg.get("sectie_aliassen"))

    # ------------------------------------------------------------ hulpfuncties

    def normalize_url(self, href: str) -> str:
        u = urlparse(urljoin(self.base + "/", href))
        return urlunparse((u.scheme, u.netloc.lower(), u.path.rstrip("/"), "", "", ""))

    def is_vacancy_url(self, url: str) -> bool:
        u = urlparse(url)
        if u.netloc.lower().removeprefix("www.") != urlparse(self.base).netloc.lower().removeprefix("www."):
            return False
        if not self.link_re.search(u.path):
            return False
        return not (self.exclude_re and self.exclude_re.search(u.path))

    def vacancy_id(self, url: str) -> str:
        """Stabiel ID op basis van de URL (blijft gelijk tussen runs)."""
        return urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]

    def allowed(self, url: str) -> bool:
        return self.robots is None or self.robots.can_fetch(USER_AGENT, url)

    def pause(self) -> None:
        time.sleep(self.delay * random.uniform(0.8, 1.3))

    def _debug_write(self, name: str, content: str) -> None:
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / name).write_text(content, encoding="utf-8")

    # ------------------------------------------------------------ setup

    def fetch_text(self, page: Page, url: str) -> str | None:
        """Haal een bestand op via de browser zelf (zelfde cookies/proxy als de pagina)."""
        try:
            res = page.evaluate(
                """async (u) => { const r = await fetch(u, {credentials: 'same-origin'});
                                  return r.ok ? await r.text() : null; }""",
                url,
            )
            return res
        except Exception as e:
            log.warning("Ophalen mislukt (%s): %s", url, e)
            return None

    def load_robots(self, page: Page) -> None:
        if not self.general.get("respect_robots_txt", True):
            return
        txt = self.fetch_text(page, self.base + "/robots.txt")
        if txt is None:
            log.warning("robots.txt niet op te halen; ga verder zonder beperkingen")
            return
        rp = RobotFileParser()
        rp.parse(txt.splitlines())
        self.robots = rp
        log.info("robots.txt geladen")

    def accept_cookies(self, page: Page) -> None:
        for sel in _COOKIE_BUTTONS:
            try:
                btn = page.locator(sel).first
                if btn.is_visible(timeout=500):
                    btn.click(timeout=2000)
                    log.info("Cookiebanner geaccepteerd (%s)", sel)
                    page.wait_for_timeout(500)
                    return
            except Exception:
                continue

    def _capture_json(self, response) -> None:
        if not self.debug_dir:
            return
        try:
            if "json" in (response.headers.get("content-type") or "") and response.request.resource_type in ("xhr", "fetch"):
                body = response.text()
                self.json_responses.append({"url": response.url, "status": response.status, "voorbeeld": body[:3000]})
        except Exception:
            pass

    # ------------------------------------------------------------ stap 1: links verzamelen

    def collect_links_on_page(self, page: Page) -> set[str]:
        hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.getAttribute('href'))")
        urls = {self.normalize_url(h) for h in hrefs if h and not h.startswith(("mailto:", "tel:", "javascript:", "#"))}
        return {u for u in urls if self.is_vacancy_url(u)}

    def click_more(self, page: Page) -> bool:
        """Klik op 'meer laden' of 'volgende'. True als er geklikt is."""
        for text in self.cfg.get("meer_laden_teksten", []):
            pattern = re.compile(rf"^\s*{re.escape(text)}\b", re.IGNORECASE)
            for role in ("button", "link"):
                loc = page.get_by_role(role, name=pattern)
                try:
                    if loc.count() and loc.first.is_visible() and loc.first.is_enabled():
                        loc.first.scroll_into_view_if_needed()
                        loc.first.click(timeout=5000)
                        try:
                            page.wait_for_load_state("networkidle", timeout=15000)
                        except PWTimeout:
                            pass
                        return True
                except Exception:
                    continue
        nxt = page.locator("a[rel=next], a[aria-label*='volgende' i], a[aria-label*='next' i]").first
        try:
            if nxt.count() and nxt.is_visible():
                nxt.click(timeout=5000)
                page.wait_for_load_state("networkidle", timeout=15000)
                return True
        except Exception:
            pass
        return False

    def open_start(self, page: Page) -> None:
        page.goto(self.start_url, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except PWTimeout:
            pass
        self.accept_cookies(page)

    def links_from_listing(self, page: Page) -> set[str]:
        self._debug_write("overzicht_start.html", page.content())
        if self.debug_dir:
            page.screenshot(path=str(self.debug_dir / "overzicht_start.png"), full_page=True)

        links = self.collect_links_on_page(page)
        log.info("Overzichtspagina: %d vacature-links", len(links))
        stale_rounds = 0
        for i in range(int(self.cfg.get("max_paginas", 200))):
            # Sommige sites laden meer bij scrollen
            page.mouse.wheel(0, 20000)
            page.wait_for_timeout(800)
            clicked = self.click_more(page)
            new = self.collect_links_on_page(page) - links
            links |= new
            log.info("Ronde %d: +%d (totaal %d)", i + 1, len(new), len(links))
            if not new:
                stale_rounds += 1
            else:
                stale_rounds = 0
            if not clicked and stale_rounds >= 1 or stale_rounds >= 3:
                break
            self.pause()
        self._debug_write("overzicht_eind.html", page.content())
        return links

    def links_from_sitemap(self, page: Page) -> set[str]:
        """Aanvulling via sitemap.xml (inclusief sitemap-indexen)."""
        found: set[str] = set()
        todo = [self.base + "/sitemap.xml"]
        if self.robots and self.robots.site_maps():
            todo = list(self.robots.site_maps()) + todo
        seen: set[str] = set()
        while todo and len(seen) < 50:
            sm = todo.pop(0)
            if sm in seen:
                continue
            seen.add(sm)
            body = self.fetch_text(page, sm)
            if not body:
                continue
            locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body)
            for loc in locs:
                if loc.endswith(".xml") or "sitemap" in loc.lower():
                    todo.append(loc)
                else:
                    u = self.normalize_url(loc)
                    if self.is_vacancy_url(u):
                        found.add(u)
        log.info("Sitemap: %d vacature-links", len(found))
        return found

    def discover(self, page: Page) -> tuple[set[str], dict[str, dict]]:
        """Geeft (vacature-URL's, extra velden per URL). Sites kunnen dit vervangen."""
        links = self.links_from_listing(page)
        sitemap_links = self.links_from_sitemap(page)
        extra = sitemap_links - links
        if extra:
            log.info("Sitemap voegt %d links toe die niet in het overzicht stonden", len(extra))
        return links | sitemap_links, {}

    # ------------------------------------------------------------ stap 2: detailpagina's

    def scrape_detail(self, page: Page, url: str) -> dict:
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=45000)
                if resp and resp.status in (404, 410):
                    return {"_verlopen": True}
                wait_ms = int(self.cfg.get("detail_wacht_ms", 10000))
                if wait_ms:
                    try:
                        page.wait_for_load_state("networkidle", timeout=wait_ms)
                    except PWTimeout:
                        pass
                html = page.content()
                rec = parse_vacancy_page(html, url, self.alias_map)
                # Pagina zonder JobPosting-data met veel vacaturelinks = overzichts-/filterpagina
                if rec.get("bron_jsonld") == "nee" and len(self.collect_links_on_page(page)) > 5:
                    return {"_geen_vacature": True}
                rec = self.postprocess(rec, page)
                return rec
            except Exception as e:
                last_err = e
                log.warning("Fout bij %s (poging %d/%d): %s", url, attempt, self.retries, e)
                time.sleep(2 * attempt)
        return {"url": url, "scrape_fout": str(last_err)[:300]}

    def postprocess(self, rec: dict, page: Page) -> dict:
        """Hook voor site-specifieke aanvullingen."""
        return rec

    # ------------------------------------------------------------ alles samen

    def run(self, limit: int | None = None) -> tuple[list[dict], bool]:
        """Geeft (records, compleet). compleet=False als de lijst onbetrouwbaar is."""
        with sync_playwright() as pw:
            # Optioneel eigen Chrome/Chromium gebruiken via omgevingsvariabele CHROMIUM_PATH
            exe = os.environ.get("CHROMIUM_PATH") or None
            browser = pw.chromium.launch(headless=bool(self.general.get("headless", True)), executable_path=exe)
            ctx = browser.new_context(user_agent=USER_AGENT, locale="nl-NL", viewport={"width": 1366, "height": 900})
            page = ctx.new_page()
            page.on("response", self._capture_json)
            try:
                self.open_start(page)
                self.load_robots(page)
                if not self.allowed(self.start_url):
                    raise RuntimeError(f"robots.txt staat {self.start_url} niet toe")
                links, meta = self.discover(page)
                if self.debug_dir:
                    self._debug_write("gevonden_links.txt", "\n".join(sorted(links)))
                    self._debug_write("json_responses.json", json.dumps(self.json_responses, ensure_ascii=False, indent=2))

                if not links:
                    log.error("Geen vacature-links gevonden. Draai met --debug en controleer vacature_link_pattern in config.toml")
                    return [], False

                urls = sorted(links)
                complete = limit is None
                if limit is not None:
                    urls = urls[:limit]
                records: list[dict] = []
                for n, url in enumerate(urls, 1):
                    if not self.allowed(url):
                        log.info("Overgeslagen door robots.txt: %s", url)
                        continue
                    rec = self.scrape_detail(page, url)
                    if rec.get("_verlopen"):
                        log.info("[%d/%d] verlopen (404): %s", n, len(urls), url)
                        continue
                    if rec.get("_geen_vacature"):
                        log.info("[%d/%d] geen vacature (overzichtspagina): %s", n, len(urls), url)
                        continue
                    for k, v in meta.get(url, {}).items():
                        if v and not rec.get(k):
                            rec[k] = v
                    rec["vacature_id"] = self.vacancy_id(url)
                    rec["url"] = url
                    records.append(rec)
                    log.info("[%d/%d] %s", n, len(urls), rec.get("titel") or url)
                    if self.debug_dir and n <= 3:
                        self._debug_write(f"vacature_{n}.html", page.content())
                    self.pause()
                return records, complete
            finally:
                ctx.close()
                browser.close()
