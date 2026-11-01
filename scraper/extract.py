"""Haalt gestructureerde vacaturedata uit de HTML van een vacaturepagina.

Werkt in drie lagen, zodat de scraper blijft werken als de site verandert:
1. JSON-LD (schema.org JobPosting): de meest betrouwbare bron, gemaakt voor Google Jobs.
2. 'Kenmerken' in de HTML: label/waarde-paren (dl/dt/dd, tabellen, 'Label: waarde' lijstjes).
3. Tekstsecties: alle koppen (h2/h3) met de tekst eronder, bv. 'Wat ga je doen'.

Alles is pure HTML -> dict, zonder netwerk, zodat het te testen is.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from bs4 import BeautifulSoup, Tag

from . import pii

# Koppen in de pagina die nooit vacature-inhoud zijn
_SKIP_HEADINGS = re.compile(
    r"^(deel|share|cookie|vergelijkbare|gerelateerde|relevante vacatures|bekijk ook|andere vacatures"
    r"|other vacancies|related|similar|populaire|nieuwsbrief|volg ons|follow us|job alert|vacature ?alert"
    r"|blijf op de hoogte|menu|zoeken|search|ja,? ik wil solliciteren|yes,? i want to apply"
    r"|deze .{0,20} gingen jou voor|these .{0,30} before you)",
    re.IGNORECASE,
)
_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _NL.sub("\n\n", text).strip()


def html_to_text(html: str | None) -> str:
    """HTML-fragment naar leesbare tekst, met opsommingstekens voor lijstjes."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return _element_text(soup)


def _element_text(el: Tag) -> str:
    for br in el.find_all("br"):
        br.replace_with("\n")
    for li in el.find_all("li"):
        li.insert_before("\n• ")
    for block in el.find_all(["p", "div", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        block.insert_after("\n")
    return clean_text(el.get_text())


def slug(text: str) -> str:
    """'Wat ga je doen?' -> 'wat_ga_je_doen'"""
    text = text.lower().strip()
    text = re.sub(r"[àáâä]", "a", text)
    text = re.sub(r"[èéêë]", "e", text)
    text = re.sub(r"[ïíî]", "i", text)
    text = re.sub(r"[öóô]", "o", text)
    text = re.sub(r"[üúû]", "u", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:60]


# ---------------------------------------------------------------- JSON-LD


def find_job_postings(soup: BeautifulSoup) -> list[dict]:
    found: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for n in node:
                walk(n)
        elif isinstance(node, dict):
            t = node.get("@type")
            types = t if isinstance(t, list) else [t]
            if "JobPosting" in types:
                found.append(node)
            for key in ("@graph", "mainEntity", "itemListElement"):
                if key in node:
                    walk(node[key])

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        try:
            walk(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    return found


def _as_list(v: Any) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _join(values: list, sep: str = " | ") -> str:
    out = []
    for v in values:
        if isinstance(v, dict):
            v = v.get("name") or v.get("value") or v.get("@value") or ""
        v = clean_text(str(v)) if v is not None else ""
        if v and v not in out:
            out.append(v)
    return sep.join(out)


def _text_field(v: Any) -> str:
    """JSON-LD velden kunnen string, lijst of object zijn, en soms HTML bevatten."""
    if v is None:
        return ""
    if isinstance(v, dict):
        v = v.get("description") or v.get("name") or v.get("value") or ""
    if isinstance(v, list):
        return "\n".join(filter(None, (_text_field(x) for x in v)))
    v = str(v)
    return html_to_text(v) if "<" in v else clean_text(v)


def parse_job_posting(jp: dict) -> dict[str, str]:
    rec: dict[str, str] = {}
    rec["titel"] = clean_text(jp.get("title") or jp.get("name"))
    ident = jp.get("identifier")
    if isinstance(ident, dict):
        ident = ident.get("value") or ident.get("name")
    rec["vacature_id_site"] = clean_text(str(ident)) if ident else ""
    rec["publicatiedatum"] = clean_text(jp.get("datePosted"))
    rec["geldig_tot"] = clean_text(jp.get("validThrough"))
    rec["dienstverband"] = _join(_as_list(jp.get("employmentType")))
    org = jp.get("hiringOrganization")
    rec["organisatie"] = clean_text(org.get("name")) if isinstance(org, dict) else clean_text(org)

    cities, regions, postcodes, countries, streets = [], [], [], [], []
    for loc in _as_list(jp.get("jobLocation")):
        addr = loc.get("address", {}) if isinstance(loc, dict) else {}
        if isinstance(addr, str):
            cities.append(addr)
            continue
        cities.append(addr.get("addressLocality"))
        regions.append(addr.get("addressRegion"))
        postcodes.append(addr.get("postalCode"))
        streets.append(addr.get("streetAddress"))
        c = addr.get("addressCountry")
        countries.append(c.get("name") if isinstance(c, dict) else c)
    rec["plaats"] = _join([c for c in cities if c])
    rec["provincie_regio"] = _join([r for r in regions if r])
    rec["postcode"] = _join([p for p in postcodes if p])
    rec["kantoor_adres"] = _join([s for s in streets if s])
    rec["land"] = _join([c for c in countries if c])
    rec["locatietype"] = _join(_as_list(jp.get("jobLocationType")))

    sal = jp.get("baseSalary")
    if isinstance(sal, dict):
        val = sal.get("value", {})
        rec["salaris_valuta"] = clean_text(sal.get("currency"))
        if isinstance(val, dict):
            rec["salaris_min"] = str(val.get("minValue", val.get("value", "")) or "")
            rec["salaris_max"] = str(val.get("maxValue", "") or "")
            rec["salaris_eenheid"] = clean_text(val.get("unitText"))
        else:
            rec["salaris_min"] = str(val or "")

    simple = {
        "werktijden": "workHours",
        "opleidingseisen": "educationRequirements",
        "ervaringseisen": "experienceRequirements",
        "kwalificaties": "qualifications",
        "verantwoordelijkheden": "responsibilities",
        "vaardigheden": "skills",
        "arbeidsvoorwaarden": "jobBenefits",
        "branche": "industry",
        "functiecategorie": "occupationalCategory",
        "aantal_posities": "totalJobOpenings",
        "directe_start": "jobImmediateStart",
        "startdatum": "jobStartDate",
    }
    for col, key in simple.items():
        if key in jp:
            rec[col] = _text_field(jp[key])
    rec["omschrijving_volledig"] = _text_field(jp.get("description"))
    return {k: v for k, v in rec.items() if v}


# ---------------------------------------------------------------- HTML-kenmerken


_LABEL_VALUE = re.compile(r"^\s*([A-Za-zÀ-ÿ /&'-]{2,40})\s*:\s*(.{1,200})$")


def extract_facts(soup: BeautifulSoup) -> dict[str, str]:
    """Label/waarde-paren zoals 'Locatie: Amstelveen' of 'Uren: 32-40'."""
    facts: dict[str, str] = {}

    def add(label: str, value: str) -> None:
        label, value = clean_text(label).rstrip(":"), clean_text(value)
        if not label or not value or len(label) > 40 or pii.is_contact_heading(label):
            return
        key = "kenmerk_" + slug(label)
        if key != "kenmerk_" and key not in facts:
            facts[key] = value

    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                add(dt.get_text(" "), dd.get_text(" "))
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) == 2:
            add(cells[0].get_text(" "), cells[1].get_text(" "))
    for li in soup.find_all("li"):
        m = _LABEL_VALUE.match(li.get_text(" ").strip())
        if m:
            add(m.group(1), m.group(2))
    return facts


# ---------------------------------------------------------------- Tekstsecties


def _main_container(soup: BeautifulSoup) -> Tag:
    for sel in ("main", "article", "[role=main]", "#content", ".content"):
        el = soup.select_one(sel)
        if el:
            return el
    return soup.body or soup


def build_alias_map(aliases: dict[str, list[str]] | None) -> dict[str, str]:
    """{'wat_ga_je_doen': ['What will you do?']} -> {'what_will_you_do': 'wat_ga_je_doen', ...}"""
    out: dict[str, str] = {}
    for canonical, variants in (aliases or {}).items():
        c = slug(canonical)
        out[c] = c
        for v in variants:
            out[slug(v)] = c
    return out


def extract_sections(soup: BeautifulSoup, alias_map: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Geeft {sectie_<kop>: tekst} en de lijst met overgeslagen (contact)koppen."""
    alias_map = alias_map or {}
    container = _main_container(soup)
    for junk in container.find_all(["script", "style", "noscript", "nav", "footer", "form", "svg", "iframe"]):
        junk.decompose()

    sections: dict[str, str] = {}
    skipped: list[str] = []
    headings = container.find_all(["h2", "h3"])
    for h in headings:
        title = re.sub(r"\s+", " ", clean_text(h.get_text(" "))).strip()
        if not title or _SKIP_HEADINGS.search(title):
            continue
        if pii.is_contact_heading(title):
            # Alleen het soort kop bewaren, nooit de kop zelf (kan een naam bevatten)
            skipped.append("contact")
            continue
        # Verzamel alle elementen tot de volgende kop van gelijk of hoger niveau
        parts: list[str] = []
        level = int(h.name[1])
        for sib in h.find_all_next():
            if sib.name in ("h1", "h2", "h3") and int(sib.name[1]) <= level:
                break
            if sib.name in ("h2", "h3"):
                break
            # Alleen 'top-level' blokken pakken, niet ook hun kinderen nog eens
            if sib.name in ("p", "ul", "ol", "table", "blockquote") and not sib.find_parent(
                ["p", "ul", "ol", "table", "blockquote"]
            ):
                parts.append(_element_text(BeautifulSoup(str(sib), "html.parser")))
        text = clean_text("\n".join(p for p in parts if p))
        if text:
            s = slug(title)
            key = "sectie_" + alias_map.get(s, s)
            if text not in sections.get(key, ""):
                sections[key] = (sections.get(key, "") + "\n" + text).strip()
    return sections, sorted(set(skipped))


# ---------------------------------------------------------------- Afgeleide velden

_MONEY = r"€\s*(\d[\d.,]*\d|\d)(?:,-)?"
_SALARY_RANGE = re.compile(_MONEY + r"\s*(?:-|–|tot(?:\s+en\s+met)?|en|to|and)\s*" + _MONEY, re.IGNORECASE)
_SALARY_SINGLE = re.compile(r"(?:salaris|salary)[^€\n]{0,60}" + _MONEY, re.IGNORECASE)
_HOURS = re.compile(r"(\d{2})\s*(?:-|–|tot|to)\s*(\d{2})\s*(?:uur|hours|u\b)", re.IGNORECASE)
_HOURS_SINGLE = re.compile(r"(\d{2})\s*(?:uur|hours)\s*(?:per|a|in de|p\.?)\s*(?:week|wk)", re.IGNORECASE)


def _money(s: str) -> str:
    """'2.640' / '3,185' / '10.000' -> hele euro's; '12,50' / '12.50' -> decimaal."""
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", s):
        return re.sub(r"[.,]", "", s)
    if re.fullmatch(r"\d+[.,]\d{1,2}", s):
        return s.replace(",", ".")
    return re.sub(r"[.,]", "", s)


def derive_fields(rec: dict[str, str]) -> dict[str, str]:
    """Salaris en uren uit de vrije tekst halen als de gestructureerde data ze niet heeft."""
    text = "\n".join(v for k, v in rec.items() if k.startswith(("sectie_", "arbeidsvoorwaarden", "omschrijving")))
    if not rec.get("salaris_min"):
        m = _SALARY_RANGE.search(text)
        if m:
            rec["salaris_min"], rec["salaris_max"] = _money(m.group(1)), _money(m.group(2))
        elif (m := _SALARY_SINGLE.search(text)):
            rec["salaris_min"] = _money(m.group(1))
        if m:
            ctx = text[max(0, m.start() - 80): m.end() + 40].lower()
            if "maand" in ctx or "month" in ctx:
                rec["salaris_eenheid"] = "MONTH"
            elif "jaar" in ctx or "year" in ctx or "annual" in ctx:
                rec["salaris_eenheid"] = "YEAR"
            elif "uur" in ctx or "hour" in ctx:
                rec["salaris_eenheid"] = "HOUR"
            rec["salaris_valuta"] = "EUR"
            rec["salaris_bron"] = "tekst"
    elif not rec.get("salaris_bron"):
        rec["salaris_bron"] = "gestructureerd"
    m = _HOURS.search(text)
    if m and 8 <= int(m.group(1)) <= int(m.group(2)) <= 48:
        rec["uren_min"], rec["uren_max"] = m.group(1), m.group(2)
    elif (m := _HOURS_SINGLE.search(text)) and 8 <= int(m.group(1)) <= 48:
        rec["uren_min"] = rec["uren_max"] = m.group(1)
    return rec


# ---------------------------------------------------------------- Samengevoegd


def parse_vacancy_page(html: str, url: str, alias_map: dict[str, str] | None = None) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    rec: dict[str, str] = {"url": url}

    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        rec["canonical_url"] = canonical["href"]

    postings = find_job_postings(soup)
    if postings:
        jp = dict(postings[0])
        for k in pii.PERSON_KEYS:
            jp.pop(k, None)
        rec.update(parse_job_posting(jp))
        rec["bron_jsonld"] = "ja"
    else:
        rec["bron_jsonld"] = "nee"

    if not rec.get("titel"):
        h1 = soup.find("h1")
        og = soup.find("meta", property="og:title")
        rec["titel"] = clean_text(h1.get_text(" ") if h1 else (og.get("content") if og else ""))

    meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    if meta_desc and meta_desc.get("content"):
        rec["korte_omschrijving"] = clean_text(meta_desc["content"])

    for k, v in extract_facts(soup).items():
        rec.setdefault(k, v)

    sections, skipped = extract_sections(soup, alias_map)
    rec.update(sections)
    if skipped:
        rec["verwijderde_contactsecties"] = " | ".join(skipped)

    # Volledige omschrijving = alle inhoudssecties met hun kop; de (vaak korte)
    # JSON-LD-omschrijving blijft apart bewaard als 'intro'.
    if sections:
        if rec.get("omschrijving_volledig"):
            rec["intro"] = rec.pop("omschrijving_volledig")
        body = "\n\n".join(f"## {k.removeprefix('sectie_').replace('_', ' ').capitalize()}\n{v}"
                           for k, v in sections.items())
        rec["omschrijving_volledig"] = (rec.get("intro", "") + "\n\n" + body).strip()

    rec = derive_fields(pii.scrub_record(rec))
    rec["aantal_woorden_omschrijving"] = str(len((rec.get("omschrijving_volledig") or "").split()))
    return rec
