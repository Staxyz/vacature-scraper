"""Verwijdert persoonsgegevens (namen, e-mail, telefoon, profielen) uit vacaturedata.

Strategie, van grof naar fijn:
1. Hele secties met contactinformatie worden overgeslagen (zie is_contact_heading).
2. JSON-LD velden die naar personen verwijzen worden niet overgenomen (zie PERSON_KEYS).
3. In alle overgebleven tekst worden e-mailadressen, telefoonnummers, LinkedIn-profielen
   en namen na typische zinnen ("neem contact op met ...") vervangen door REDACTED.
"""

from __future__ import annotations

import re

REDACTED = "[verwijderd]"

# Koppen van secties die (bijna) altijd over een contactpersoon gaan
_CONTACT_HEADING = re.compile(
    r"contact|recruiter|vragen\??$|meer weten|meer informatie|jouw (aanspreekpunt|contactpersoon)"
    r"|wie (is|zijn) (je|jouw)|interesse\??$|solliciteren\??$",
    re.IGNORECASE,
)

# JSON-LD velden die personen beschrijven
PERSON_KEYS = {"applicationContact", "contactPoint", "author", "creator", "recruiter", "employee"}

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
# Nederlandse en internationale nummers: 06-12345678, 020 656 7890, +31 (0)20 656 7890, 0031...
_PHONE = re.compile(
    r"(?<!\w)(?:\+|00)?(?:31[\s.-]?(?:\(0\)[\s.-]?)?|0)[1-9](?:[\s.-]?\d){7,9}(?!\w)"
)
_LINKEDIN = re.compile(r"(https?://)?([a-z]{2,3}\.)?linkedin\.com/in/[\w%-]+/?", re.IGNORECASE)

# Naam = 1 tot 4 woorden met hoofdletter, met optionele tussenvoegsels (van, de, der, ...)
_NAME = (
    r"[A-Z][a-zà-ÿ'-]+(?:\s+(?:van|der|den|de|het|ten|ter|te|op|in|'t|el|al|da|di|du|la|le|von)\b(?=\s)"
    r"|\s+[A-Z][a-zà-ÿ'-]+){0,4}"
)
_NAME_TRIGGERS = re.compile(
    r"(?P<pre>(?:contact\s+op\s+met|contact\s+opnemen\s+met|contactpersoon(?:\s+is)?:?|recruiter(?:\s+is)?:?"
    r"|bel\s+(?:met\s+)?|mail\s+(?:naar\s+)?|vraag\s+(?:het\s+)?aan|stuur\s+(?:een\s+)?(?:bericht|mail)\s+naar"
    r"|benaderen\s+van|is\s+(?:jouw|je)\s+(?:recruiter|contactpersoon)\s*:?)\s+)"
    r"(?P<name>" + _NAME + r")",
)
# "Jan de Vries (Recruiter)" of "Jan de Vries, recruiter"
_NAME_ROLE = re.compile(
    r"(?P<name>" + _NAME + r")"
    r"(?P<post>(?:\s*[,(]|\s+[–-])\s*(?i:(?:senior\s+|junior\s+|corporate\s+)?(?:recruiter|talent acquisition)))",
)


def is_contact_heading(heading: str) -> bool:
    return bool(_CONTACT_HEADING.search(heading.strip()))


def scrub(text: str | None) -> str | None:
    """Vervang persoonsgegevens in vrije tekst."""
    if not text:
        return text
    text = _EMAIL.sub(REDACTED, text)
    text = _LINKEDIN.sub(REDACTED, text)
    text = _PHONE.sub(REDACTED, text)
    text = _NAME_TRIGGERS.sub(lambda m: m.group("pre") + REDACTED, text)
    text = _NAME_ROLE.sub(lambda m: REDACTED + m.group("post"), text)
    return text


# Technische velden die nooit persoonsgegevens bevatten maar wel op een telefoonnummer kunnen lijken
_NO_SCRUB = {"url", "canonical_url", "vacature_id", "vacature_id_site", "publicatiedatum", "geldig_tot"}


def scrub_record(record: dict) -> dict:
    """Scrub alle tekstwaarden in een vacature-record."""
    return {
        k: (scrub(v) if isinstance(v, str) and k not in _NO_SCRUB else v) for k, v in record.items()
    }
