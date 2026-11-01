from pathlib import Path

from scraper.extract import parse_vacancy_page

HTML = (Path(__file__).parent / "fixtures" / "vacature.html").read_text(encoding="utf-8")


def parse():
    return parse_vacancy_page(HTML, "https://www.example.nl/vacatures/test-auditor-12345")


def test_jsonld_velden():
    r = parse()
    assert r["titel"] == "Test Auditor"
    assert r["vacature_id_site"] == "12345"
    assert r["publicatiedatum"] == "2026-10-01"
    assert r["plaats"] == "Amstelveen | Utrecht"
    assert r["provincie_regio"] == "Noord-Holland"
    assert r["salaris_min"] == "3500" and r["salaris_max"] == "4800"
    assert r["opleidingseisen"] == "HBO/WO"
    assert r["dienstverband"] == "FULL_TIME"
    assert "• Punt een" in r["omschrijving_volledig"]


def test_html_kenmerken_en_secties():
    r = parse()
    assert r["kenmerk_ervaringsniveau"] == "Medior"
    assert r["kenmerk_business_unit"] == "Audit"
    assert r["kenmerk_uren"] == "32 - 40"
    assert "jaarrekeningen" in r["sectie_wat_ga_je_doen"]
    assert "• Klanten bezoeken" in r["sectie_wat_ga_je_doen"]
    assert "sectie_wat_vragen_wij" in r
    assert "sectie_deel_deze_vacature" not in r


def test_geen_persoonsgegevens():
    r = parse()
    alles = " ".join(r.values())
    for verboden in ("Lisa", "Groot", "Geheime", "geheim@", "lisa@", "11223344"):
        assert verboden not in alles, verboden
    assert "kenmerk_contactpersoon" not in r
    assert "sectie_heb_je_vragen" not in r
    assert r["verwijderde_contactsecties"] == "Heb je vragen?"
