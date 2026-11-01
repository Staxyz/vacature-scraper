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
    assert r["verwijderde_contactsecties"] == "contact"


def test_afgeleide_velden_en_aliassen():
    from scraper.extract import build_alias_map

    html = """<html><body><main><h1>X</h1>
    <h2>Interesse? Let’s meet Anna Bakker</h2><p>Recruiter</p>
    <h2>This is what we offer</h2><ul><li>Brutomaandsalaris tussen €2.640,- en €4.565,-</li>
    <li>Een werkweek van 32 tot 40 uur</li></ul>
    <h2>Relevante vacatures</h2><p>Andere baan</p></main></body></html>"""
    amap = build_alias_map({"dit_bieden_wij": ["Dit bieden wij", "This is what we offer"]})
    r = parse_vacancy_page(html, "https://x.nl/en/vacancy/x", amap)
    assert "sectie_dit_bieden_wij" in r
    assert not any("anna" in k or "meet" in k or "relevante" in k for k in r)
    assert "Anna" not in " ".join(r.values())
    assert (r["salaris_min"], r["salaris_max"], r["salaris_eenheid"]) == ("2640", "4565", "MONTH")
    assert (r["uren_min"], r["uren_max"]) == ("32", "40")


def test_bedragen():
    from scraper.extract import derive_fields

    gevallen = {
        "Gross salary between €3,185 and €4,700 per month": ("3185", "4700"),
        "Brutomaandsalaris tussen € 5.085,- en € 6.230,-": ("5085", "6230"),
        "between € 5.100 and € 6,250 per month": ("5100", "6250"),
        "between € 5.000 and € 10.000 per month": ("5000", "10000"),
    }
    for tekst, verwacht in gevallen.items():
        r = derive_fields({"sectie_x": tekst})
        assert (r["salaris_min"], r["salaris_max"]) == verwacht, tekst
