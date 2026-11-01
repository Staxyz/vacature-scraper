from scraper.pii import REDACTED, is_contact_heading, scrub


def test_email_en_telefoon():
    t = "Mail naar jan.devries@kpmg.nl of bel 06-12345678 / +31 (0)20 656 7890 of 020 656 7890."
    out = scrub(t)
    assert "@" not in out
    assert "12345678" not in out and "656 7890" not in out


def test_naam_na_contactzin():
    out = scrub("Heb je vragen? Neem dan contact op met Sanne van der Berg via onderstaand nummer.")
    assert "Sanne" not in out and "Berg" not in out
    assert REDACTED in out


def test_naam_met_rol():
    out = scrub("Jouw aanspreekpunt: Pieter Jansen (Recruiter)")
    assert "Pieter" not in out


def test_linkedin():
    assert "linkedin" not in scrub("Zie https://www.linkedin.com/in/jan-jansen-123/").lower()


def test_gewone_tekst_blijft_intact():
    t = "Wij zoeken een collega, recruiter of niet, voor 32-40 uur per week in Amstelveen sinds 2024."
    assert scrub(t) == t


def test_contactkoppen():
    assert is_contact_heading("Contact")
    assert is_contact_heading("Heb je vragen?")
    assert is_contact_heading("Jouw recruiter")
    assert not is_contact_heading("Wat ga je doen?")
    assert not is_contact_heading("Wat bieden wij")


def test_kpmg_recruiterkop():
    assert is_contact_heading("Interesse? Let’s meet Lotte Enzler")
    assert is_contact_heading("Interested?Let’s meet Rohan Koetse")
