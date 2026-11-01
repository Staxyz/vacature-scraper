from datetime import datetime

from scraper.storage import Store


def rec(vid, titel, pub=""):
    return {"vacature_id": vid, "titel": titel, "url": f"https://x.nl/vacatures/{vid}", "publicatiedatum": pub}


def test_nieuw_en_verdwenen(tmp_path):
    s = Store(tmp_path, "test")
    r1 = s.finalize_run([rec("a", "A", "2026-10-01"), rec("b", "B")], datetime(2026, 11, 1, 9, 0))
    assert r1["wijzigingen"] is None and r1["totaal"] == 2

    r2 = s.finalize_run([rec("b", "B"), rec("c", "C")], datetime(2026, 11, 8, 9, 0))
    assert r2["nieuw"] == 1 and r2["verdwenen"] == 1
    rows = {r["vacature_id"]: r for r in s.read_csv(r2["wijzigingen"])}
    assert rows["c"]["wijziging"] == "nieuw"
    assert rows["a"]["wijziging"] == "verdwenen"
    assert rows["a"]["dagen_online"] == "31"  # 1 okt -> laatst gezien 1 nov

    hoofd = {r["vacature_id"]: r for r in s.read_csv(r2["csv"])}
    assert hoofd["b"]["status"] == "bestaand"
    assert hoofd["b"]["eerst_gezien"] == "2026-11-01"
    assert hoofd["b"]["dagen_online"] == "7"
    assert s.load_history()["a"]["verdwenen_op"] == "2026-11-08"


def test_onvolledige_run_markeert_niets_als_verdwenen(tmp_path):
    s = Store(tmp_path, "test")
    s.finalize_run([rec("a", "A"), rec("b", "B")], datetime(2026, 11, 1, 9, 0))
    r = s.finalize_run([rec("a", "A")], datetime(2026, 11, 2, 9, 0), complete=False)
    assert r["verdwenen"] == 0
