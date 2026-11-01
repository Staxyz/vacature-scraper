"""End-to-end test van de crawler tegen een lokale nep-carrièresite (geen internet nodig).

Controleert: links verzamelen, 'meer laden'-knop, cookiebanner, detailpagina's, 404's,
filterpagina's overslaan en de koppeling met Store.
"""

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("playwright")

from scraper.crawler import CareerSiteCrawler  # noqa: E402

DETAIL = """<!doctype html><html><head><script type="application/ld+json">
{{"@type":"JobPosting","title":"{title}","datePosted":"2026-10-01",
"jobLocation":{{"address":{{"addressLocality":"Amstelveen"}}}},"description":"<p>Tekst {title}</p>"}}
</script></head><body><main><h1>{title}</h1><h2>Wat ga je doen?</h2><p>Werk voor {title}</p>
<h2>Contact</h2><p>Bel Jan Jansen 06-12345678</p></main></body></html>"""

LISTING = """<!doctype html><html><body>
<div id="cookie"><button id="onetrust-accept-btn-handler" onclick="this.parentNode.remove()">Accepteren</button></div>
<a href="/vacatures/filter-pagina">Alle vacatures in Amstelveen</a>
<ul id="list">
  <li><a href="/vacatures/vac-1">1</a></li><li><a href="/vacatures/vac-2">2</a></li>
  <li><a href="/vacatures/vac-404">weg</a></li>
</ul>
<button id="more" onclick="
  document.getElementById('list').insertAdjacentHTML('beforeend',
    '<li><a href=/vacatures/vac-3>3</a></li><li><a href=/vacatures/vac-4>4</a></li>');
  this.remove();">Meer vacatures</button>
<a href="/over-ons">Over ons</a>
</body></html>"""

FILTER = "<html><body>" + "".join(f'<a href="/vacatures/vac-{i}">x</a>' for i in range(1, 8)) + "</body></html>"


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/vacatures":
            body = LISTING
        elif path == "/vacatures/filter-pagina":
            body = FILTER
        elif path.startswith("/vacatures/vac-") and not path.endswith("404"):
            body = DETAIL.format(title=path.rsplit("/", 1)[-1].upper())
        else:
            self.send_error(404)
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def site():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_crawler(site, tmp_path):
    cfg = {
        "start_url": f"{site}/vacatures",
        "vacature_link_pattern": "/vacatures?/[^/?#]+$",
        "uitsluit_pattern": "",
        "meer_laden_teksten": ["meer vacatures"],
        "max_paginas": 5,
    }
    general = {"delay_seconds": 0, "max_retries": 1, "headless": True}
    records, complete = CareerSiteCrawler("test", cfg, general, debug_dir=tmp_path / "debug").run()

    ids = sorted(r["vacature_id"] for r in records)
    assert ids == ["vac-1", "vac-2", "vac-3", "vac-4"]  # 404 en filterpagina eruit, 'meer laden' werkt
    assert complete
    r = next(r for r in records if r["vacature_id"] == "vac-3")
    assert r["titel"] == "VAC-3" and r["plaats"] == "Amstelveen"
    assert "sectie_contact" not in r and "Jansen" not in " ".join(r.values())
    assert (tmp_path / "debug" / "gevonden_links.txt").exists()
