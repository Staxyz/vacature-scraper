# Vacature-scraper

Scrapet alle vacatures van **werkenbijkpmg.nl** naar CSV voor recruitment-analyse.
De scraper vergelijkt elke run met de vorige, zodat je ziet welke vacatures nieuw zijn
en welke verdwenen zijn. Persoonsgegevens worden automatisch verwijderd.

Opgezet zodat er later eenvoudig andere werkgevers bij kunnen.

## Installatie (eenmalig)

Vereist **Python 3.11 of nieuwer**.

```bash
git clone https://github.com/Staxyz/vacature-scraper.git
cd vacature-scraper
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Gebruik

```bash
python -m scraper kpmg
```

| Optie | Wat het doet |
|---|---|
| `--limit 5` | Alleen de eerste 5 vacatures (snelle test). Testruns gaan naar `output/testruns/` en tellen niet mee in de vergelijking. |
| `--debug` | Bewaart HTML, screenshots, gevonden links en API-responses in `output/kpmg/debug/`. |
| `--zichtbaar` | Laat de browser zien terwijl hij werkt. |

**Eerste keer?** Draai eerst `python -m scraper kpmg --debug --limit 5` en controleer
`output/kpmg/debug/.../gevonden_links.txt`. Staan daar de juiste vacature-URL's in?
Zo niet, pas dan `vacature_link_pattern` aan in `config.toml`.

## Output

Per run in `output/kpmg/`:

| Bestand | Inhoud |
|---|---|
| `kpmg_2026-11-01_0930.csv` | Alle vacatures van deze run, één rij per vacature |
| `kpmg_2026-11-01_0930_wijzigingen.csv` | Vacatures die `nieuw` of `verdwenen` zijn t.o.v. de vorige run |
| `geschiedenis.json` | Per vacature: eerst gezien / laatst gezien / verdwenen op (bijgehouden over alle runs) |

CSV's gebruiken `;` en UTF-8, zodat ze direct goed openen in Nederlandse Excel
(instelbaar in `config.toml`).

### Kolommen

**Classificatie (uit de zoekfilters van KPMG):** `markt` (Advisory / Audit / Staff),
`interessegebied`, `ervaringsniveau`, `opleidingsniveau`, `locaties` (alle vestigingen), `taal` (NL/EN).

**Basis:** `vacature_id`, `status` (eerste_run / nieuw / bestaand), `titel`, `organisatie`, `plaats`,
`land`, `branche`, `url`.

**Voorwaarden:** `salaris_min`, `salaris_max`, `salaris_eenheid` (MONTH/YEAR), `salaris_valuta`,
`salaris_bron` (tekst = uit de vacaturetekst gehaald), `uren_min`, `uren_max`.

**Tijd:** `publicatiedatum` (laatst gepubliceerd volgens de site), `aangemaakt_op` (aangemaakt in het
systeem van KPMG), `eerst_gezien`, `laatst_gezien`, `dagen_online`.

**Tekst:** `korte_omschrijving`, `intro`, `omschrijving_volledig` (alle secties samen), en per sectie
een eigen kolom: `sectie_jouw_uitdagingen`, `sectie_wat_ga_je_doen`, `sectie_dit_neem_jij_mee`,
`sectie_dit_bieden_wij`, `sectie_sollicitatieprocedure`. Engelstalige vacatures komen in dezelfde
kolommen (instelbaar via `sectie_aliassen` in `config.toml`). Overige label/waarde-paren
komen in `kenmerk_*`-kolommen.

**Kwaliteit:** `aantal_woorden_omschrijving`, `bron_jsonld`, `verwijderde_contactsecties`,
`scrape_fout`, `gescraped_op`.

`dagen_online` is gebaseerd op de publicatiedatum van de site. Ontbreekt die, dan
wordt geteld vanaf de eerste run waarin de scraper de vacature zag.

## Privacy: wat wordt verwijderd

- Secties met contactkoppen ("Contact", "Heb je vragen?", "Jouw recruiter", …) worden overgeslagen.
- Contactvelden in de gestructureerde data (`applicationContact` e.d.) worden niet overgenomen.
- In alle tekst worden e-mailadressen, telefoonnummers, LinkedIn-profielen en namen na
  zinnen als "neem contact op met …" of "… (Recruiter)" vervangen door `[verwijderd]`.

- Koppen als "Interesse? Let's meet [naam]" (KPMG zet daar de recruiter in) worden volledig overgeslagen.

Namen herkennen met regels is nooit 100% sluitend. Controleer de eerste output steekproefsgewijs.

## Hoe het werkt

1. **Vacatures vinden:** werkenbijkpmg.nl laadt de vacaturelijst uit een zoekindex
   (`/_lunr/vacancies`), met álle vacatures en hun filterwaarden. De scraper leest die index
   (`scraper/sites/kpmg.py`). Werkt de index niet meer, dan valt hij automatisch terug op de
   overzichtspagina ('meer laden' doorklikken) en `sitemap.xml`.
2. **Vacatures uitlezen** (`scraper/extract.py`) in drie lagen:
   gestructureerde JSON-LD-data (schema.org JobPosting), label/waarde-paren en tekstsecties.
   Salaris en uren worden uit de tekst gehaald als ze niet gestructureerd beschikbaar zijn.
3. **Opslaan en vergelijken** (`scraper/storage.py`).

Een volledige run (±185 vacatures) duurt ongeveer 10 minuten.

Netjes scrapen: `robots.txt` wordt gerespecteerd en er zit een pauze tussen pagina's
(`delay_seconds`). Pagina's die mislukken worden opnieuw geprobeerd.
Als er 0 vacatures gevonden worden, wordt er niets weggeschreven, zodat niet ten onrechte
alles als "verdwenen" wordt gemarkeerd.

## Een andere werkgever toevoegen

1. Voeg een blok `[sites.<naam>]` toe aan `config.toml`.
2. Meestal werkt de generieke crawler direct. Zo niet, maak dan `scraper/sites/<naam>.py`
   met een subclass van `CareerSiteCrawler` en registreer die in `scraper/sites/__init__.py`.

## Automatisch draaien (later)

- **Windows:** Taakplanner → actie `python -m scraper kpmg`, map: deze projectmap (met venv-python).
- **Mac/Linux:** `crontab -e` →
  `0 7 * * 1 cd /pad/naar/vacature-scraper && .venv/bin/python -m scraper kpmg`

## Tests

```bash
pytest
```
