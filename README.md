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

**Vast:** `vacature_id`, `status` (eerste_run / nieuw / bestaand), `titel`, `organisatie`, `plaats`,
`provincie_regio`, `land`, `dienstverband`, `werktijden`, `salaris_min/max/eenheid/valuta`,
`opleidingseisen`, `ervaringseisen`, `functiecategorie`, `branche`, `publicatiedatum`, `geldig_tot`,
`eerst_gezien`, `laatst_gezien`, `dagen_online`, `url`, `korte_omschrijving`, `omschrijving_volledig`.

**Dynamisch:** alles wat de site extra toont, zoals:
- `kenmerk_*`: label/waarde-paren van de pagina, bijvoorbeeld `kenmerk_ervaringsniveau` of `kenmerk_business_unit`
- `sectie_*`: tekstblokken per kop, bijvoorbeeld `sectie_wat_ga_je_doen` of `sectie_wat_bieden_wij`

**Kwaliteit:** `aantal_woorden_omschrijving`, `bron_jsonld` (ja = gestructureerde data
gevonden), `verwijderde_contactsecties`, `scrape_fout`, `gescraped_op`.

`dagen_online` is gebaseerd op de publicatiedatum van de site. Ontbreekt die, dan
wordt geteld vanaf de eerste run waarin de scraper de vacature zag.

## Privacy: wat wordt verwijderd

- Secties met contactkoppen ("Contact", "Heb je vragen?", "Jouw recruiter", …) worden overgeslagen.
- Contactvelden in de gestructureerde data (`applicationContact` e.d.) worden niet overgenomen.
- In alle tekst worden e-mailadressen, telefoonnummers, LinkedIn-profielen en namen na
  zinnen als "neem contact op met …" of "… (Recruiter)" vervangen door `[verwijderd]`.

Namen herkennen met regels is nooit 100% sluitend. Controleer de eerste output steekproefsgewijs.

## Hoe het werkt

1. **Links verzamelen:** de overzichtspagina wordt geopend, de cookiebanner geaccepteerd en
   'meer laden'/'volgende' doorgeklikt. Daarnaast wordt `sitemap.xml` gelezen als aanvulling.
2. **Vacatures uitlezen** (`scraper/extract.py`) in drie lagen:
   gestructureerde JSON-LD-data (schema.org JobPosting), label/waarde-paren en tekstsecties.
3. **Opslaan en vergelijken** (`scraper/storage.py`).

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
