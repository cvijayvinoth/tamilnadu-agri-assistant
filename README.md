# 🌾 Tamil Nadu Agriculture Schemes Assistant

A retrieval-augmented chatbot that answers questions about Tamil Nadu
agricultural schemes **only** from official government documents, with
page-level citations and an explicit refusal when the answer is not in the
knowledge base.

## How it works

The build path and the serve path are **deliberately separate**. Documents are
fetched and embedded once, offline; the app only ever reads the finished index.
No scraping happens at launch, so startup is instant and there is no downtime
when sources change.

```
sources.csv ──► scripts/bootstrap.py ──► data/vector_store/ ──► streamlit run app.py
                (fetch + embed once)      (FAISS + manifest)     (query only)
```

## Requirements

- Python 3.8+
- An OpenAI API key (embeddings + chat completions)
- *Optional:* [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) —
  only needed for scanned PDFs that have no text layer

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your OPENAI_API_KEY
```

### 1. Build the knowledge base (once)

```bash
python scripts/bootstrap.py              # fetch + index every source
python scripts/bootstrap.py --dry-run    # show the plan, download nothing
python scripts/bootstrap.py --force      # discard the index and rebuild
python scripts/bootstrap.py --include-raw # also index PDFs in data/raw/
```

Safe to re-run. Every document is keyed by a SHA-256 content hash, so an
interrupted run resumes instead of re-embedding work that already succeeded,
and the same file downloaded from two different URLs is only ever stored once.

### 2. Run the app

```bash
streamlit run app.py
```

If `bootstrap.py` rebuilds the index while the app is running, the app detects
the new index and reloads it on the next interaction — no restart needed.

## Optional: OCR for scanned PDFs

Most TN government PDFs have a real text layer and need no OCR. For scanned
ones, install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki), then:

```bash
python scripts/setup_ocr.py   # provisions eng + tam into data/tessdata/
```

The Tesseract binary is located via `TESSERACT_CMD` in `config.py` (override it
in `.env`), so **your system PATH is never modified** and no admin rights are
required. OCR runs *only* on pages with no extractable text, and those chunks
are tagged `extraction: "ocr"` so the UI can mark them as lower-confidence.

## Configuration

Everything tunable lives in `config.py`:

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 512 dims, to keep cost down |
| `LLM_MODEL` | `gpt-4o-mini` | temperature 0.1 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | |
| `DEFAULT_TOP_K` | 4 | passages per answer |
| `OCR_LANG` | `eng+tam` | |
| `CRAWL_MAX_PAGES` / `CRAWL_MAX_DEPTH` | 10 / 2 | hard crawl caps |

## Layout

```
app.py                  Streamlit UI (Assistant / Catalogue / Sources / KB)
config.py               All tunable settings
scripts/bootstrap.py    Offline knowledge-base builder
scripts/setup_ocr.py    One-shot OCR language provisioning
src/ingestion.py        Loading, chunking, embedding, atomic index swap
src/retrieval.py        Vector (MMR) retrieval + district/category filters
src/chain.py            Streaming LCEL chain, history-aware query rewriting
src/scraper.py          Bounded, robots-aware crawler with SSRF guards
src/taxonomy.py         SQLite vocabulary for the filter dropdowns
src/extract.py          Rule-based district/category tagging
src/guardrails.py       Input validation, injection defence, grounding checks
src/voice.py            On-device English speech-to-text
src/ui/speaker.py       Browser read-aloud (Web Speech API, no deps/cost)
```

> **Speech asymmetry:** input is **English only** (Whistle has no Tamil model),
> but output read-aloud *can* speak Tamil if a `ta-IN` voice is installed in the
> OS (Windows: Settings → Time & Language → Speech → Add voices). Synthesis runs
> in the browser tab, so answer text is never sent to any TTS service.

## Safety notes

- **Citations or refusal.** The prompt requires grounded answers; when the
  retrieved passages do not contain the answer, the model says so.
- **Scraping is bounded.** robots.txt is honoured, private/loopback addresses
  are refused (SSRF guard), responses are size-capped, and crawls are limited
  to 10 pages / depth 2 with a polite delay.
- **Dead links are detected.** A `.pdf` URL that returns an HTML error page is
  rejected rather than silently indexed as if it were scheme guidance.

## Demo

Screenshots of the running app are in [`demo/`](demo/).

| | |
|---|---|
| ![Assistant](demo/Screenshot%201.png) | ![Citations](demo/Screenshot%202.png) |
| ![Catalogue](demo/Screenshot%203.png) | ![Sources](demo/Screenshot%204.png) |

## Tests

```bash
pytest -q
```

`tests/test_core.py` covers guardrails, extraction and retrieval helpers;
`tests/test_speaker.py` covers the browser read-aloud component. The suite
makes **no network calls** and needs no API key.

## License

Released under the [MIT License](LICENSE).


