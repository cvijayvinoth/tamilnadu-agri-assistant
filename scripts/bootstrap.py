"""One-shot knowledge-base builder.

Reads data/sources.csv, downloads every distinct source once, and indexes it so
the Streamlit app launches instantly with a ready-to-query knowledge base.

    python scripts/bootstrap.py              # build (skips what is done)
    python scripts/bootstrap.py --dry-run    # show the plan, fetch nothing
    python scripts/bootstrap.py --force      # rebuild from scratch

Safe to re-run: content-hash dedupe means an interrupted run resumes instead of
re-embedding work that already succeeded.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when run as scripts/bootstrap.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                  # noqa: E402
from src import catalogue, ingestion, scraper, taxonomy   # noqa: E402


def collect_sources() -> list[dict]:
    """Distinct URLs from the catalogue, de-duplicated by normalized URL."""
    df = catalogue.load_catalogue()
    rows = catalogue.candidate_urls(df)

    unique: dict[str, dict] = {}
    for r in rows:
        key = scraper.normalize_url(r["url"])
        if key not in unique:          # the CSV repeats one PDF across 6 rows
            unique[key] = r
    return list(unique.values())


def ingest_one(item: dict, force: bool) -> tuple[str, str]:
    """Fetch and index a single source. Returns (status, message)."""
    url = item["url"]
    scheme = item.get("scheme", "")

    try:
        resp = scraper.fetch(url)
    except Exception as exc:
        return "error", f"fetch failed: {exc}"

    digest = ingestion.content_hash(resp.content)
    if not force and taxonomy.is_already_ingested(digest):
        return "skip", "already indexed (same content hash)"

    try:
        docs = ingestion.load_url(url)
    except Exception as exc:
        return "error", f"extract failed: {exc}"

    if not docs:
        return "error", "no text extracted"

    report = ingestion.build_or_update_index(
        docs, scheme=scheme, url=url, doc_hash=digest,
    )
    if not report.ok:
        return "error", "; ".join(report.errors) or "indexing failed"

    return "ok", f"{report.added_chunks} chunks in {report.elapsed_s:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the TN Agri knowledge base.")
    ap.add_argument("--force", action="store_true",
                    help="re-index even if the content was already ingested")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be ingested, then exit")
    ap.add_argument("--include-raw", action="store_true",
                    help="also index the PDFs sitting in data/raw")
    args = ap.parse_args()

    print("=" * 66)
    print(" TN Agriculture Schemes - knowledge base bootstrap")
    print("=" * 66)

    taxonomy.init_db()
    sources = collect_sources()
    print(f"\nCatalogue: {len(sources)} distinct source URL(s) "
          f"from {config.CATALOGUE_CSV.name}")

    if args.dry_run:
        for i, s in enumerate(sources, 1):
            kind = "PDF" if s["is_pdf"] else "HTML"
            print(f"  {i:2}. [{kind}] {s['scheme'][:48]}")
            print(f"      {s['url']}")
        print("\nDry run - nothing was downloaded or indexed.")
        return 0

    if not config.api_key_present():
        print("\nERROR: OPENAI_API_KEY is missing. Add it to .env first.")
        return 1

    if args.force:
        print("\n--force: deleting the existing index first.")
        ingestion.reset_index()

    started = time.perf_counter()
    counts = {"ok": 0, "skip": 0, "error": 0}

    for i, item in enumerate(sources, 1):
        label = (item["scheme"] or item["url"])[:52]
        print(f"\n[{i}/{len(sources)}] {label}")
        status, message = ingest_one(item, args.force)
        counts[status] += 1
        print(f"      {status.upper():5} - {message}")

    if args.include_raw:
        print("\nIndexing bundled PDFs in data/raw …")
        rep = ingestion.ingest_seed_pdfs()
        for w in rep.warnings:
            print(f"      WARN  - {w}")
        if rep.ok:
            print(f"      OK    - {rep.added_chunks} chunks")
            counts["ok"] += 1
        else:
            for e in rep.errors:
                print(f"      ERROR - {e}")
            counts["error"] += 1

    stats = ingestion.index_stats()
    print("\n" + "=" * 66)
    print(f" Indexed {counts['ok']} | skipped {counts['skip']} | "
          f"failed {counts['error']}  ({time.perf_counter() - started:.1f}s)")
    print(f" Knowledge base: {stats.get('chunk_count', 0)} chunks "
          f"from {stats.get('file_count', 0)} document(s)")
    print("=" * 66)
    print("\nNow run:  streamlit run app.py\n")

    # Non-zero exit if nothing at all is queryable, so CI/cron can detect it.
    return 0 if stats.get("chunk_count", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
