"""Dynamic district/category vocabulary, persisted in SQLite.

Design rule
-----------
The vector store is the source of truth for *which chunk has which tag*; this
database only holds the **vocabulary** (the distinct values + counts) used to
populate the dropdowns. If this file is ever lost it can be rebuilt from the
FAISS docstore via ``rebuild_from_index``.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager

from config import CATEGORIES, DISTRICTS, TAXONOMY_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS taxonomy (
    kind      TEXT NOT NULL,
    value     TEXT NOT NULL,
    slug      TEXT NOT NULL,
    doc_count INTEGER DEFAULT 0,
    is_seed   INTEGER DEFAULT 0,
    PRIMARY KEY (kind, slug)
);
CREATE TABLE IF NOT EXISTS documents (
    source       TEXT PRIMARY KEY,
    source_type  TEXT,
    url          TEXT,
    title        TEXT,
    district     TEXT,
    category     TEXT,
    scheme       TEXT,
    content_hash TEXT,
    chunk_count  INTEGER DEFAULT 0,
    ingested_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash);
"""

# Common ways the same district shows up in scraped government text.
_ALIASES = {
    "trichy": "Tiruchirappalli",
    "tiruchi": "Tiruchirappalli",
    "kanchipuram": "Kancheepuram",
    "tuticorin": "Thoothukudi",
    "nilgiris": "The Nilgiris",
    "tirunelvelli": "Tirunelveli",
}


def normalize(value: str) -> str:
    """Slug used for matching: lowercase, no punctuation, no admin suffixes."""
    if not value:
        return ""
    s = value.strip().lower()
    s = re.sub(r"\b(district|dist|taluk|block|region)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def canonical(kind: str, value: str) -> str:
    """Snap a scraped value onto a seed term when the slug matches."""
    slug = normalize(value)
    if not slug:
        return ""
    if slug in _ALIASES:
        return _ALIASES[slug]
    seeds = DISTRICTS if kind == "district" else CATEGORIES
    for seed in seeds:
        if normalize(seed) == slug:
            return seed
    return value.strip().title()


@contextmanager
def _conn():
    conn = sqlite3.connect(TAXONOMY_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and seed the known TN districts/categories once."""
    with _conn() as c:
        c.executescript(_SCHEMA)
        for kind, seeds in (("district", DISTRICTS), ("category", CATEGORIES)):
            for s in seeds:
                if s == "All":
                    continue
                c.execute(
                    "INSERT OR IGNORE INTO taxonomy (kind, value, slug, doc_count,"
                    " is_seed) VALUES (?,?,?,0,1)",
                    (kind, s, normalize(s)),
                )


def upsert_term(kind: str, value: str, delta: int = 1) -> str:
    """Record a discovered term and bump its document count."""
    if not value or value == "All":
        return ""
    canon = canonical(kind, value)
    slug = normalize(canon)
    if not slug:
        return ""
    with _conn() as c:
        c.execute(
            "INSERT INTO taxonomy (kind, value, slug, doc_count, is_seed)"
            " VALUES (?,?,?,?,0)"
            " ON CONFLICT(kind, slug) DO UPDATE SET doc_count = doc_count + ?",
            (kind, canon, slug, max(delta, 0), max(delta, 0)),
        )
    return canon


def get_terms(kind: str, only_used: bool = False) -> list[str]:
    """Dropdown values: 'All' first, seeds next, then discovered terms."""
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT value, doc_count, is_seed FROM taxonomy WHERE kind=?"
                " ORDER BY is_seed DESC, value ASC", (kind,),
            ).fetchall()
    except sqlite3.Error:
        rows = []

    values = [
        r["value"] for r in rows
        if not only_used or r["doc_count"] > 0 or r["is_seed"]
    ]
    if not values:
        values = [v for v in (DISTRICTS if kind == "district" else CATEGORIES)
                  if v != "All"]
    return ["All"] + values


def get_districts(only_used: bool = False) -> list[str]:
    return get_terms("district", only_used)


def get_categories(only_used: bool = False) -> list[str]:
    return get_terms("category", only_used)


def get_terms_with_counts(kind: str) -> list[tuple]:
    """Same vocabulary as ``get_terms`` but paired with each term's document
    count, so the UI can show e.g. 'Seeds (2)' and make an empty filter
    ("Irrigation (0)") visibly empty *before* it is picked, rather than after
    it silently returns the whole corpus unfiltered.
    """
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT value, doc_count FROM taxonomy WHERE kind=?"
                " ORDER BY is_seed DESC, value ASC", (kind,),
            ).fetchall()
    except sqlite3.Error:
        rows = []

    seeds = [v for v in (DISTRICTS if kind == "district" else CATEGORIES)
             if v != "All"]
    if not rows:
        pairs = [(v, 0) for v in seeds]
    else:
        pairs = [(r["value"], r["doc_count"]) for r in rows]

    total = sum(n for _, n in pairs)
    return [("All", total)] + pairs


# ------------------------------------------------------------- document registry
def register_document(source: str, *, source_type: str = "", url: str = "",
                      title: str = "", district: str = "All",
                      category: str = "All", scheme: str = "",
                      content_hash: str = "", chunk_count: int = 0,
                      ingested_at: str = "") -> None:
    """Record one ingested document and fold its tags into the vocabulary."""
    d = upsert_term("district", district) or "All"
    k = upsert_term("category", category) or "All"
    with _conn() as c:
        c.execute(
            "INSERT INTO documents (source, source_type, url, title, district,"
            " category, scheme, content_hash, chunk_count, ingested_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(source) DO UPDATE SET"
            " source_type=excluded.source_type, url=excluded.url,"
            " title=excluded.title, district=excluded.district,"
            " category=excluded.category, scheme=excluded.scheme,"
            " content_hash=excluded.content_hash,"
            " chunk_count=excluded.chunk_count, ingested_at=excluded.ingested_at",
            (source, source_type, url, title, d, k, scheme,
             content_hash, chunk_count, ingested_at),
        )


def is_already_ingested(content_hash: str) -> bool:
    """Content-hash dedupe - stops the same PDF being embedded twice."""
    if not content_hash:
        return False
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT 1 FROM documents WHERE content_hash=? LIMIT 1",
                (content_hash,),
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def known_urls() -> set:
    try:
        with _conn() as c:
            return {r["url"] for r in c.execute(
                "SELECT url FROM documents WHERE url != ''").fetchall()}
    except sqlite3.Error:
        return set()


def list_documents() -> list[dict]:
    try:
        with _conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()]
    except sqlite3.Error:
        return []


def rebuild_from_index(store) -> int:
    """Recreate the vocabulary by scanning the FAISS docstore.

    The vector store is the source of truth, so this is always safe to run.
    """
    if store is None:
        return 0
    init_db()
    with _conn() as c:
        c.execute("UPDATE taxonomy SET doc_count = 0")

    seen: set[tuple] = set()
    for doc in store.docstore._dict.values():
        m = doc.metadata
        key = (m.get("source", ""), m.get("district", "All"), m.get("category", "All"))
        if key in seen:
            continue
        seen.add(key)
        upsert_term("district", m.get("district", ""))
        upsert_term("category", m.get("category", ""))
    return len(seen)
