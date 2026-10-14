"""Document ingestion: PDF/URL -> text -> chunks -> FAISS vector store.

Uses the LangChain loader + splitter stack so chunks never span a page boundary
and every chunk carries source/page metadata for citations.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_MANIFEST,
    OCR_LANG,
    OCR_MIN_CHARS,
    RAW_DATA_DIR,
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
)


@dataclass
class IngestionReport:
    """What happened during one ingestion run, for display in the UI."""

    added_chunks: int = 0
    files: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.added_chunks > 0


def get_embeddings():
    """Embedding model, 512-dim to keep indexing cost low."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)


def get_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ----------------------------------------------------------------- extraction
def load_pdf(path: str | Path) -> list[Document]:
    """One Document per page, with 1-indexed page numbers for humans.

    PyMuPDF is preferred (faster, better layout handling); pypdf is the
    fallback so ingestion never dies on a single awkward file.
    """
    path = Path(path)
    pages: list[Document] = []
    try:
        from langchain_community.document_loaders import PyMuPDFLoader

        pages = PyMuPDFLoader(str(path)).load()
    except Exception:
        try:
            from langchain_community.document_loaders import PyPDFLoader

            pages = PyPDFLoader(str(path)).load()
        except Exception as exc:
            raise RuntimeError(f"Could not read PDF '{path.name}': {exc}") from exc

    for p in pages:
        p.metadata["source"] = path.name
        p.metadata["source_type"] = "pdf"
        p.metadata.setdefault("extraction", "text")
        raw_page = p.metadata.get("page", 0)
        try:
            p.metadata["page"] = int(raw_page) + 1
        except (TypeError, ValueError):
            p.metadata["page"] = 1

    # Recover pages that have no usable text layer. Pages that extracted fine
    # are never touched, so this costs nothing for the PDFs that already work.
    blank = [i for i, p in enumerate(pages)
             if len(p.page_content.strip()) < OCR_MIN_CHARS]
    if blank:
        try:
            from src import ocr

            if ocr.available():
                recovered = ocr.ocr_pdf_pages(path, blank)
                for idx, text in recovered.items():
                    if len(text.strip()) >= OCR_MIN_CHARS:
                        pages[idx].page_content = text
                        # Tag provenance: OCR text is less reliable than a real
                        # text layer, so the UI can flag it as lower-confidence.
                        pages[idx].metadata["extraction"] = "ocr"
                        pages[idx].metadata["ocr_lang"] = OCR_LANG
        except Exception:
            pass                       # OCR is best-effort, never fatal

    return pages


def ocr_page_count(pages: list[Document], min_chars: int = 40) -> int:
    """How many pages look scanned (little or no extractable text)."""
    return sum(1 for p in pages if len(p.page_content.strip()) < min_chars)


def load_url(url: str) -> list[Document]:
    """Fetch an HTML page or a remote PDF into Documents."""
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; TNAgriRAG/2.0)"}
    resp = requests.get(url, headers=headers, timeout=45)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()

    looks_like_pdf = "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")

    if looks_like_pdf:
        # Trust the BYTES, not the URL suffix or Content-Type. Government
        # servers routinely answer a dead .pdf link with 200 OK and an HTML
        # error page; without this check that "404 Not Found" boilerplate gets
        # embedded and cited as if it were official scheme guidance.
        if not resp.content.startswith(b"%PDF"):
            head = resp.content[:600].decode("utf-8", "ignore").lower()
            if "404" in head or "not found" in head or "<html" in head:
                raise RuntimeError(
                    f"That link returned a web error page, not a PDF: {url}"
                )
            raise RuntimeError(f"Downloaded file is not a valid PDF: {url}")

        name = Path(url.split("?")[0]).name or "remote.pdf"
        dest = UPLOAD_DIR / name
        dest.write_bytes(resp.content)
        docs = load_pdf(dest)
        for d in docs:
            d.metadata["url"] = url
        return docs

    try:
        import trafilatura

        text = trafilatura.extract(resp.text) or ""
    except ImportError:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)

    if not text.strip():
        raise RuntimeError("No readable text could be extracted from that page.")

    # NOTE: this return is load_url's HTML branch. It must stay attached to
    # this function - without it, scraping an HTML page silently returns None.
    title = ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    except Exception:
        pass

    return [Document(
        page_content=text,
        metadata={
            "source": title or url,
            "source_type": "url",
            "page": 1,
            "url": url,
            "title": title,
        },
    )]



# ------------------------------------------------------------------- indexing
def content_hash(data: bytes) -> str:
    """SHA256 of the raw bytes - the dedupe key for 'already ingested?'."""
    return hashlib.sha256(data).hexdigest()


def _tag(docs: list[Document], district: str, category: str, scheme: str) -> None:
    """Attach the filter metadata the sidebar actually uses at query time.

    setdefault, not assignment: auto-detected per-document tags must win over
    the broad manual selection.
    """
    for d in docs:
        d.metadata.setdefault("district", district or "All")
        d.metadata.setdefault("category", category or "All")
        if scheme:
            d.metadata.setdefault("scheme", scheme)
        d.metadata.setdefault("ingested_at", time.strftime("%Y-%m-%d %H:%M"))


def autotag(docs: list[Document], title: str = "") -> tuple[str, str]:
    """Infer district/category from the text and stamp them onto every doc."""
    from src.extract import detect_tags

    district, category = detect_tags(docs, title)
    for d in docs:
        if district != "All":
            d.metadata["district"] = district
        if category != "All":
            d.metadata["category"] = category
    return district, category


def build_or_update_index(
    docs: list[Document],
    district: str = "All",
    category: str = "All",
    scheme: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    auto_tag: bool = True,
    url: str = "",
    doc_hash: str = "",
) -> IngestionReport:
    """Split, embed and merge documents into the on-disk FAISS store."""
    from langchain_community.vectorstores import FAISS

    report = IngestionReport()
    started = time.perf_counter()

    if not docs:
        report.errors.append("Nothing to index.")
        return report

    # Infer tags from the content first; _tag then only fills what is missing.
    if auto_tag:
        try:
            title = docs[0].metadata.get("title", "") or docs[0].metadata.get("source", "")
            autotag(docs, title)
        except Exception:
            pass                      # tagging must never block ingestion

    _tag(docs, district, category, scheme)
    chunks = get_splitter(chunk_size, chunk_overlap).split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        report.errors.append("No extractable text found (scanned PDF without OCR?).")
        return report

    embeddings = get_embeddings()
    try:
        new_store = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        report.errors.append(f"Embedding failed: {exc}")
        return report

    existing = load_index()
    if existing is not None:
        existing.merge_from(new_store)
        store = existing
    else:
        store = new_store

    _atomic_save(store)

    report.added_chunks = len(chunks)
    report.files = dict(Counter(c.metadata.get("source", "unknown") for c in chunks))
    report.elapsed_s = time.perf_counter() - started
    _write_manifest(store)

    # Record each document in the taxonomy DB so the facets stay in sync.
    try:
        from src import taxonomy

        taxonomy.init_db()
        per_source = Counter(c.metadata.get("source", "unknown") for c in chunks)
        for src_name, n in per_source.items():
            meta = next(
                (c.metadata for c in chunks if c.metadata.get("source") == src_name),
                {},
            )
            taxonomy.register_document(
                src_name,
                source_type=meta.get("source_type", ""),
                url=meta.get("url", url),
                title=meta.get("title", ""),
                district=meta.get("district", district),
                category=meta.get("category", category),
                scheme=meta.get("scheme", scheme),
                content_hash=doc_hash,
                chunk_count=n,
                ingested_at=time.strftime("%Y-%m-%d %H:%M"),
            )
    except Exception:
        pass                          # bookkeeping must never fail an ingest

    return report


def _atomic_save(store) -> None:
    """Write the index to a temp dir, then swap it in with one rename.

    save_local() writes index.faiss and index.pkl in place, so a reader that
    loads midway through can see a torn/mismatched pair. Writing aside and
    renaming means readers always see either the whole old index or the whole
    new one - no downtime window.
    """
    target = Path(VECTOR_STORE_DIR)
    tmp = target.with_name(target.name + ".tmp")
    old = target.with_name(target.name + ".old")

    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    store.save_local(str(tmp))

    # Windows is far stricter than POSIX here: os.replace() onto an EXISTING
    # directory raises Access denied, and config.py recreates VECTOR_STORE_DIR
    # on every import - so a second process can resurrect the target between
    # our rmtree and our rename. Retry a few times to ride that out.
    last_exc = None
    for attempt in range(6):
        try:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            os.replace(tmp, target)            # atomic on NTFS
            shutil.rmtree(old, ignore_errors=True)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.4)

    # Last resort: copy the files in rather than leave the app with no index.
    try:
        target.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            shutil.copy2(f, target / f.name)
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
        return
    except Exception:
        pass

    raise RuntimeError(f"Could not swap the index into place: {last_exc}")


def index_version() -> str:
    """Cheap fingerprint of the index on disk, used to invalidate caches.

    Lets a running Streamlit app notice that scripts/bootstrap.py rebuilt the
    index underneath it, without needing a restart.
    """
    try:
        parts = []
        for name in ("index.faiss", "index.pkl"):
            f = Path(VECTOR_STORE_DIR) / name
            if f.exists():
                stat = f.stat()
                parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}")
        return "|".join(parts) or "empty"
    except Exception:
        return "unknown"


def load_index():
    """Load the persisted FAISS store, or None if it was never built."""
    from langchain_community.vectorstores import FAISS

    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        return None
    try:
        return FAISS.load_local(
            str(VECTOR_STORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,  # we wrote this file ourselves
        )
    except Exception:
        return None


def _write_manifest(store) -> None:
    """Sidecar JSON describing the index, so the UI can show stats cheaply."""
    try:
        docs = list(store.docstore._dict.values())
        per_file = Counter(d.metadata.get("source", "unknown") for d in docs)
        INDEX_MANIFEST.write_text(
            json.dumps({
                "chunk_count": len(docs),
                "file_count": len(per_file),
                "per_file": dict(per_file),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Stats are a nicety - never fail an ingestion because of them.
        pass


def index_stats() -> dict:
    """Read the manifest. Falls back to live inspection if it is missing."""
    if INDEX_MANIFEST.exists():
        try:
            return json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            pass

    store = load_index()
    if store is None:
        return {"chunk_count": 0, "file_count": 0, "per_file": {}}

    docs = list(store.docstore._dict.values())
    per_file = Counter(d.metadata.get("source", "unknown") for d in docs)
    return {
        "chunk_count": len(docs),
        "file_count": len(per_file),
        "per_file": dict(per_file),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def ingest_seed_pdfs() -> IngestionReport:
    """Index every PDF bundled in data/raw."""
    report = IngestionReport()
    pdfs = sorted(Path(RAW_DATA_DIR).glob("*.pdf"))
    if not pdfs:
        report.errors.append("No PDFs found in data/raw.")
        return report

    try:
        from src import taxonomy

        taxonomy.init_db()
    except Exception:
        taxonomy = None

    docs: list[Document] = []
    seen_hashes: set[str] = set()
    for pdf in pdfs:
        # Content-hash guard: the same document often sits in data/raw under a
        # hashed filename AND gets downloaded from its source URL. Without this
        # the identical text is embedded twice and crowds out top-k retrieval.
        try:
            digest = content_hash(pdf.read_bytes())
        except Exception:
            digest = ""

        if digest and (digest in seen_hashes or
                       (taxonomy and taxonomy.is_already_ingested(digest))):
            report.warnings.append(
                f"{pdf.name}: skipped - identical content is already indexed."
            )
            continue
        if digest:
            seen_hashes.add(digest)

        try:
            pages = load_pdf(pdf)
        except Exception as exc:
            report.errors.append(f"{pdf.name}: {exc}")
            continue
        scanned = ocr_page_count(pages)
        if scanned:
            report.warnings.append(
                f"{pdf.name}: {scanned} page(s) appear scanned - "
                "text extraction may be incomplete."
            )
        for p in pages:
            p.metadata["content_hash"] = digest
        docs.extend(pages)

    if not docs:
        report.errors.append("No readable text extracted from data/raw.")
        return report

    built = build_or_update_index(docs)
    built.warnings = report.warnings + built.warnings
    built.errors = report.errors + built.errors
    return built


def reset_index() -> bool:
    """Delete the vector store and its manifest. Source PDFs are untouched."""
    ok = True
    try:
        if VECTOR_STORE_DIR.exists():
            shutil.rmtree(VECTOR_STORE_DIR)
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        if INDEX_MANIFEST.exists():
            INDEX_MANIFEST.unlink()
    except Exception:
        ok = False
    return ok
