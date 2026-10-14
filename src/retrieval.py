"""Retrieval strategies: Vector (MMR), BM25 keyword, and Hybrid ensemble.

Every strategy returns LangChain Documents carrying ``source``/``page`` metadata
so the answer layer can cite them as [1], [2], ...
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from config import DEFAULT_TOP_K, MMR_FETCH_K, MMR_LAMBDA


@dataclass
class RetrievalResult:
    docs: list
    mode: str
    note: str = ""

    @property
    def empty(self) -> bool:
        return not self.docs


def _metadata_filter(district: str, category: str):
    """Build a FAISS metadata predicate from the sidebar selections.

    Chunks ingested before filters existed are tagged "All", so they stay
    visible under every selection rather than silently disappearing.
    """
    wanted = {}
    if district and district != "All":
        wanted["district"] = district
    if category and category != "All":
        wanted["category"] = category
    if not wanted:
        return None

    def _predicate(meta: dict) -> bool:
        for key, value in wanted.items():
            actual = meta.get(key, "All")
            if actual not in (value, "All"):
                return False
        return True

    return _predicate


def vector_search(store, query: str, k: int, flt=None) -> list[Document]:
    """MMR search: pulls MMR_FETCH_K candidates then re-ranks for diversity,
    so three near-identical chunks of the same paragraph don't fill the context.
    """
    kwargs = {"k": k, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA}
    if flt is not None:
        kwargs["filter"] = flt
    try:
        return store.max_marginal_relevance_search(query, **kwargs)
    except Exception:
        # Older/newer FAISS wrappers differ on *filtered* MMR support; retry
        # plain MMR without the filter kwarg, then similarity search as a last
        # resort. Post-filter the results ourselves so a district/category
        # selection is never silently dropped - that would make the sidebar
        # lie about what it is actually searching.
        try:
            docs = store.max_marginal_relevance_search(
                query, k=MMR_FETCH_K if flt is not None else k,
                fetch_k=MMR_FETCH_K, lambda_mult=MMR_LAMBDA,
            )
        except Exception:
            docs = store.similarity_search(query, k=MMR_FETCH_K if flt is not None else k)
        if flt is not None:
            docs = [d for d in docs if flt(d.metadata)][:k]
        return docs


def bm25_search(store, query: str, k: int, flt=None) -> list[Document]:
    """Lexical search - better than embeddings for exact codes like 'G.O. 123'."""
    from langchain_community.retrievers import BM25Retriever

    docs = list(store.docstore._dict.values())
    if flt is not None:
        docs = [d for d in docs if flt(d.metadata)]
    if not docs:
        return []

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever.invoke(query)


def hybrid_search(store, query: str, k: int, flt=None) -> list[Document]:
    """Reciprocal-rank fusion of semantic + keyword hits via EnsembleRetriever."""
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    docs = list(store.docstore._dict.values())
    if flt is not None:
        docs = [d for d in docs if flt(d.metadata)]
    if not docs:
        return []

    try:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = k
        search_kwargs = {"k": k, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA}
        if flt is not None:
            search_kwargs["filter"] = flt
        dense = store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

        ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])
        return ensemble.invoke(query)[:k]
    except Exception:
        return vector_search(store, query, k, flt)


def retrieve(
    store,
    query: str,
    mode: str = "Hybrid",
    k: int = DEFAULT_TOP_K,
    district: str = "All",
    category: str = "All",
) -> RetrievalResult:
    """Single entry point used by the chat layer."""
    if store is None:
        return RetrievalResult([], mode, "The knowledge base is empty.")
    if not query or not query.strip():
        return RetrievalResult([], mode, "Empty question.")

    flt = _metadata_filter(district, category)
    note = ""

    if mode == "Vector":
        docs = vector_search(store, query, k, flt)
    elif mode == "BM25":
        docs = bm25_search(store, query, k, flt)
    else:
        docs = hybrid_search(store, query, k, flt)

    # A filter that excludes everything is worse than no filter at all.
    if not docs and flt is not None:
        docs = vector_search(store, query, k, None)
        if docs:
            note = "No match for the selected district/category filters — showing unfiltered results."

    return RetrievalResult(docs, mode, note)


def format_context(docs: list[Document]) -> str:
    """Number every passage so the model can cite it precisely."""
    return "\n\n".join(
        f"[{i}] (file: {d.metadata.get('source', 'unknown')}, "
        f"page: {d.metadata.get('page', 1)})\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


def to_sources(docs: list[Document]) -> list[dict]:
    """Flatten Documents into plain dicts for session-state storage."""
    return [
        {
            "n": i,
            "file": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", 1),
            "url": d.metadata.get("url", ""),
            "scheme": d.metadata.get("scheme", ""),
            "text": d.page_content.strip(),
        }
        for i, d in enumerate(docs, 1)
    ]


def group_sources(sources: list[dict]) -> str:
    """Compact 'file (p. 1, 4)' summary line shown under each answer."""
    by_file: dict[str, set] = {}
    for s in sources:
        by_file.setdefault(s["file"], set()).add(s["page"])
    return "; ".join(
        f"{f} (p. {', '.join(str(p) for p in sorted(pages))})"
        for f, pages in by_file.items()
    )
