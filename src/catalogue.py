"""Scheme catalogue backed by data/sources.csv.

The CSV was already curated in this repo but never surfaced in the app. This
module turns it into a browsable, filterable registry that can also hand URLs
straight to the ingestion pipeline.
"""

from __future__ import annotations

from config import CATALOGUE_CSV

EXPECTED_COLUMNS = [
    "id", "department", "scheme", "scheme_type", "what_it_covers",
    "primary_source_url", "evidence_level", "secondary_source_url",
]


def load_catalogue():
    """Read the curated CSV into a DataFrame. Returns an empty frame if absent."""
    import pandas as pd

    if not CATALOGUE_CSV.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    try:
        df = pd.read_csv(CATALOGUE_CSV)
    except Exception:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df.fillna("")


def departments(df) -> list[str]:
    if df.empty:
        return ["All"]
    return ["All"] + sorted({d for d in df["department"].astype(str) if d.strip()})


def scheme_types(df) -> list[str]:
    if df.empty:
        return ["All"]
    return ["All"] + sorted({t for t in df["scheme_type"].astype(str) if t.strip()})


def filter_catalogue(df, search: str = "", department: str = "All",
                     scheme_type: str = "All"):
    """Free-text + facet filtering over the catalogue."""
    if df.empty:
        return df

    out = df
    if department != "All":
        out = out[out["department"].astype(str) == department]
    if scheme_type != "All":
        out = out[out["scheme_type"].astype(str) == scheme_type]

    if search and search.strip():
        needle = search.strip().lower()
        mask = (
            out["scheme"].astype(str).str.lower().str.contains(needle, na=False)
            | out["what_it_covers"].astype(str).str.lower().str.contains(needle, na=False)
            | out["department"].astype(str).str.lower().str.contains(needle, na=False)
        )
        out = out[mask]
    return out


def candidate_urls(df) -> list[dict]:
    """Every distinct source URL in the catalogue, ready for one-click ingest."""
    seen: set[str] = set()
    rows: list[dict] = []
    if df.empty:
        return rows

    for _, row in df.iterrows():
        for col in ("primary_source_url", "secondary_source_url"):
            url = str(row.get(col, "")).strip()
            if not url or not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            rows.append({
                "scheme": str(row.get("scheme", "")).strip(),
                "department": str(row.get("department", "")).strip(),
                "url": url,
                "is_pdf": url.lower().split("?")[0].endswith(".pdf"),
                "evidence_level": str(row.get("evidence_level", "")).strip(),
            })
    return rows


def catalogue_stats(df) -> dict:
    if df.empty:
        return {"schemes": 0, "departments": 0, "urls": 0}
    return {
        "schemes": int(len(df)),
        "departments": int(df["department"].astype(str).nunique()),
        "urls": len(candidate_urls(df)),
    }
