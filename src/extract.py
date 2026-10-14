"""Rule-based district/category detection for ingested documents.

Deliberately NOT an LLM call per chunk: embedding a document is cheap, but
classifying every chunk with a model is not. Keyword rules handle the common
cases and cost nothing.
"""

from __future__ import annotations

import re

from config import DISTRICTS

# Scanned/searched against the title plus the opening of the document.
_HEAD_CHARS = 2500

_DISTRICT_ALIASES = {
    "Tiruchirappalli": ["trichy", "tiruchi", "tiruchirapalli"],
    "Kancheepuram": ["kanchipuram", "kanchi"],
    "Thoothukudi": ["tuticorin"],
    "The Nilgiris": ["nilgiris", "ooty", "udhagamandalam"],
}

_CATEGORY_KEYWORDS = {
    "Irrigation": ["drip", "sprinkler", "micro irrigation", "borewell",
                   "tank desilting", "canal", "water management"],
    "Insurance": ["insurance", "pmfby", "crop insurance", "premium"],
    "Seeds": ["seed", "seedling", "nursery", "germination", "certified seed"],
    "Machinery": ["machinery", "tractor", "implement", "custom hiring",
                  "harvester", "power tiller"],
    "Horticulture": ["horticulture", "fruit", "vegetable", "coconut",
                     "banana", "flower", "orchard"],
    "Organic Farming": ["organic", "biofertiliser", "biofertilizer",
                        "vermicompost", "green manure", "natural farming"],
    "Credit & Loans": ["loan", "credit", "kisan credit", "interest subvention",
                       "waiver", "bank linkage"],
    "Subsidies": ["subsidy", "subsidies", "assistance", "incentive", "grant"],
}


def detect_district(text: str, title: str = "") -> str:
    """First matching TN district name, else 'All'."""
    hay = f"{title}\n{text[:_HEAD_CHARS]}".lower()
    for d in DISTRICTS:
        if d == "All":
            continue
        if re.search(rf"\b{re.escape(d.lower())}\b", hay):
            return d
    for canon, aliases in _DISTRICT_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", hay):
                return canon
    return "All"


def detect_category(text: str, title: str = "") -> str:
    """Highest-scoring category by keyword hits, else 'All'."""
    hay = f"{title}\n{text[:_HEAD_CHARS]}".lower()
    best, best_score = "All", 0
    for cat, words in _CATEGORY_KEYWORDS.items():
        score = sum(len(re.findall(rf"\b{re.escape(w)}\b", hay)) for w in words)
        if score > best_score:
            best, best_score = cat, score
    return best if best_score else "All"


def detect_tags(docs: list, title: str = "") -> tuple[str, str]:
    """Infer (district, category) from the first few pages of a document."""
    if not docs:
        return "All", "All"
    sample = "\n".join(d.page_content for d in docs[:3])
    return detect_district(sample, title), detect_category(sample, title)
