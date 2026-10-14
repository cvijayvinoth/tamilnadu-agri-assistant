"""Input validation and output grounding checks.

These are lightweight, deterministic guards that run before any API call, so
obviously bad input never costs a token.
"""

from __future__ import annotations

import re

MAX_QUESTION_CHARS = 1200
MIN_QUESTION_CHARS = 3

# Crude prompt-injection probes. Content retrieved from PDFs is DATA, never
# instructions - this catches the obvious attempts to flip that.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(the\s+)?(system|previous)\s+prompt",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"\bact\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(dan|jailbreak)",
]

_OFF_TOPIC_HINTS = (
    "write code", "python script", "stock price", "cricket score",
    "movie", "song lyrics", "horoscope",
)


def validate_question(text: str) -> tuple[bool, str]:
    """Return (is_ok, message). Message is user-facing when not ok."""
    if text is None:
        return False, "Please type or speak a question."

    cleaned = text.strip()
    if len(cleaned) < MIN_QUESTION_CHARS:
        return False, "That question is too short — please add a little more detail."
    if len(cleaned) > MAX_QUESTION_CHARS:
        return False, (
            f"That question is {len(cleaned)} characters long. "
            f"Please shorten it to under {MAX_QUESTION_CHARS}."
        )

    lowered = cleaned.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, (
                "That request looks like an attempt to change the assistant's "
                "instructions, so it was not sent. Please ask about a Tamil Nadu "
                "agriculture scheme instead."
            )

    return True, ""


def is_probably_off_topic(text: str) -> bool:
    """Soft signal only — used to show a hint, never to block a question."""
    lowered = text.lower()
    return any(hint in lowered for hint in _OFF_TOPIC_HINTS)


def sanitize_passage(text: str, limit: int = 4000) -> str:
    """Neutralise instruction-like lines inside retrieved passages.

    A PDF could contain "ignore previous instructions". We keep the text (it may
    be legitimate scheme wording) but defang the imperative framing.
    """
    out = text[:limit]
    for pattern in _INJECTION_PATTERNS:
        out = re.sub(pattern, "[redacted instruction-like text]", out, flags=re.I)
    return out


def cited_numbers(answer: str) -> set:
    """Every [n] citation the model produced."""
    return {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}


def grounding_report(answer: str, source_count: int) -> dict:
    """Did the answer actually cite the passages it was given?"""
    cites = cited_numbers(answer)
    invalid = {c for c in cites if c < 1 or c > source_count}
    refused = any(
        phrase in answer.lower()
        for phrase in ("could not find", "not contain", "கிடைக்கவில்லை", "no information")
    )
    return {
        "citation_count": len(cites),
        "invalid_citations": sorted(invalid),
        "is_grounded": bool(cites) and not invalid,
        "is_refusal": refused,
    }
