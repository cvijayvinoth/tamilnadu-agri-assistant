"""Design system: a single injected stylesheet plus small render helpers.

Colours are driven by CSS variables so the app reads well in Streamlit's light
and dark themes without duplicating rules.
"""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "green_900": "#0b3d20",
    "green_700": "#0f6b3a",
    "green_500": "#2f9e58",
    "gold": "#c98a1f",
    "amber": "#f59e0b",
    "red": "#dc2626",
    "slate": "#64748b",
}

_CSS = """
<style>
:root {
  --agri-green-900: #0b3d20;
  --agri-green-700: #0f6b3a;
  --agri-green-500: #2f9e58;
  --agri-gold: #c98a1f;
  --agri-amber: #f59e0b;
  --agri-radius: 14px;
  --agri-border: rgba(148, 163, 184, 0.28);
}

/* ---------------------------------------------------------------- layout */
.block-container { padding-top: 1.6rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }

/* ------------------------------------------------------------------ hero */
.agri-hero {
  background: linear-gradient(120deg, var(--agri-green-900) 0%,
              var(--agri-green-700) 55%, var(--agri-green-500) 100%);
  border-radius: var(--agri-radius);
  border-bottom: 3px solid var(--agri-gold);
  padding: 1.5rem 1.8rem;
  margin-bottom: 1.1rem;
  color: #f0fdf4;
  box-shadow: 0 10px 28px rgba(11, 61, 32, 0.32);
}
.agri-hero h1 {
  margin: 0; font-size: 1.85rem; font-weight: 750;
  letter-spacing: -0.02em; color: #ffffff;
  display: flex; align-items: center; gap: .55rem;
}
.agri-hero .agri-logo svg { display: block; }
.agri-hero p { margin: .38rem 0 0; opacity: .93; font-size: .97rem; }
.agri-hero .agri-chips { margin-top: .8rem; display: flex;
  flex-wrap: wrap; gap: .4rem; }
.agri-hero .agri-chip {
  background: rgba(255,255,255,.16); border: 1px solid rgba(201,138,31,.45);
  padding: .2rem .62rem; border-radius: 999px; font-size: .76rem;
  backdrop-filter: blur(4px);
}

/* ----------------------------------------------------------------- cards */
.agri-card {
  border: 1px solid var(--agri-border); border-radius: var(--agri-radius);
  padding: .95rem 1.1rem; background: rgba(255,255,255,.035);
}
.agri-metric { text-align: center; }
.agri-metric .v {
  font-size: 1.6rem; font-weight: 750; color: var(--agri-green-500);
  line-height: 1.1;
}
.agri-metric .l {
  font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;
  opacity: .7; margin-top: .18rem;
}

/* ----------------------------------------------------------------- pills */
.agri-pill {
  display: inline-flex; align-items: center; gap: .32rem;
  padding: .16rem .6rem; border-radius: 999px;
  font-size: .74rem; font-weight: 600; margin: .12rem .2rem .12rem 0;
}
.agri-pill.ok   { background: rgba(34,197,94,.16);  color: #16a34a;
                  border: 1px solid rgba(34,197,94,.34); }
.agri-pill.warn { background: rgba(245,158,11,.16); color: #d97706;
                  border: 1px solid rgba(245,158,11,.34); }
.agri-pill.err  { background: rgba(220,38,38,.14);  color: #dc2626;
                  border: 1px solid rgba(220,38,38,.3); }
.agri-pill.info { background: rgba(100,116,139,.16); color: #64748b;
                  border: 1px solid rgba(100,116,139,.3); }

/* ------------------------------------------------------------------ chat */
[data-testid="stChatMessage"] {
  border-radius: var(--agri-radius); border: 1px solid var(--agri-border);
  padding: .85rem 1rem; margin-bottom: .55rem;
}
.agri-sources {
  font-size: .82rem; opacity: .85; margin-top: .55rem;
  padding-top: .5rem; border-top: 1px dashed var(--agri-border);
}
.agri-passage {
  border-left: 3px solid var(--agri-green-500);
  padding: .45rem .75rem; margin: .4rem 0;
  background: rgba(34,197,94,.06);
  border-radius: 0 8px 8px 0; font-size: .85rem;
}
.agri-passage .meta {
  font-weight: 700; color: var(--agri-green-700);
  font-size: .78rem; margin-bottom: .2rem;
}

/* --------------------------------------------------------------- widgets */
.stButton > button {
  border-radius: 9px; font-weight: 600; border: 1px solid var(--agri-border);
  transition: transform .08s ease, box-shadow .15s ease;
}
.stButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 4px 12px rgba(20,83,45,.18);
  border-color: var(--agri-green-500);
}
.stButton > button[kind="primary"] {
  background: var(--agri-green-700); border-color: var(--agri-green-700);
}
.stButton > button[kind="primary"]:hover { background: var(--agri-green-900); }

/* Mic widget: compact so it sits neatly beside the chat input */
[data-testid="stAudioInput"] {
  border: 1px dashed var(--agri-green-500); border-radius: var(--agri-radius);
  padding: .3rem; background: rgba(34,197,94,.05);
}

.agri-muted { opacity: .62; font-size: .8rem; }
.agri-divider {
  height: 1px; background: var(--agri-border); margin: 1rem 0 .85rem;
  border: none;
}
</style>
"""


def apply_theme() -> None:
    """Inject the stylesheet once per rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, tagline: str, chips: list[str] | None = None) -> None:
    chip_html = ""
    if chips:
        chip_html = '<div class="agri-chips">' + "".join(
            f'<span class="agri-chip">{c}</span>' for c in chips
        ) + "</div>"
    st.markdown(
        f'<div class="agri-hero"><h1>{title}</h1><p>{tagline}</p>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(value, label: str) -> None:
    st.markdown(
        f'<div class="agri-card agri-metric"><div class="v">{value}</div>'
        f'<div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "info") -> str:
    """Return pill markup so several can be composed on one line."""
    return f'<span class="agri-pill {kind}">{text}</span>'


def pills(items: list[tuple]) -> None:
    st.markdown("".join(pill(t, k) for t, k in items), unsafe_allow_html=True)


def passage(meta: str, text: str) -> None:
    safe = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="agri-passage"><div class="meta">{meta}</div>{safe}</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<hr class="agri-divider"/>', unsafe_allow_html=True)
