"""Brand mark: an inline SVG farmer logo.

Inline SVG (not a PNG) so the mark stays crisp at every size, inherits the
theme's CSS variables for light/dark mode, and needs no extra HTTP request or
binary asset in the repo.
"""

from __future__ import annotations

import base64

# Farmer in a wide-brim hat against a rising sun, framed by paddy stalks.
# viewBox is square so it scales cleanly from favicon to hero size.
LOGO_SVG = """
<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Tamil Nadu Agriculture Schemes Assistant">
  <defs>
    <linearGradient id="agriSky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0b3d20"/>
      <stop offset="100%" stop-color="#15803d"/>
    </linearGradient>
    <linearGradient id="agriSun" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fde047"/>
      <stop offset="100%" stop-color="#f59e0b"/>
    </linearGradient>
  </defs>

  <circle cx="60" cy="60" r="58" fill="url(#agriSky)"/>
  <circle cx="60" cy="56" r="26" fill="url(#agriSun)" opacity="0.95"/>

  <!-- paddy stalks, left and right -->
  <g stroke="#bbf7d0" stroke-width="2.4" fill="none" stroke-linecap="round">
    <path d="M22 96 Q26 74 34 60"/>
    <path d="M34 60 Q28 60 25 65"/>
    <path d="M34 60 Q40 60 43 65"/>
    <path d="M30 72 Q24 72 21 77"/>
    <path d="M30 72 Q36 72 39 77"/>
    <path d="M98 96 Q94 74 86 60"/>
    <path d="M86 60 Q92 60 95 65"/>
    <path d="M86 60 Q80 60 77 65"/>
    <path d="M90 72 Q96 72 99 77"/>
    <path d="M90 72 Q84 72 81 77"/>
  </g>

  <!-- farmer: shoulders then head, drawn over the sun -->
  <path d="M38 102 Q38 80 60 80 Q82 80 82 102 Z" fill="#0b3d20"/>
  <circle cx="60" cy="64" r="11" fill="#0b3d20"/>
  <!-- wide-brim hat -->
  <path d="M34 58 Q60 44 86 58 Q60 64 34 58 Z" fill="#14532d"/>
  <path d="M49 56 Q52 44 60 43 Q68 44 71 56 Z" fill="#14532d"/>
</svg>
"""


def logo_html(size: int = 64) -> str:
    """The mark sized for inline use (hero, empty states, avatars)."""
    svg = LOGO_SVG.replace(
        "<svg ", f'<svg width="{size}" height="{size}" ', 1
    )
    return f'<span class="agri-logo" style="line-height:0">{svg}</span>'


def favicon_data_uri() -> str:
    """Base64 data URI so st.set_page_config can use the real brand mark."""
    encoded = base64.b64encode(LOGO_SVG.strip().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
