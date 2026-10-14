"""OCR for scanned PDF pages.

Only runs on pages that have no usable text layer. Pages that extract cleanly
are never re-processed, so adding OCR costs nothing for the PDFs that already
work.

Rendering is done with PyMuPDF (already a dependency) rather than pdf2image, so
no Poppler system install is required.
"""

from __future__ import annotations

import config

_configured = False


def available() -> bool:
    """Point pytesseract at the configured binary; report whether it is usable."""
    global _configured
    if not config.OCR_ENABLED or not config.tesseract_available():
        return False
    try:
        import pytesseract
    except ImportError:
        return False

    if not _configured:
        # Option A: configure the path here instead of touching the machine PATH.
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        _configured = True
    return True


def _tess_config() -> str:
    """--tessdata-dir REPLACES the search path, so eng must also live there."""
    if config.TESSDATA_DIR.exists():
        return f'--tessdata-dir "{config.TESSDATA_DIR}"'
    return ""


def ocr_page_text(pdf_path, page_number: int) -> str:
    """OCR a single 0-indexed page of a PDF. Returns '' on any failure."""
    if not available():
        return ""
    try:
        import io

        import pymupdf
        import pytesseract
        from PIL import Image

        with pymupdf.open(str(pdf_path)) as doc:
            if page_number < 0 or page_number >= doc.page_count:
                return ""
            pix = doc[page_number].get_pixmap(dpi=config.OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

        return pytesseract.image_to_string(
            img, lang=config.OCR_LANG, config=_tess_config(),
        ).strip()
    except Exception:
        return ""                      # OCR is best-effort, never fatal


def ocr_pdf_pages(pdf_path, page_numbers: list[int]) -> dict[int, str]:
    """OCR several pages in one document open. Returns {page_number: text}."""
    if not available() or not page_numbers:
        return {}

    pages = page_numbers[: config.OCR_MAX_PAGES]
    out: dict[int, str] = {}
    try:
        import io

        import pymupdf
        import pytesseract
        from PIL import Image

        cfg = _tess_config()
        with pymupdf.open(str(pdf_path)) as doc:
            for n in pages:
                if n < 0 or n >= doc.page_count:
                    continue
                try:
                    pix = doc[n].get_pixmap(dpi=config.OCR_DPI)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(
                        img, lang=config.OCR_LANG, config=cfg,
                    ).strip()
                    if text:
                        out[n] = text
                except Exception:
                    continue           # skip the bad page, keep the rest
    except Exception:
        return out
    return out
