"""Provision OCR language data into the project (no admin rights needed).

C:\\Program Files\\Tesseract-OCR\\tessdata is writable only by Administrators, so
instead of elevating we keep the language files in data/tessdata and point
Tesseract at them with --tessdata-dir.

Important: --tessdata-dir REPLACES the search path, it does not add to it. So
eng.traineddata must be copied in alongside tam.traineddata, otherwise English
OCR silently stops working.

    python scripts/setup_ocr.py

Idempotent: files already present are left alone.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Official tessdata_best would be more accurate but is ~4x larger; the standard
# tessdata models are the right trade-off for scanned government circulars.
_BASE_URL = "https://github.com/tesseract-ocr/tessdata/raw/main"
_LANGS = ("eng", "tam")


def system_tessdata() -> Path:
    return Path(config.TESSERACT_CMD).parent / "tessdata"


def provision(lang: str, dest_dir: Path) -> tuple[str, str]:
    """Ensure <lang>.traineddata exists in dest_dir. Returns (status, detail)."""
    dest = dest_dir / f"{lang}.traineddata"
    if dest.exists() and dest.stat().st_size > 100_000:
        return "skip", f"already present ({dest.stat().st_size:,} bytes)"

    # Prefer copying from the local install - no network, no download.
    local = system_tessdata() / f"{lang}.traineddata"
    if local.exists():
        try:
            shutil.copy2(local, dest)
            return "copy", f"copied from {local.parent} ({dest.stat().st_size:,} bytes)"
        except Exception as exc:
            return "error", f"copy failed: {exc}"

    # Otherwise download it from the official tessdata repository.
    url = f"{_BASE_URL}/{lang}.traineddata"
    try:
        import requests

        resp = requests.get(url, timeout=180, stream=True)
        resp.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(1 << 16):
                fh.write(block)
        tmp.replace(dest)                      # atomic, never a half file
        return "download", f"downloaded ({dest.stat().st_size:,} bytes)"
    except Exception as exc:
        return "error", f"download failed: {exc}"


def main() -> int:
    print("=" * 66)
    print(" OCR language setup")
    print("=" * 66)

    if not config.tesseract_available():
        print(f"\nERROR: Tesseract not found at {config.TESSERACT_CMD}")
        print("Install it, or set TESSERACT_CMD in .env to the real path.")
        return 1
    print(f"\nTesseract binary : {config.TESSERACT_CMD}")

    dest_dir = config.TESSDATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Project tessdata : {dest_dir}")

    failed = False
    for lang in _LANGS:
        status, detail = provision(lang, dest_dir)
        print(f"  {lang:4} {status.upper():9} {detail}")
        if status == "error":
            failed = True

    # Verify against Tesseract itself rather than trusting the file copy.
    print("\nVerifying with --list-langs …")
    try:
        import subprocess

        out = subprocess.run(
            [config.TESSERACT_CMD, "--list-langs", "--tessdata-dir", str(dest_dir)],
            capture_output=True, text=True, timeout=60,
        )
        found = {ln.strip() for ln in (out.stdout + out.stderr).splitlines()
                 if ln.strip() and " " not in ln.strip()}
        for lang in _LANGS:
            ok = lang in found
            print(f"  {lang:4} {'OK' if ok else 'MISSING'}")
            if not ok:
                failed = True
    except Exception as exc:
        print(f"  verification failed: {exc}")
        failed = True

    if failed:
        print("\nOCR setup incomplete - see the errors above.")
        return 1

    print(f"\nReady. OCR_LANG = {config.OCR_LANG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
