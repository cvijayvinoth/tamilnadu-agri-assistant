"""Central configuration for the TN Agri Schemes assistant.

Everything tunable lives here so the UI, ingestion and retrieval layers never
hardcode a model name or a path.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Must run before the first model call so the env vars are visible.
load_dotenv(override=True)

# --------------------------------------------------------------- credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ------------------------------------------------------------------- models
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512          # kept at 512 to hold embedding cost down
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
# No hard token cap any more: the old 100-token ceiling truncated real answers
# mid-sentence. Length is controlled by the prompt instead.
LLM_MAX_TOKENS = 900

# -------------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
CATALOGUE_CSV = DATA_DIR / "sources.csv"
INDEX_MANIFEST = VECTOR_STORE_DIR / "manifest.json"

# Legacy paths kept so the original tests/store.py shim keeps working.
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
METADATA_STORE_PATH = str(DATA_DIR / "metadata.pkl")

for _d in (DATA_DIR, RAW_DATA_DIR, UPLOAD_DIR, VECTOR_STORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
MMR_FETCH_K = 20             # candidates pulled before MMR diversity re-ranking
MMR_LAMBDA = 0.5
RETRIEVAL_MODES = ("Vector",)
DEFAULT_RETRIEVAL_MODE = "Vector"

# --------------------------------------------------------------- web scraping
CRAWL_MAX_PAGES = 10
CRAWL_MAX_DEPTH = 2
CRAWL_DELAY_SECONDS = 1.0
CRAWL_TIMEOUT = 45
RESPECT_ROBOTS_TXT = True
MAX_PAGE_BYTES = 15 * 1024 * 1024
USER_AGENT = "Mozilla/5.0 (compatible; TNAgriRAG/2.0; +research)"

# ------------------------------------------------------------------ taxonomy
TAXONOMY_DB = DATA_DIR / "metadata.db"

# ------------------------------------------------------------------------ OCR
# Option A: point pytesseract at the binary from config instead of mutating the
# machine PATH. Env-overridable so teammates/CI can relocate it.
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
# Language data lives in-project: C:\Program Files\...\tessdata is writable only
# by Administrators, and we do not want to require an elevated shell.
TESSDATA_DIR = DATA_DIR / "tessdata"
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") == "1"
OCR_LANG = os.getenv("OCR_LANG", "eng+tam")
OCR_MIN_CHARS = 40           # same threshold ocr_page_count() already uses
OCR_DPI = 300
OCR_MAX_PAGES = 60           # cost/time guard on very large scanned books


def tesseract_available() -> bool:
    """True when the Tesseract binary configured above actually exists."""
    import shutil

    return bool(TESSERACT_CMD) and (
        Path(TESSERACT_CMD).exists() or shutil.which(TESSERACT_CMD) is not None
    )

# ------------------------------------------------------------------- voice
# Whistle (cactus-needle) runs fully on-device and supports English, German,
# French, Spanish, Italian, Dutch and Polish. Tamil is NOT supported, so the UI
# advertises English-only speech input rather than failing silently.
VOICE_LANGUAGE = "en"
VOICE_SAMPLE_RATE = 16000    # st.audio_input default; exactly what Whistle wants
VOICE_KEYWORDS = [
    "Tamil Nadu", "subsidy", "hectare", "drip irrigation", "micro irrigation",
    "borewell", "millet", "redgram", "oilseeds", "horticulture", "Uzhavar",
    "Government Order", "crop insurance", "PMFBY", "KAVIADP", "agriculture",
]

# --------------------------------------------------------------- read-aloud
# Output speech is synthesised by the *browser* (Web Speech API), not on the
# server and not by Whistle -- cactus-needle has no TTS. That keeps the
# "no audio leaves this machine" promise intact and adds zero dependencies.
# Note the asymmetry: speech INPUT is English-only (Whistle), but speech
# OUTPUT can read Tamil if a ta-IN voice is installed in the OS.
TTS_ENABLED = True
TTS_RATE = 0.95              # slightly under normal; scheme text is number-dense
TTS_PITCH = 1.0
TTS_LANG_PREFS = ["en-IN", "en-US", "ta-IN"]
TTS_MAX_CHARS = 3000         # guard against a runaway long answer
# Chrome ships some voices that synthesise server-side. Restricting to voices
# reporting localService=true keeps the privacy claim literally true.
TTS_LOCAL_VOICES_ONLY = True

# ---------------------------------------------------------------- filtering
DISTRICTS = [
    "All", "Ariyalur", "Coimbatore", "Cuddalore", "Dindigul", "Erode",
    "Kancheepuram", "Madurai", "Nagapattinam", "Salem", "Thanjavur",
    "Tiruchirappalli", "Tirunelveli", "Vellore",
]
CATEGORIES = [
    "All", "Irrigation", "Subsidies", "Seeds", "Machinery", "Insurance",
    "Horticulture", "Organic Farming", "Credit & Loans",
]

# ----------------------------------------------------------------- metadata
APP_TITLE = "Tamil Nadu Agriculture Schemes Assistant"
APP_TAGLINE = "Grounded answers on TN farm schemes — voice, citations, live streaming"
APP_VERSION = "2.0.0"


def api_key_present() -> bool:
    """Cheap guard used by the UI before any network call is attempted."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"))


def langsmith_enabled() -> bool:
    """Tracing is switched on purely by env vars, never by code."""
    flag = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", ""))
    return flag.lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    )
