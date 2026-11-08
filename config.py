import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5


DEFAULT_RELEVANCE_THRESHOLD = 0.45

DONT_KNOW = "I don't know based on the provided documents."


def pdf_dir() -> Path:
    return Path(os.environ.get("PDF_DIR", BASE_DIR / "pdfs"))


def chroma_dir() -> Path:
    return Path(os.environ.get("CHROMA_DIR", BASE_DIR / "chroma_db"))


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_timeout() -> float:
    raw = os.environ.get("OLLAMA_TIMEOUT", "120")
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    return value if value > 0 else 120.0


def relevance_threshold() -> float:
    raw = os.environ.get("RELEVANCE_THRESHOLD", str(DEFAULT_RELEVANCE_THRESHOLD))
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "RELEVANCE_THRESHOLD must be a number such as 0.35. "
            "It is the minimum cosine similarity kept during retrieval."
        ) from exc
