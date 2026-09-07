"""
Application configuration.

All settings are read from environment variables or provided at runtime.
Nothing is hardcoded — defaults are only for non-sensitive, non-data values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths (all relative to project root, never hardcoded absolute paths) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
DB_PATH = PROJECT_ROOT / "concord.db"

# Ensure upload directory exists at startup
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# --- LLM ---
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini/gemini-2.0-flash")

# --- Embeddings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # Matches all-MiniLM-L6-v2 output dimension

# --- Normalization Registry ---
NORMALIZATION_SIMILARITY_THRESHOLD = float(
    os.getenv("NORMALIZATION_SIMILARITY_THRESHOLD", "85")
)

# --- Candidate Matching ---
EMBEDDING_SIMILARITY_THRESHOLD = float(
    os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.75")
)
EMBEDDING_TOP_K = int(os.getenv("EMBEDDING_TOP_K", "5"))
