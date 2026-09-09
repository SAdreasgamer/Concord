"""
Application configuration.

All settings are read from environment variables or .env file.
Nothing is hardcoded — defaults are only for non-sensitive, non-data values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Prevent LiteLLM from blocking on GitHub SSL timeouts for remote cost maps
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
try:
    import litellm
    litellm.telemetry = False
    litellm.drop_params = True
except Exception:
    pass

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
DB_PATH = PROJECT_ROOT / "concord.db"

UPLOAD_DIR.mkdir(exist_ok=True)

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# --- LLM Models & Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "groq/qwen/qwen3.8-27b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
DEFAULT_PAGE_LIMIT = int(os.getenv("DEFAULT_PAGE_LIMIT", "15"))
