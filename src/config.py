import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
STT_MODEL = os.getenv("STT_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.5-flash").split(",")
    if m.strip()
]
API_TIMEOUT_MS = int(os.getenv("API_TIMEOUT_MS", "15000"))
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.5"))
MAX_HISTORY_TURNS = 5
DEBUG = os.getenv("DEBUG", "0") == "1"
# RAG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
DOCS_DIR = os.getenv("DOCS_DIR", "data/documents")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.45"))