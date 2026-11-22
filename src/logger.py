import json
import os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "chat_log.json"


def log_conversation(
    mode: str,
    question: str,
    answer: str,
    chunks: list[str] | None = None,
    sources: list[str] | None = None,
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "question": question,
        "answer": answer,
        "rag_sources": sorted(set(sources)) if sources else [],
        "rag_chunks": chunks or [],
    }

    existing = []
    if LOG_FILE.exists():
        try:
            existing = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(entry)
    LOG_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")