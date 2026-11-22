# Vietnamese Voice-Interactive AI Assistant for Public Administrative Services

A voice-first chatbot that helps citizens navigate Vietnamese public administrative procedures — running on Gemini API with a local RAG knowledge base built from official PDF documents.

Built as a graduation project, designed to run on both laptop (dev) and Raspberry Pi 4B (deployment).

---

## What it does

Citizens can speak or type a question about administrative procedures (ID cards, social insurance, residence registration, etc.). The assistant transcribes the speech, retrieves relevant context from indexed government documents, generates a concise spoken-style answer, and reads it back aloud.

When a question falls outside the indexed documents, the system automatically searches the web via DuckDuckGo and answers from live results, with a disclaimer to verify with the relevant authority.

Two interaction modes:

- **Voice** — Microphone → STT → RAG + LLM → TTS → Speaker
- **Keyboard** — Text input → RAG + LLM → TTS → Speaker

---

## Current status

| # | Feature | Status |
|---|---------|--------|
| 1 | Voice chat (mic → STT → LLM → TTS → speaker) | ✅ Done |
| 2 | Keyboard chat (text → LLM → TTS → speaker) | ✅ Done |
| 3 | Document ingestion into RAG vector store | ✅ Done |
| 4 | RAG retrieval test (no LLM call) | ✅ Done |
| 5 | Audio hardware diagnostic (mic + speaker test) | ✅ Done |
| 6 | System config viewer (API keys, models, RAG status) | ✅ Done |

---

## Stack

| Layer | Technology |
|-------|------------|
| LLM | `gemini-2.5-flash-lite` (primary) + fallback chain |
| STT | `gemini-2.5-flash` via Gemini API |
| TTS | `gTTS` (Google Text-to-Speech) |
| Playback speed | `audiotsm` — WSOLA algorithm, default 1.3× |
| Embeddings | `models/gemini-embedding-2` |
| Vector store | `ChromaDB` (local, persistent) |
| PDF parsing | `pdf2image` + `pytesseract` (OCR for scanned docs) |
| Web search fallback | `ddgs` (DuckDuckGo, no API key required) |
| Audio I/O | `sounddevice` + `soundfile` |
| Language | Python 3.11 |

---

## Project layout

```
Foundation/
├── main.py                  # Entry point — wires up services, launches menu
├── requirements.txt
├── .env.example
├── data/
│   ├── documents/           # (not in repo) Put your PDF/Word files here before ingesting
│   ├── chroma_db/           # (not in repo) Auto-generated ChromaDB vector store
│   └── logs/
│       └── chat_log.json    # (not in repo) Conversation log (auto-generated)
├── poppler-26.09.0/         # (not in repo) Poppler binaries for Windows — download separately
└── src/
    ├── config.py            # Loads .env, exposes typed settings
    ├── logger.py            # Logs each conversation turn to chat_log.json
    ├── audio/
    │   └── audio_io.py      # Record from mic, play audio, WSOLA speed adjust, HW test
    ├── rag/
    │   └── rag_service.py   # PDF OCR ingestion, ChromaDB indexing, semantic retrieval
    ├── services/
    │   ├── gemini_utils.py  # Gemini API calls + automatic model fallback + streaming
    │   ├── stt_service.py   # WAV → Vietnamese text
    │   ├── llm_service.py   # Question → answer, web search fallback, 5-turn memory, streaming
    │   └── tts_service.py   # Text normalization → chunking → parallel gTTS fetch → assemble
    └── ui/
        └── menu.py          # CLI menu, orchestrates the full pipeline
```

Drop your administrative procedure PDF files into `data/documents/` before running the ingestion step (menu option 3).

---

## Files not included in this repo

The following are excluded from version control and must be set up manually:

| Path | Why excluded | What to do |
|------|-------------|------------|
| `.env` | Contains API key | Copy `.env.example` → `.env`, fill in your key |
| `data/chroma_db/` | Auto-generated vector store | Run menu option **3** to rebuild after ingesting documents |
| `data/logs/` | Runtime logs | Auto-created on first run |
| `poppler-26.09.0/` | Large binary (~100 MB) | Download separately — see Windows setup below |

---

## Setup

The project depends on two system-level tools that must be installed before the Python packages: **Tesseract OCR** and **Poppler**. These handle PDF reading and text extraction from scanned documents.

### Windows

**Tesseract OCR**

Download and run the installer from the [UB-Mannheim releases page](https://github.com/UB-Mannheim/tesseract/wiki). During installation, expand *Additional language data* and check **Vietnamese**. The default install path is `C:\Program Files\Tesseract-OCR\` — `pytesseract` will find it automatically there.

**Poppler**

Poppler is **not bundled** in this repo (too large). Download the Windows build from [github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases), extract it, and place the folder at `poppler-26.09.0/` in the project root. The expected path is `poppler-26.09.0\Library\bin\`.

### Linux / Raspberry Pi

```bash
sudo apt install tesseract-ocr tesseract-ocr-vie poppler-utils \
                 libportaudio2 libsndfile1
```

---

### Python environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# Linux / macOS / Raspberry Pi
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env   # Linux/macOS
copy .env.example .env  # Windows
```

Open `.env` and fill in your `GEMINI_API_KEY` (get one at [aistudio.google.com](https://aistudio.google.com)).

### Run

```bash
python main.py
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `LLM_MODEL` | `gemini-2.5-flash-lite` | Primary model for answering questions |
| `STT_MODEL` | `gemini-2.5-flash` | Model used for speech transcription |
| `FALLBACK_MODELS` | `gemini-2.5-flash-lite, gemini-2.5-flash` | Comma-separated fallback chain on rate limit / server errors |
| `API_TIMEOUT_MS` | `15000` | Per-request API timeout in milliseconds |
| `TTS_SPEED` | `1.3` | Playback speed multiplier (1.0 = natural) |
| `DEBUG` | `0` | Set to `1` to print per-step latency (STT / LLM / TTS) |

---

## Implementation notes

**RAG with similarity threshold** — At query time, only chunks with a cosine distance below `0.45` are used as context. If no chunk passes the threshold the system falls back to web search, avoiding hallucinated answers from loosely related content.

**Web search fallback** — When RAG has no relevant context, the system queries DuckDuckGo (Vietnamese region), injects the top 5 snippets into the LLM context, and appends a disclaimer reminding the citizen to verify with the relevant authority. No API key required.

**Streaming TTS** — The LLM response is streamed token by token, accumulated into complete sentences, and submitted to gTTS in parallel as each sentence finishes. Audio playback begins as soon as the first sentence is synthesized — without waiting for the full response — significantly reducing perceived latency.

**Batch embedding** — `embed_content` is called once per batch of up to 100 chunks rather than once per chunk, cutting API round-trips during document ingestion.

**Model fallback** — On `429 Rate Limit` or `5xx` server errors, the system automatically cycles through `FALLBACK_MODELS` without interrupting the conversation.

**Vietnamese abbreviation normalization** — Before TTS, a regex pass expands common administrative abbreviations (`CCCD` → *căn cước công dân*, `UBND` → *ủy ban nhân dân*, etc.) so the speech output sounds natural.

**Spoken-style LLM output** — The system prompt constrains the LLM to produce short sentences (< 15 words each), one step per line — optimized for TTS rather than reading on screen.

**Parallel TTS fetch** — Long responses are split into chunks (< 98 chars), fetched from gTTS in parallel using `ThreadPoolExecutor(max_workers=3)`, then assembled with calibrated pauses between sentences (0.3 s) and clauses (0.12 s).

**Conversation logging** — Every turn is appended to `data/logs/chat_log.json` with timestamp, mode, question, answer, retrieved chunks, and source documents.