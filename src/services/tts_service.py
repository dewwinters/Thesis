import io
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from gtts import gTTS

ABBREVIATIONS = {
    r"\bCCCD\b": "căn cước công dân",
    r"\bCMND\b": "chứng minh nhân dân",
    r"\bUBND\b": "ủy ban nhân dân",
    r"\bBHXH\b": "bảo hiểm xã hội",
    r"\bBHYT\b": "bảo hiểm y tế",
    r"\bVNeID\b": "vi en ai đi",
    r"\bQĐ\b": "quyết định",
}

MAX_CHUNK_CHARS = 98
MAX_WORKERS = 3
PAUSE_SENTENCE, PAUSE_CLAUSE, PAUSE_WORD = 0.30, 0.12, 0.05
SILENCE_THRESHOLD = 0.005
LEAD_MARGIN_SEC = 0.05
TAIL_MARGIN_SEC = 0.15


def normalize_text(text: str) -> str:
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[*#_\[\]()]", "", text)
    text = re.sub(r"\s*\n\s*[-•]\s*", ". ", text)
    text = re.sub(r"^\s*[-•]\s*", "", text)
    text = " ".join(text.split())
    text = re.sub(r"\s[-–—]\s", ". ", text)
    return re.sub(r"([.!?,;:])(\s*\.)+", r"\1", text)


def _split(text: str, end_pause: float, max_len: int) -> list[tuple[str, float]]:
    n = len(text)
    if n <= max_len:
        return [(text, end_pause)]

    commas = [
        i + 1
        for i, c in enumerate(text)
        if c in ",;:" and 12 <= i + 1 <= n - 12 and max(i + 1, n - i - 1) <= max_len
    ]
    if commas:
        cut, pause = min(commas, key=lambda c: abs(c - n / 2)), PAUSE_CLAUSE
    else:
        spaces = [i for i, c in enumerate(text) if c == " "]
        cut = min(spaces, key=lambda i: abs(i - n / 2)) if spaces else n // 2
        pause = PAUSE_WORD
    return _split(text[:cut].strip(), pause, max_len) + _split(text[cut:].strip(), end_pause, max_len)


def split_text(text: str, max_len: int = MAX_CHUNK_CHARS) -> list[tuple[str, float]]:
    """Split into chunks of at most max_len chars; each chunk comes with the pause that follows it."""
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if sentence.strip():
            chunks += _split(sentence.strip(), PAUSE_SENTENCE, max_len)
    return chunks


def _fetch_chunk(text: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=text, lang="vi", timeout=15).write_to_fp(buf)
    return buf.getvalue()


def _trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    voiced = np.flatnonzero(np.abs(audio) > SILENCE_THRESHOLD)
    if voiced.size == 0:
        return audio
    start = max(voiced[0] - int(LEAD_MARGIN_SEC * sample_rate), 0)
    end = voiced[-1] + int(TAIL_MARGIN_SEC * sample_rate)
    return audio[start:end]


def assemble(parts: list[bytes], pauses: list[float]) -> bytes:
    pieces, sample_rate = [], None
    for data, pause in zip(parts, pauses):
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sample_rate = sample_rate or sr
        pieces += [_trim_silence(audio, sr), np.zeros(int(pause * sr), dtype="float32")]

    out = io.BytesIO()
    sf.write(out, np.concatenate(pieces[:-1]), sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


class TTSService:
    def synthesize(self, text: str) -> bytes | None:
        chunks = split_text(normalize_text(text))
        if not chunks:
            return None
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            parts = list(pool.map(_fetch_chunk, [t for t, _ in chunks]))
        return assemble(parts, [p for _, p in chunks])