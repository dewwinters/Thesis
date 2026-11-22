import hashlib
import os
import chromadb
import docx
import pytesseract


from pathlib import Path
from google import genai
from pypdf import PdfReader
from src import config
from pdf2image import convert_from_bytes


COLLECTION_NAME = "hanh_chinh_cong"


TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Dew\Study\IU\#PRETHESIS\Foundation\poppler-26.09.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        if text:
            return text
        print(f"   [OCR] {path.name} không có text layer, đang OCR...")
        images = convert_from_bytes(path.read_bytes(), dpi=200, poppler_path=POPPLER_PATH)
        return "\n".join(pytesseract.image_to_string(img, lang="vie") for img in images)
    if suffix in (".docx", ".doc"):
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 30]


def _file_id(path: Path) -> str:
    return hashlib.md5(path.name.encode()).hexdigest()[:12]


class RAGService:
    def __init__(self, client: genai.Client):
        self.client = client
        Path(config.DOCS_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self._db = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self._col = self._db.get_or_create_collection(COLLECTION_NAME)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        BATCH_SIZE = 100
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            result = self.client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=texts[i:i + BATCH_SIZE],
            )
            all_embeddings.extend(e.values for e in result.embeddings)
        return all_embeddings


    def ingest(self, path: Path) -> tuple[int, int]:
        """Return (chunks_added, chunks_skipped)."""
        text = _extract_text(path)
        if not text.strip():
            return 0, 0

        chunks = _chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        file_id = _file_id(path)

        # build ids, skip existing
        ids = [f"{file_id}_{i}" for i in range(len(chunks))]
        existing = set(self._col.get(ids=ids)["ids"])
        new_pairs = [(id_, ch) for id_, ch in zip(ids, chunks) if id_ not in existing]

        if not new_pairs:
            return 0, len(chunks)

        new_ids, new_chunks = zip(*new_pairs)
        embeddings = self._embed(list(new_chunks))
        self._col.add(
            ids=list(new_ids),
            embeddings=embeddings,
            documents=list(new_chunks),
            metadatas=[{"source": path.name}] * len(new_ids),
        )
        return len(new_ids), len(chunks) - len(new_ids)

    def ingest_all(self) -> dict[str, tuple[int, int]]:
        """Ingest every PDF/Word in DOCS_DIR. Return {filename: (added, skipped)}."""
        results = {}
        docs_path = Path(config.DOCS_DIR)
        for f in docs_path.iterdir():
            if f.suffix.lower() in (".pdf", ".docx", ".doc"):
                results[f.name] = self.ingest(f)
        return results

    def query(self, question: str, distance_threshold: float = config.RAG_DISTANCE_THRESHOLD):
        """Return (chunks, sources). Filters out chunks with distance above threshold."""
        if self._col.count() == 0:
            return [], []
        embedding = self._embed([question])[0]
        result = self._col.query(
            query_embeddings=[embedding],
            n_results=min(config.RAG_TOP_K, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not result["documents"]:
            return [], []

        chunks, sources = [], []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            if dist <= distance_threshold:
                chunks.append(doc)
                sources.append(meta["source"])

        return chunks, sources

    def stats(self) -> dict:
        count = self._col.count()
        sources = set()
        if count:
            all_meta = self._col.get(include=["metadatas"])["metadatas"]
            sources = {m["source"] for m in all_meta}
        return {"total_chunks": count, "sources": sorted(sources)}