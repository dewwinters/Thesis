from google import genai
from google.genai import types

from src import config
from src.rag.rag_service import RAGService
from src.services.llm_service import LLMService
from src.services.stt_service import STTService
from src.services.tts_service import TTSService
from src.ui.menu import Menu


def build_services() -> dict:
    if not config.GEMINI_API_KEY:
        return {}
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=config.API_TIMEOUT_MS),
    )
    return {
        "llm": LLMService(client),
        "stt": STTService(client),
        "tts": TTSService(),
        "rag": RAGService(client),
    }


if __name__ == "__main__":
    Menu(**build_services()).run()