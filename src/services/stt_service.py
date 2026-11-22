from google.genai import types

from src import config
from src.services.gemini_utils import generate_with_fallback, make_config

PROMPT = (
    "Transcribe this Vietnamese speech. The topic is Vietnamese public administrative services; "
    "expect terms such as: căn cước công dân, CCCD, VNeID, định danh điện tử mức 2, giấy khai sinh, "
    "đăng ký cư trú, UBND, bảo hiểm xã hội, bảo hiểm y tế, hộ chiếu, giấy phép lái xe. "
    "Return only the transcript, no extra text. Return an empty string if there is no speech."
)


class STTService:
    def __init__(self, client):
        self.client = client

    def transcribe(self, wav_bytes: bytes) -> str:
        response = generate_with_fallback(
            self.client,
            config.STT_MODEL,
            contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), PROMPT],
            config=make_config(temperature=0),
        )
        return (response.text or "").strip()