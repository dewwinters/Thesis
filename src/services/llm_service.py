import re


from ddgs import DDGS
from google.genai import types
from src import config
from src.services.gemini_utils import generate_with_fallback, make_config, stream_with_fallback


SYSTEM_PROMPT = (
    "Bạn là trợ lý AI hỗ trợ người dân về dịch vụ hành chính công tại Việt Nam. "
    "Trả lời bằng tiếng Việt, chính xác, lịch sự. "
    "Nếu câu hỏi nghe không rõ hoặc khó hiểu, hãy hỏi lại thay vì đoán."
)
GROUNDED_RULE = (
    " Chỉ dựa vào phần 'Tài liệu tham khảo' được cung cấp để trả lời; "
    "nếu tài liệu không đủ thông tin, hãy nói rõ là chưa có thông tin thay vì đoán."
)
NO_CONTEXT_RULE = " Trả lời theo hiểu biết chung, chỉ nhắc người dân xác nhận lại với cơ quan có thẩm quyền một lần ngắn ở cuối."
WEB_SEARCH_RULE = (
    " Tìm kiếm thông tin từ các trang web chính thức của chính phủ Việt Nam để trả lời. "
    "Ưu tiên các nguồn như dichvucong.gov.vn, chinhphu.vn, hoặc cổng thông tin của các bộ ngành. "
    "Cuối câu trả lời, thêm một câu ngắn nhắc người dân xác nhận lại thông tin với cơ quan có thẩm quyền."
)
DISCLAIMER = "\nLưu ý: Thông tin trên được tổng hợp từ internet. Bạn nên xác nhận lại với cơ quan có thẩm quyền trước khi thực hiện."
SPOKEN_HINT = (
    " Câu trả lời sẽ được đọc thành tiếng: dùng câu ngắn, mỗi câu dưới 15 từ, "
    "có dấu chấm và dấu phẩy hợp lý, không dùng markdown hay ký hiệu đặc biệt nào khác ngoài dấu gạch đầu dòng. "
    "Với các bước, mỗi bước viết trên một dòng riêng và bắt đầu bằng dấu gạch đầu dòng. "
    "Độ dài tùy câu hỏi: hỏi khái niệm hoặc câu đơn giản thì trả lời 2 đến 3 câu; "
    "hỏi cách làm hoặc thủ tục thì nêu đủ các bước theo thứ tự (khoảng 5 đến 8 bước), "
    "dùng các từ nối như 'trước tiên', 'tiếp theo', 'cuối cùng'."
)


def _msg(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


class LLMService:
    def __init__(self, client):
        self.client = client
        self.history: list[types.Content] = []

    def answer(self, question: str, context: str | None = None, spoken: bool = False) -> str:
        user_msg = f"Tài liệu tham khảo:\n{context}\n\nCâu hỏi: {question}" if context else question
        contents = [*self.history[-config.MAX_HISTORY_TURNS * 2:], _msg("user", user_msg)]
        instruction = (
            SYSTEM_PROMPT
            + (GROUNDED_RULE if context else NO_CONTEXT_RULE)
            + (SPOKEN_HINT if spoken else "")
        )

        response = generate_with_fallback(
            self.client,
            config.LLM_MODEL,
            contents=contents,
            config=make_config(system_instruction=instruction, temperature=0.2),
        )
        text = (response.text or "").strip() or "Xin lỗi, tôi chưa thể trả lời câu hỏi này."
        if spoken:
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        self.history += [_msg("user", question), _msg("model", text)]
        return text

    @staticmethod
    def _web_search(question: str, max_results: int = 5) -> str | None:
        try:
            results = DDGS().text(question, region="vn-vi", max_results=max_results)
            if not results:
                return None
            snippets = [f"- {r['title']}: {r['body']}" for r in results]
            return "\n".join(snippets)
        except Exception:
            return None

    def answer_with_search(self, question: str, spoken: bool = False) -> str:
        web_context = self._web_search(question)
        if web_context:
            user_msg = f"Thông tin tìm kiếm từ web:\n{web_context}\n\nCâu hỏi: {question}"
            instruction = SYSTEM_PROMPT + WEB_SEARCH_RULE + (SPOKEN_HINT if spoken else "")
        else:
            user_msg = question
            instruction = SYSTEM_PROMPT + NO_CONTEXT_RULE + (SPOKEN_HINT if spoken else "")

        contents = [*self.history[-config.MAX_HISTORY_TURNS * 2:], _msg("user", user_msg)]
        response = generate_with_fallback(
            self.client,
            config.LLM_MODEL,
            contents=contents,
            config=make_config(system_instruction=instruction, temperature=0.2),
        )
        text = (response.text or "").strip() or "Xin lỗi, tôi chưa thể trả lời câu hỏi này."
        if spoken:
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if web_context:
            text += DISCLAIMER
        self.history += [_msg("user", question), _msg("model", text)]
        return text

    def reset(self):
        self.history.clear()

    def answer_stream(self, question: str, context: str | None = None):
        """Yields hoàn chỉnh từng câu khi LLM stream về."""
        user_msg = f"Tài liệu tham khảo:\n{context}\n\nCâu hỏi: {question}" if context else question
        contents = [*self.history[-config.MAX_HISTORY_TURNS * 2:], _msg("user", user_msg)]
        instruction = (
                SYSTEM_PROMPT
                + (GROUNDED_RULE if context else NO_CONTEXT_RULE)
                + SPOKEN_HINT
        )

        stream = stream_with_fallback(
            self.client,
            config.LLM_MODEL,
            contents=contents,
            config=make_config(system_instruction=instruction, temperature=0.2),
        )

        buffer = ""
        full_text = ""
        sentence_end = re.compile(r"(?<=[.!?,;:\n])\s+")

        for chunk in stream:
            token = (chunk.text or "") if chunk is not None else ""  # <-- thêm check chunk
            buffer += token
            full_text += token

            # tách câu hoàn chỉnh ra khỏi buffer
            parts = sentence_end.split(buffer)
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    yield sentence
            buffer = parts[-1]  # phần chưa kết thúc câu

        # flush phần còn lại
        if buffer.strip():
            yield buffer.strip()

        # cập nhật history sau khi stream xong
        full_text = full_text.strip() or "Xin lỗi, tôi chưa thể trả lời câu hỏi này."
        self.history += [_msg("user", question), _msg("model", full_text)]