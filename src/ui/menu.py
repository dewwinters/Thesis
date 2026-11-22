import shutil
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src import config
from src.audio import audio_io
from src.audio.audio_io import test_audio_hardware
from src.logger import log_conversation


class Menu:
    TITLE = "TRỢ LÝ AI TƯƠNG TÁC BẰNG GIỌNG NÓI HỖ TRỢ DỊCH VỤ HÀNH CHÍNH CÔNG"

    def __init__(self, rag=None, llm=None, stt=None, tts=None):
        self.rag, self.llm, self.stt, self.tts = rag, llm, stt, tts
        self.is_running = True
        self.actions = {
            "1": ("🎙   Giao tiếp bằng giọng nói (Voice ➔ STT ➔ RAG ➔ LLM ➔ TTS)", self.handle_voice_input),
            "2": ("⌨   Giao tiếp bằng bàn phím (Text ➔ RAG ➔ LLM ➔ TTS)", self.handle_keyboard_input),
            "3": ("📂   Nạp & cập nhật tài liệu hành chính công (PDF/Word)", self.handle_ingest_documents),
            "4": ("🔍   Kiểm tra tìm kiếm RAG (không gọi API)", self.handle_test_rag),
            "5": ("🔊   Test thiết bị Micro & Loa", self.handle_test_audio_devices),
            "6": ("⚙   Xem cấu hình hệ thống & kết nối API", self.handle_show_config),
        }

    # ---- display ----

    def display_header(self):
        print("\n" + "=" * 75)
        print(f"🤖  {self.TITLE}")
        print("=" * 75)

    def display_menu(self):
        print("\n--- BẢNG ĐIỀU KHIỂN CHÍNH ---")
        for key, (label, _) in self.actions.items():
            print(f"{key}. {label}")
        print("0. 🚪   Thoát chương trình")
        print("-" * 75)

    # ---- shared helpers ----

    def _llm_ready(self) -> bool:
        if self.llm is None:
            print("❌ Chưa cấu hình GEMINI_API_KEY trong file .env")
            return False
        return True

    def _get_context(self, question: str) -> tuple[str | None, list[str], list[str]]:
        if self.rag is None:
            return None, [], []
        chunks, sources = self.rag.query(question)
        context = "\n\n".join(chunks) if chunks else None
        return context, chunks, sources

    @staticmethod
    def _timed(label: str, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        if config.DEBUG:
            print(f"   ({label}: {time.time() - start:.1f}s)")
        return result

    @staticmethod
    def _print_answer(answer: str):
        width = min(shutil.get_terminal_size().columns, 100) - 2
        first, *rest = answer.splitlines() or [answer]
        print(textwrap.fill(first, width, initial_indent="🤖 Trợ lý: ", subsequent_indent="   "))
        for line in rest:
            print(textwrap.fill(line, width, initial_indent="   ", subsequent_indent="     "))

    def _respond_stream(self, question: str, mode: str = "keyboard"):
        print("⏳ Đang tìm câu trả lời...")
        context, chunks, sources = self._get_context(question)

        if context is None:
            print("🌐 Không tìm thấy trong tài liệu, đang tìm kiếm trên web...")
            answer = self._timed("LLM", self.llm.answer_with_search, question, spoken=True)
            self._print_answer(answer)
            log_conversation(mode=mode, question=question, answer=answer, chunks=[], sources=[])
            if self.tts:
                audio = self._timed("TTS", self.tts.synthesize, answer)
                if audio:
                    audio_io.play_audio(audio)
            return

        print("🤖 Trợ lý: ", end="", flush=True)
        sentences = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []

            for sentence in self.llm.answer_stream(question, context):
                print(sentence, end=" ", flush=True)
                sentences.append(sentence)
                if self.tts:
                    future = pool.submit(self.tts.synthesize, sentence)
                    futures.append((future, sentence))

                while futures and futures[0][0].done():
                    audio = futures.pop(0)[0].result()
                    if audio:
                        audio_io.play_audio(audio)

            print()

            for future, _ in futures:
                audio = future.result()
                if audio:
                    audio_io.play_audio(audio)

        full_answer = " ".join(sentences)
        log_conversation(mode=mode, question=question, answer=full_answer, chunks=chunks, sources=sources)

    # ---- functions 1 & 2 ----

    def handle_voice_input(self):
        if not (self._llm_ready() and self.stt and self.tts):
            return
        try:
            while True:
                if input("\n⏎  Nhấn [Enter] để bắt đầu nói ('0' để thoát): ").strip() == "0":
                    break

                print("🎙  Đang thu âm... nhấn [Enter] để dừng.")
                wav = audio_io.record_until_enter()
                if wav is None:
                    print("⚠️  Không thu được âm thanh, vui lòng thử lại.")
                    continue

                try:
                    print("⏳ Đang nhận diện giọng nói...")
                    question = self._timed("STT", self.stt.transcribe, wav)
                    if not question:
                        print("⚠️  Không nhận diện được giọng nói.")
                        continue
                    print(f"👤 Bạn: {question}")
                    self._respond_stream(question, mode="voice")
                except Exception as e:
                    print(f"❌ Lỗi: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\n↩  Quay lại menu.")

    def handle_keyboard_input(self):
        if not self._llm_ready():
            return
        try:
            while True:
                print("\n[Chế độ bàn phím] Gõ '0' để quay lại menu.")
                question = input("\n👉 Câu hỏi: ").strip()
                if question == "0":
                    break
                if not question:
                    print("⚠️  Bạn chưa nhập nội dung câu hỏi!")
                    continue

                try:
                    self._respond_stream(question, mode="keyboard")
                except Exception as e:
                    print(f"❌ Lỗi: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\n↩  Quay lại menu.")

    # ---- functions 3-6 ----

    def handle_ingest_documents(self):
        if self.rag is None:
            print("❌ RAG chưa khởi tạo (thiếu API key).")
            return

        docs_dir = Path(config.DOCS_DIR)
        print(f"\n📂 Thư mục tài liệu: {docs_dir.resolve()}")
        print("   Đặt file PDF/Word vào thư mục trên, sau đó nhấn Enter để nạp.\n")

        files = [f for f in docs_dir.iterdir() if f.suffix.lower() in (".pdf", ".docx", ".doc")]
        if not files:
            print("⚠️  Không tìm thấy tài liệu nào. Hãy thêm file PDF/Word vào thư mục và thử lại.")
            return

        print(f"📄 Tìm thấy {len(files)} file: {', '.join(f.name for f in files)}")
        if input("👉 Bắt đầu nạp? [Enter / '0' để hủy]: ").strip() == "0":
            return

        print("⏳ Đang xử lý...")
        results = self.rag.ingest_all()

        total_added = total_skipped = 0
        for name, (added, skipped) in results.items():
            status = "✅" if added > 0 else "⏭"
            print(f"   {status} {name}: +{added} chunk mới, {skipped} đã có")
            total_added += added
            total_skipped += skipped

        print(f"\n✅ Hoàn tất: {total_added} chunk mới, {total_skipped} chunk bỏ qua (đã tồn tại).")
        stats = self.rag.stats()
        print(f"📊 Tổng trong DB: {stats['total_chunks']} chunks từ {len(stats['sources'])} tài liệu.")

    def handle_test_rag(self):
        if self.rag is None:
            print("❌ RAG chưa khởi tạo (thiếu API key).")
            return

        stats = self.rag.stats()
        print(f"\n📊 Vector DB: {stats['total_chunks']} chunks | Nguồn: {', '.join(stats['sources']) or 'trống'}")
        if stats["total_chunks"] == 0:
            print("⚠️  DB trống, hãy nạp tài liệu ở chức năng 3 trước.")
            return

        try:
            while True:
                query = input("\n🔍 Nhập câu truy vấn ('0' để thoát): ").strip()
                if query == "0":
                    break
                if not query:
                    continue

                chunks, sources = self.rag.query(query)
                if not chunks:
                    print("⚠️  Không tìm thấy chunk liên quan.")
                    continue

                print(f"\n📄 Top {len(chunks)} chunk liên quan:")
                for i, (chunk, source) in enumerate(zip(chunks, sources), 1):
                    preview = chunk[:200].replace("\n", " ")
                    print(f"\n  [{i}] [{source}] {preview}{'...' if len(chunk) > 200 else ''}")
        except (KeyboardInterrupt, EOFError):
            print("\n↩  Quay lại menu.")

    def handle_test_audio_devices(self):
        print("\n[Module 5: Hardware Audio Test]")
        test_audio_hardware()

    def handle_show_config(self):
        print("\n[Module 6: System Configuration]")
        print(f"--> 🧠 LLM Model: {config.LLM_MODEL}")
        print(f"--> 🎙  STT Model: {config.STT_MODEL}")
        print(f"--> 🔁 Model dự phòng: {', '.join(config.FALLBACK_MODELS)}")
        print(f"--> 🔑 Gemini API key: {'đã cấu hình' if config.GEMINI_API_KEY else 'chưa cấu hình (.env)'}")
        print(f"--> 🔊 Tốc độ đọc: x{config.TTS_SPEED}")
        print(f"--> 📦 RAG: {'đã khởi tạo' if self.rag else 'chưa khởi tạo'}")

    # ---- main loop ----

    def run(self):
        self.display_header()
        try:
            while self.is_running:
                self.display_menu()
                choice = input(f"Mời chọn chức năng (0-{len(self.actions)}): ").strip()
                if choice == "0":
                    print("\n👋 Cảm ơn bạn đã sử dụng hệ thống! Đang thoát...")
                    self.is_running = False
                elif choice in self.actions:
                    self.actions[choice][1]()
                else:
                    print(f"\n❌ Lựa chọn không hợp lệ! Vui lòng chọn từ 0 đến {len(self.actions)}.")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Đã thoát.")