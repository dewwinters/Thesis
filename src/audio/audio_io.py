import io

import numpy as np
import sounddevice as sd
import soundfile as sf
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter

from src import config

SAMPLE_RATE = 16000
MIN_SAMPLES = SAMPLE_RATE // 3
END_PADDING_SEC = 0.4


def record_until_enter() -> bytes | None:
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    if len(audio) < MIN_SAMPLES:
        return None

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def change_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio
    reader = ArrayReader(audio.reshape(1, -1))
    writer = ArrayWriter(channels=1)
    wsola(channels=1, speed=speed).run(reader, writer)
    return writer.data[0]


def play_audio(data: bytes, speed: float = None):
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    effective_speed = speed if speed is not None else config.TTS_SPEED
    audio = change_speed(audio, effective_speed)

    audio = np.concatenate([audio, np.zeros(int(END_PADDING_SEC * sample_rate), dtype="float32")])
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        raise

def get_audio_devices_info() -> dict:
    try:
        input_dev = sd.query_devices(kind="input")
        output_dev = sd.query_devices(kind="output")
        return {
            "input": input_dev.get('name', 'Không xác định'),
            "output": output_dev.get('name', 'Không xác định')
        }
    except Exception as e:
        return {
            "input": f"Lỗi đọc thiết bị ({e})",
            "output": f"Lỗi đọc thiết bị ({e})"
        }

def test_audio_hardware():
    devices = get_audio_devices_info()
    print("\n" + "=" * 65)
    print("🎧 THÔNG TIN THIết BỊ ÂM THANH ĐANG KẾT NỐI")
    print("=" * 65)
    print(f"🎙️ Micro mặc định (Input) : {devices['input']}")
    print(f"🔊 Loa mặc định (Output) : {devices['output']}")
    print("=" * 65)

    # Bước 1: Chờ người dùng nhấn Enter để sẵn sàng BẮT ĐẦU
    input("\n👉 Nhấn Enter để BẮT ĐẦU thu âm thử...")

    # Bước 2: Hiển thị trạng thái đang thu âm
    print("🎙️ [ĐANG THU ÂM...] Hãy nói vào Micro (Nhấn Enter để KẾT THÚC)...")

    try:
        # Bước 3: Kích hoạt Micro và chờ Enter để KẾT THÚC
        audio_bytes = record_until_enter()
        if audio_bytes:
            print("✅ Thu âm thành công!")
            print("🔊 Đang phát lại đoạn âm thanh vừa thu qua Loa...")
            play_audio(audio_bytes, speed = 1.0)
            print("🎉 Kiểm tra phần cứng Micro và Loa hoàn tất!")
        else:
            print("⚠️ Không nhận được dữ liệu âm thanh (Đoạn thu quá ngắn < 0.3s hoặc Micro bị tắt tiếng).")
    except Exception as e:
        print(f"❌ Lỗi trong quá trình test thiết bị: {e}")