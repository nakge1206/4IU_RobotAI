from gtts import gTTS
from pydub import AudioSegment
import io
import simpleaudio as sa  # 재생용 (경량)

def speak_gtts(text, lang="ko"):
    """gTTS를 메모리 상에서 mp3 디코딩 후 재생 (파일 저장 없이)"""
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # mp3 → PCM 변환
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        raw_data = audio.raw_data
        sample_rate = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width

        # 재생
        play_obj = sa.play_buffer(raw_data, num_channels=channels,
                                  bytes_per_sample=sample_width,
                                  sample_rate=sample_rate)
        play_obj.wait_done()

    except Exception as e:
        print("[gTTS 재생 오류]", e)

speak_gtts("안녕하세요요")