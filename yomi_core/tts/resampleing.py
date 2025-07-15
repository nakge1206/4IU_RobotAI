import os
import librosa
import soundfile as sf

def resample_wavs(root_dir, target_sr=22050):
    count = 0
    for folder, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.wav'):
                wav_path = os.path.join(folder, filename)
                try:
                    # 기존 오디오 로드
                    y, sr = librosa.load(wav_path, sr=None)
                    if sr != target_sr:
                        # 리샘플링
                        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                        # 덮어쓰기
                        sf.write(wav_path, y_resampled, target_sr)
                        print(f"✅ Resampled: {wav_path} ({sr} → {target_sr})")
                        count += 1
                except Exception as e:
                    print(f"⚠️ Error: {wav_path} - {e}")
    print(f"\n총 {count}개 파일 리샘플링 완료.")

# 사용 예시
resample_wavs(r"C:\Users\COM\Desktop\데이터\011.한국어 아동 음성 데이터\01.데이터\2.Validation\원천데이터\kor_formatted")
