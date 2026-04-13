# preprocess_perturbation.py
import glob
import parselmouth
from parselmouth.praat import call
import random
import os

def shift_formant(wav_path):
    out_path = wav_path.replace('.wav', '_pert.wav')
    if os.path.exists(out_path):
        return

    # 1. 载入音频
    sound = parselmouth.Sound(wav_path)
    
    # 2. 随机生成扰动比例 (比如 0.8 到 1.2 之间)
    formant_shift_ratio = random.uniform(0.8, 1.2)
    pitch_shift_ratio = 1.0 # 保持音高不变，只改变音色(共振峰)
    
    # 3. 使用 Praat 的 Change gender 功能来实现共振峰平移
    # 参数：sound, pitch_floor, pitch_ceiling, formant_shift_ratio, pitch_shift_ratio, pitch_range_ratio, duration_factor
    try:
        perturbed_sound = call(sound, "Change gender", 75, 600, formant_shift_ratio, pitch_shift_ratio, 1.0, 1.0)
        # 4. 保存扰动后的音频
        perturbed_sound.save(out_path, "WAV")
        print(f"Saved perturbed audio to {out_path}")
    except Exception as e:
        print(f"Failed to process {wav_path}: {e}")

if __name__ == "__main__":
    wav_files = glob.glob("dataset/**/*.wav", recursive=True)
    # 过滤掉已经是 pert 的音频
    wav_files = [w for w in wav_files if not w.endswith('_pert.wav')]
    for w in wav_files:
        shift_formant(w)