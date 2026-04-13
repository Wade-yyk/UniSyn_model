# preprocess_f0.py
import os
import glob
import torch
import librosa
import pyworld as pw
import numpy as np

def extract_f0(wav_path, hop_length=300, sampling_rate=24000):
    # 1. 读取音频
    wav, sr = librosa.load(wav_path, sr=sampling_rate)
    
    # 2. 将音频转为 64 位浮点数 (pyworld 要求)
    wav = wav.astype(np.float64)
    
    # 3. 使用 WORLD 提取 F0
    frame_period = (hop_length / sampling_rate) * 1000.0
    f0, t = pw.dio(wav, sampling_rate, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, t, sampling_rate)
    
    # === 新增：安全对数转换 (只对有声部分取对数) ===
    f0_log = np.zeros_like(f0)
    voiced_indices = f0 > 0
    f0_log[voiced_indices] = np.log(f0[voiced_indices])
    
    # 4. 转为 tensor 并保存
    f0_tensor = torch.FloatTensor(f0_log)
    out_path = wav_path.replace('.wav', '.f0.pt')
    torch.save(f0_tensor, out_path)
    print(f"Saved Log-F0 to {out_path}")

if __name__ == "__main__":
    # 替换为你实际的 wav 存放路径
    wav_files = glob.glob("dataset/**/*.wav", recursive=True)
    for w in wav_files:
        extract_f0(w)

