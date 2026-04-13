import os
import glob
import librosa
import soundfile as sf
from tqdm import tqdm

def resample_dataset(dataset_dir="dataset", target_sr=24000):
    # 找到 dataset 文件夹下所有的 wav 文件
    wav_files = glob.glob(f"{dataset_dir}/**/*.wav", recursive=True)
    
    print(f"找到 {len(wav_files)} 个音频文件，开始统一重采样至 {target_sr} Hz...")
    
    for wav_path in tqdm(wav_files):
        try:
            # librosa.load 会自动将读取的音频重采样到目标 sr
            y, sr = librosa.load(wav_path, sr=target_sr)
            # 覆盖保存为 24000 Hz
            sf.write(wav_path, y, target_sr)
        except Exception as e:
            print(f"处理 {wav_path} 时出错: {e}")

if __name__ == "__main__":
    resample_dataset()

    