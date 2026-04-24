# UniSyn 复刻实现

基于 [VITS](https://github.com/jaywalnut310/vits) 仓库，对论文 [**UniSyn: An End-to-End Unified Model for Text-to-Speech and Singing Voice Synthesis**](https://arxiv.org/abs/2212.01546) (Lei et al., AAAI 2023) 的非官方复刻实现。

UniSyn 是一个端到端的统一语音/歌声合成模型，允许在**只拥有目标说话人的语音数据或歌声数据之一**的情况下，同时合成该说话人的说话和歌唱声音。

---

## ✨ 主要特性

- **统一建模**：单一模型同时支持 TTS 与 SVS 两种任务
- **统一语言学特征**：文本（phoneme + tone）与乐谱（phoneme + note pitch）共用同一套输入表示
- **MC-VAE**：多条件变分自编码器，将潜空间分解为说话人子空间 `z_s` 与其余信息子空间 `z_rst`
- **Guided-VAE**：通过对 `z_s` 的 speaker 监督和对 `z_rst` 的 pitch 监督强化解耦
- **Speaker Timbre Perturbation**：共振峰扰动 + Wasserstein 距离约束，进一步解耦说话人音色
- **端到端**：直接从文本/乐谱生成波形，无需单独的声码器

---

## 📐 模型架构

```
           ┌──────────────┐
Text/Score │ Prior Model  │       Speaker ID ──► p(z_s | c_s)
─────────► │ (Text Enc +  │
           │ Length Reg + │──► p(z_rst | c_rst)
           │ Frame Prior) │              │
           └──────────────┘              ▼
                                      ┌─────┐
Audio ──► Posterior Encoder ──► z ──► │ Dec │──► Waveform
                                      └─────┘
                       ▲
                       │
                    KL / GVAE / Perturbation 约束
```

详细结构参见 [论文 Figure 1](https://arxiv.org/abs/2212.01546)。

---

## 🗂️ 项目结构

```
.
├── configs/
│   └── unisyn_base.json         # 训练/模型超参
├── dataset/
│   ├── tts/                     # TTS 数据集（如 baker 中文女声）
│   │   ├── wavs/
│   │   └── meta/
│   │       ├── ProsodyLabeling/
│   │       └── PhoneLabeling/
│   └── svs/                     # SVS 数据集（如 Opencpop）
│       ├── wavs/
│       └── meta/
│           ├── transcriptions.txt
│           ├── train.txt
│           └── test.txt
├── filelists/                   # 由 prepare_filelists.py 生成
├── text/
│   └── phone_vocab.py           # 音素词表
├── models.py                    # UniSyn 模型（SynthesizerTrn 等）
├── data_utils.py                # 数据加载与 collate
├── train.py                     # 训练入口
├── inference.py                 # 推理脚本
├── losses.py                    # KL / GVAE / Wasserstein 损失
├── prepare_filelists.py         # 生成训练用 filelist
├── preprocess_f0.py             # WORLD 提取 log-F0
├── preprocess_perturbation.py   # Praat 共振峰扰动
├── resample.py                  # 统一采样率到 24kHz
├── attentions.py / commons.py / modules.py / mel_processing.py
└── utils.py
```

---

## 🔧 环境配置

```bash
# Python >= 3.9
pip install -r requirements.txt
```

主要依赖：
- PyTorch (>= 2.0, 支持 CUDA)
- librosa, soundfile, scipy
- pyworld（F0 提取）
- praat-parselmouth（共振峰扰动）
- tensorboard, tqdm

---

## 📦 数据准备

本项目在两类数据上训练：

| 类型 | 推荐数据集 | 说话人数 | 时长 |
|------|-----------|----------|------|
| TTS  | [DataBaker 中文女声](https://www.data-baker.com/open_source.html) | 1 | ~10h |
| SVS  | [Opencpop](https://wenet.org.cn/opencpop/) | 1 | ~5h |

目录组织示例见「项目结构」一节。

### 1. 统一采样率

```bash
python resample.py
```
将 `dataset/` 下所有 wav 文件统一重采样到 24 kHz。

### 2. 提取 log-F0（用于 GVAE pitch 监督）

```bash
python preprocess_f0.py
```
每个 `.wav` 会生成一个对应的 `.f0.pt`。

### 3. 共振峰扰动（用于 Speaker Perturbation）

```bash
python preprocess_perturbation.py
```
每个 `.wav` 会生成一个 `_pert.wav`。

### 4. 生成 filelist

```bash
python prepare_filelists.py
```
产出：
- `filelists/unisyn_train.txt`
- `filelists/unisyn_val.txt`

每行格式：
```
<wav_path>|<phone_ids>|<pitch_ids>|<durations>|<pos>|<style_id>|<spk_id>
```
其中 `style_id`: `0=TTS`, `1=SVS`。

---

## 🚀 训练

```bash
python train.py -c configs/unisyn_base.json -m unisyn_base
```

- checkpoint 保存在 `logs/unisyn_base/`
- TensorBoard：
  ```bash
  tensorboard --logdir logs/unisyn_base
  ```

主要损失项（详见 `configs/unisyn_base.json`）：

| 名称 | 系数 | 说明 |
|------|------|------|
| `c_mel` | 60.0 | Mel 重构损失 |
| `c_kl_s` | 12.0 | `z_s` 的 KL |
| `c_kl_rst` | 1.5 | `z_rst` 的 KL |
| `c_gvae_s` | 0.0¹ | GVAE speaker 监督 |
| `c_gvae_p` | 10.0 | GVAE pitch 监督 |
| `c_dur` | 1.5 | Duration 预测 L1 |
| `c_fm` | 2.0 | Feature matching |
| `c_adv` | 2.0 | 对抗损失 |
| `c_pert` | 0.02 | Wasserstein 扰动约束 |

¹ 单说话人场景下建议设为 0；多说话人训练请调为正数。

---

## 🎤 推理

```bash
# TTS：让 SVS 的说话人开口说话
python inference.py \
    --checkpoint logs/unisyn_base/G_100000.pth \
    --style 0 \
    --noise_scale 0.3 \
    --length_scale 1.0 \
    --out_prefix tts_demo

# SVS：让 TTS 的说话人唱歌
python inference.py \
    --checkpoint logs/unisyn_base/G_100000.pth \
    --style 1 \
    --noise_scale 0.2 \
    --length_scale 1.0 \
    --out_prefix svs_demo \
    --do_recon
```

参数说明：
- `--style`: `0` 代表 TTS，`1` 代表 SVS
- `--noise_scale`: 潜变量采样的噪声幅度，越大生成越多样、音色越不稳定
- `--length_scale`: 语速缩放系数（仅 TTS 生效；SVS 由乐谱决定时长）
- `--do_recon`: 同时跑一次 voice conversion 的重建 sanity check

---

## 🔍 与 VITS 的主要差异

| 方面 | VITS | 本项目 |
|------|------|--------|
| 任务 | 单任务 TTS | TTS + SVS 统一 |
| 潜空间 | 单一 `z` | 切分为 `z_s` + `z_rst`（MC-VAE） |
| 对齐 | Monotonic Alignment Search | Length Regulator + Duration Predictor |
| 先验 | 文本编码 + Flow | Text Encoder + Frame Prior Network |
| 解耦 | Flow 隐式 | GVAE 显式 + 共振峰扰动 |
| 输入 | phoneme | phoneme + tone/pitch + dur + pos + style |

---

## 📝 与论文的偏差 / 已知限制

由于论文未开源官方实现，以下细节为复刻推断，可能与原论文有差异：

- **Duration 输入**：论文中 TTS 的 `dur_note` 为占位符；本项目 TTS 训练时 `note_dur_input=0`，SVS 训练时喂入 GT note duration，推理时 SVS 依赖乐谱给定的时长。
- **Pos 特征**：按"音节内相对位置"重新计算（SVS 按音符分组，TTS 按带调韵母边界分组），非论文显式描述。
- **Prior Pitch Supervision**：额外在 FramePriorNetwork 输出端加了一个 pitch predictor 并对其计算 MSE，强化 prior 端音高学习（论文未明说）。
- **FramePriorNetwork**：增加了 `style_id` embedding 输入，让先验网络显式区分说话/唱歌风格。
- **数据规模**：论文使用 2×speaker + 2×singer，本项目目前仅验证了 1 说话人 + 1 歌手的设置。

---

## 🙏 致谢

- [VITS](https://github.com/jaywalnut310/vits) - 基础代码框架
- [UniSyn 论文](https://arxiv.org/abs/2212.01546) - 模型设计
- [Opencpop](https://wenet.org.cn/opencpop/) - 中文歌声数据集
- [DataBaker 中文标准女声](https://www.data-baker.com/open_source.html) - 中文 TTS 数据集
- [WORLD](https://github.com/mmorise/World), [Praat](https://www.fon.hum.uva.nl/praat/) - 信号处理工具

---

## 📄 引用

如果本项目对你有帮助，请考虑引用原始 UniSyn 论文：

```bibtex
@inproceedings{lei2023unisyn,
  title={UniSyn: An End-to-End Unified Model for Text-to-Speech and Singing Voice Synthesis},
  author={Lei, Yi and Yang, Shan and Wang, Xinsheng and Xie, Qicong and Yao, Jixun and Xie, Lei and Su, Dan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

以及 VITS：

```bibtex
@inproceedings{kim2021conditional,
  title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={ICML},
  year={2021}
}
```

---

## ⚠️ 免责声明

本项目为学习与研究用途的非官方复刻，不代表论文作者或任何机构的官方立场。请勿用做商业用途，合成的音频不得用于未授权冒充他人身份、欺诈或任何违法用途。