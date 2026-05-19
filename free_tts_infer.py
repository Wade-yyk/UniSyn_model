import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write

import utils
from models import SynthesizerTrn
from text.phone_vocab import phone_to_id


INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
    "y", "w",
]

PAUSE_CHARS = set(" \t\r\n,.;:!?，。！？；：、")


def normalize_phone_token(token):
    token = token.strip()
    if token in {"sp", "sil", "pau"}:
        return "SP"
    if token == "AP":
        return "AP"

    m = re.match(r"^([a-z]+)r([1-5])$", token)
    if m:
        token = f"{m.group(1)}{m.group(2)}"

    special_map = {
        "ui": "uei",
        "ui1": "uei1", "ui2": "uei2", "ui3": "uei3", "ui4": "uei4", "ui5": "uei5",
        "un": "uen",
        "un1": "uen1", "un2": "uen2", "un3": "uen3", "un4": "uen4", "un5": "uen5",
        "iu": "iou",
        "iu1": "iou1", "iu2": "iou2", "iu3": "iou3", "iu4": "iou4", "iu5": "iou5",
        "v": "v", "u:": "v", "ü": "v",
        "v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4", "v5": "v5",
        "ve1": "ve1", "ve2": "ve2", "ve3": "ve3", "ve4": "ve4", "ve5": "ve5",
        "ue1": "ve1", "ue2": "ve2", "ue3": "ve3", "ue4": "ve4", "ue5": "ve5",
        "y": "y", "w": "w",
    }
    return special_map.get(token, token)


def split_pinyin_syllable(syllable):
    syllable = syllable.lower().replace("u:", "v").replace("ü", "v")
    if normalize_phone_token(syllable) in phone_to_id:
        return "", normalize_phone_token(syllable)
    for initial in INITIALS:
        if syllable.startswith(initial):
            final = syllable[len(initial):]
            return initial, final
    return "", syllable


def pinyin_to_phone_tokens(pinyin_items):
    tokens = []
    for item in pinyin_items:
        item = item.strip()
        if not item:
            continue
        if item.upper() in {"SP", "AP"}:
            tokens.append(item.upper())
            continue

        initial, final = split_pinyin_syllable(item)
        if initial:
            tokens.append(initial)
        if final:
            tokens.append(normalize_phone_token(final))
    return tokens


def text_to_phone_tokens(text):
    try:
        from pypinyin import Style, pinyin
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pypinyin，不能把中文转成拼音。请先安装 requirements.txt，"
            "或改用 --pinyin/--phones 手动输入。"
        ) from exc

    tokens = []
    for ch in text:
        if ch in PAUSE_CHARS:
            if tokens and tokens[-1] != "SP":
                tokens.append("SP")
            continue
        if ch.isascii() and not ch.isalpha():
            continue

        initial = pinyin(ch, style=Style.INITIALS, strict=False)[0][0]
        final = pinyin(
            ch,
            style=Style.FINALS_TONE3,
            strict=False,
            neutral_tone_with_five=True,
        )[0][0]

        if initial == ch and final == ch:
            raise ValueError(f"无法把字符转换成拼音: {ch!r}")
        if initial:
            tokens.append(normalize_phone_token(initial))
        if final:
            tokens.append(normalize_phone_token(final))

    while tokens and tokens[0] == "SP":
        tokens.pop(0)
    while tokens and tokens[-1] == "SP":
        tokens.pop()
    return tokens


def phone_tokens_to_ids(tokens):
    ids = []
    unknown = []
    for token in tokens:
        token = normalize_phone_token(token)
        if token not in phone_to_id:
            unknown.append(token)
            continue
        ids.append(phone_to_id[token])
    if unknown:
        raise ValueError(f"这些音素不在 text/phone_vocab.py 里: {' '.join(unknown)}")
    if not ids:
        raise ValueError("输入为空，无法推理。")
    return ids


def pitch_ids_from_tokens(tokens):
    ids = []
    for token in tokens:
        m = re.search(r"([1-5])$", token)
        ids.append(int(m.group(1)) if m else 0)
    return ids


def build_inputs(args, device):
    if args.phones:
        tokens = args.phones.strip().split()
    elif args.pinyin:
        tokens = pinyin_to_phone_tokens(args.pinyin.strip().split())
    else:
        tokens = text_to_phone_tokens(args.text)

    phone_ids = phone_tokens_to_ids(tokens)
    pitch_ids = pitch_ids_from_tokens(tokens)
    note_dur = [0] * len(phone_ids)
    pos = [1.0] * len(phone_ids)

    pho = torch.LongTensor(phone_ids).unsqueeze(0).to(device)
    pho_lengths = torch.LongTensor([len(phone_ids)]).to(device)
    pitch = torch.LongTensor(pitch_ids).unsqueeze(0).to(device)
    note_dur = torch.LongTensor(note_dur).unsqueeze(0).to(device)
    pos = torch.FloatTensor(pos).unsqueeze(0).to(device)
    style_id = torch.LongTensor([args.style]).to(device)
    spk_id = torch.LongTensor([args.spk]).to(device)
    return tokens, pho, pho_lengths, pitch, note_dur, pos, style_id, spk_id


def latest_checkpoint(model_dir):
    ckpts = glob.glob(str(Path(model_dir) / "G_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"在 {model_dir} 下找不到 G_*.pth")

    def step(path):
        nums = re.findall(r"\d+", Path(path).stem)
        return int(nums[-1]) if nums else -1

    return max(ckpts, key=step)


def resolve_config_and_checkpoint(args):
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = latest_checkpoint(args.model_dir)

    config = args.config
    if config is None:
        model_config = Path(args.model_dir) / "config.json"
        config = str(model_config if model_config.exists() else Path("configs") / "unisyn_base.json")
    return config, checkpoint


def load_model(config_path, checkpoint_path, device):
    hps = utils.get_hparams_from_file(config_path)
    hps.model.n_speakers = hps.data.n_speakers
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    emb = state_dict.get("enc_p.emb_pho.weight")
    vocab_size = emb.shape[0] if emb is not None else max(phone_to_id.values()) + 1

    net_g = SynthesizerTrn(
        vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)

    utils.load_checkpoint(checkpoint_path, net_g, None)
    net_g.eval()
    return hps, net_g


def save_wav(path, audio_tensor, sampling_rate, max_wav_value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio_tensor[0, 0].detach().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * max_wav_value).clip(-max_wav_value, max_wav_value - 1).astype(np.int16)
    write(str(path), int(sampling_rate), audio)


def infer_one(text, args, hps, net_g, device, index=None):
    args.text = text
    tokens, pho, pho_lengths, pitch, note_dur, pos, style_id, spk_id = build_inputs(args, device)
    max_id = int(pho.max().item())
    if max_id >= net_g.n_vocab:
        raise ValueError(
            f"输入里用到了 id={max_id}，但当前 checkpoint 只训练了 {net_g.n_vocab} 个 phone token。"
            "这通常说明文本触发了刚补进 phone_vocab.py 的新音素；旧模型不能直接使用新 token，"
            "需要用新的 phone_vocab.py 重新生成 filelist 并重新训练。"
        )

    with torch.no_grad():
        audio, mask, _ = net_g.infer(
            pho,
            pho_lengths,
            pitch,
            None if args.free_duration else note_dur,
            pos,
            style_id,
            spk_id,
            noise_scale=args.noise_scale,
            length_scale=args.length_scale,
        )
        y_len = int((mask.sum([1, 2]).long() * hps.data.hop_length)[0].item())
        if y_len > 0:
            audio = audio[:, :, :y_len]

    if index is None:
        out_path = Path(args.out)
    else:
        stem = Path(args.out).stem
        suffix = Path(args.out).suffix or ".wav"
        out_path = Path(args.out).with_name(f"{stem}_{index:03d}{suffix}")

    save_wav(out_path, audio, hps.data.sampling_rate, hps.data.max_wav_value)
    print(f"[done] {text}")
    print(f"[phones] {' '.join(tokens)}")
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Free text TTS inference for the UniSyn/VITS checkpoint.")
    parser.add_argument("--text", default="", help='要合成的中文文本，例如: "我喜欢你"')
    parser.add_argument("--pinyin", default="", help='可选：直接输入带调拼音，例如: "wo3 xi3 huan1 ni3"')
    parser.add_argument("--phones", default="", help='可选：直接输入音素，例如: "w uo3 x i3 h uan1 n i3"')
    parser.add_argument("--config", default=None, help="config.json；默认优先用 --model-dir/config.json")
    parser.add_argument("--checkpoint", default=None, help="G_*.pth；不填则自动取 --model-dir 下最新的 G_*.pth")
    parser.add_argument("--model-dir", default="logs/unisyn_svs_first_edit2")
    parser.add_argument("--out", default="free_tts.wav")
    parser.add_argument("--spk", type=int, default=0)
    parser.add_argument("--style", type=int, default=0, help="TTS 固定用 0")
    parser.add_argument("--noise-scale", type=float, default=0.3)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--free-duration", action="store_true", default=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动改用 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)

    config_path, checkpoint_path = resolve_config_and_checkpoint(args)
    print(f"[config] {config_path}")
    print(f"[checkpoint] {checkpoint_path}")
    hps, net_g = load_model(config_path, checkpoint_path, device)

    if args.text or args.pinyin or args.phones:
        text = args.text or args.pinyin or args.phones
        infer_one(text, args, hps, net_g, device)
        return

    print("进入交互模式。直接输入中文并回车；输入 q/quit/exit 退出。")
    idx = 1
    while True:
        text = input("text> ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue
        infer_one(text, args, hps, net_g, device, index=idx)
        idx += 1


if __name__ == "__main__":
    main()
