import os
import re
from pathlib import Path
from text.phone_vocab import phone_to_id

SVS_SPK_ID = 0
TTS_SPK_ID = 0

SVS_STYLE_ID = 1
TTS_STYLE_ID = 0


# ---------- 通用工具 ----------

SR = 24000
HOP_LENGTH = 300

def compute_pos_from_notes(phones, notes):
    """
    根据音素和对应的音符，计算每个音素在其音符组内的相对位置
    """
    pos = []
    i = 0
    while i < len(phones):
        cur_note = notes[i]
        # 找出属于同一音符的所有音素
        j = i
        while j < len(phones) and notes[j] == cur_note:
            j += 1
        group_size = j - i
        for rank in range(group_size):
            pos.append((rank + 1) / group_size)  # 1/n, 2/n, ..., 1.0
        i = j
    return pos

def sec_to_frames(sec: float) -> int:
    """将秒换算为帧数，至少保证有1帧，防止特征消失"""
    frames = int(round(sec * SR / HOP_LENGTH))
    return max(1, frames)

def normalize_phone_token(t: str) -> str:
    t = t.strip()

    # 停顿统一
    if t in ["sp1", "sp", "sil", "pau"]:
        return "SP"

    # 保留 AP
    if t == "AP":
        return "AP"

    # 儿化规约：uanr1 -> uan1, air4 -> ai4, ar2 -> a2
    m = re.match(r"^([a-z]+)r([1-5])$", t)
    if m:
        base = m.group(1)
        tone = m.group(2)
        t = f"{base}{tone}"

    # 再做一轮特殊归并
    special_map = {
        # 常见无调/缩写
        "ui": "uei",
        "un": "uen",
        "iu": "iou",

        # 儿化后仍然可能残留的特殊形式
        "ir1": "iii1",
        "ir2": "iii2",
        "ir3": "iii3",
        "ir4": "iii4",
        "ir5": "iii5",

        "iiir4": "iii4",

        # 特殊拼写兼容
        "io5": "iou5",
        "iour1": "iou1",
        "ueir1": "uei1",
        "ueir3": "uei3",
        "ueir4": "uei4",

        # y / w 作为介音时直接保留
        "y": "y",
        "w": "w",

        # 特殊音节
        "ng1": "ng1",
        "pl": "SP",   # 先当停顿占位，后面你真想保留再单独加
        "iyl4": "i4", # 先粗略规约，保证能跑
    }

    if t in special_map:
        return special_map[t]

    return t


def phones_to_ids(tokens):
    ids = []
    for t in tokens:
        t = normalize_phone_token(t)
        if t not in phone_to_id:
            raise ValueError(f"Unknown phone token: {t}")
        ids.append(phone_to_id[t])
    return ids

def normalize_note_token(note: str) -> int:
    """
    把:
      C4, D#4/Eb4, A#3/Bb3, rest
    转成整数 pitch id (近似 MIDI)
    """
    note = note.strip()
    if note.lower() == "rest":
        return 0

    if "/" in note:
        note = note.split("/")[0]

    note_map = {
        "C": 0, "C#": 1, "Db": 1,
        "D": 2, "D#": 3, "Eb": 3,
        "E": 4,
        "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10,
        "B": 11
    }

    if len(note) < 2:
        return 0

    if len(note) >= 3 and note[1] in ["#", "b"]:
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[:1]
        octave = int(note[1:])

    midi = (octave + 1) * 12 + note_map[name]
    return midi


def normalize_utt_id(x: str) -> str:
    x = x.strip()
    x = x.replace("\\", "/")

    # 如果这一行是完整 metadata，用第一列
    if "|" in x:
        x = x.split("|")[0].strip()

    # 如果这一列里还带路径或 .wav，再继续清理
    x = Path(x).stem
    return x


def load_id_set(path: str) -> set:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.add(normalize_utt_id(x))
    return ids


# ---------- SVS 解析 ----------

def parse_svs_transcriptions(path: str):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 7:
                continue

            utt_id = normalize_utt_id(parts[0])
            text = parts[1]
            pho = parts[2].split()
            notes = parts[3].split()
            durs = [sec_to_frames(float(x)) for x in parts[4].split()]
            pos = compute_pos_from_notes(pho, notes)

            pitch_ids = [normalize_note_token(x) for x in notes]

            if not (len(pho) == len(pitch_ids) == len(durs) == len(pos)):
                print(f"[WARN][SVS] length mismatch: {utt_id}")
                continue

            data[utt_id] = {
                "text": text,
                "pho": pho,
                "pitch": pitch_ids,
                "dur": durs,
                "pos": pos
            }
    return data


def build_svs_lines():
    meta = parse_svs_transcriptions("dataset/svs/meta/transcriptions.txt")
    train_ids = load_id_set("dataset/svs/meta/train.txt")
    val_ids = load_id_set("dataset/svs/meta/test.txt")

    print(f"[SVS] total transcription items: {len(meta)}")
    print(f"[SVS] train ids: {len(train_ids)}")
    print(f"[SVS] val ids:   {len(val_ids)}")

    train_lines = []
    val_lines = []

    for utt_id, item in meta.items():
        wav_path = f"dataset/svs/wavs/{utt_id}.wav"
        if not os.path.exists(wav_path):
            print(f"[WARN][SVS] wav missing: {wav_path}")
            continue

        if not (len(item['pho']) == len(item['pitch']) == len(item['dur']) == len(item['pos'])):
            print(f"[WARN][SVS] length mismatch: {utt_id}")
            continue

        phone_ids = phones_to_ids(item["pho"])

        line = "|".join([
            wav_path,
            " ".join(map(str, phone_ids)),
            " ".join(map(str, item["pitch"])),
            " ".join(map(str, item["dur"])),
            " ".join(map(str, item["pos"])),
            str(SVS_STYLE_ID),
            str(SVS_SPK_ID),
        ])

        if utt_id in train_ids:
            train_lines.append(line)
        elif utt_id in val_ids:
            val_lines.append(line)
        else:
            print(f"[WARN][SVS] utt_id not found in split files: {utt_id}")

    print(f"[SVS] final train lines: {len(train_lines)}")
    print(f"[SVS] final val lines:   {len(val_lines)}")
    return train_lines, val_lines


# ---------- TTS 解析 ----------

def parse_prosody_txt(path: str):
    """
    解析 000001-010000.txt
    形式大致是：
      000001\t文本
      \tka2 er2 pu3 ...
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.rstrip("\n") for x in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip("\n")
        if not line.strip():
            i += 1
            continue

        if "\t" in line:
            parts = line.split("\t")
            utt_id = parts[0].strip()

            # 下一行通常是拼音
            if i + 1 < len(lines):
                py_line = lines[i + 1].strip()
                if py_line.startswith("\t"):
                    py_line = py_line[1:].strip()
                pinyins = py_line.split()
                mapping[utt_id] = pinyins
                i += 2
            else:
                i += 1
        else:
            i += 1

    return mapping


def parse_interval_file(path: str):
    """
    解析 Praat TextGrid 文本格式的 .interval
    返回:
      phones: list[str]
      durs: list[float]
    只保留非 sil 的 phone
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]

    phones = []
    durs = []

    # 找出所有带引号的 phone token，对应前两个数是 start/end
    # 你的文件是：
    # start
    # end
    # "phone"
    for i in range(len(lines) - 2):
        try:
            start = float(lines[i])
            end = float(lines[i + 1])
            phone = lines[i + 2]
        except Exception:
            continue

        if not (phone.startswith('"') and phone.endswith('"')):
            continue

        phone = phone[1:-1].strip()
        start_frame = int(round(start * SR / HOP_LENGTH))
        end_frame = int(round(end * SR / HOP_LENGTH))
        dur = max(1, end_frame - start_frame)

        phones.append(phone)
        durs.append(dur)

    return phones, durs


def phone_to_pitch_id(phone: str) -> int:
    """
    如果 phone 末尾有声调数字，就取该数字。
    比如:
      a2 -> 2
      er2 -> 2
      u3 -> 3
      k -> 0
      p -> 0
    """
    m = re.search(r'([0-9])$', phone)
    if m:
        return int(m.group(1))
    return 0

def compute_pos_from_syllables(phones):
    """
    TTS: 按拼音音节边界计算相对位置
    声母（辅音、无数字结尾）+ 韵母（带数字结尾）构成一个音节
    """
    pos = []
    i = 0
    while i < len(phones):
        # 找出当前音节的范围：从 i 开始，直到遇到带数字的音素（韵母）
        j = i
        while j < len(phones):
            if re.search(r'[0-9]$', phones[j]):  # 遇到带声调的韵母
                j += 1
                break
            j += 1
        group_size = j - i
        if group_size == 0:
            group_size = 1
            j = i + 1
        for rank in range(group_size):
            pos.append((rank + 1) / group_size)
        i = j
    return pos

def build_tts_lines():
    # 目前你的 ProsodyLabeling 看起来集中在一个大文件里
    prosody_map = parse_prosody_txt("dataset/tts/meta/ProsodyLabeling/000001-010000.txt")

    interval_files = sorted(Path("dataset/tts/meta/PhoneLabeling").glob("*.interval"))

    all_lines = []

    for interval_path in interval_files:
        utt_id = interval_path.stem  # 000001
        wav_path = f"dataset/tts/wavs/{utt_id}.wav"
        if not os.path.exists(wav_path):
            print(f"[WARN][TTS] wav missing: {wav_path}")
            continue

        phones, durs = parse_interval_file(str(interval_path))
        if len(phones) == 0:
            print(f"[WARN][TTS] empty phones: {utt_id}")
            continue

        # pitch 直接从 phone 的 tone 数字取
        pitch_ids = [phone_to_pitch_id(p) for p in phones]

        # pos 先简单全 1
        pos = compute_pos_from_syllables(phones)

        if not (len(phones) == len(pitch_ids) == len(durs) == len(pos)):
            print(f"[WARN][TTS] length mismatch: {utt_id}")
            continue

        phone_ids = phones_to_ids(phones)

        line = "|".join([
            wav_path,
            " ".join(map(str, phone_ids)),
            " ".join(map(str, pitch_ids)),
            " ".join(map(str, durs)),
            " ".join(map(str, pos)),
            str(TTS_STYLE_ID),
            str(TTS_SPK_ID),
        ])
        all_lines.append(line)

    # 简单切 95/5 做 train/val
    n = len(all_lines)
    split = max(1, int(n * 0.95))
    train_lines = all_lines[:split]
    val_lines = all_lines[split:]

    return train_lines, val_lines


# ---------- 主函数 ----------

def main():
    os.makedirs("filelists", exist_ok=True)

    tts_train, tts_val = build_tts_lines()
    svs_train, svs_val = build_svs_lines()

    train_lines = tts_train + svs_train
    val_lines = tts_val + svs_val

    with open("filelists/unisyn_train.txt", "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open("filelists/unisyn_val.txt", "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print(f"[OK] train lines: {len(train_lines)}")
    print(f"[OK] val lines:   {len(val_lines)}")


if __name__ == "__main__":
    main()