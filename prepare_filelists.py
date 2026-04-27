import os
import re
from pathlib import Path
from text.phone_vocab import phone_to_id

# 建议改成两个不同的 spk_id，让 MC-VAE 的 speaker 子空间真正起作用。
# 如果你暂时只想跑通，保持 0/0 + n_speakers=1 也能训，但论文里那套解耦故事就不成立。
SVS_SPK_ID = 1
TTS_SPK_ID = 0

SVS_STYLE_ID = 1
TTS_STYLE_ID = 0


# ---------- 通用工具 ----------

SR = 24000
HOP_LENGTH = 300

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

    special_map = {
        "ui": "uei",
        "un": "uen",
        "iu": "iou",
        "ir1": "iii1", "ir2": "iii2", "ir3": "iii3", "ir4": "iii4", "ir5": "iii5",
        "iiir4": "iii4",
        "io5": "iou5",
        "iour1": "iou1",
        "ueir1": "uei1", "ueir3": "uei3", "ueir4": "uei4",
        "y": "y", "w": "w",
        "ng1": "ng1",
        "pl": "SP",
        "iyl4": "i4",
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
        "B": 11,
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
    if "|" in x:
        x = x.split("|")[0].strip()
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


# ---------- 计算 pos ----------

def compute_pos_by_note(notes_str_list, note_durs_sec):
    """
    把连续相同 note + 相同 note_dur 的音素视为一个 syllable group，
    pos = (rank_in_group) / group_size，范围 (0, 1]。
    论文里说 pos 是 "rank / total"，这里就是这个意思。
    """
    n = len(notes_str_list)
    pos = [0.0] * n
    i = 0
    while i < n:
        j = i
        while (j < n
               and notes_str_list[j] == notes_str_list[i]
               and abs(note_durs_sec[j] - note_durs_sec[i]) < 1e-6):
            j += 1
        group_size = j - i
        for k in range(group_size):
            pos[i + k] = (k + 1) / group_size
        i = j
    return pos


# ---------- SVS 解析 ----------

def parse_svs_transcriptions(path: str):
    """
    Opencpop transcriptions.txt 每行7列：
      0: utt_id
      1: text
      2: phonemes (空格分隔)
      3: notes (per phoneme)
      4: note_durations  (秒)        <-- 之前被错当成 phoneme_dur
      5: phoneme_durations (秒)      <-- 之前被错当成 pos
      6: slur tag
    """
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

            note_durs_sec = [float(x) for x in parts[4].split()]   # 音符时长（秒）
            pho_durs_sec  = [float(x) for x in parts[5].split()]   # 音素时长（秒）

            note_dur_frames  = [sec_to_frames(d) for d in note_durs_sec]
            align_dur_frames = [sec_to_frames(d) for d in pho_durs_sec]

            pitch_ids = [normalize_note_token(x) for x in notes]
            pos = compute_pos_by_note(notes, note_durs_sec)

            if not (len(pho) == len(pitch_ids) == len(align_dur_frames) == len(note_dur_frames) == len(pos)):
                print(f"[WARN][SVS] length mismatch: {utt_id}")
                continue

            data[utt_id] = {
                "text": text,
                "pho": pho,
                "pitch": pitch_ids,
                "note_dur": note_dur_frames,    # 喂给 dp 的 note 条件
                "align_dur": align_dur_frames,  # 喂给 length regulator 的真实音素时长
                "pos": pos,
            }
    return data


def build_svs_lines():
    meta = parse_svs_transcriptions("dataset/svs/meta/transcriptions.txt")
    train_ids = load_id_set("dataset/svs/meta/train.txt")
    val_ids = load_id_set("dataset/svs/meta/test.txt")

    print(f"[SVS] total transcription items: {len(meta)}")
    print(f"[SVS] train ids: {len(train_ids)}")
    print(f"[SVS] val ids:   {len(val_ids)}")

    train_lines, val_lines = [], []

    for utt_id, item in meta.items():
        wav_path = f"dataset/svs/wavs/{utt_id}.wav"
        if not os.path.exists(wav_path):
            print(f"[WARN][SVS] wav missing: {wav_path}")
            continue

        phone_ids = phones_to_ids(item["pho"])

        # 8 列：wav | pho | pitch | note_dur | align_dur | pos | style | spk
        line = "|".join([
            wav_path,
            " ".join(map(str, phone_ids)),
            " ".join(map(str, item["pitch"])),
            " ".join(map(str, item["note_dur"])),
            " ".join(map(str, item["align_dur"])),
            " ".join(f"{x:.4f}" for x in item["pos"]),
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
      durs:   list[int]   (帧数)
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]

    phones = []
    durs = []

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
    TTS 的 'tp' 用声调数字 1~5；轻声/无声调当 0。
    注意：和 SVS 的 MIDI（一般 >=36）不重叠，可以共用同一个 emb_pitch。
    """
    m = re.search(r'([0-9])$', phone)
    if m:
        return int(m.group(1))
    return 0


def build_tts_lines():
    prosody_map = parse_prosody_txt("dataset/tts/meta/ProsodyLabeling/000001-010000.txt")

    interval_files = sorted(Path("dataset/tts/meta/PhoneLabeling").glob("*.interval"))

    all_lines = []

    for interval_path in interval_files:
        utt_id = interval_path.stem
        wav_path = f"dataset/tts/wavs/{utt_id}.wav"
        if not os.path.exists(wav_path):
            print(f"[WARN][TTS] wav missing: {wav_path}")
            continue

        phones, durs = parse_interval_file(str(interval_path))
        if len(phones) == 0:
            print(f"[WARN][TTS] empty phones: {utt_id}")
            continue

        pitch_ids = [phone_to_pitch_id(p) for p in phones]

        # TTS 没有真正的 note dur，全 0 占位喂给 dp
        note_dur = [0] * len(phones)

        # TTS 的 pos：没有 textgrid 给的 syllable 边界，做个粗略近似
        # 这里把每个音素当作一个独立 syllable -> pos 全为 1.0
        # 想做更细可以按 prosody_map 的 pinyin 切，但不是阻塞项，先简单处理
        pos = [1.0] * len(phones)

        if not (len(phones) == len(pitch_ids) == len(durs) == len(note_dur) == len(pos)):
            print(f"[WARN][TTS] length mismatch: {utt_id}")
            continue

        phone_ids = phones_to_ids(phones)

        # 8 列：wav | pho | pitch | note_dur | align_dur | pos | style | spk
        line = "|".join([
            wav_path,
            " ".join(map(str, phone_ids)),
            " ".join(map(str, pitch_ids)),
            " ".join(map(str, note_dur)),
            " ".join(map(str, durs)),
            " ".join(f"{x:.4f}" for x in pos),
            str(TTS_STYLE_ID),
            str(TTS_SPK_ID),
        ])
        all_lines.append(line)

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