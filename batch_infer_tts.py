import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

# Project imports (must be run from the user's project root, or with PYTHONPATH set)
import utils
from models import SynthesizerTrn
from text.phone_vocab import phone_to_id


def _recursive_to_obj(d):
    if isinstance(d, dict):
        return type('HParams', (), {k: _recursive_to_obj(v) for k, v in d.items()})()
    return d


def load_hparams(config_path: Path):
    if hasattr(utils, "get_hparams_from_file"):
        return utils.get_hparams_from_file(str(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        return _recursive_to_obj(json.load(f))


def try_load_checkpoint(ckpt_path: Path, model):
    # First try project helper
    if hasattr(utils, "load_checkpoint"):
        try:
            utils.load_checkpoint(str(ckpt_path), model, None)
            return
        except Exception:
            pass

    # Fallback: common checkpoint layouts
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict):
        if "model" in obj:
            state = obj["model"]
        elif "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj
    else:
        state = obj

    # Strip possible 'module.' prefix
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)


class SimpleTTSFilelistDataset(Dataset):
    def __init__(self, filelist_path: Path, task: str = "tts"):
        self.items = []
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 7:
                    continue
                wav_path = parts[0].strip()
                if not wav_path.endswith(".wav"):
                    # skip malformed/header-like lines
                    continue
                norm = wav_path.replace("\\", "/").lower()
                if task == "tts" and "/tts/" not in norm:
                    continue
                if task == "svs" and "/svs/" not in norm:
                    continue

                try:
                    pho = [int(x) for x in parts[1].strip().split() if x]
                    pitch = [int(float(x)) for x in parts[2].strip().split() if x]
                    align_dur = [int(float(x)) for x in parts[3].strip().split() if x]
                    # parts[4] exists but infer() in your train.py does not use note_dur_input
                    style_id = int(parts[5].strip())
                    spk_id = int(parts[6].strip())
                except Exception:
                    continue

                if len(pho) == 0:
                    continue
                if len(pitch) != len(pho) or len(align_dur) != len(pho):
                    # Keep only well-formed examples
                    continue

                pos = list(range(1, len(pho) + 1))
                self.items.append({
                    "wav_path": wav_path,
                    "pho": pho,
                    "pitch": pitch,
                    "align_dur": align_dur,
                    "pos": pos,
                    "style_id": style_id,
                    "spk_id": spk_id,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "wav_path": it["wav_path"],
            "pho": torch.LongTensor(it["pho"]),
            "pitch": torch.LongTensor(it["pitch"]),
            "align_dur": torch.LongTensor(it["align_dur"]),
            "pos": torch.LongTensor(it["pos"]),
            "style_id": torch.LongTensor([it["style_id"]]),
            "spk_id": torch.LongTensor([it["spk_id"]]),
            "pho_len": len(it["pho"]),
        }


def collate_fn(batch):
    assert len(batch) == 1, "This script currently uses batch_size=1 for simplicity and safety."
    b = batch[0]
    return {
        "wav_path": b["wav_path"],
        "pho": b["pho"].unsqueeze(0),
        "pho_lengths": torch.LongTensor([b["pho_len"]]),
        "pitch": b["pitch"].unsqueeze(0),
        "align_dur": b["align_dur"].unsqueeze(0),
        "pos": b["pos"].unsqueeze(0),
        "style_id": b["style_id"],
        "spk_id": b["spk_id"],
    }


def save_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr)


def main():
    parser = argparse.ArgumentParser(description="Batch inference for TTS-only examples from UniSyn/VITS-style checkpoints.")
    parser.add_argument("--project-root", required=True, help="Project root so filelist wav paths like dataset/tts/wavs/... resolve correctly")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to G_*.pth or G_final_*.pth")
    parser.add_argument("--filelist", required=True, help="Validation/test filelist, e.g. filelists/unisyn_val.txt")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--task", default="tts", choices=["tts", "svs", "all"], help="Filter entries by task")
    parser.add_argument("--length-scale", type=float, default=1.2)
    parser.add_argument("--max-items", type=int, default=0, help="Optional limit for quick tests; 0 means all")
    parser.add_argument("--run-name", default="", help="Name for prediction subfolder; default inferred from checkpoint name")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    filelist_path = Path(args.filelist).resolve()
    outdir = Path(args.outdir).resolve()
    run_name = args.run_name.strip() or ckpt_path.stem

    gt_dir = outdir / "gt"
    pred_dir = outdir / "pred" / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    hps = load_hparams(config_path)
    current_spks = getattr(hps.data, 'n_speakers', 1)
    setattr(hps.model, 'n_speakers', current_spks)

    vocab_size = max(phone_to_id.values()) + 1
    segment_frames = hps.train.segment_size // hps.data.hop_length
    net_g = SynthesizerTrn(
        vocab_size,
        hps.data.filter_length // 2 + 1,
        segment_frames,
        **vars(hps.model) if hasattr(hps.model, '__dict__') else hps.model
    ).cuda().eval()
    try_load_checkpoint(ckpt_path, net_g)
    net_g.eval()

    dataset = SimpleTTSFilelistDataset(filelist_path, task=args.task)
    if len(dataset) == 0:
        raise RuntimeError(f"No usable items found in {filelist_path} for task={args.task}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    manifest_path = outdir / f"manifest_{run_name}.csv"
    count = 0
    skipped = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "source_wav", "gt_wav", "pred_wav", "task", "checkpoint", "length_scale"])

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if args.max_items and count >= args.max_items:
                    break
                wav_rel = batch["wav_path"]
                wav_src = (project_root / wav_rel).resolve()
                if not wav_src.exists():
                    print(f"[warn] missing GT wav, skip: {wav_src}")
                    skipped += 1
                    continue

                pho = batch["pho"].cuda(non_blocking=True)
                pho_lengths = batch["pho_lengths"].cuda(non_blocking=True)
                pitch = batch["pitch"].cuda(non_blocking=True)
                align_dur = batch["align_dur"].cuda(non_blocking=True)
                pos = batch["pos"].cuda(non_blocking=True)
                style_id = batch["style_id"].cuda(non_blocking=True)
                spk_id = batch["spk_id"].cuda(non_blocking=True)

                try:
                    y_hat, mask, _ = net_g.infer(
                        pho, pho_lengths, pitch,
                        align_dur, pos, style_id, spk_id,
                        length_scale=args.length_scale
                    )
                    y_len = int((mask.sum([1, 2]).long() * hps.data.hop_length)[0].item())
                    if y_len <= 0:
                        print(f"[warn] inferred zero length for {wav_rel}")
                        skipped += 1
                        continue
                    audio = y_hat[0, 0, :y_len].detach().cpu().numpy()
                except Exception as e:
                    print(f"[warn] inference failed for {wav_rel}: {e}")
                    skipped += 1
                    continue

                base = Path(wav_rel).name
                pred_path = pred_dir / base
                gt_path = gt_dir / base
                save_wav(pred_path, audio, int(hps.data.sampling_rate))
                if not gt_path.exists():
                    shutil.copy2(wav_src, gt_path)

                task_name = "tts" if "/tts/" in wav_rel.replace("\\", "/").lower() else ("svs" if "/svs/" in wav_rel.replace("\\", "/").lower() else "unknown")
                writer.writerow([idx, wav_rel, str(gt_path), str(pred_path), task_name, str(ckpt_path), args.length_scale])
                count += 1
                if count % 20 == 0:
                    print(f"[info] generated {count} files...")

    summary = {
        "project_root": str(project_root),
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "filelist": str(filelist_path),
        "task": args.task,
        "run_name": run_name,
        "generated_count": count,
        "skipped_count": skipped,
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "manifest_csv": str(manifest_path),
    }
    with open(outdir / f"summary_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[done]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


#python batch_infer_tts.py --project-root "D:\Vits\baseline\vits" --config "D:\Vits\baseline\vits\logs\unisyn_base_2\config.json" --checkpoint "D:\Vits\baseline\vits\logs\unisyn_base_2\G_65000.pth" --filelist "D:\Vits\baseline\vits\filelists\unisyn_val.txt" --outdir "D:\Vits\baseline\vits\eval_pack" --task tts --length-scale 1.2