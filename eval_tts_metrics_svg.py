from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "Missing dependency. Please install: pip install librosa soundfile numpy"
    ) from e

MCD_CONST = 10.0 / math.log(10.0) * math.sqrt(2.0)


def load_audio(path: Path, sr: int) -> np.ndarray:
    wav, file_sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav.astype(np.float32)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / max(peak, 1e-8)
    return wav


def aligned_waveforms(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(gt), len(pred))
    if n <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return gt[:n], pred[:n]


def waveform_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_a, pred_a = aligned_waveforms(gt, pred)
    return float(np.mean((gt_a - pred_a) ** 2))


def log_mel(
    wav: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


def mel_mse(
    gt: np.ndarray,
    pred: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
) -> float:
    gt_mel = log_mel(gt, sr, n_fft, hop, win, n_mels, fmin, fmax)
    pred_mel = log_mel(pred, sr, n_fft, hop, win, n_mels, fmin, fmax)
    t = min(gt_mel.shape[1], pred_mel.shape[1])
    if t <= 0:
        return float("nan")
    return float(np.mean((gt_mel[:, :t] - pred_mel[:, :t]) ** 2))


def mfcc_features(
    wav: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    n_mfcc: int,
    fmin: float,
    fmax: Optional[float],
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=40,
        fmin=fmin,
        fmax=fmax,
    )
    return mfcc.astype(np.float32)


def mcd(
    gt: np.ndarray,
    pred: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    n_mfcc: int,
    exclude_c0: bool,
    use_dtw: bool,
) -> float:
    gt_mfcc = mfcc_features(gt, sr, n_fft, hop, win, n_mfcc, 0.0, None)
    pred_mfcc = mfcc_features(pred, sr, n_fft, hop, win, n_mfcc, 0.0, None)
    if exclude_c0 and gt_mfcc.shape[0] > 1 and pred_mfcc.shape[0] > 1:
        gt_mfcc = gt_mfcc[1:, :]
        pred_mfcc = pred_mfcc[1:, :]
    if gt_mfcc.shape[1] == 0 or pred_mfcc.shape[1] == 0:
        return float("nan")
    if use_dtw:
        _, wp = librosa.sequence.dtw(X=gt_mfcc, Y=pred_mfcc, metric="euclidean")
        wp = np.asarray(wp)[::-1]
        gt_aligned = gt_mfcc[:, wp[:, 0]]
        pred_aligned = pred_mfcc[:, wp[:, 1]]
    else:
        t = min(gt_mfcc.shape[1], pred_mfcc.shape[1])
        gt_aligned = gt_mfcc[:, :t]
        pred_aligned = pred_mfcc[:, :t]
    dist = np.sqrt(np.sum((gt_aligned - pred_aligned) ** 2, axis=0))
    return float(MCD_CONST * np.mean(dist))


def find_wavs(root: Path) -> Dict[str, Path]:
    files = list(root.rglob("*.wav"))
    rel_map: Dict[str, Path] = {}
    stem_map: Dict[str, Path] = {}
    for f in files:
        rel_key = str(f.relative_to(root)).replace("\\", "/")
        rel_map[rel_key] = f
        stem_map.setdefault(f.stem, f)
    merged = {**stem_map}
    for k, v in rel_map.items():
        merged[f"rel::{k}"] = v
    return merged


def pair_files(gt_dir: Path, pred_dir: Path) -> List[Tuple[str, Path, Path]]:
    gt_files = list(gt_dir.rglob("*.wav"))
    pred_lookup = find_wavs(pred_dir)
    pairs: List[Tuple[str, Path, Path]] = []
    for gt in gt_files:
        rel_part = str(gt.relative_to(gt_dir)).replace("\\", "/")
        rel_key = f"rel::{rel_part}"
        pred = pred_lookup.get(rel_key)
        if pred is None:
            pred = pred_lookup.get(gt.stem)
        if pred is not None:
            pairs.append((rel_part, gt, pred))
    return pairs


def finite_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_svg_barplot(summary_rows: List[Dict[str, object]], metric: str, out_path: Path) -> None:
    width, height = 1000, 600
    ml, mr, mt, mb = 90, 40, 60, 130
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    vals = [float(r.get(f"{metric}_mean", float("nan"))) for r in summary_rows]
    errs = [float(r.get(f"{metric}_std", 0.0)) for r in summary_rows]
    labels = [str(r.get("run_name", "")) for r in summary_rows]
    finite_upper = [v + e for v, e in zip(vals, errs) if np.isfinite(v + e)]
    y_max = max(finite_upper) if finite_upper else 1.0
    y_max = 1.0 if y_max <= 0 else y_max * 1.15

    n = max(len(labels), 1)
    group_w = plot_w / n
    bar_w = group_w * 0.55

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="30" font-size="24" text-anchor="middle">{svg_escape(metric)} by run</text>')

    # axes
    x0, y0 = ml, mt + plot_h
    x1, y1 = ml + plot_w, mt
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black" stroke-width="2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black" stroke-width="2"/>')

    # y ticks
    ticks = 5
    for i in range(ticks + 1):
        frac = i / ticks
        y = y0 - frac * plot_h
        val = frac * y_max
        parts.append(f'<line x1="{x0-6}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{x0-10}" y="{y+5:.1f}" font-size="14" text-anchor="end">{val:.3g}</text>')

    # bars
    for idx, (label, v, e) in enumerate(zip(labels, vals, errs)):
        cx = ml + group_w * (idx + 0.5)
        bx = cx - bar_w / 2
        if np.isfinite(v):
            bh = (v / y_max) * plot_h
            by = y0 - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="#4C78A8"/>')
            if np.isfinite(e) and e > 0:
                ey1 = y0 - ((v - e) / y_max) * plot_h
                ey2 = y0 - ((v + e) / y_max) * plot_h
                ey1 = max(y1, min(y0, ey1))
                ey2 = max(y1, min(y0, ey2))
                parts.append(f'<line x1="{cx:.1f}" y1="{ey1:.1f}" x2="{cx:.1f}" y2="{ey2:.1f}" stroke="black" stroke-width="2"/>')
                parts.append(f'<line x1="{cx-10:.1f}" y1="{ey1:.1f}" x2="{cx+10:.1f}" y2="{ey1:.1f}" stroke="black" stroke-width="2"/>')
                parts.append(f'<line x1="{cx-10:.1f}" y1="{ey2:.1f}" x2="{cx+10:.1f}" y2="{ey2:.1f}" stroke="black" stroke-width="2"/>')
            parts.append(f'<text x="{cx:.1f}" y="{by-8:.1f}" font-size="12" text-anchor="middle">{v:.3g}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y0+22:.1f}" font-size="13" text-anchor="end" transform="rotate(-35 {cx:.1f},{y0+22:.1f})">{svg_escape(label)}</text>')

    parts.append(f'<text x="{ml + plot_w/2:.1f}" y="{height-25}" font-size="18" text-anchor="middle">Run</text>')
    parts.append(f'<text x="25" y="{mt + plot_h/2:.1f}" font-size="18" text-anchor="middle" transform="rotate(-90 25,{mt + plot_h/2:.1f})">{svg_escape(metric)}</text>')
    parts.append('</svg>')
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_run(
    gt_dir: Path,
    pred_dir: Path,
    run_name: str,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    n_mels: int,
    n_mfcc: int,
    fmin: float,
    fmax: Optional[float],
    exclude_c0: bool,
    use_dtw: bool,
) -> List[Dict[str, object]]:
    pairs = pair_files(gt_dir, pred_dir)
    rows: List[Dict[str, object]] = []
    for key, gt_path, pred_path in pairs:
        row: Dict[str, object] = {
            "run_name": run_name,
            "file": key,
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
        }
        try:
            gt = load_audio(gt_path, sr)
            pred = load_audio(pred_path, sr)
            row.update(
                {
                    "waveform_mse": waveform_mse(gt, pred),
                    "mel_mse": mel_mse(gt, pred, sr, n_fft, hop, win, n_mels, fmin, fmax),
                    "mcd": mcd(gt, pred, sr, n_fft, hop, win, n_mfcc, exclude_c0, use_dtw),
                    "gt_sec": len(gt) / sr,
                    "pred_sec": len(pred) / sr,
                    "error": "",
                }
            )
        except Exception as e:
            row.update(
                {
                    "waveform_mse": float("nan"),
                    "mel_mse": float("nan"),
                    "mcd": float("nan"),
                    "gt_sec": float("nan"),
                    "pred_sec": float("nan"),
                    "error": str(e),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TTS metrics and output CSV/JSON/SVG plots without matplotlib.")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dirs", nargs="+", required=True)
    parser.add_argument("--run-names", nargs="*", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--include-c0", action="store_true")
    parser.add_argument("--no-dtw", action="store_true")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dirs = [Path(p) for p in args.pred_dirs]
    run_names = args.run_names or [p.name for p in pred_dirs]
    if len(run_names) != len(pred_dirs):
        raise SystemExit("--run-names must match number of --pred-dirs")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    file_fields = [
        "run_name", "file", "gt_path", "pred_path", "waveform_mse", "mel_mse", "mcd", "gt_sec", "pred_sec", "error"
    ]

    for pred_dir, run_name in zip(pred_dirs, run_names):
        rows = evaluate_run(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            run_name=run_name,
            sr=args.sr,
            n_fft=args.n_fft,
            hop=args.hop_length,
            win=args.win_length,
            n_mels=args.n_mels,
            n_mfcc=args.n_mfcc,
            fmin=args.fmin,
            fmax=args.fmax,
            exclude_c0=not args.include_c0,
            use_dtw=not args.no_dtw,
        )
        all_rows.extend(rows)
        write_csv(outdir / f"{run_name}_metrics_by_file.csv", file_fields, rows)

        wave_stats = finite_stats([float(r["waveform_mse"]) for r in rows])
        mel_stats = finite_stats([float(r["mel_mse"]) for r in rows])
        mcd_stats = finite_stats([float(r["mcd"]) for r in rows])
        summary_rows.append(
            {
                "run_name": run_name,
                "num_pairs": len(rows),
                "waveform_mse_mean": wave_stats["mean"],
                "waveform_mse_std": wave_stats["std"],
                "waveform_mse_median": wave_stats["median"],
                "mel_mse_mean": mel_stats["mean"],
                "mel_mse_std": mel_stats["std"],
                "mel_mse_median": mel_stats["median"],
                "mcd_mean": mcd_stats["mean"],
                "mcd_std": mcd_stats["std"],
                "mcd_median": mcd_stats["median"],
            }
        )

    summary_fields = list(summary_rows[0].keys()) if summary_rows else [
        "run_name", "num_pairs", "waveform_mse_mean", "waveform_mse_std", "waveform_mse_median",
        "mel_mse_mean", "mel_mse_std", "mel_mse_median", "mcd_mean", "mcd_std", "mcd_median"
    ]
    write_csv(outdir / "summary_by_run.csv", summary_fields, summary_rows)
    write_csv(outdir / "all_metrics_by_file.csv", file_fields, all_rows)

    summary_json = {
        "sample_rate": args.sr,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "mcd_excludes_c0": not args.include_c0,
        "mcd_uses_dtw": not args.no_dtw,
        "runs": summary_rows,
    }
    (outdir / "summary_by_run.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    if summary_rows:
        save_svg_barplot(summary_rows, "waveform_mse", outdir / "waveform_mse_by_run.svg")
        save_svg_barplot(summary_rows, "mel_mse", outdir / "mel_mse_by_run.svg")
        save_svg_barplot(summary_rows, "mcd", outdir / "mcd_by_run.svg")

    print(f"[done] results saved to: {outdir.resolve()}")
    print("[done] created:")
    print("  - per-run *_metrics_by_file.csv")
    print("  - all_metrics_by_file.csv")
    print("  - summary_by_run.csv")
    print("  - summary_by_run.json")
    print("  - waveform_mse_by_run.svg")
    print("  - mel_mse_by_run.svg")
    print("  - mcd_by_run.svg")


if __name__ == "__main__":
    main()
