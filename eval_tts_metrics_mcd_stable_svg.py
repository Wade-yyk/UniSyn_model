
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

# Optional, preferred backends for more standard MCD
HAS_PYWORLD = False
HAS_PYSPTK = False
try:
    import pyworld as pw
    HAS_PYWORLD = True
except Exception:
    pass

try:
    import pysptk
    HAS_PYSPTK = True
except Exception:
    pass


MCD_CONST = 10.0 / math.log(10.0) * math.sqrt(2.0)


def load_audio(path: Path, sr: int, peak_norm: bool) -> np.ndarray:
    wav, file_sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav.astype(np.float64)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    if peak_norm:
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-8:
            wav = wav / peak
    return wav.astype(np.float64)


def trim_silence_pair(gt: np.ndarray, pred: np.ndarray, top_db: float) -> Tuple[np.ndarray, np.ndarray]:
    def _trim(x: np.ndarray) -> np.ndarray:
        yt, _ = librosa.effects.trim(x.astype(np.float32), top_db=top_db)
        if len(yt) == 0:
            return x
        return yt.astype(np.float64)
    return _trim(gt), _trim(pred)


def aligned_waveforms(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(gt), len(pred))
    if n <= 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
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
        y=wav.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return np.log(np.maximum(mel, 1e-10)).astype(np.float64)


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
    gt_mel = gt_mel[:, :t]
    pred_mel = pred_mel[:, :t]
    return float(np.mean((gt_mel - pred_mel) ** 2))


def mfcc_features(wav: np.ndarray, sr: int, n_fft: int, hop: int, win: int, n_mfcc: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=wav.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=40,
        fmin=0.0,
        fmax=None,
    )
    return mfcc.astype(np.float64)


def mcd_mfcc_proxy(
    gt: np.ndarray,
    pred: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    n_mfcc: int,
    exclude_c0: bool,
    use_dtw: bool,
) -> Tuple[float, str]:
    gt_mfcc = mfcc_features(gt, sr, n_fft, hop, win, n_mfcc)
    pred_mfcc = mfcc_features(pred, sr, n_fft, hop, win, n_mfcc)

    if exclude_c0 and gt_mfcc.shape[0] > 1 and pred_mfcc.shape[0] > 1:
        gt_mfcc = gt_mfcc[1:, :]
        pred_mfcc = pred_mfcc[1:, :]

    if gt_mfcc.shape[1] == 0 or pred_mfcc.shape[1] == 0:
        return float("nan"), "librosa_mfcc_proxy"

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
    return float(MCD_CONST * np.mean(dist)), "librosa_mfcc_proxy"


def mcd_world_mcep(
    gt: np.ndarray,
    pred: np.ndarray,
    sr: int,
    hop: int,
    order: int = 24,
    alpha: Optional[float] = None,
    use_dtw: bool = True,
) -> Tuple[float, str]:
    if not (HAS_PYWORLD and HAS_PYSPTK):
        return float("nan"), "world_mcep_unavailable"

    if alpha is None:
        if sr >= 48000:
            alpha = 0.554
        elif sr >= 44100:
            alpha = 0.544
        elif sr >= 24000:
            alpha = 0.466
        elif sr >= 22050:
            alpha = 0.455
        elif sr >= 16000:
            alpha = 0.42
        else:
            alpha = 0.35

    frame_period = hop / sr * 1000.0

    def _mcep(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        _f0, t = pw.harvest(x, sr, frame_period=frame_period)
        sp = pw.cheaptrick(x, _f0, t, sr)
        mc = pysptk.sp2mc(sp, order=order, alpha=alpha)
        return mc.T  # [dim, T]

    gt_mc = _mcep(gt)
    pred_mc = _mcep(pred)

    if gt_mc.shape[1] == 0 or pred_mc.shape[1] == 0:
        return float("nan"), "world_mcep"

    # Remove c0 for distance as is common in MCD practice
    if gt_mc.shape[0] > 1 and pred_mc.shape[0] > 1:
        gt_mc = gt_mc[1:, :]
        pred_mc = pred_mc[1:, :]

    if use_dtw:
        _, wp = librosa.sequence.dtw(X=gt_mc, Y=pred_mc, metric="euclidean")
        wp = np.asarray(wp)[::-1]
        gt_aligned = gt_mc[:, wp[:, 0]]
        pred_aligned = pred_mc[:, wp[:, 1]]
    else:
        t = min(gt_mc.shape[1], pred_mc.shape[1])
        gt_aligned = gt_mc[:, :t]
        pred_aligned = pred_mc[:, :t]

    dist = np.sqrt(np.sum((gt_aligned - pred_aligned) ** 2, axis=0))
    return float(MCD_CONST * np.mean(dist)), "world_mcep"


def stable_mcd(
    gt: np.ndarray,
    pred: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    n_mfcc: int,
    exclude_c0: bool,
    use_dtw: bool,
) -> Tuple[float, str]:
    mcd_val, backend = mcd_world_mcep(gt, pred, sr, hop, order=24, alpha=None, use_dtw=use_dtw)
    if np.isfinite(mcd_val):
        return mcd_val, backend
    return mcd_mfcc_proxy(gt, pred, sr, n_fft, hop, win, n_mfcc, exclude_c0, use_dtw)


def find_wavs(root: Path) -> Dict[str, Path]:
    files = list(root.rglob("*.wav"))
    rel_map: Dict[str, Path] = {}
    stem_map: Dict[str, Path] = {}
    for f in files:
        rel_key = str(f.relative_to(root)).replace("\\", "/")
        rel_map[rel_key] = f
        stem_map.setdefault(f.stem, f)
    merged = dict(stem_map)
    for k, v in rel_map.items():
        merged["rel::" + k] = v
    return merged


def pair_files(gt_dir: Path, pred_dir: Path) -> List[Tuple[str, Path, Path]]:
    gt_files = list(gt_dir.rglob("*.wav"))
    pred_lookup = find_wavs(pred_dir)
    pairs: List[Tuple[str, Path, Path]] = []
    for gt in gt_files:
        rel_str = str(gt.relative_to(gt_dir)).replace("\\", "/")
        rel_key = "rel::" + rel_str
        pred = pred_lookup.get(rel_key)
        if pred is None:
            pred = pred_lookup.get(gt.stem)
        if pred is not None:
            pairs.append((rel_str, gt, pred))
    return pairs


def mean_std_median(values: List[float]) -> Tuple[float, float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    median = float(np.median(vals))
    return mean, std, median


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def eval_one_run(
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
    trim_db: float,
    peak_norm: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    pairs = pair_files(gt_dir, pred_dir)
    rows: List[Dict[str, object]] = []
    mcd_backends: Dict[str, int] = {}
    for rel_name, gt_path, pred_path in pairs:
        try:
            gt = load_audio(gt_path, sr, peak_norm=peak_norm)
            pred = load_audio(pred_path, sr, peak_norm=peak_norm)
            gt, pred = trim_silence_pair(gt, pred, top_db=trim_db)

            wav_mse = waveform_mse(gt, pred)
            mel_val = mel_mse(gt, pred, sr, n_fft, hop, win, n_mels, fmin, fmax)
            mcd_val, backend = stable_mcd(
                gt, pred, sr, n_fft, hop, win, n_mfcc, exclude_c0=exclude_c0, use_dtw=use_dtw
            )
            mcd_backends[backend] = mcd_backends.get(backend, 0) + 1

            rows.append({
                "run_name": run_name,
                "file": rel_name,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "waveform_mse": wav_mse,
                "mel_mse": mel_val,
                "mcd": mcd_val,
                "mcd_backend_used": backend,
                "gt_sec": len(gt) / sr,
                "pred_sec": len(pred) / sr,
                "error": "",
            })
        except Exception as e:
            rows.append({
                "run_name": run_name,
                "file": rel_name,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "waveform_mse": "",
                "mel_mse": "",
                "mcd": "",
                "mcd_backend_used": "",
                "gt_sec": "",
                "pred_sec": "",
                "error": str(e),
            })

    wavs = [float(r["waveform_mse"]) for r in rows if r["waveform_mse"] != ""]
    mels = [float(r["mel_mse"]) for r in rows if r["mel_mse"] != ""]
    mcds = [float(r["mcd"]) for r in rows if r["mcd"] != ""]

    wav_mean, wav_std, wav_med = mean_std_median(wavs)
    mel_mean, mel_std, mel_med = mean_std_median(mels)
    mcd_mean, mcd_std, mcd_med = mean_std_median(mcds)

    dominant_backend = ""
    if mcd_backends:
        dominant_backend = sorted(mcd_backends.items(), key=lambda x: (-x[1], x[0]))[0][0]

    summary = {
        "run_name": run_name,
        "num_pairs": len(rows),
        "waveform_mse_mean": wav_mean,
        "waveform_mse_std": wav_std,
        "waveform_mse_median": wav_med,
        "mel_mse_mean": mel_mean,
        "mel_mse_std": mel_std,
        "mel_mse_median": mel_med,
        "mcd_mean": mcd_mean,
        "mcd_std": mcd_std,
        "mcd_median": mcd_med,
        "mcd_backend_used": dominant_backend,
    }
    return rows, summary


def svg_barplot(
    out_path: Path,
    title: str,
    ylabel: str,
    run_names: List[str],
    means: List[float],
    stds: List[float],
    width: int = 980,
    height: int = 560,
) -> None:
    left, right, top, bottom = 90, 40, 70, 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    finite_vals = [v for v in means + [m + s for m, s in zip(means, stds)] if np.isfinite(v)]
    ymax = max(finite_vals) if finite_vals else 1.0
    ymax = ymax * 1.15 if ymax > 0 else 1.0

    def x_of(i: int, n: int) -> float:
        step = plot_w / max(n, 1)
        return left + i * step + step * 0.2

    def bar_w(n: int) -> float:
        step = plot_w / max(n, 1)
        return step * 0.6

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - max(0.0, min(v / ymax, 1.0)))

    n = len(run_names)
    bw = bar_w(n)
    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>')

    # axes
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black" stroke-width="2"/>')

    # y grid/ticks
    num_ticks = 5
    for i in range(num_ticks + 1):
        v = ymax * i / num_ticks
        y = y_of(v)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-size="14" font-family="Arial">{v:.3f}</text>')

    # ylabel
    parts.append(
        f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" '
        f'text-anchor="middle" font-size="18" font-family="Arial">{ylabel}</text>'
    )

    for i, (name, mean, std) in enumerate(zip(run_names, means, stds)):
        x = x_of(i, n)
        base_y = top + plot_h
        y = y_of(mean if np.isfinite(mean) else 0.0)
        h = base_y - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#4C78A8" opacity="0.85"/>')
        if np.isfinite(mean) and np.isfinite(std):
            y1 = y_of(mean + std)
            y2 = y_of(max(mean - std, 0.0))
            xc = x + bw / 2.0
            parts.append(f'<line x1="{xc:.1f}" y1="{y1:.1f}" x2="{xc:.1f}" y2="{y2:.1f}" stroke="black" stroke-width="2"/>')
            parts.append(f'<line x1="{xc-8:.1f}" y1="{y1:.1f}" x2="{xc+8:.1f}" y2="{y1:.1f}" stroke="black" stroke-width="2"/>')
            parts.append(f'<line x1="{xc-8:.1f}" y1="{y2:.1f}" x2="{xc+8:.1f}" y2="{y2:.1f}" stroke="black" stroke-width="2"/>')
        parts.append(f'<text x="{x + bw/2:.1f}" y="{base_y + 24:.1f}" text-anchor="middle" font-size="14" font-family="Arial">{name}</text>')
        if np.isfinite(mean):
            parts.append(f'<text x="{x + bw/2:.1f}" y="{max(y-8, 50):.1f}" text-anchor="middle" font-size="12" font-family="Arial">{mean:.3f}</text>')

    parts.append('</svg>')
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stable TTS metrics with stronger MCD and SVG plots.")
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
    parser.add_argument("--trim-top-db", type=float, default=35.0)
    parser.add_argument("--peak-norm", action="store_true")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dirs = [Path(p) for p in args.pred_dirs]
    run_names = args.run_names if args.run_names else [p.name for p in pred_dirs]
    if len(run_names) != len(pred_dirs):
        raise SystemExit("--run-names must match number of --pred-dirs")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    per_file_fields = [
        "run_name", "file", "gt_path", "pred_path", "waveform_mse", "mel_mse", "mcd",
        "mcd_backend_used", "gt_sec", "pred_sec", "error"
    ]

    for pred_dir, run_name in zip(pred_dirs, run_names):
        rows, summary = eval_one_run(
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
            trim_db=args.trim_top_db,
            peak_norm=args.peak_norm,
        )
        all_rows.extend(rows)
        summary_rows.append(summary)
        write_csv(outdir / f"{run_name}_metrics_by_file.csv", rows, per_file_fields)

    summary_fields = [
        "run_name", "num_pairs",
        "waveform_mse_mean", "waveform_mse_std", "waveform_mse_median",
        "mel_mse_mean", "mel_mse_std", "mel_mse_median",
        "mcd_mean", "mcd_std", "mcd_median",
        "mcd_backend_used"
    ]
    write_csv(outdir / "summary_by_run.csv", summary_rows, summary_fields)
    write_csv(outdir / "all_metrics_by_file.csv", all_rows, per_file_fields)

    meta = {
        "sample_rate": args.sr,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "mcd_excludes_c0": not args.include_c0,
        "mcd_uses_dtw": not args.no_dtw,
        "trim_top_db": args.trim_top_db,
        "peak_norm": args.peak_norm,
        "has_pyworld": HAS_PYWORLD,
        "has_pysptk": HAS_PYSPTK,
        "runs": summary_rows,
    }
    with open(outdir / "summary_by_run.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    names = [r["run_name"] for r in summary_rows]
    wav_means = [float(r["waveform_mse_mean"]) if r["waveform_mse_mean"] == r["waveform_mse_mean"] else float("nan") for r in summary_rows]
    wav_stds = [float(r["waveform_mse_std"]) if r["waveform_mse_std"] == r["waveform_mse_std"] else 0.0 for r in summary_rows]
    mel_means = [float(r["mel_mse_mean"]) if r["mel_mse_mean"] == r["mel_mse_mean"] else float("nan") for r in summary_rows]
    mel_stds = [float(r["mel_mse_std"]) if r["mel_mse_std"] == r["mel_mse_std"] else 0.0 for r in summary_rows]
    mcd_means = [float(r["mcd_mean"]) if r["mcd_mean"] == r["mcd_mean"] else float("nan") for r in summary_rows]
    mcd_stds = [float(r["mcd_std"]) if r["mcd_std"] == r["mcd_std"] else 0.0 for r in summary_rows]

    svg_barplot(outdir / "waveform_mse_by_run.svg", "Waveform MSE by run", "waveform_mse", names, wav_means, wav_stds)
    svg_barplot(outdir / "mel_mse_by_run.svg", "Mel MSE by run", "mel_mse", names, mel_means, mel_stds)
    svg_barplot(outdir / "mcd_by_run.svg", "Stable MCD by run", "mcd", names, mcd_means, mcd_stds)

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
