import json
import shutil
import subprocess
from pathlib import Path

BASE_CONFIG = Path("configs/unisyn_base.json")

TRIALS = [
    ("unisyn_base_3", 1235),
    ("unisyn_base_4", 1236),
    ("unisyn_base_5", 1237),
    ("unisyn_base_6", 1238),
]

SRC_MODEL_DIR = Path("logs/unisyn_base_2")
SRC_G = SRC_MODEL_DIR / "G_13000.pth"
SRC_D = SRC_MODEL_DIR / "D_13000.pth"


def main():
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    use_branch_ckpt = False

    if use_branch_ckpt:
        print(f"Found base checkpoint: {SRC_G} and {SRC_D}")
        print("Trials will continue from the copied 13000-step checkpoint.")
    else:
        print("Base checkpoint not found.")
        print("Trials will start from scratch.")

    for model_name, seed in TRIALS:
        model_dir = Path("logs") / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 如果基础 checkpoint 存在，就复制进去；不存在就什么都不做
        if use_branch_ckpt:
            dst_g = model_dir / "G_13000.pth"
            dst_d = model_dir / "D_13000.pth"

            if not dst_g.exists():
                shutil.copy2(SRC_G, dst_g)
            if not dst_d.exists():
                shutil.copy2(SRC_D, dst_d)

        cfg = json.loads(json.dumps(base_cfg))
        cfg["train"]["seed"] = seed

        trial_cfg_path = Path("configs") / f"{model_name}.json"
        with open(trial_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        cmd = [
            "python", "train.py",
            "-c", str(trial_cfg_path),
            "-m", model_name,
        ]

        print(f"\n===== Running {model_name} | seed={seed} =====")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()