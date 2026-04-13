import argparse
import torch
import utils
from data_utils import UniSynTextAudioLoader


def check_dataset(filelist_path, data_cfg, name):
    print(f"\nChecking dataset: {name}")
    ds = UniSynTextAudioLoader(filelist_path, data_cfg)
    print(f"dataset size = {len(ds)}")

    bad = 0
    for i in range(len(ds)):
        try:
            item = ds[i]

            # Adjust this unpack if your loader returns different structure
            (
                pho, pitch, note_dur, pos, style_id,
                spec, wav, spk_id, real_f0, spec_pert
            ) = item

            issues = []

            if pho.size(0) <= 0:
                issues.append("pho len <= 0")
            if note_dur.size(0) <= 0:
                issues.append("duration len <= 0")
            if pho.size(0) != note_dur.size(0):
                issues.append(f"pho/dur mismatch: {pho.size(0)} vs {note_dur.size(0)}")
            if int(note_dur.sum().item()) <= 0:
                issues.append(f"duration sum <= 0 ({int(note_dur.sum().item())})")
            if spec.size(-1) <= 0:
                issues.append("spec length <= 0")
            if wav.size(-1) <= 0:
                issues.append("wav length <= 0")

            if issues:
                bad += 1
                print(f"[BAD] idx={i}: {', '.join(issues)}")

        except Exception as e:
            bad += 1
            print(f"[BAD] idx={i}: exception: {e}")

    print(f"Done: total={len(ds)}, bad={bad}, good={len(ds)-bad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    check_dataset(hps.data.training_files, hps.data, "train")
    check_dataset(hps.data.validation_files, hps.data, "val")


if __name__ == "__main__":
    main()