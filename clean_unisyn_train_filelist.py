import argparse
import utils
from data_utils import UniSynTextAudioLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    ds = UniSynTextAudioLoader(hps.data.training_files, hps.data)

    kept = []
    removed = []

    for i in range(len(ds.audiopaths_and_text)):
        raw = ds.audiopaths_and_text[i]
        try:
            item = ds[i]
            (
                pho, pitch, note_dur, pos, style_id,
                spec, wav, spk_id, real_f0, spec_pert
            ) = item

            if pho.size(0) <= 0 or note_dur.size(0) <= 0:
                removed.append((i, raw, "empty pho or duration"))
                continue
            if pho.size(0) != note_dur.size(0):
                removed.append((i, raw, f"pho/dur mismatch: {pho.size(0)} vs {note_dur.size(0)}"))
                continue
            if int(note_dur.sum().item()) <= 0:
                removed.append((i, raw, f"duration sum <= 0 ({int(note_dur.sum().item())})"))
                continue
            if spec.size(-1) <= 0 or wav.size(-1) <= 0:
                removed.append((i, raw, "empty spec or wav"))
                continue

            kept.append(raw)
        except Exception as e:
            removed.append((i, raw, f"exception: {e}"))

    with open(args.out, "w", encoding="utf-8") as f:
        for row in kept:
            if isinstance(row, (list, tuple)):
                f.write("|".join(str(x) for x in row) + "\n")
            else:
                f.write(str(row).rstrip("\n") + "\n")

    print(f"original = {len(ds.audiopaths_and_text)}")
    print(f"kept     = {len(kept)}")
    print(f"removed  = {len(removed)}")
    for idx, raw, reason in removed[:20]:
        print(f"[REMOVED] idx={idx} reason={reason}")
        print(raw)

if __name__ == "__main__":
    main()
    