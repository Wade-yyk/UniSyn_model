import torch
import utils
from data_utils import UniSynTextAudioLoader, UniSynTextAudioCollate
from torch.utils.data import DataLoader
from models import SynthesizerTrn
from text.phone_vocab import phone_to_id


def main():
    hps = utils.get_hparams_from_file("configs/unisyn_base.json")

    torch.cuda.set_device(0)

    val_dataset = UniSynTextAudioLoader(hps.data.validation_files, hps.data)
    collate_fn = UniSynTextAudioCollate()
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    vocab_size = max(phone_to_id.values()) + 1
    current_spks = getattr(hps.data, 'n_speakers', 10)
    setattr(hps.model, 'n_speakers', max(current_spks, 10))

    net_g = SynthesizerTrn(
        vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda(0)

    net_g.eval()

    bad = 0
    total = 0

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            total += 1
            try:
                (
                    pho_padded, pho_lengths, pitch_padded, note_dur_padded, pos_padded,
                    style_ids, spec_padded, spec_lengths, wav_padded, wav_lengths,
                    spk_ids, real_f0_padded, spec_pert_padded
                ) = batch

                cur_pho_len = int(pho_lengths[0].item())
                cur_spec_len = int(spec_lengths[0].item())
                cur_wav_len = int(wav_lengths[0].item())

                if cur_pho_len <= 0 or cur_spec_len <= 0 or cur_wav_len <= 0:
                    print(f"[BAD] idx={idx}: empty length")
                    bad += 1
                    continue

                total_frames = int(note_dur_padded[0, :cur_pho_len].sum().item())
                if total_frames <= 0:
                    print(f"[BAD] idx={idx}: total_frames={total_frames}")
                    bad += 1
                    continue

                pho_padded = pho_padded.cuda(0)
                pho_lengths = pho_lengths.cuda(0)
                pitch_padded = pitch_padded.cuda(0)
                note_dur_padded = note_dur_padded.cuda(0)
                pos_padded = pos_padded.cuda(0)
                style_ids = style_ids.cuda(0)
                spk_ids = spk_ids.cuda(0)

                y_hat, mask, _ = net_g.infer(
                    pho_padded, pho_lengths, pitch_padded,
                    note_dur_padded, pos_padded, style_ids, spk_ids
                )

                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                if int(y_hat_lengths[0].item()) <= 0:
                    print(f"[BAD] idx={idx}: inferred length=0")
                    bad += 1
                    continue

            except Exception as e:
                print(f"[BAD] idx={idx}: infer failed: {e}")
                bad += 1

    print(f"Done: total={total}, bad={bad}, good={total-bad}")


if __name__ == "__main__":
    main()