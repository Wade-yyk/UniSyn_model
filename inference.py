import argparse
import os
import torch
import utils
from scipy.io.wavfile import write
from data_utils import UniSynTextAudioLoader
from models import SynthesizerTrn
from text.phone_vocab import phone_to_id


def save_audio(path, audio_tensor, max_wav_value, sampling_rate):
    audio = (
        audio_tensor[0, 0].detach().cpu().float().numpy() * max_wav_value
    ).clip(-max_wav_value, max_wav_value - 1).astype("int16")
    write(path, sampling_rate, audio)


def find_sample(dataset, target_style=None, sample_index=None):
    if sample_index is not None:
        return dataset[sample_index], sample_index

    if target_style is None:
        return dataset[0], 0

    for i in range(len(dataset)):
        item = dataset[i]
        style_id = int(item[5].item())  # 新顺序里 style 在 index 5
        if style_id == target_style:
            return item, i

    raise RuntimeError(f"没有找到 style_id={target_style} 的样本")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/unisyn_base.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--filelist", type=str, default="filelists/unisyn_val.txt")
    parser.add_argument("--style", type=int, default=None, help="0=TTS, 1=SVS, 默认不筛选")
    parser.add_argument("--index", type=int, default=None, help="直接指定 filelist 中的第几条")
    parser.add_argument("--noise_scale", type=float, default=0.3)
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument("--do_recon", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="demo")
    parser.add_argument(
        "--free_tts",
        action="store_true",
        help="仅对 TTS(style=0) 生效：infer 时传 note_dur=None，测试自由时长推理"
    )
    args = parser.parse_args()

    print("加载配置...")
    hps = utils.get_hparams_from_file(args.config)

    vocab_size = max(phone_to_id.values()) + 1

    # 不要再强行 max(..., 10)
    hps.model.n_speakers = hps.data.n_speakers

    print("初始化模型...")
    net_g = SynthesizerTrn(
        vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到 checkpoint: {args.checkpoint}")

    print(f"加载 checkpoint: {args.checkpoint}")
    _ = utils.load_checkpoint(args.checkpoint, net_g, None)
    net_g.eval()

    print(f"读取数据集: {args.filelist}")
    dataset = UniSynTextAudioLoader(args.filelist, hps.data)
    sample, real_index = find_sample(dataset, target_style=args.style, sample_index=args.index)

    # 新版 data_utils 返回顺序
    (
        pho, pitch, note_dur_input, align_dur, pos,
        style_id, spec, wav, spk_id,
        real_f0, spec_pert
    ) = sample

    style_scalar = int(style_id.item())
    spk_scalar = int(spk_id.item())

    print(f"使用样本 index = {real_index}")
    print(f"style_id = {style_scalar}, spk_id = {spk_scalar}")
    print(f"pho_len = {pho.size(0)}, spec_len = {spec.size(1)}, wav_len = {wav.size(1)}")
    print(f"align_dur_sum = {int(align_dur.sum().item())}")

    pho = pho.unsqueeze(0).cuda()
    pho_lengths = torch.LongTensor([pho.size(1)]).cuda()
    pitch = pitch.unsqueeze(0).cuda()
    pos = pos.unsqueeze(0).cuda()
    style_id = style_id.view(-1).cuda()
    spk_id = spk_id.view(-1).cuda()

    spec = spec.unsqueeze(0).cuda()
    spec_lengths = torch.LongTensor([spec.size(2)]).cuda()

    # 推理时给 infer 的时长
    # - 默认：用 align_dur 做 teacher-forced duration inference
    # - 如果是 TTS 且指定 --free_tts：传 None，测试自由时长预测
    if style_scalar == 0:
        infer_note_dur = None
        print("当前模式：TTS 自由时长推理（note_dur=None）")
    else:
        infer_note_dur = note_dur_input.unsqueeze(0).cuda()
        print("当前模式：使用 align_dur 做时长条件推理")

    with torch.no_grad():
        # 1) infer 生成
        print("开始 infer 生成...")
        audio_infer, mask, _ = net_g.infer(
            pho, pho_lengths, pitch, infer_note_dur, pos, style_id, spk_id,
            noise_scale=args.noise_scale,
            length_scale=args.length_scale
        )
        infer_path = f"{args.out_prefix}_infer.wav"
        save_audio(infer_path, audio_infer, hps.data.max_wav_value, hps.data.sampling_rate)
        print(f"已保存: {infer_path}")

        # 2) recon / voice_conversion sanity check
        if args.do_recon:
            print("开始 voice_conversion / recon 测试...")
            audio_recon, _, _ = net_g.voice_conversion(
                spec, spec_lengths, target_spk_id=spk_id
            )
            recon_path = f"{args.out_prefix}_recon.wav"
            save_audio(recon_path, audio_recon, hps.data.max_wav_value, hps.data.sampling_rate)
            print(f"已保存: {recon_path}")

    print("完成。")


if __name__ == "__main__":
    main()

# python inference.py --checkpoint logs/unisyn_base_2/G_15000.pth --style 1 --noise_scale 0.2 --length_scale 1.0 --out_prefix svs_test --do_recon
# python inference.py --checkpoint logs/unisyn_base_2/G_15000.pth --style 1 --noise_scale 0.2 --length_scale 1.0 --out_prefix svs_test --do_recon
# python inference.py --checkpoint logs/unisyn_base_2/G_15000.pth --style 0 --free_tts --noise_scale 0.3 --length_scale 1.0 --out_prefix tts_free
