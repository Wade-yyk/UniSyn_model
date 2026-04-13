import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import random
import numpy as np

from text.phone_vocab import phone_to_id

import commons
import utils
from data_utils import (
  UniSynTextAudioLoader,    # <--- 修改这里
  UniSynTextAudioCollate,   # <--- 修改这里
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
# 修改 train.py 顶部 import
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    unisyn_kl_loss,
    gvae_excitation_loss,
    gvae_pitch_loss,
    duration_loss,
    wasserstein_distance_gaussian # <--- 新增这行
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  # 1. 依然需要获取配置参数对象，否则 run 函数没法工作
  hps = utils.get_hparams()

  # 2. 强制设为单卡模式
  n_gpus = 1 

  # 3. 设置必要的环境变量（单卡有时也需要这些基础设置）
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12345'

  # 4. 直接跳过 mp.spawn，单进程启动
  print("检测到 Windows 环境访问冲突，正在切换至单进程稳定模式...")
  run(0, n_gpus, hps)


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  device = torch.device("cuda:0")
  torch.manual_seed(hps.train.seed)
  torch.cuda.manual_seed_all(hps.train.seed)
  random.seed(hps.train.seed)
  np.random.seed(hps.train.seed)
  torch.cuda.set_device(0)

  train_dataset = UniSynTextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,300,400,500,600,700,800,900,1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
        seed=hps.train.seed
  )
  collate_fn = UniSynTextAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=0, shuffle=False, pin_memory=False,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = UniSynTextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=False,
        drop_last=False, collate_fn=collate_fn)

  vocab_size = max(phone_to_id.values()) + 1 
  current_spks = getattr(hps.data, 'n_speakers', 1)
  setattr(hps.model, 'n_speakers', current_spks)

  net_g = SynthesizerTrn(
      vocab_size,
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model
  ).cuda(0)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(0)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  # net_g = DDP(net_g, device_ids=[rank])
  # net_d = DDP(net_d, device_ids=[rank])

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()
  
  if rank == 0:
    # 判断：如果最后一步刚好不是 eval_interval 的整数倍，才需要额外存盘
    if global_step % hps.train.eval_interval != 0:
        logger.info('🎉 Training Complete! Saving final models at step {}...'.format(global_step))
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_final_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_final_{}.pth".format(global_step)))
    else:
        # 如果刚好是整数倍，说明循环里已经存过了，报个喜就行
        logger.info('🎉 Training Complete! Final models already saved at step {}.'.format(global_step))


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, batch in enumerate(train_loader):    
    # 1. 解包我们新的 UniSyn data loader 数据
    (pho_padded, pho_lengths, pitch_padded, note_dur_input_padded, align_dur_padded, pos_padded,
        style_ids, spec_padded, spec_lengths, wav_padded, wav_lengths, spk_ids, real_f0_padded, spec_pert_padded) = batch

    # 2. 推送到 GPU (注意 rank 变量来源于 VITS 原有架构)
    pho_padded = pho_padded.cuda(0, non_blocking=True)
    pho_lengths = pho_lengths.cuda(0, non_blocking=True)
    pitch_padded = pitch_padded.cuda(0, non_blocking=True)
    note_dur_input_padded = note_dur_input_padded.cuda(0, non_blocking=True)
    align_dur_padded = align_dur_padded.cuda(0, non_blocking=True)
    pos_padded = pos_padded.cuda(0, non_blocking=True)
    style_ids = style_ids.cuda(0, non_blocking=True)
    spec_padded = spec_padded.cuda(0, non_blocking=True)
    spec_lengths = spec_lengths.cuda(0, non_blocking=True)
    wav_padded = wav_padded.cuda(0, non_blocking=True)
    wav_lengths = wav_lengths.cuda(0, non_blocking=True)
    spk_ids = spk_ids.cuda(0, non_blocking=True)
    real_f0_padded = real_f0_padded.cuda(0, non_blocking=True)
    spec_pert_padded = spec_pert_padded.cuda(0, non_blocking=True)

    with torch.amp.autocast(device_type='cuda', enabled=hps.train.fp16_run):
        # 3. Generator 前向传播 (必须在判别器之前运行，因为判别器需要用到假音频 y_hat)
        (y_hat, ids_slice, spec_mask, frame_mask,
        z_s_posterior, z_rst_posterior,
        z_s_mu, z_rst_mu, z_s_logs, z_rst_logs,
        prior_s_mu, prior_s_logs, prior_rst_mu, prior_rst_logs,
        logw, pred_spk_logits, pred_pitch, pred_spk_adversary) = net_g(
            pho_padded, pho_lengths, pitch_padded,
            note_dur_input_padded, align_dur_padded,
            pos_padded, style_ids, spec_padded, spec_lengths, spk_ids
        )

        mel = spec_to_mel_torch(
            spec_padded, 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate,
            hps.data.mel_fmin, 
            hps.data.mel_fmax
        )

        # 对齐真实音频切片，使其与生成音频长度一致
        if wav_padded.size(2) < hps.train.segment_size:
          # 使用零填充到 segment_size 长度
          padding = hps.train.segment_size - wav_padded.size(2)
          wav_padded = F.pad(wav_padded, (0, padding)) # 在最后一维补零
        
        target_spec_len = hps.train.segment_size // hps.data.hop_length
        if spec_padded.size(2) < target_spec_len:
            padding_spec = target_spec_len - spec_padded.size(2)
            spec_padded = F.pad(spec_padded, (0, padding_spec))
            # 同时也要补齐 spec_pert_padded
            spec_pert_padded = F.pad(spec_pert_padded, (0, padding_spec))
        mel_slice = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        wav_slice = commons.slice_segments(wav_padded, ids_slice * hps.data.hop_length, hps.train.segment_size)

        # 将假音频转换为 mel 频谱，用于计算重构 Loss
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), 
            hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, 
            hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax
        )
    
    #real_f0_slice = commons.slice_segments(real_f0_padded, ids_slice, hps.train.segment_size // hps.data.hop_length)

    # ==========================================
    # 4. 判别器 (Discriminator) 训练步骤
    # ==========================================
    # 注意使用 y_hat.detach()，防止梯度回传给生成器
    with torch.amp.autocast(device_type='cuda', enabled=hps.train.fp16_run): # <--- 添加这行，包裹住判别器的前向传播
        # 注意使用 y_hat.detach()，防止梯度回传给生成器
        y_d_hat_r, y_d_hat_g, _, _ = net_d(wav_slice, y_hat.detach())
    
    with torch.amp.autocast(device_type='cuda', enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    # ==========================================
    # 5. 生成器 (Generator) 损失计算与训练步骤
    # ==========================================
    with torch.amp.autocast(device_type='cuda', enabled=hps.train.fp16_run):
        # 重新让假音频过一遍判别器，这次不 detach，为了获取特征匹配 (feature map)
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_f = net_d(wav_slice, y_hat)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            text_mask = torch.unsqueeze(commons.sequence_mask(pho_lengths, pho_padded.size(1)), 1).to(pho_padded.dtype)

            loss_mel = F.l1_loss(mel_slice, y_hat_mel) * hps.train.c_mel
            loss_kl_s = unisyn_kl_loss(prior_s_mu, prior_s_logs, z_s_mu, z_s_logs, spec_mask) * hps.train.c_kl_s
            loss_kl_rst = unisyn_kl_loss(prior_rst_mu, prior_rst_logs, z_rst_mu, z_rst_logs, spec_mask) * hps.train.c_kl_rst

            loss_gvae_s_exc = gvae_excitation_loss(pred_spk_logits, spk_ids, spec_mask)
            loss_gvae_s_inh = gvae_excitation_loss(pred_spk_adversary, spk_ids, spec_mask)
            loss_gvae_s = (loss_gvae_s_exc + loss_gvae_s_inh) * hps.train.c_gvae_s

            loss_gvae_p = gvae_pitch_loss(pred_pitch, real_f0_padded, spec_mask) * hps.train.c_gvae_p

            loss_dur = duration_loss(logw, align_dur_padded.unsqueeze(1), text_mask) * hps.train.c_dur

            loss_fm = feature_loss(fmap_r, fmap_f) * hps.train.c_fm
            loss_adv_g, losses_gen = generator_loss(y_d_hat_g)
            loss_adv_g = loss_adv_g * hps.train.c_adv

            z_pert, z_mu_pert, z_logs_pert, _ = net_g.enc_q(spec_pert_padded, spec_lengths)
            _, z_rst_mu_pert = torch.split(z_mu_pert, [net_g.z_s_dim, net_g.z_rst_dim], dim=1)
            _, z_rst_logs_pert = torch.split(z_logs_pert, [net_g.z_s_dim, net_g.z_rst_dim], dim=1)

            loss_pert = wasserstein_distance_gaussian(
                z_rst_mu, z_rst_logs,
                z_rst_mu_pert, z_rst_logs_pert,
                spec_mask
            ) * hps.train.c_pert

            # UniSyn 总损失合并
            loss_gen_all = loss_mel + loss_kl_s + loss_kl_rst + loss_gvae_s + loss_gvae_p + loss_dur + loss_fm + loss_adv_g + loss_pert

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        # 1. 更新你要打印的 loss 列表
        losses = [loss_disc_all, loss_gen_all, loss_fm, loss_mel, loss_dur, loss_kl_s, loss_kl_rst, loss_gvae_s, loss_gvae_p, loss_pert]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        # 2. 更新 tensorboard 的标量记录
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl_s": loss_kl_s, "loss/g/kl_rst": loss_kl_rst, "loss/g/gvae_s": loss_gvae_s, "loss/g/gvae_p": loss_gvae_p, "loss/g/pert": loss_pert})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        
        # 3. 移除不存在的 attn 和 y_mel
        # image_dict = { 
        #     "slice/mel_org": utils.plot_spectrogram_to_numpy(mel_slice[0].data.cpu().numpy()),
        #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
        #     "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
        # }
        image_dict = {}
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval, global_step)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval, global_step):
    generator.eval()
    y_hat = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            (pho_padded, pho_lengths, pitch_padded, note_dur_input_padded, align_dur_padded, pos_padded,
            style_ids, spec_padded, spec_lengths, wav_padded, wav_lengths, spk_ids, real_f0_padded, spec_pert_padded) = batch

            # 只看 batch 里的第一个样本
            cur_pho_len = int(pho_lengths[0].item())
            cur_spec_len = int(spec_lengths[0].item())
            cur_wav_len = int(wav_lengths[0].item())

            if cur_pho_len <= 0 or cur_spec_len <= 0 or cur_wav_len <= 0:
                continue
            

            # 只统计有效音素范围内的 duration
            cur_durs = align_dur_padded[0, :cur_pho_len]
            total_frames = int(cur_durs.sum().item())

            # 关键：如果 duration 总帧数是 0，就直接跳过
            if total_frames <= 0:
                print(f"[WARN] Skip eval sample {batch_idx}: total_frames={total_frames}, pho_len={cur_pho_len}")
                continue

            pho_padded = pho_padded[:1].cuda(0)
            pho_lengths = pho_lengths[:1].cuda(0)
            pitch_padded = pitch_padded[:1].cuda(0)
            note_dur_input_padded = note_dur_input_padded[:1].cuda(0)
            align_dur_padded = align_dur_padded[:1].cuda(0)
            pos_padded = pos_padded[:1].cuda(0)
            style_ids = style_ids[:1].cuda(0)
            spk_ids = spk_ids[:1].cuda(0)
            spec_padded = spec_padded[:1].cuda(0)
            wav_padded = wav_padded[:1].cuda(0)
            wav_lengths = wav_lengths[:1].cuda(0)
            note_dur_input_padded = note_dur_input_padded.cuda(0, non_blocking=True)
            align_dur_padded = align_dur_padded.cuda(0, non_blocking=True)

            try:
                print("batch_idx =", batch_idx)
                print("cur_pho_len =", cur_pho_len)
                print("cur_spec_len =", cur_spec_len)
                print("cur_wav_len =", cur_wav_len)
                print("cur_durs shape =", cur_durs.shape)
                print("cur_durs sum =", int(cur_durs.sum().item()))
                print("cur_durs first 30 =", cur_durs[:30].tolist())
                print("pitch first shape =", pitch_padded[:1].shape)
                print("note_dur_input shape =", note_dur_input_padded[:1].shape)
                print("align_dur shape =", align_dur_padded[:1].shape)
                print("pos shape =", pos_padded[:1].shape)
                print(f"DEBUG: batch_idx {batch_idx}, pho_lengths: {pho_lengths[0]}, text: {pho_padded[0][:10]}")
                dur_scale = 1.0
                if cur_durs.sum() < 50: # 如果总帧数小于 50 (对于 VITS 来说太短了)
                    # 我们认为数据可能单位不对，尝试将其放大 10 倍或更多
                    # 或者直接在 infer 时增加 noise_scale 和 length_scale
                    print(f"[DEBUG] Duration too short ({cur_durs.sum()}), applying length_scale.")
                
                y_hat, mask, _ = generator.infer(
                    pho_padded, pho_lengths, pitch_padded,
                    align_dur_padded, pos_padded, style_ids, spk_ids,
                    length_scale=1.2
                )
            except Exception as e:
                print(f"[ERROR] eval sample {batch_idx} failed: {e}")
                raise

            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            # 再补一道保险
            if int(y_hat_lengths[0].item()) <= 0:
                print(f"[WARN] Skip eval sample {batch_idx}: inferred y_hat length is 0")
                y_hat = None
                continue
            mel = spec_to_mel_torch(
                spec_padded,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            common_mel_len = min(mel.size(2), y_hat_mel.size(2))
            val_mel_loss = F.l1_loss(
                mel[:, :, :common_mel_len],
                y_hat_mel[:, :, :common_mel_len]
            ).item()

            break

    if y_hat is not None:
        audio_dict = {"gen/audio": y_hat[0, :, :y_hat_lengths[0]]}
        scalar_dict = {
            "val/mel_loss": float(val_mel_loss),
            "val/generated_audio_len": float(y_hat_lengths[0].item())
        }
        utils.summarize(
            writer=writer_eval,
            global_step=global_step,
            images={},
            audios=audio_dict,
            scalars=scalar_dict,
            audio_sampling_rate=hps.data.sampling_rate
        )
        metrics_path = os.path.join(hps.model_dir, "eval_metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as f:
            record = {
                "global_step": int(global_step),
                "val_mel_loss": float(val_mel_loss),
                "generated_audio_len": int(y_hat_lengths[0].item())
            }
            f.write(json.dumps(record) + "\n")

    generator.train()

                           
if __name__ == "__main__":
  main()
