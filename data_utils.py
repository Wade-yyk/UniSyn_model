import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence


class UniSynTextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and auxiliary features pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        # 你的 filelist 格式必须是按照 '|' 分割的7列:
        # audiopath | phonemes | pitch_ids | note_durations | positions | style_id | speaker_id
        # 例如: dataset/wavs/001.wav|p i n y i n|1 1 2 2 3|0.5 0.5 1.0 1.0 2.0|0.1 0.5 0.1 0.5 0.1|0|1
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        #torch.manual_seed(1234)

        # === 新增：计算数据集中每条音频的估算帧长，用于加速训练的分桶采样 ===
        self.lengths = []
        for row in self.audiopaths_and_text:
            audiopath = row[0]
            # 根据文件大小估算帧数: 文件字节数 // (16bit(2字节) * hop_length)
            self.lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        
    def get_audio_text_pair(self, audiopath_and_text):
        # 解析 UniSyn 需要的统一特征
        file_path, pho, pitch, note_dur, pos, style_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3], audiopath_and_text[4], audiopath_and_text[5], audiopath_and_text[6]
        
        # 1. 提取音素 (pho)
        pho_tensor = self.get_text(pho)
        
        pitch_ids = [int(x) for x in pitch.split(" ")]
        align_dur_ids = [max(1, int(x)) for x in note_dur.split(" ")]
        pos_ids = [float(x) for x in pos.split(" ")]

        style_val = int(style_id)

        # 给 DP 的输入时长
        if style_val == 0:  # TTS
            note_dur_input_ids = [0] * len(align_dur_ids)
        else:               # SVS
            note_dur_input_ids = align_dur_ids.copy()

        if self.add_blank:
            pitch_ids = commons.intersperse(pitch_ids, 0)
            note_dur_input_ids = commons.intersperse(note_dur_input_ids, 0)
            align_dur_ids = commons.intersperse(align_dur_ids, 0)
            pos_ids = commons.intersperse(pos_ids, 0.0)

        pitch_tensor = torch.LongTensor(pitch_ids)
        note_dur_input_tensor = torch.LongTensor(note_dur_input_ids)
        align_dur_tensor = torch.LongTensor(align_dur_ids)
        pos_tensor = torch.FloatTensor(pos_ids)
        
        # 5. 提取 Style ID 和 Speaker ID
        style_tensor = torch.LongTensor([style_val])
        spk_tensor = torch.LongTensor([int(spk_id)])
        
        # 6. 处理音频和频谱
        spec, wav = self.get_audio(file_path)
        real_f0 = self.get_f0(file_path)
        # === 新增：读取共振峰扰动音频的频谱 ===
        pert_file_path = file_path.replace(".wav", "_pert.wav")
        if os.path.exists(pert_file_path):
            # 只需要频谱，不需要波形
            spec_pert, _ = self.get_audio(pert_file_path) 
        else:
            # 如果某条音频没有成功生成 _pert.wav，为了防止报错，直接退化为使用原频谱
            spec_pert = spec

        min_len = min(spec.size(1), real_f0.size(1), spec_pert.size(1))
        spec = spec[:, :min_len]
        real_f0 = real_f0[:, :min_len]
        spec_pert = spec_pert[:, :min_len]
        
        # 确保文本特征长度一致，方便后续合并
        assert pho_tensor.size(0) == pitch_tensor.size(0) == note_dur_input_tensor.size(0) == align_dur_tensor.size(0) == pos_tensor.size(0), \
            f"Feature length mismatch in file {file_path}"
            
        return (
            pho_tensor, pitch_tensor, note_dur_input_tensor, align_dur_tensor, pos_tensor,
            style_tensor, spec, wav, spk_tensor, real_f0, spec_pert
        )

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm
    
    def get_f0(self, filename):
        f0_filename = filename.replace(".wav", ".f0.pt")
        if not os.path.exists(f0_filename):
            raise FileNotFoundError(
                f"Missing F0 file: {f0_filename}\n"
                f"Please run preprocess_f0.py before training."
            )

        f0 = torch.load(f0_filename)

        # 防止保存出来的是空 tensor
        if f0.numel() == 0:
            raise ValueError(f"Empty F0 tensor: {f0_filename}")

        return f0.unsqueeze(0)  # [1, T]

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = [int(x) for x in text.split(" ")]
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class UniSynTextAudioCollate():
    """ Zero-pads model inputs and targets """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # 找出 batch 中各个特征的最大长度用于 padding
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[6].size(1) for x in batch]),
            dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[6].size(1) for x in batch])
        max_wav_len = max([x[7].size(1) for x in batch])

        pho_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        # 初始化 padding 后的张量
        pho_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        pitch_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        note_dur_input_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        align_dur_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        pos_padded = torch.FloatTensor(len(batch), max_text_len).zero_()
        
        style_ids = torch.LongTensor(len(batch))
        spk_ids = torch.LongTensor(len(batch))
        
        spec_padded = torch.FloatTensor(len(batch), batch[0][6].size(0), max_spec_len).zero_()
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len).zero_()

        real_f0_padded = torch.FloatTensor(len(batch), 1, max_spec_len).zero_()

        spec_pert_padded = torch.FloatTensor(len(batch), batch[0][6].size(0), max_spec_len).zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            pho, pitch, note_dur_input, align_dur, pos, style, spec, wav, spk, real_f0, spec_pert = row
            
            # Padding text features
            pho_lengths[i] = pho.size(0)
            pho_padded[i, :pho.size(0)] = pho
            pitch_padded[i, :pitch.size(0)] = pitch
            note_dur_input_padded[i, :note_dur_input.size(0)] = note_dur_input
            align_dur_padded[i, :align_dur.size(0)] = align_dur
            pos_padded[i, :pos.size(0)] = pos
            
            # Scalar features
            style_ids[i] = style[0]
            spk_ids[i] = spk[0]

            # Padding audio features
            spec_lengths[i] = spec.size(1)
            spec_padded[i, :, :spec.size(1)] = spec

            real_f0_padded[i, :, :real_f0.size(1)] = real_f0
            
            wav_lengths[i] = wav.size(1)
            wav_padded[i, :, :wav.size(1)] = wav

            spec_pert_padded[i, :, :spec_pert.size(1)] = spec_pert

        if self.return_ids:
            return (
                pho_padded, pho_lengths, pitch_padded,
                note_dur_input_padded, align_dur_padded, pos_padded,
                style_ids, spec_padded, spec_lengths,
                wav_padded, wav_lengths, spk_ids,
                real_f0_padded, spec_pert_padded, ids_sorted_decreasing
            )
        return (
            pho_padded, pho_lengths, pitch_padded,
            note_dur_input_padded, align_dur_padded, pos_padded,
            style_ids, spec_padded, spec_lengths,
            wav_padded, wav_lengths, spk_ids,
            real_f0_padded, spec_pert_padded
        )


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_f0(self, filename):
        f0_filename = filename.replace(".wav", ".f0.pt")
        if not os.path.exists(f0_filename):
            raise FileNotFoundError(
                f"Missing F0 file: {f0_filename}\n"
                f"Please run preprocess_f0.py before training."
            )
        f0 = torch.load(f0_filename)
        if f0.numel() == 0:
            raise ValueError(f"Empty F0 tensor: {f0_filename}")
        return f0.unsqueeze(0)

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True, seed=1234):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.seed = seed
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
