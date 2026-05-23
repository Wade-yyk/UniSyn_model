import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
#import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding



class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



class SynthesizerTrn(nn.Module):
    """
    UniSyn 核心生成器 (MC-VAE 架构)
    """
    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0, **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.n_speakers = n_speakers

        # 定义潜变量 z 的维度切分 (总维度 inter_channels 通常是 192)
        self.z_s_dim = 16   # z_s (speaker) 的维度
        self.z_rst_dim = inter_channels - self.z_s_dim # z_rst (剩余信息) 的维度

        # 1. 统一文本编码器 (替代原版 TextEncoder)
        self.enc_p = UniSynTextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        
        # 2. UniSyn 时长预测器与长度调节器
        self.dp = UniSynDurationPredictor(hidden_channels, filter_channels, 3, 0.5)
        self.length_regulator = LengthRegulator()
        
        # 3. 框架先验网络 (输出 z_rst 的先验)
        self.frame_prior_net = FramePriorNetwork(hidden_channels, filter_channels, n_heads, 6, kernel_size, p_dropout, self.z_rst_dim)

        # 4. 后验编码器 (从声学频谱提取完整的 z，保持不变)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=0)
        
        # 5. 波形解码器 (HiFi-GAN，保持不变)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0)

        # --- GVAE 专属模块 ---
        # Speaker 均值表 (用于生成 z_s 的先验 mu_cs)
        self.spk_emb = nn.Embedding(n_speakers, self.z_s_dim)
        
        # Speaker 分类器 (Excitation: 强迫 z_s 包含 speaker 信息)
        self.spk_classifier = nn.Sequential(
            nn.Conv1d(self.z_s_dim, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, n_speakers, 1)
        )
        
        # Pitch 预测器 (GVAE: 强迫 z_rst 包含 pitch 信息)
        self.pitch_predictor = nn.Sequential(
            nn.Conv1d(self.z_rst_dim, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, 1)
        )

        # GVAE Inhibition: 试图从 z_rst 中猜出 speaker，用于对抗
        self.spk_adversary = nn.Sequential(
            nn.Conv1d(self.z_rst_dim, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, n_speakers, 1)
        )

    def forward(self, pho, pho_lengths, pitch, note_dur_input, align_dur, pos, style_id, spec, spec_lengths, spk_id):
        # 1. 后验编码 (Posterior): 从音频中提取出完整的 z
        z, z_mu, z_logs, spec_mask = self.enc_q(spec, spec_lengths)
        #z = z_mu + torch.randn_like(z_mu) * torch.exp(z_logs) * spec_mask
        
        # 关键步骤：潜空间切分 (MC-VAE)
        z_s_posterior, z_rst_posterior = torch.split(z, [self.z_s_dim, self.z_rst_dim], dim=1)
        
        z_s_mu, z_rst_mu = torch.split(z_mu, [self.z_s_dim, self.z_rst_dim], dim=1)
        z_s_logs, z_rst_logs = torch.split(z_logs, [self.z_s_dim, self.z_rst_dim], dim=1)

        # 2. 先验编码 (Prior Model)
        # a) 处理文本
        text_hidden, _, text_mask = self.enc_p(pho, pitch, pho_lengths)
        
        # b) 时长预测
        logw = self.dp(text_hidden, text_mask, note_dur_input, pos, style_id)
        valid_text = text_mask.squeeze(1).bool()
        dur_for_lr = torch.where(
            valid_text,
            torch.clamp(align_dur.long(), min=1),
            torch.zeros_like(align_dur.long()),
        )
        
        # c) 训练阶段用真实 align_dur 展开，推理阶段才使用预测 duration 展开
        #    这样 frame prior 不会被训练早期不稳定的 duration prediction 带偏。
        frame_hidden, frame_mask, _ = self.length_regulator(
            text_hidden, dur_for_lr, y_lengths=spec_lengths
        )
        
        # d) 生成 z_rst 的先验
        prior_rst_mu, prior_rst_logs = self.frame_prior_net(frame_hidden, frame_mask)
        
        # e) 生成 z_s 的先验 (基于 speaker ID)
        prior_s_mu = self.spk_emb(spk_id).unsqueeze(-1).expand(-1, -1, z_s_posterior.size(2)) * spec_mask
        # 论文提到 zs 几乎是不随时间变化的，且方差很小(0.01)，log(0.01) ≈ -4.6
        prior_s_logs = (torch.zeros_like(prior_s_mu) - 2.3) * spec_mask 

        # 3. GVAE 预测 (分离监督)
        pred_spk_logits = self.spk_classifier(z_s_posterior)  # [B, n_speakers, T]
        pred_pitch = self.pitch_predictor(z_rst_posterior)   # [B, 1, T]

        # 4. 波形解码 (生成)
        # 重新拼接 z_s 和 z_rst
        z_combine = torch.cat([z_s_posterior, z_rst_posterior], dim=1)
        
        # 训练时同样对 z 进行切片以节省显存 (和原版VITS一致)
        z_slice, ids_slice = commons.rand_slice_segments(z_combine, spec_lengths, self.segment_size)
        # prior_s_mu_slice = commons.slice_segments(prior_s_mu, ids_slice, self.segment_size) # 假设 hop 是 300
        # prior_s_logs_slice = commons.slice_segments(prior_s_logs, ids_slice, self.segment_size)
        # prior_rst_mu_slice = commons.slice_segments(prior_rst_mu, ids_slice, self.segment_size)
        # prior_rst_logs_slice = commons.slice_segments(prior_rst_logs, ids_slice, self.segment_size)
        o = self.dec(z_slice)

        # 加上梯度反转层，传给 adversarial 分类器
        z_rst_rev = grad_reverse(z_rst_posterior, alpha=1.0)
        pred_spk_adversary = self.spk_adversary(z_rst_rev)

        return (o, ids_slice, spec_mask, frame_mask, 
                z_s_posterior, z_rst_posterior, 
                z_s_mu, z_rst_mu, z_s_logs, z_rst_logs,
                prior_s_mu, prior_s_logs, prior_rst_mu, prior_rst_logs,
                logw, pred_spk_logits, pred_pitch, pred_spk_adversary)

    def infer(self, pho, pho_lengths, pitch, note_dur, pos, style_id, spk_id,
          noise_scale=0.667, length_scale=1.0, force_dur=None):
        """
        推理模式
        - SVS: 可以传入真实 note_dur
        - TTS: 可以传入 None，此时用 0 占位喂给 DP，再用预测时长展开
        """
        text_hidden, _, text_mask = self.enc_p(pho, pitch, pho_lengths)

        #use_given_dur = note_dur is not None

        # 给 DP 的输入时长：如果没有，就喂全 0 占位
        if note_dur is None:
            note_dur_input = torch.zeros(
                pho.size(0), pho.size(1),
                device=pho.device,
                dtype=text_hidden.dtype
            )
        else:
            note_dur_input = note_dur.to(text_hidden.dtype)

        logw = self.dp(text_hidden, text_mask, note_dur_input, pos, style_id)
        w = torch.exp(logw) * text_mask * length_scale

        # 给 LengthRegulator 的展开时长
        if force_dur is not None:
            dur_for_lr = torch.clamp(force_dur.long(), min=1)
        else:
            dur_for_lr = torch.clamp(torch.ceil(w.squeeze(1)).long(), min=1)

        frame_hidden, frame_mask, _ = self.length_regulator(text_hidden, dur_for_lr)

        prior_rst_mu, prior_rst_logs = self.frame_prior_net(frame_hidden, frame_mask)
        z_rst = prior_rst_mu + torch.randn_like(prior_rst_mu) * torch.exp(prior_rst_logs) * noise_scale

        prior_s_mu = self.spk_emb(spk_id).unsqueeze(-1).expand(-1, -1, z_rst.size(2)) * frame_mask
        prior_s_logs = (torch.zeros_like(prior_s_mu) - 2.3) * frame_mask
        z_s = prior_s_mu + torch.randn_like(prior_s_mu) * torch.exp(prior_s_logs) * noise_scale

        z_combine = torch.cat([z_s, z_rst], dim=1)
        o = self.dec(z_combine * frame_mask)

        return o, frame_mask, (z_s, z_rst)

    def voice_conversion(self, spec, spec_lengths, target_spk_id):
        """
        变声模式 (Voice Conversion)
        得益于 MC-VAE 的解耦，我们只需要替换 zs 即可
        """
        # 1. 从源音频提取完整的后验 z
        z, z_mu, z_logs, spec_mask = self.enc_q(spec, spec_lengths)
        #z = z_mu + torch.randn_like(z_mu) * torch.exp(z_logs) * spec_mask
        
        # 2. 潜空间切分，直接抛弃源音频的 zs
        _, z_rst_posterior = torch.split(z, [self.z_s_dim, self.z_rst_dim], dim=1)
        
        # 3. 生成目标说话人的 zs
        target_s_mu = self.spk_emb(target_spk_id).unsqueeze(-1).expand(-1, -1, z_rst_posterior.size(2)) * spec_mask
        # 变声时为了稳定，通常直接使用均值，不加噪声
        target_z_s = target_s_mu 
        
        # 4. 拼接并解码
        z_combine = torch.cat([target_z_s, z_rst_posterior], dim=1)
        o = self.dec(z_combine * spec_mask)
        
        return o, spec_mask, (target_z_s, z_rst_posterior)

class LengthRegulator(nn.Module):
    """
    将音素级别的隐藏状态，根据每个音素对应的帧数(时长)，
    在时间维度上进行复制扩展，使其与声学频谱的帧率对齐。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, dur, y_lengths=None):
        """
        x: [B, hidden_channels, T_text] (音素级别的特征)
        dur: [B, T_text] (每个音素需要重复的帧数，必须是整数)
        y_lengths: [B] (真实的声学特征长度，训练时用于截断或强制对齐)
        """
        B, C, T_text = x.size()
        outputs = []
        out_lengths = []

        # 遍历 Batch 中的每一个样本
        for i in range(B):
            # 获取当前样本的重复次数 (时长)
            repeats = dur[i].long()
            
            # 过滤掉 padding 部分导致的负数或异常值
            repeats = torch.clamp(repeats, min=0) 
            
            # 在时间维度 (dim=1) 上复制特征
            expanded = torch.repeat_interleave(x[i], repeats, dim=1)
            outputs.append(expanded)
            out_lengths.append(expanded.size(1))

        # 找出当前 batch 扩展后的最大长度
        max_len = max(out_lengths)
        if y_lengths is not None:
            # 训练时，为保证与 target mel 长度完全一致，取最大值
            max_len = int(y_lengths.max().item())

        # 初始化 padding 后的输出 Tensor
        out_padded = torch.zeros(B, C, max_len).to(x.device)
        
        for i in range(B):
            length = outputs[i].size(1)
            if y_lengths is not None:
                # 训练模式下：严格对齐真实声学长度
                target_len = y_lengths[i].item()
                if length > target_len:
                    # 如果预测长度大于真实长度，截断
                    out_padded[i, :, :target_len] = outputs[i][:, :target_len]
                else:
                    # 如果预测长度小于真实长度，末尾补零 (或补最后一个特征)
                    out_padded[i, :, :length] = outputs[i]
                out_lengths[i] = target_len
            else:
                # 推理模式下：直接赋予
                out_padded[i, :, :length] = outputs[i]

        out_lengths = torch.LongTensor(out_lengths).to(x.device)
        
        # 生成对应的 mask
        mask = torch.unsqueeze(commons.sequence_mask(out_lengths, max_len), 1).to(x.dtype)
        
        return out_padded * mask, mask, out_lengths


class UniSynDurationPredictor(nn.Module):
    """
    统一时长预测器，接收 pho, dur_note, pos, style 作为输入。
    按照 UniSyn 论文，使用 3 层带有 Dropout 的一维卷积。
    """
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # 1. Style Embedding (区分说话 0 和 唱歌 1)
        # 假设 in_channels 已经包含了音素特征的维度
        self.style_emb = nn.Embedding(2, filter_channels)

        # 2. 特征融合层：将 text_hidden(in_channels) + note_dur(1) + pos(1) 融合
        self.pre_conv = nn.Conv1d(in_channels + 2, filter_channels, 1)

        # 3. 三层卷积模块 (论文标准配置)
        self.drop = nn.Dropout(p_dropout)
        
        self.conv_1 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        
        self.conv_3 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_3 = modules.LayerNorm(filter_channels)
        
        # 4. 输出层，预测对数时长
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, note_dur, pos, style_id):
        """
        x: [B, C, T] (来自 TextEncoder 的音素隐藏状态)
        note_dur: [B, T] (乐谱音符时长)
        pos: [B, T] (相对位置)
        style_id: [B] (风格 ID，0 for TTS, 1 for SVS)
        """
        # 调整额外特征的维度以拼接
        note_dur = note_dur.unsqueeze(1).float() # [B, 1, T]
        note_dur = torch.log1p(note_dur)
        pos = pos.unsqueeze(1).float()           # [B, 1, T]
        
        # 沿着通道维度拼接: [B, C+2, T]
        x = torch.cat([x, note_dur, pos], dim=1) 
        
        # 预卷积，统一通道数到 filter_channels
        x = self.pre_conv(x) * x_mask
        
        # 加入 Style Embedding 调节
        style_vector = self.style_emb(style_id).unsqueeze(-1) # [B, filter_channels, 1]
        x = x + style_vector # 广播加法机制
        
        # 经过三层卷积
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.conv_3(x * x_mask)
        x = torch.relu(x)
        x = self.norm_3(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x
    
class UniSynTextEncoder(nn.Module):
    """
    统一文本编码器：接收音素(pho)和音高/声调(tp)
    在 VITS 原版 TextEncoder 基础上增加 pitch embedding
    """
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        self.emb_pho = nn.Embedding(n_vocab, hidden_channels)
        self.emb_pitch = nn.Embedding(256, hidden_channels) # 假设 256 种 pitch/tone
        nn.init.normal_(self.emb_pho.weight, 0.0, hidden_channels**-0.5)
        nn.init.normal_(self.emb_pitch.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, pho, pitch, x_lengths):
        x = self.emb_pho(pho) * math.sqrt(self.hidden_channels) # [B, T, H]
        p = self.emb_pitch(pitch) * math.sqrt(self.hidden_channels)
        
        x = x + p # 融合音素和音高信息
        x = torch.transpose(x, 1, -1) # [B, H, T]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        
        # 返回隐藏状态和 stats (为了和原有框架兼容)
        return x, stats, x_mask


class FramePriorNetwork(nn.Module):
    """
    Frame Prior Network: 接收对齐后的帧级别特征，输出 z_rst 的先验分布 p(z_rst|c_rst)
    论文指出包含 6 个 Transformer blocks
    """
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, out_channels):
        super().__init__()
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        # 输出均值和方差，所以 out_channels * 2
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1) 

    def forward(self, x, x_mask):
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, stats.size(1)//2, dim=1)
        return m, logs
