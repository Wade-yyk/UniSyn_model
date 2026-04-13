import torch 
from torch.nn import functional as F

import commons


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # 1. 如果形状完全一样，直接算
            if rl.shape == gl.shape:
                loss += torch.mean(torch.abs(rl - gl))
            else:
                # 2. 如果形状不一样，获取两个张量中每一个维度的最小值
                # 例如 rl 是 [16, 32, 1366], gl 是 [16, 32, 1351]
                # min_shape 就会变成 [16, 32, 1351]
                min_shape = [min(s1, s2) for s1, s2 in zip(rl.shape, gl.shape)]
                
                # 3. 根据最小维度构造切片索引
                # 这会自动处理 2D, 3D, 4D 等所有情况
                slices = tuple(slice(0, s) for s in min_shape)
                
                # 4. 强制对齐并相减
                loss += torch.mean(torch.abs(rl[slices] - gl[slices]))
    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l


def unisyn_kl_loss(z_p_mu, z_p_logs, z_q_mu, z_q_logs, z_mask):
    """
    分离计算的 KL 散度。
    由于 z_s 和 z_rst 分别服从不同的分布，我们需要单独计算它们与各自先验的 KL 散度。
    """
    # z_p: prior (先验), z_q: posterior (后验)
    kl = z_p_logs - z_q_logs - 0.5
    kl += 0.5 * ((z_q_mu - z_p_mu)**2) * torch.exp(-2. * z_p_logs)
    kl += 0.5 * torch.exp(2. * z_q_logs - 2. * z_p_logs)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss

def gvae_excitation_loss(pred_logits, target_labels, mask):
    """
    GVAE 的 Excitation 损失 (迫使 z_s 包含说话人信息)
    pred_logits: [B, n_speakers, T]
    target_labels: [B]
    mask: [B, 1, T]
    """
    # 将 target_labels 扩展至时间维度 [B, T]
    B, _, T = pred_logits.size()
    targets = target_labels.unsqueeze(1).expand(B, T)
    
    # 将 [B, C, T] 转换为 [B*T, C]，targets 转换为 [B*T]
    pred_logits_flat = pred_logits.transpose(1, 2).reshape(-1, pred_logits.size(1))
    targets_flat = targets.reshape(-1)
    mask_flat = mask.view(-1)
    
    # 仅对 mask 范围内的帧计算 Cross Entropy
    loss = F.cross_entropy(pred_logits_flat, targets_flat, reduction='none')
    loss = torch.sum(loss * mask_flat) / torch.sum(mask_flat)
    return loss

def gvae_pitch_loss(pred_pitch, target_pitch, mask):
    """
    GVAE 的 Pitch 约束 (迫使 z_rst 包含正确的音高信息)
    pred_pitch: [B, 1, T]
    target_pitch: [B, 1, T]
    """
    loss = F.mse_loss(pred_pitch * mask, target_pitch * mask, reduction='sum')
    loss = loss / torch.sum(mask)
    return loss

def duration_loss(logw, target_dur, text_mask):
    """
    对数时长的 L1 损失 (论文中 L_dur)
    logw: [B, 1, T_text]
    target_dur: [B, 1, T_text] (这里要求 target_dur 是时长的真值)
    """
    # 为了防止 log(0)，通常在外面会加一个小常数或者直接对预测的对数值做回归
    # 假设 target_dur 已经是原始帧数，取对数
    target_logw = torch.log(target_dur.float() + 1e-6) * text_mask
    loss = F.l1_loss(logw * text_mask, target_logw, reduction='sum')
    loss = loss / torch.sum(text_mask)
    return loss

def wasserstein_distance_gaussian(mu1, log_var1, mu2, log_var2, mask):
    """
    计算两个对角高斯分布之间的 Wasserstein 距离
    """
    # 将对数方差转为标准差
    std1 = torch.exp(0.5 * log_var1)
    std2 = torch.exp(0.5 * log_var2)
    
    # 计算 W_2^2 = (mu1 - mu2)^2 + (std1 - std2)^2
    w2_sq = (mu1 - mu2)**2 + (std1 - std2)**2
    
    # 按照 mask 求平均
    loss = torch.sum(w2_sq * mask) / torch.sum(mask)
    return loss