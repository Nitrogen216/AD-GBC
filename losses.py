import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'BCEDiceWithGeometryLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


def wasserstein_diversity_loss(centers, mode='legacy'):
    """Wasserstein spectral diversity loss on the ball centers (K, D).

    mode (A000 spec / Prompt 1 Part D):
      legacy     : released repo loss, clamp(||mu||^2 + tr(Σ) - 2 tr(√Σ), min=0).
                   (coefficient differs from the paper and the outer clamp can
                   zero the gradient at small-center init — kept only as a
                   reproduction baseline.)
      paper      : ||mu||^2 + tr(Σ) - (2/√D) tr(√Σ), NO clamp (paper Eq.8).
      rank_aware : ||mu||^2 + Σ_j (s_j - 1/√D)^2 over r=min(D,K-1) singular
                   values of X=(C-mean)/√K. Equals `paper` + r/D (gradient-
                   identical), is non-negative and stable. RECOMMENDED.
    """
    K, D = centers.shape
    if K <= 1:
        return torch.tensor(0.0, device=centers.device)

    mu_hat = torch.mean(centers, dim=0)
    term_mean = torch.sum(mu_hat ** 2)

    if mode == 'legacy':
        centers_centered = centers - mu_hat
        Sigma_hat = (centers_centered.t() @ centers_centered) / K
        term_trace_Sigma = torch.trace(Sigma_hat)
        try:
            eigenvalues = torch.linalg.eigvalsh(Sigma_hat)
            term_trace_sqrt_Sigma = torch.sum(torch.sqrt(torch.clamp(eigenvalues, min=0)))
        except torch.linalg.LinAlgError:
            return torch.tensor(0.0, device=centers.device)
        w2_squared = term_mean + term_trace_Sigma - 2 * term_trace_sqrt_Sigma
        return torch.clamp(w2_squared, min=0)

    # paper / rank_aware: singular values of X = (C - mean)/sqrt(K) in float32.
    X = (centers - mu_hat) / math.sqrt(K)
    s = torch.linalg.svdvals(X.float()).to(centers.dtype)        # length min(K,D)
    r = min(D, K - 1)
    s = s[:r]
    inv_sqrt_D = 1.0 / math.sqrt(D)
    if mode == 'paper':
        return term_mean + (s ** 2).sum() - 2 * inv_sqrt_D * s.sum()
    if mode == 'rank_aware':
        return term_mean + ((s - inv_sqrt_D) ** 2).sum()
    raise ValueError(f'unknown wdiv mode: {mode}')

class BCEDiceWithGeometryLoss(nn.Module):
    def __init__(self, div_weight=0.1, scale_weight=0.1, wdiv_mode='legacy'):
        """
        复合损失函数.
        param div_weight: 多样性损失的权重 (λ_W)
        param scale_weight: 尺度一致性损失的权重 (λ_S)
        param wdiv_mode: 'legacy' | 'paper' | 'rank_aware' (see wasserstein_diversity_loss)
        """
        super().__init__()
        # 1. 在 __init__ 中实例化基础损失，并传入其超参数
        self.bce_dice = BCEDiceLoss()

        # 2. 将多样性损失的权重也作为超参数存储起来
        self.div_weight = div_weight
        # 3. 将尺度一致性损失的权重存储起来
        self.scale_weight = scale_weight # <--- 新增
        self.wdiv_mode = wdiv_mode

    def calculate_scale_loss(self, att, dif, log_sigma):
        """
        计算各向异性尺度一致性损失 (L_scale_con).
        att: [B, N, K] (软分配)
        dif: [B, N, K, D] (z_i - c_k)
        log_sigma: [K, D] (GBC模块的可学习参数)
        """
        # 我们只在各向异性模式下计算此损失
        if not log_sigma.requires_grad: 
            return torch.tensor(0.0, device=att.device)

        with torch.no_grad():
            # 我们 detach 'att'，因为我们不希望 L_scale_con 通过 alpha 回传梯度
            # alpha 仅作为统计权重，这能防止循环依赖并稳定训练
            weight = att.detach().unsqueeze(-1) # [B, N, K, 1]
            # Mk: 每个球的软样本数, [B, 1, K, 1]
            Mk = weight.sum(dim=1, keepdim=True) + 1e-6 

        # (z_i - c_k)^2, 逐元素平方
        diff_elem_sq = dif.pow(2) # [B, N, K, D]

        # s_k^2 (观测方差), [B, 1, K, D]
        s_k_sq_batched = (weight * diff_elem_sq).sum(dim=1, keepdim=True) / Mk

        # 在 batch 维度上取平均，得到最终的 s_k^2: [K, D]
        s_k_sq = s_k_sq_batched.mean(dim=0).squeeze(0) # [K, D]

        # sigma_k^2 (学习方差)
        sigma_k_sq = F.softplus(log_sigma).pow(2)  # [K, D]

        # 计算 L_scale_con
        # 我们 detach s_k_sq，因为它在这里是“目标” (target)
        # 梯度应该只流向 log_sigma (模型参数)
        scale_consistency_loss = F.mse_loss(s_k_sq.detach(), sigma_k_sq)

        return scale_consistency_loss

    def forward(self, model_output, target, model):
        """
        计算总损失.
        :param model_output: 模型的预测输出 (在训练时是一个元组)
        :param target: 真实标签
        :param model: 传入整个模型，以便从中获取 centers
        """
        # --- 解包模型输出 ---
        if isinstance(model_output, tuple):
            # 训练模式: (seg_output, loss_intermediates)
            seg_output, loss_intermediates = model_output
        else:
            # 评估模式: seg_output
            seg_output = model_output
            loss_intermediates = None

        # --- 1. 计算主要分割损失 ---
        main_loss = self.bce_dice(seg_output, target)

        # --- 2. 计算多样性损失 ---
        # 从传入的模型中安全地获取 GBC 模块
        if isinstance(model, torch.nn.DataParallel):
            gbc_module = model.module.gbc
        else:
            gbc_module = model.gbc

        centers = gbc_module.centers
        div_loss = wasserstein_diversity_loss(centers, mode=self.wdiv_mode)

        # --- 3. 计算尺度一致性损失 (L_scale_con) ---
        scale_loss = torch.tensor(0.0, device=main_loss.device)
        if loss_intermediates is not None and self.scale_weight > 0 and gbc_module.use_diag_cov:
            log_sigma = gbc_module.log_sigma
            # 计算第一个GBC调用的loss
            scale_loss_1 = self.calculate_scale_loss(
                loss_intermediates["att_1"],
                loss_intermediates["dif_1"],
                log_sigma
            )
            # 计算第二个GBC调用的loss
            scale_loss_2 = self.calculate_scale_loss(
                loss_intermediates["att_2"],
                loss_intermediates["dif_2"],
                log_sigma
            )
            # 取平均
            scale_loss = (scale_loss_1 + scale_loss_2) / 2.0

        # --- 4. 组合损失 ---
        total_loss = main_loss + self.div_weight * div_loss + self.scale_weight * scale_loss

        return total_loss
