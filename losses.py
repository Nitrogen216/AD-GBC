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


def wasserstein_diversity_loss(centers):
    """
    基于ICLR 2024论文思想，使用瓦瑟斯坦距离计算多样性损失。
    
    Args:
        centers (torch.Tensor): GBC的中心点张量, 形状为 (K, D)。
    """
    K, D = centers.shape
    if K <= 1:
        return torch.tensor(0.0, device=centers.device)

    # --- 1. 计算样本均值和协方差矩阵  ---
    mu_hat = torch.mean(centers, dim=0)
    centers_centered = centers - mu_hat
    # Sigma_hat 形状: (D, D)
    Sigma_hat = (centers_centered.t() @ centers_centered) / K

    # --- 2. 计算瓦瑟斯坦距离的各个组成部分 ---
    # a) 均值部分的损失: ||mu_hat||^2
    term_mean = torch.sum(mu_hat ** 2)

    # b) 协方差矩阵的迹: Tr(Sigma_hat)
    term_trace_Sigma = torch.trace(Sigma_hat)
    
    # c) 协方差矩阵平方根的迹: Tr(sqrt(Sigma_hat))
    #    最高效、最稳定的方法是通过特征值分解来计算
    try:
        # eigh 专门用于对称/厄米矩阵，返回特征值和特征向量
        eigenvalues = torch.linalg.eigvalsh(Sigma_hat)
        # 为防止数值不稳定导致极小的负特征值，用clamp截断
        term_trace_sqrt_Sigma = torch.sum(torch.sqrt(torch.clamp(eigenvalues, min=0)))
    except torch.linalg.LinAlgError:
        # 如果矩阵奇异或在训练中出现问题，返回0损失，避免训练崩溃
        return torch.tensor(0.0, device=centers.device)

    # d) 根据论文公式(7)的变体 (目标分布为N(0,I))，计算距离的平方
    # W^2 = ||mu||^2 + Tr(Sigma) + Tr(I) - 2*Tr(sqrt(Sigma))
    # Tr(I) = D (特征维度)
    # 论文中目标是N(0, I/m)，所以有1/m项。我们简化目标为N(0,I)，更关注各向同性
    w2_squared = term_mean + term_trace_Sigma - 2 * term_trace_sqrt_Sigma
    
    # 返回一个非负的损失值
    return torch.clamp(w2_squared, min=0)

class BCEDiceWithGeometryLoss(nn.Module):
    def __init__(self, div_weight=0.1, scale_weight=0.1):
        """
        复合损失函数.
        param div_weight: 多样性损失的权重
        param scale_weight: 尺度一致性损失的权重
        """
        super().__init__()
        # 1. 在 __init__ 中实例化基础损失，并传入其超参数
        self.bce_dice = BCEDiceLoss()
        
        # 2. 将多样性损失的权重也作为超参数存储起来
        self.div_weight = div_weight
        # 3. 将尺度一致性损失的权重存储起来
        self.scale_weight = scale_weight # <--- 新增

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
        div_loss = wasserstein_diversity_loss(centers)

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
