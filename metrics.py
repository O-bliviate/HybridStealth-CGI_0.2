import torch
import torch.nn.functional as F
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
import numpy as np


def compute_feature_metrics(feat_pred, feat_gt):
    """
    [Phase 1 评估] 计算特征空间的相似度
    Args:
        feat_pred: 恢复出的特征 [D]
        feat_gt: 真实特征 [D]
    """
    # 1. 余弦相似度 (Cosine Similarity) - 最重要的指标，衡量方向/语义一致性
    # dim=0 因为输入是 1D 向量
    cos_sim = F.cosine_similarity(feat_pred, feat_gt, dim=0).item()

    # 2. MSE (幅度差异)
    mse = F.mse_loss(feat_pred, feat_gt).item()

    return {'cos': cos_sim, 'mse': mse}


class Evaluator:
    """
    [Phase 2 评估] 专业的图像质量评估
    在 main 函数外初始化以避免重复加载 LPIPS 模型
    """

    def __init__(self, device):
        self.device = device
        print("Loading LPIPS model for rigorous evaluation (AlexNet)...")
        # LPIPS 需要下载预训练模型，第一次运行可能需要一点时间
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        self.lpips_loss.eval()

    def compute_img_metrics(self, img_pred, img_gt):
        """
        计算 MSE, PSNR, SSIM, LPIPS
        """
        # 1. 统一归一化到 [0, 1] 用于 PSNR/SSIM
        pred_01 = self._to_01(img_pred)
        gt_01 = self._to_01(img_gt)

        # 2. MSE & PSNR
        mse = torch.mean((pred_01 - gt_01) ** 2).item()
        psnr = psnr_func(pred_01, gt_01, data_range=1.0).item()

        # 3. SSIM (使用 torchmetrics 的严谨实现)
        ssim = ssim_func(pred_01, gt_01, data_range=1.0).item()

        # 4. LPIPS (需要映射到 [-1, 1])
        pred_norm = pred_01 * 2 - 1
        gt_norm = gt_01 * 2 - 1

        with torch.no_grad():
            # LPIPS 返回的是 tensor，需转 float
            lpips_score = self.lpips_loss(pred_norm, gt_norm).item()

        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips_score
        }

    def _to_01(self, x):
        """将任意范围 Tensor 映射到 [0, 1]"""
        x = x.detach().float()
        min_v, max_v = x.min(), x.max()
        if max_v - min_v > 1e-8:
            return (x - min_v) / (max_v - min_v)
        return torch.zeros_like(x)