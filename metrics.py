import torch
import torch.nn.functional as F
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
import numpy as np
import warnings

class Evaluator:
    def __init__(self, device):
        self.device = device
        # 屏蔽 LPIPS 的冗余打印
        # [Fix] Suppress LPIPS warnings by filtering them or updating usage if possible.
        # LPIPS library might internally use deprecated torchvision calls.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.lpips_loss = lpips.LPIPS(net='alex', verbose=False).to(device)
        self.lpips_loss.eval()

    def compute_img_metrics(self, img_pred, img_gt):
        if img_pred.ndim == 3: img_pred = img_pred.unsqueeze(0)
        if img_gt.ndim == 3: img_gt = img_gt.unsqueeze(0)

        pred_01 = self._to_01(img_pred)
        gt_01 = self._to_01(img_gt)

        mse = torch.mean((pred_01 - gt_01) ** 2).item()
        psnr = psnr_func(pred_01, gt_01, data_range=1.0).item()
        ssim = ssim_func(pred_01, gt_01, data_range=1.0).item()

        pred_norm = pred_01 * 2 - 1
        gt_norm = gt_01 * 2 - 1
        with torch.no_grad():
            lpips_score = self.lpips_loss(pred_norm, gt_norm).item()

        return {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips_score}

    def _to_01(self, x):
        x = x.detach().float()
        min_v, max_v = x.min(), x.max()
        if max_v - min_v > 1e-8:
            return (x - min_v) / (max_v - min_v)
        return torch.zeros_like(x)

def compute_feature_metrics(feat_pred, feat_gt):
    cos_sim = F.cosine_similarity(feat_pred, feat_gt, dim=0).item()
    mse = F.mse_loss(feat_pred, feat_gt).item()
    return {'cos': cos_sim, 'mse': mse}
