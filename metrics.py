import torch
import torch.nn.functional as F
import numpy as np


def compute_metrics(img_pred, img_gt):
    def to_01(x):
        x = x.float()
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    img_pred = to_01(img_pred)
    img_gt = to_01(img_gt)

    mse = F.mse_loss(img_pred, img_gt).item()
    psnr = 10 * np.log10(1 / (mse + 1e-9))

    # 简易 SSIM 估计
    ssim = 1.0 - np.sqrt(mse)
    if ssim < 0: ssim = 0

    return {'mse': mse, 'psnr': psnr, 'ssim': ssim}