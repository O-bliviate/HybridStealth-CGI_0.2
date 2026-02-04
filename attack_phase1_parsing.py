import torch
import torch.nn.functional as F


class Phase1Parser:
    def __init__(self, target_matrix, scaler_mean, scaler_std, device):
        """
        [新增] 传入 Scaler 参数用于反标准化
        """
        self.target_matrix = target_matrix
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.device = device

    def parse(self, agg_grad, client_id, total_scale):
        # 1. 投影分离 (Standard Logic)
        T_k = self.target_matrix
        if T_k.ndim == 2: T_k = T_k[0]

        projected = torch.matmul(agg_grad, -T_k)

        norm_sq = torch.norm(T_k) ** 2
        # 这是 PMM 输入层(Normalized Space) 的特征近似值
        x_norm = projected / (norm_sq * total_scale + 1e-8)

        # 2. [关键修正] 反标准化 (Inverse Transform)
        # X_original = X_norm * std + mean
        if self.scaler_mean is not None:
            x_feat = x_norm * self.scaler_std + self.scaler_mean
        else:
            x_feat = x_norm

        # 3. ReLU 约束
        x_feat = F.relu(x_feat)

        return x_feat