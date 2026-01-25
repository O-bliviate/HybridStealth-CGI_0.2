import torch
import numpy as np
from sklearn.decomposition import FastICA


class Phase1Parser:
    def __init__(self, target_matrix, device):
        self.target_matrix = target_matrix
        self.device = device

    def parse(self, aggregated_grad, client_id, total_clients):
        """
        执行解析流程: 频段锁定 -> 线性求逆 -> 信号分离
        """
        # 1. 频段锁定 (Energy Slicing)
        block_size = max(1, self.target_matrix.shape[0] // total_clients)
        start = client_id * block_size
        end = start + block_size

        # 提取梯度与目标矩阵切片
        grad_subset = aggregated_grad[start:end, :].detach()
        T_subset = self.target_matrix[start:end].detach()

        # 2. 线性逆问题求解 (Robbing the Fed Logic)
        # 理论: Grad \approx X^T * (-T)
        # 求解: X = (Grad^T * E) / ||E||^2, where E = -T
        E = -T_subset
        E_norm_sq = torch.dot(E, E) + 1e-8

        numerator = torch.matmul(grad_subset.t(), E)
        x_low_feat = numerator / E_norm_sq

        # 3. (可选) 盲源分离 ICA
        # 如果 block_size 较大，可用 ICA 进一步提纯
        if block_size >= 5:
            try:
                x_low_feat = self._apply_ica(grad_subset, x_low_feat)
            except:
                pass  # Fallback to linear solution

        return x_low_feat

    def _apply_ica(self, grad_subset, reference_feat):
        X_mixed = grad_subset.cpu().numpy().T
        ica = FastICA(n_components=1, whiten='unit-variance')
        S_ = ica.fit_transform(X_mixed)
        x_ica = torch.from_numpy(S_).squeeze().to(self.device)

        # 符号校正
        if torch.cosine_similarity(x_ica, reference_feat, dim=0) < 0:
            x_ica = -x_ica

        # 尺度恢复
        scale = torch.dot(reference_feat, x_ica) / torch.dot(x_ica, x_ica)
        return x_ica * scale