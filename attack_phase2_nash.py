import torch
import torch.nn.functional as F
import torch.optim as optim


class MinNormSolver:
    """
    [MGDA 核心] 寻找多梯度的最小范数组合
    解决 Server A (Feature) 和 Server B (Prior) 的梯度冲突
    """

    def find_min_norm_element(self, grads):
        g1, g2 = grads[0], grads[1]
        g1_flat = g1.view(-1)
        g2_flat = g2.view(-1)

        g1_g1 = torch.dot(g1_flat, g1_flat)
        g2_g2 = torch.dot(g2_flat, g2_flat)
        g1_g2 = torch.dot(g1_flat, g2_flat)

        denominator = g1_g1 + g2_g2 - 2 * g1_g2
        if denominator < 1e-8:
            return 0.5 * g1 + 0.5 * g2

        alpha = (g2_g2 - g1_g2) / denominator
        alpha = torch.clamp(alpha, 0, 1)

        g_star = alpha * g1 + (1 - alpha) * g2
        return g_star, alpha


class Phase2Inverter:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.solver = MinNormSolver()

    def tv_loss(self, img):
        """计算 Total Variation Loss，用于去噪和平滑"""
        diff1 = img[:, :, :, :-1] - img[:, :, :, 1:]
        diff2 = img[:, :, :-1, :] - img[:, :, 1:, :]
        return torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2))

    def invert(self, target_feat):
        """
        Returns: img_init (LBFGS Result), img_final (Nash/Adam Result)
        """
        # =========================================================
        # Step 0: 强初始化 (Warm-up with LBFGS)
        # LBFGS 擅长在平滑曲面上快速收敛，非常适合初期从噪声恢复轮廓
        # =========================================================
        dummy_data = torch.randn(1, 3, 32, 32).to(self.args.device).requires_grad_(True)
        target_feat_fixed = target_feat.detach().unsqueeze(0)

        # 使用 LBFGS 进行预热
        # history_size=100 是常用配置
        optimizer_lbfgs = optim.LBFGS([dummy_data], max_iter=20, history_size=100, lr=1)

        # LBFGS 需要 closure 函数
        def closure():
            optimizer_lbfgs.zero_grad()
            _, _, curr_feat = self.model(dummy_data)
            # 预热阶段：混合 Loss (Feature + 小部分 TV)
            loss = F.mse_loss(curr_feat, target_feat_fixed) + 0.0001 * self.tv_loss(dummy_data)
            loss.backward()
            return loss

        # 执行预热
        for i in range(self.args.warmup_iter // 20):  # LBFGS step 一次包含多次迭代
            optimizer_lbfgs.step(closure)
            with torch.no_grad():
                dummy_data.clamp_(-2.5, 2.5)

        img_init = dummy_data.clone().detach()

        # =========================================================
        # Step 1: 纳什博弈主循环 (Game with Adam + MGDA)
        # =========================================================
        # 使用 Adam 进行精细博弈
        optimizer = optim.Adam([dummy_data], lr=0.01)
        # 引入学习率衰减，防止后期震荡导致 LPIPS 上升
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 400], gamma=0.5)

        for step in range(self.args.attack_iter):
            optimizer.zero_grad()

            # --- 1. Server A (Feature Fidelity) ---
            # 目标: 尽可能匹配特征
            _, _, feat_a = self.model(dummy_data)

            # 使用 Cosine (方向) + MSE (强度)
            # 重点: 这里的权重决定了 A 的“固执程度”
            loss_a = 1.0 - F.cosine_similarity(feat_a, target_feat_fixed) + 0.1 * F.mse_loss(feat_a, target_feat_fixed)
            g_a = torch.autograd.grad(loss_a, dummy_data, retain_graph=True)[0]

            # --- 2. Server B (Image Prior) ---
            # 目标: 图像自然度 (TV Loss + L2 Norm)
            # 重点: 这里的 loss_b 如果太大，MGDA 会倾向于完全平滑图像；如果太小，压不住噪点。

            # [CGI 策略] 动态调整 TV 权重? 这里我们先固定一个较强的值
            # 0.001 的 TV Loss 系数对于 CIFAR100 来说通常是合理的
            val_tv = self.tv_loss(dummy_data)
            val_norm = torch.norm(dummy_data)

            loss_b = 0.005 * val_tv + 0.01 * val_norm
            g_b = torch.autograd.grad(loss_b, dummy_data, retain_graph=True)[0]

            # --- 3. MGDA (Min-Norm) ---
            # 寻找帕累托最优方向
            final_grad, alpha = self.solver.find_min_norm_element([g_a, g_b])

            # 更新
            dummy_data.grad = final_grad
            optimizer.step()
            scheduler.step()  # 更新学习率

            # 约束
            with torch.no_grad():
                dummy_data.clamp_(-2.5, 2.5)

        img_final = dummy_data.detach()
        return img_init, img_final