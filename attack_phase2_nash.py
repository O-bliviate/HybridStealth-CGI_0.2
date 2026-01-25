import torch
import torch.nn.functional as F
import torch.optim as optim


class NashBargainingOptimizer:
    """
    CGI 纳什议价优化器
    """

    def __init__(self, num_servers, lr=0.1):
        self.num_servers = num_servers
        self.alpha = torch.ones(num_servers, requires_grad=True)
        self.lr = lr

    def get_weights(self, grads, target_grads):
        optimizer = optim.Adam([self.alpha], lr=self.lr)
        for _ in range(10):  # 内循环求解纳什均衡
            optimizer.zero_grad()
            weights = F.softmax(self.alpha, dim=0)

            utilities = []
            for k in range(self.num_servers):
                # 效用 = Cosine Similarity (梯度方向一致性)
                u = F.cosine_similarity(grads[k].view(-1).unsqueeze(0),
                                        target_grads[k].view(-1).unsqueeze(0))
                utilities.append(u)

            # Maximize Nash Product: sum log(u_k)
            loss = -sum([weights[k] * torch.log(torch.clamp(utilities[k], min=1e-6))
                         for k in range(self.num_servers)])
            loss.backward()
            optimizer.step()

        return F.softmax(self.alpha, dim=0).detach()


class Phase2Inverter:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.nash = NashBargainingOptimizer(args.num_colluding_servers, args.nash_lr)

    def invert(self, target_feat):
        """
        特征反演主循环
        """
        dummy_data = torch.randn(1, 3, 32, 32).to(self.args.device).requires_grad_(True)
        optimizer = optim.Adam([dummy_data], lr=0.05)

        # 模拟合谋: Server 2 看到带噪版本
        target_feat_s1 = target_feat.detach().unsqueeze(0)
        target_feat_s2 = target_feat_s1 + torch.randn_like(target_feat_s1) * 0.05

        for step in range(self.args.attack_iter):
            optimizer.zero_grad()
            _, _, curr_feat = self.model(dummy_data)

            # --- 构建博弈 ---
            # Player 1: 严格特征匹配
            loss1 = 1.0 - F.cosine_similarity(curr_feat, target_feat_s1)
            # Player 2: 带噪匹配 + 图像先验 (TV)
            tv = torch.sum(torch.abs(dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:])) + \
                 torch.sum(torch.abs(dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :]))
            loss2 = 1.0 - F.cosine_similarity(curr_feat, target_feat_s2) + 0.005 * tv

            # 计算各自梯度
            g1 = torch.autograd.grad(loss1, dummy_data, retain_graph=True)[0]
            g2 = torch.autograd.grad(loss2, dummy_data, retain_graph=True)[0]

            # 纳什聚合
            weights = self.nash.get_weights([g1, g2], [g1, g1])
            final_grad = weights[0] * g1 + weights[1] * g2

            dummy_data.grad = final_grad
            optimizer.step()

            with torch.no_grad(): dummy_data.clamp_(-2.5, 2.5)

        return dummy_data.detach()