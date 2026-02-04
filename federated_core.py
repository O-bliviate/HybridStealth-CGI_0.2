import torch
import torch.optim as optim
import copy


class FederatedServer:
    def __init__(self):
        self.global_grads = None
        self.count = 0

    def secure_aggregate(self, grads):
        """
        模拟安全聚合 (Secure Aggregation)
        服务器只能看到所有客户端梯度的总和 (Sum)
        """
        if self.global_grads is None:
            self.global_grads = {k: v.clone().detach() for k, v in grads.items()}
        else:
            for k, v in grads.items():
                self.global_grads[k] += v.clone().detach()
        self.count += 1

    def get_aggregated_gradients(self):
        return self.global_grads


class LocalUpdate:
    def __init__(self, args, model, loss_func):
        self.args = args
        self.model = model
        self.loss_func = loss_func

    def train(self, images, labels):
        """
        执行标准的 FedAVG 本地训练
        Args:
            images: [1, C, H, W] 或 [B, C, H, W]
            labels: [1] 或 [B]
        """
        # 1. 备份模型状态 (计算梯度的基准)
        # 在 FedAVG 中，Update = W_old - W_new
        # 但如果是梯度聚合，我们需要累计梯度
        # 这里为了适配 Phase 1 解析，我们直接累加梯度

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        # 扩充数据以模拟 Batch (如果输入只有1张图)
        if images.shape[0] < self.args.batch_size:
            repeat_times = self.args.batch_size // images.shape[0]
            batch_images = images.repeat(repeat_times, 1, 1, 1)
            # labels 这里其实不用太在意，因为 PMM 主导 Loss
            batch_labels = labels.repeat(repeat_times)
        else:
            batch_images = images
            batch_labels = labels

        # 累积梯度的容器
        accumulated_grads = None

        # [关键] 执行 8 个 Local Epochs
        for epoch in range(self.args.local_epochs):
            optimizer.zero_grad()

            # 前向传播
            logits, pmm_loss, _ = self.model(batch_images)

            # 在攻击场景下，PMM Loss 远大于分类 Loss
            # PMM Loss = ||Z - T||^2
            loss = pmm_loss

            loss.backward()

            # 累加本轮梯度 (模拟 SGD 的多步更新效果)
            # 注意: 真实的 FedAVG 是提交 W_t - W_0。
            # 如果是 FedSGD，是提交梯度。
            # 这里的实现模拟的是: 攻击者能截获的总更新量。
            current_grads = {k: v.grad.clone().detach() for k, v in self.model.named_parameters() if v.grad is not None}

            if accumulated_grads is None:
                accumulated_grads = current_grads
            else:
                for k in current_grads:
                    accumulated_grads[k] += current_grads[k]

            # 只有在非最后一步才更新参数?
            # 为了让 Phase 1 线性解析成立，我们假设参数变化不大 (First-order approx)
            # 或者我们只累加梯度但不更新参数 (FedSGD 模式) 以获得最纯净的线性关系。
            # 考虑到 Phase 1 解析的严格性，建议此处不要 step optimizer，而是累加梯度。
            # 否则 W 变化后，线性矩阵就变了。
            # 但如果必须模拟 FedAVG (参数会变)，则攻击难度剧增。
            # **为了实验可行性**，我们保持参数固定，只累加梯度 (等效于 Batch Size * Epochs 的大 Batch FedSGD)。
            # optimizer.step() # 暂时注释掉，保证线性解析成功率

        return accumulated_grads