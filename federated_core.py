import torch
import copy


class FederatedServer:
    """
    模拟安全聚合服务器
    """

    def __init__(self):
        self.global_gradients = {}

    def secure_aggregate(self, client_grads):
        """
        [关键] 安全聚合逻辑
        Server 仅能看到 sum(g_i)，无法看到单独的 g_i
        """
        for name, grad in client_grads.items():
            if name not in self.global_gradients:
                self.global_gradients[name] = torch.zeros_like(grad)
            self.global_gradients[name] += grad

    def get_aggregated_gradients(self):
        return self.global_gradients

    def reset(self):
        self.global_gradients = {}


class LocalUpdate:
    """
    模拟客户端本地训练
    """

    def __init__(self, args, model, loss_func):
        self.args = args
        self.model = model
        self.loss_func = loss_func

    def train(self, img, label):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        optimizer.zero_grad()

        # Forward (model returns 3 values as per previous design)
        output, pmm_loss, _ = self.model(img)

        # Loss Calculation
        cls_loss = self.loss_func(output, label)
        total_loss = cls_loss + self.args.pmm_scale * pmm_loss

        total_loss.backward()

        # Return gradients
        return {k: v.grad.clone().detach() for k, v in self.model.named_parameters() if v.grad is not None}