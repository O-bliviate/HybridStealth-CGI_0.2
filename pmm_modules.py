import torch
import torch.nn as nn
import torchvision


class ParsableLinear(nn.Module):
    """
    [PMM 核心]
    1. 线性层设计
    2. 辅助损失函数 L_pmm
    3. Soft LOKI 指纹注入
    """

    def __init__(self, input_dim, output_dim, target_scale=1000.0, device='cpu'):
        super(ParsableLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # 无 Bias 线性层，简化逆问题求解
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        # 预设目标矩阵 T
        self.target_matrix = torch.randn(output_dim).to(device) * target_scale

    def inject_statistical_fingerprint(self, client_id, total_clients):
        """
        [Soft LOKI] 注入块对角统计指纹
        """
        block_size = max(1, self.output_dim // total_clients)
        start = client_id * block_size
        end = start + block_size

        with torch.no_grad():
            # 背景噪声 (极小)
            self.linear.weight.data.normal_(0, 1e-5)
            # 专属频段 (高增益正交)
            fingerprint = torch.empty(end - start, self.input_dim)
            nn.init.orthogonal_(fingerprint, gain=2.0)
            self.linear.weight.data[start:end, :] = fingerprint.to(self.device)

    def forward(self, x):
        return self.linear(x)

    def compute_malicious_loss(self, out):
        # L = 1/2 || Z - T ||^2
        if out.shape[0] != self.target_matrix.shape[0]:
            target = self.target_matrix.expand_as(out)
        else:
            target = self.target_matrix
        return 0.5 * torch.norm(out - target, p='fro') ** 2


class MaliciousModel(nn.Module):
    def __init__(self, num_classes=100, device='cpu'):
        super(MaliciousModel, self).__init__()
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base.maxpool = nn.Identity()
        self.base.fc = nn.Identity()

        self.pmm = ParsableLinear(512, num_classes, device=device)
        self.device = device

    def forward(self, x):
        feat = self.base.conv1(x)
        feat = self.base.bn1(feat)
        feat = self.base.relu(feat)
        feat = self.base.layer1(feat)
        feat = self.base.layer2(feat)
        feat = self.base.layer3(feat)
        feat = self.base.layer4(feat)
        feat = self.base.avgpool(feat)
        feat = torch.flatten(feat, 1)

        logits = self.pmm(feat)
        loss = self.pmm.compute_malicious_loss(logits)

        # 返回 feat 供 Phase 2 特征匹配
        return logits, loss, feat


def inject_fingerprint(model, client_id, total_clients):
    model.pmm.inject_statistical_fingerprint(client_id, total_clients)