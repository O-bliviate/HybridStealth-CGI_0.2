import torch
import torch.nn as nn
import torchvision
import os
import joblib  # 需要 pip install joblib
import numpy as np


class ParsableLinear(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu', root_path='./data'):
        super(ParsableLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        weights_path = os.path.join(root_path, 'model_weights', 'pmm_weights.pth')
        target_path = os.path.join(root_path, 'model_weights', 'target_matrix.pth')
        scaler_path = os.path.join(root_path, 'model_weights', 'scaler.gz')

        self.pretrained = False
        self.scaler_mean = None
        self.scaler_std = None

        if os.path.exists(weights_path):
            try:
                self.linear.weight.data = torch.load(weights_path, map_location=device)
                self.all_targets = torch.load(target_path, map_location=device)
                self.register_buffer('target_matrix', self.all_targets[0])

                # [新增] 加载 Scaler 参数
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    # 转换为 Tensor 这里的 scaler.mean_ 是 numpy array
                    self.register_buffer('scaler_mean', torch.from_numpy(scaler.mean_).float().to(device))
                    self.register_buffer('scaler_std', torch.from_numpy(scaler.scale_).float().to(device))

                self.pretrained = True
                print("[PMM] ✅ Loaded Weights & Scaler.")
            except Exception as e:
                print(f"[PMM] Error: {e}")

        if not self.pretrained:
            # Fallback
            nn.init.kaiming_normal_(self.linear.weight)
            self.register_buffer('target_matrix', torch.randn(output_dim).to(device))

    def set_target_for_client(self, client_id, target_scale):
        if self.pretrained:
            idx = client_id % len(self.all_targets)
            self.target_matrix = self.all_targets[idx] * target_scale

    def forward(self, x):
        # [关键] 在进入 Linear 层之前，应用标准化
        # 这一步模拟了预训练时的 StandardScaler
        if self.scaler_mean is not None:
            x_normalized = (x - self.scaler_mean) / (self.scaler_std + 1e-8)
            return self.linear(x_normalized)
        else:
            return self.linear(x)

    # compute_malicious_loss 保持不变...
    def compute_malicious_loss(self, out):
        if out.shape[0] != self.target_matrix.shape[0]:
            target = self.target_matrix.expand_as(out)
        else:
            target = self.target_matrix
        return 0.5 * torch.norm(out - target, p='fro') ** 2


# MaliciousModel 类保持不变...
class MaliciousModel(nn.Module):
    def __init__(self, num_classes=100, device='cpu', root_path='./data'):
        super(MaliciousModel, self).__init__()
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base.maxpool = nn.Identity()
        self.base.fc = nn.Identity()
        self.pmm = ParsableLinear(512, 512, device=device, root_path=root_path)

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
        return logits, loss, feat


def inject_fingerprint(model, client_id, total_clients, target_scale):
    model.pmm.set_target_for_client(client_id, target_scale)