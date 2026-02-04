import torch
import time

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)


class Arguments:
    def __init__(self, logger):
        self.logger = logger

        # === 基础环境 ===
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = SEED

        # === 路径与数据集 ===
        self.dataset = 'cifar100'
        self.root_path = './data'

        # === 联邦学习配置 ===
        self.num_clients = 50
        self.batch_size = 4
        self.local_epochs = 8
        self.lr = 0.01

        # === 网络配置 ===
        self.net = 'resnet18'

        # === HybridStealth-CGI 核心参数 ===
        self.num_colluding_servers = 2
        # [核心修正] 目标放大倍数必须足够大，才能让 PMM 主导梯度方向
        # 建议设为 10000.0，解决 Cos Sim 只有 0.5 的问题
        self.target_scale = 100.0

        # === Phase 2 配置 (CGI Style) ===
        self.nash_lr = 0.05
        self.attack_iter = 500  # 增加迭代次数
        self.tv_reg = 0.001  # TV 正则化系数

        # [新增] LBFGS 预热轮数
        self.warmup_iter = 100

        # 实验输出配置
        self.log_interval = 10

    def log(self):
        self.logger.debug(f"Arguments: {str(self)}")

    def __str__(self):
        return f"HybridStealth-CGI Args | Scale: {self.target_scale} | Epochs: {self.local_epochs}"