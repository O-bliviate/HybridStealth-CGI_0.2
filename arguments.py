import torch.nn.functional as F
import torch
import json
import time
import os

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)


class Arguments:
    def __init__(self, logger):
        self.logger = logger

        # === 基础环境配置 (来自 Config) ===
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.int_time = int(time.time())
        self.seed = SEED

        # === 路径与数据集 ===
        self.debugOrRun = 'results'
        self.dataset = 'cifar100'  # 'cifar100', 'lfw', 'mnist', 'celebA', 'stl10'
        self.root_path = './data'  # [合并] 使用 Config 的路径，更整洁
        self.model_path = './model'

        # === 联邦学习基础配置 ===
        self.num_clients = 50  # [合并] 更新为 100 (Config的值)
        self.batch_size = 4  # 客户端本地训练 BatchSize
        self.num_dummy = 1  # 攻击重建时的 BatchSize
        self.iteration = 1000  # 攻击迭代次数 (或 FL 轮次，视具体逻辑)
        self.lr = 0.01  # [合并] 更新为 0.01 (FL 学习率)

        # === 网络与优化器 ===
        self.net = 'lenet'  # 'lenet', 'resnet20-4', 'resnet34', 'vgg11'
        self.net_mt_diff = True
        self.optim = 'Adam'
        self.scheduler = False

        # === 攻击与防御配置 ===
        self.methods = ['HybridStealth']
        self.defense_method = 'none'
        self.noise_std = 0.0001
        self.max_grad_norm_clipping = 4.0
        self.sparsification_defense_sparsity = 90
        self.save_final_img = False

        # === HybridStealth-CGI 核心参数 (合并自 Config) ===
        self.num_servers = 2  # 服务器总数
        self.num_colluding = 2  # 合谋服务器数量
        self.num_colluding_servers = 2  # [别名] 为了兼容可能使用旧名的地方

        self.pmm_scale = 10.0  # PMM 损失权重
        self.target_scale = 1000.0  # [新增] T矩阵量级 (保证 Z << T)
        self.nash_lr = 0.05  # [合并] 纳什博弈学习率
        self.attack_iter = 300  # [新增] Phase 2 攻击迭代轮数
        self.tv_reg = 0.001  # TV 正则化权重
        self.imidx = 10080  # 目标图片索引
        self.set_imidx = 10080  # 兼容旧名

        # === 其他杂项 ===
        self.use_game = True
        self.earlystop = 1e-9
        self.num_exp = 1
        self.log_interval = 5
        self.diff_task_agg = 'random'
        self.inv_loss = 'sim'
        self.eval_metrics = ['mse', 'lpips', 'psnr', 'ssim']

        # Data Loader Paths
        self.train_data_loader_pickle_path = "data_loaders/cifar100/train_data_loader.pickle"
        self.test_data_loader_pickle_path = "data_loaders/cifar100/test_data_loader.pickle"

    # Getter 方法 (保持兼容性)
    def get_logger(self): return self.logger

    def get_dataset(self): return self.dataset

    def get_root_path(self): return self.root_path

    def get_lr(self): return self.lr

    def get_iteration(self): return self.iteration

    def get_methods(self): return self.methods

    def get_net(self): return self.net

    def get_imidx(self): return self.set_imidx

    def get_num_dummy(self): return self.num_dummy

    def get_debugOrRun(self): return self.debugOrRun

    def get_num_exp(self): return self.num_exp

    def get_log_interval(self): return self.log_interval

    def log(self):
        self.logger.debug(f"Arguments: {str(self)}")

    def __str__(self):
        return (f"\nDataset: {self.dataset}\n"
                f"Device: {self.device}\n"
                f"Num Clients: {self.num_clients}\n"
                f"Model: {self.net}\n"
                f"PMM Scale: {self.pmm_scale}, Target Scale: {self.target_scale}\n"
                f"Attack Iter: {self.attack_iter}, Nash LR: {self.nash_lr}\n")