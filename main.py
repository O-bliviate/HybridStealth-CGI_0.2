from loguru import logger
import torch
import torch.nn.functional as F
import arguments  # 确保导入了最新的 arguments.py

# [修改]: 移除 from conf import Config

# 导入各层模块
from datasets import get_dataloader
from federated_core import FederatedServer, LocalUpdate
from pmm_modules import MaliciousModel, inject_fingerprint
from attack_phase1_parsing import Phase1Parser
from attack_phase2_nash import Phase2Inverter
from metrics import compute_metrics


def main():
    # 1. 初始化配置与环境 (使用统一的 Arguments)
    args = arguments.Arguments(logger)

    # 打印参数以供检查
    args.log()
    logger.info(f"Initializing HybridStealth-CGI on device: {args.device}...")

    # 2. 准备数据与模型
    # get_dataloader 内部会使用 args.root_path 和 args.dataset
    tt, dst, idx_shuffle, num_classes = get_dataloader(args)

    # 初始化恶意模型
    # [注意]: args.device 现在直接存在于 Arguments 类中
    global_model = MaliciousModel(num_classes, device=args.device).to(args.device)

    # 3. 初始化联邦组件
    server = FederatedServer()
    # 容器: 存储真实数据用于评估
    gt_data_list = []

    logger.info(f"Simulating FL with {args.num_clients} clients...")

    # 4. 联邦训练循环 (Fingerprint -> Train -> Secure Aggregation)
    for i in range(args.num_clients):
        # A. 准备数据
        curr_idx = idx_shuffle[i]
        gt_img = tt(dst[curr_idx][0]).float().to(args.device).unsqueeze(0)
        gt_label = torch.tensor([dst[curr_idx][1]]).long().to(args.device)
        gt_data_list.append(gt_img)

        # B. 注入 Soft LOKI 指纹
        inject_fingerprint(global_model, i, args.num_clients)

        # C. 本地训练 & 计算梯度
        client = LocalUpdate(args, global_model, F.cross_entropy)
        grads = client.train(gt_img, gt_label)

        # D. 安全聚合 (服务器只做累加)
        server.secure_aggregate(grads)

        if (i + 1) % 20 == 0:
            logger.info(f"Processed Client {i + 1}/{args.num_clients}")

    # 获取聚合梯度
    agg_grads = server.get_aggregated_gradients()
    # 提取 PMM 层梯度 (Key 需要与 MaliciousModel 定义一致)
    # 通常是 'pmm.linear.weight'
    try:
        pmm_grad = agg_grads['pmm.linear.weight']
    except KeyError:
        # 兼容部分 PyTorch 版本命名差异
        pmm_grad = agg_grads['pmm.weight']

    # 5. 攻击引擎启动
    logger.info("Starting Attack Engine...")

    # 初始化 Phase 1 & 2 组件
    # Phase 1 需要目标矩阵 T
    parser = Phase1Parser(global_model.pmm.target_matrix, args.device)
    # Phase 2 需要完整的 args (包含 nash_lr, attack_iter 等)
    inverter = Phase2Inverter(global_model, args)

    total_mse, total_psnr, total_ssim, success = 0, 0, 0, 0

    # 循环执行攻击恢复
    for i in range(args.num_clients):
        # --- Phase 1: 解析 ---
        # 从聚合梯度中解出特征 (利用 args.num_clients 进行切片计算)
        feat_recovered = parser.parse(pmm_grad, i, args.num_clients)

        # --- Phase 2: 反演 ---
        # 纳什博弈恢复图像
        img_recon = inverter.invert(feat_recovered)

        # --- 评估 ---
        metrics = compute_metrics(img_recon, gt_data_list[i])
        total_mse += metrics['mse']
        total_psnr += metrics['psnr']
        total_ssim += metrics['ssim']

        # 成功判定阈值 (SSIM > 0.6)
        if metrics['ssim'] > 0.6:
            success += 1

        if (i + 1) % 10 == 0:
            logger.info(
                f"[Client {i}] MSE: {metrics['mse']:.4f} | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.3f}")

    # 6. 最终报告
    leak_rate = (success / args.num_clients) * 100
    avg_psnr = total_psnr / args.num_clients
    logger.info(f"FINAL RESULT: Leakage Rate: {leak_rate:.2f}%, Avg PSNR: {avg_psnr:.2f}")


if __name__ == '__main__':
    # 环境检查
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ 警告: 未检测到 GPU，正在使用 CPU。请检查是否安装了 CPU 版的 PyTorch。")

    main()