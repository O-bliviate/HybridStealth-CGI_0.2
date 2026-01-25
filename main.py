from loguru import logger
import torch
import torch.nn.functional as F
import arguments
import os  # [新增] 用于路径操作
import torchvision  # [新增] 用于保存图像

# 导入各层模块
from datasets import get_dataloader
from federated_core import FederatedServer, LocalUpdate
from pmm_modules import MaliciousModel, inject_fingerprint
from attack_phase1_parsing import Phase1Parser
from attack_phase2_nash import Phase2Inverter
# 引入新的评估模块
from metrics import Evaluator, compute_feature_metrics


def main():
    # 1. 初始化配置与环境
    args = arguments.Arguments(logger)
    args.log()
    logger.info(f"Initializing HybridStealth-CGI on device: {args.device}...")

    # 2. 准备数据与模型
    tt, dst, idx_shuffle, num_classes = get_dataloader(args)
    global_model = MaliciousModel(num_classes, device=args.device).to(args.device)

    # 初始化专业评估器 (加载 LPIPS)
    evaluator = Evaluator(args.device)

    # 3. 初始化联邦组件
    server = FederatedServer()
    gt_data_list = []

    logger.info(f"Simulating FL with {args.num_clients} clients...")

    # 4. 联邦训练循环
    for i in range(args.num_clients):
        # A. 准备数据
        curr_idx = idx_shuffle[i]
        gt_img = tt(dst[curr_idx][0]).float().to(args.device).unsqueeze(0)
        gt_label = torch.tensor([dst[curr_idx][1]]).long().to(args.device)
        gt_data_list.append(gt_img)

        # B. 注入指纹
        inject_fingerprint(global_model, i, args.num_clients)

        # C. 本地训练
        client = LocalUpdate(args, global_model, F.cross_entropy)
        grads = client.train(gt_img, gt_label)

        # D. 安全聚合
        server.secure_aggregate(grads)

        if (i + 1) % 20 == 0:
            logger.info(f"Processed Client {i + 1}/{args.num_clients}")

    # 获取 PMM 梯度
    agg_grads = server.get_aggregated_gradients()
    try:
        pmm_grad = agg_grads['pmm.linear.weight']
    except KeyError:
        pmm_grad = agg_grads['pmm.weight']

    # 5. 攻击引擎启动
    logger.info("Starting Attack Engine...")
    parser = Phase1Parser(global_model.pmm.target_matrix, args.device)
    inverter = Phase2Inverter(global_model, args)

    # 准备保存图像的目录 [新增]
    save_dir = os.path.join(args.root_path, 'results', 'reconstructions')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info(f"Reconstructed images will be saved to: {save_dir}")

    # 统计容器
    stats = {
        'ph1_cos': 0, 'ph1_mse': 0,
        'ph2_psnr': 0, 'ph2_ssim': 0, 'ph2_lpips': 0
    }
    success = 0

    # 打印表头
    print(f"\n{'ID':<4} | {'Ph1 Cos':<10} | {'Ph2 PSNR':<10} | {'Ph2 SSIM':<10} | {'Ph2 LPIPS':<10}")
    print("-" * 55)

    for i in range(args.num_clients):
        # --- 准备 Ground Truth Feature 用于 Phase 1 评估 ---
        with torch.no_grad():
            _, _, gt_feat = global_model(gt_data_list[i])
            gt_feat = gt_feat.squeeze(0)  # [512]

        # --- Phase 1: 解析 ---
        feat_recovered = parser.parse(pmm_grad, i, args.num_clients)

        # Phase 1 评估
        ph1_metrics = compute_feature_metrics(feat_recovered, gt_feat)
        stats['ph1_cos'] += ph1_metrics['cos']
        stats['ph1_mse'] += ph1_metrics['mse']

        # --- Phase 2: 反演 ---
        img_recon = inverter.invert(feat_recovered)

        # --- Phase 2: 评估 ---
        ph2_metrics = evaluator.compute_img_metrics(img_recon, gt_data_list[i])

        stats['ph2_psnr'] += ph2_metrics['psnr']
        stats['ph2_ssim'] += ph2_metrics['ssim']
        stats['ph2_lpips'] += ph2_metrics['lpips']

        if ph2_metrics['ssim'] > 0.6:
            success += 1

        # --- [新增] 保存对比图像 ---
        # 拼接 Ground Truth 和 Reconstructed Image (左右排列)
        # 注意：数据已经过 Normalize，save_image 的 normalize=True 会将其自动映射回 0-1 可视化范围
        comparison = torch.cat([gt_data_list[i], img_recon], dim=0)
        save_path = os.path.join(save_dir, f'client_{i}.png')
        torchvision.utils.save_image(comparison, save_path, nrow=2, normalize=True, value_range=(-2.5, 2.5))

        # 实时打印
        if (i + 1) % 10 == 0:
            print(
                f"{i:<4} | {ph1_metrics['cos']:<10.4f} | {ph2_metrics['psnr']:<10.2f} | {ph2_metrics['ssim']:<10.3f} | {ph2_metrics['lpips']:<10.3f}")

    # 6. 最终报告
    n = args.num_clients
    logger.info("-" * 30)
    logger.info(f"FINAL RESULTS (N={n})")
    logger.info(f"Phase 1 Avg Cosine Sim: {stats['ph1_cos'] / n:.4f}  (Target > 0.9)")
    logger.info(f"Phase 2 Avg PSNR:       {stats['ph2_psnr'] / n:.2f} dB")
    logger.info(f"Phase 2 Avg SSIM:       {stats['ph2_ssim'] / n:.4f}")
    logger.info(f"Phase 2 Avg LPIPS:      {stats['ph2_lpips'] / n:.4f} (Lower is better)")
    logger.info(f"Leakage Rate (SSIM>0.6): {(success / n) * 100:.2f}%")
    logger.info(f"Check images at: {save_dir}")


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    main()