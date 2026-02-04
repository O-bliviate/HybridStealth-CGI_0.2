from loguru import logger
import torch
import torch.nn.functional as F
import arguments
import os
import torchvision
import pretrain_pmm  # [恢复] 必须导入

from datasets import get_dataloader
from federated_core import FederatedServer, LocalUpdate
from pmm_modules import MaliciousModel, inject_fingerprint
from attack_phase1_parsing import Phase1Parser
from attack_phase2_nash import Phase2Inverter
from metrics import Evaluator, compute_feature_metrics


def main():
    args = arguments.Arguments(logger)
    args.log()

    # 0. 自动执行元学习 (Meta-Learning)
    weight_path = os.path.join(args.root_path, 'model_weights', 'pmm_weights.pth')
    if not os.path.exists(weight_path):
        logger.warning("⚠️ Training PMM from scratch (Meta-Learning)...")
        # 这会调用带有 Non-IID 数据的训练脚本
        pretrain_pmm.train_pmm()

    # 1. 环境准备
    tt, dst, idx_shuffle, num_classes = get_dataloader(args)
    # 加载刚刚训练好的权重
    global_model = MaliciousModel(num_classes, device=args.device, root_path=args.root_path).to(args.device)

    evaluator = Evaluator(args.device)
    server = FederatedServer()
    gt_data_list = []

    logger.info(f"Experiment: {args.num_clients} Clients, Scale={args.target_scale}")

    # 2. 联邦训练
    for i in range(args.num_clients):
        curr_idx = idx_shuffle[i]
        gt_img = tt(dst[curr_idx][0]).float().to(args.device).unsqueeze(0)
        gt_label = torch.tensor([dst[curr_idx][1]]).long().to(args.device)
        gt_data_list.append(gt_img)

        # 注入目标 (设置 C)
        inject_fingerprint(global_model, i, args.num_clients, args.target_scale)

        client = LocalUpdate(args, global_model, F.cross_entropy)
        grads = client.train(gt_img, gt_label)
        server.secure_aggregate(grads)

        if (i + 1) % 10 == 0: logger.info(f"Client {i + 1} Training Completed.")

    agg_grads = server.get_aggregated_gradients()
    try:
        pmm_grad = agg_grads['pmm.linear.weight']
    except KeyError:
        pmm_grad = agg_grads['pmm.weight']

    # 3. 攻击阶段
    logger.info("Starting Attack Engine...")
    # [新增] 传入 scaler 参数
    parser = Phase1Parser(
        global_model.pmm.target_matrix,
        global_model.pmm.scaler_mean,
        global_model.pmm.scaler_std,
        args.device
    )
    inverter = Phase2Inverter(global_model, args)  # 内部包含 LBFGS + Nash

    save_dir = os.path.join(args.root_path, 'results', 'reconstructions')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    print("\n" + "=" * 165)
    print(
        f"{'ID':<3} | {'Ph1 Feature':<11} | {'Phase 1 (LBFGS Init)':<24} | {'Phase 2 (Nash Final)':<24} | {'Improvement':<26}")
    print(
        f"{'':<3} | {'Cos Sim':<11} | {'PSNR':<6} {'SSIM':<8} {'LPIPS':<6} | {'PSNR':<6} {'SSIM':<8} {'LPIPS':<6} | {'d_PSNR':<8} {'d_SSIM':<8} {'d_LPIPS':<6}")
    print("-" * 165)

    stats = {'ph2_psnr': 0, 'ph2_ssim': 0}

    for i in range(args.num_clients):
        with torch.no_grad():
            _, _, gt_feat = global_model(gt_data_list[i])
            gt_feat = gt_feat.squeeze(0)

        # Phase 1: 解析
        global_model.pmm.set_target_for_client(i, args.target_scale)
        parser.target_matrix = global_model.pmm.target_matrix

        total_scale = args.batch_size * args.local_epochs
        feat_recovered = parser.parse(pmm_grad, i, total_scale)

        ph1_feat = compute_feature_metrics(feat_recovered, gt_feat)

        # Phase 2 (LBFGS + Nash)
        img_init, img_final = inverter.invert(feat_recovered)

        ph1_img = evaluator.compute_img_metrics(img_init, gt_data_list[i])
        ph2_img = evaluator.compute_img_metrics(img_final, gt_data_list[i])

        stats['ph2_psnr'] += ph2_img['psnr']
        stats['ph2_ssim'] += ph2_img['ssim']

        d_psnr = ph2_img['psnr'] - ph1_img['psnr']
        d_ssim = ph2_img['ssim'] - ph1_img['ssim']
        d_lpips = ph2_img['lpips'] - ph1_img['lpips']

        print(f"{i:<3} | {ph1_feat['cos']:<11.4f} | "
              f"{ph1_img['psnr']:<6.2f} {ph1_img['ssim']:<8.3f} {ph1_img['lpips']:<6.3f} | "
              f"{ph2_img['psnr']:<6.2f} {ph2_img['ssim']:<8.3f} {ph2_img['lpips']:<6.3f} | "
              f"{d_psnr:<+8.2f} {d_ssim:<+8.3f} {d_lpips:<+6.3f}")

        comparison = torch.cat([gt_data_list[i], img_init, img_final], dim=0)
        torchvision.utils.save_image(comparison, os.path.join(save_dir, f'client_{i}.png'), nrow=3, normalize=True,
                                     value_range=(-2.5, 2.5))

    n = args.num_clients
    logger.info("=" * 60)
    logger.info(f"Avg SSIM: {stats['ph2_ssim'] / n:.4f} | Avg PSNR: {stats['ph2_psnr'] / n:.2f}")


if __name__ == '__main__':
    main()