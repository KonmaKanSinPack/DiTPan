# =============================================================================
# DiTPan 训练脚本
# =============================================================================
#
# 训练流程:
#   1. 加载数据集 (WV3/GF2/QB/CAVE/Harvard)
#   2. 初始化DiTPan模型 + ShiftDiffusion
#   3. 训练循环: 每步随机采样时间步, 前向加噪, 模型预测, 反向传播
#   4. 定期验证: 完整采样并计算SAM/ERGAS/PSNR/CC/SSIM
#   5. 保存最优模型 (按SAM指标)
#
# 使用方式:
#   python train.py --dataset wv3 --train_path data/wv3/train_wv3.h5 \
#                   --valid_path data/wv3/test_wv3_multiExm1.h5
# =============================================================================

import argparse
import os
import time
from copy import deepcopy
from functools import partial

import einops
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.dit_pan import DiTPan_S, DiTPan_B
from diffusion.shift_diffusion import ShiftDiffusion, make_sqrt_etas_schedule
from dataset.pan_dataset import PanDataset
from dataset.hisr import HISRDataSets
from utils.logger import TensorboardLogger
from utils.lr_scheduler import StepsAll
from utils.metric import AnalysisPanAcc
from utils.misc import grad_clip, model_load, path_legal_checker, exist
from utils.optim_utils import EmaUpdater


# =============================================================================
#  数据集配置
# =============================================================================

# 各数据集的归一化除数
DIVISION_DICT = {
    "wv3": 2047.0, "wv2": 2047.0, "gf2": 1023.0,
    "qb": 2047.0, "cave": 1.0, "harvard": 1.0,
}

# 各数据集的RGB通道索引 (用于可视化)
RGB_CHANNEL_DICT = {
    "wv3": [4, 2, 0], "wv2": [4, 2, 0], "gf2": [0, 1, 2],
    "qb": [0, 1, 2], "cave": [29, 19, 9], "harvard": [29, 19, 9],
}


def get_dataset_config(dataset_name):
    """
    获取数据集相关配置

    返回: (image_n_channel, add_n_channel)
      - image_n_channel: MS图像通道数
      - add_n_channel: 附加条件通道数 (PAN)
    """
    if dataset_name in ["harvard", "cave"]:
        return 31, 3
    elif dataset_name == "wv3":
        return 8, 1
    elif dataset_name in ["gf2", "qb", "wv2"]:
        return 4, 1
    else:
        raise NotImplementedError(f"不支持的数据集: {dataset_name}")


# =============================================================================
#  训练主函数
# =============================================================================

def train(args):
    """
    DiTPan 训练主函数

    自检:
      1. 数据加载: 正确分离 pan, lms, hr, wavelets ✓
      2. 条件打包: cond = [lms, pan, wavelets_upsampled] ✓
      3. 损失计算: model预测e_0, 与(hr-lms)计算MSE ✓
      4. 验证: 完整DDPM采样 + 指标计算 ✓
      5. EMA: 指数移动平均模型用于推理 ✓
    """
    device = args.device
    torch.cuda.set_device(device)

    # =================== 配置 ===================
    image_n_channel, add_n_channel = get_dataset_config(args.dataset_name)

    # =================== 日志 ===================
    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    comment = f"DiTPan_{args.dataset_name}"
    logger = TensorboardLogger(
        file_logger_name=f"{stf_time}-{comment}",
        place=os.path.join(args.log_dir, "runs"),
        file_dir=os.path.join(args.log_dir, "logs"),
    )
    logger.print(f"数据集: {args.dataset_name}")
    logger.print(f"归一化除数: {DIVISION_DICT[args.dataset_name]}")

    # =================== 模型初始化 ===================
    model_fn = DiTPan_S if args.model_size == "S" else DiTPan_B
    denoise_fn = model_fn(
        input_size=args.image_size,
        in_channels=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=add_n_channel,
        self_condition=True,
    ).to(device)

    n_params = sum(p.numel() for p in denoise_fn.parameters()) / 1e6
    logger.print(f"模型参数量: {n_params:.2f}M")

    # 加载预训练权重 (可选)
    if args.pretrain_weight is not None:
        model_load(args.pretrain_weight, denoise_fn, strict=False, device=device)
        logger.print(f"加载预训练权重: {args.pretrain_weight}")

    # =================== 数据集 ===================
    d_train = h5py.File(args.train_path, "r")
    d_valid = h5py.File(args.valid_path, "r")

    if args.dataset_name in ["wv3", "wv2", "gf2", "qb"]:
        DatasetUsed = partial(
            PanDataset,
            full_res=False,
            norm_range=False,
            division=DIVISION_DICT[args.dataset_name],
            aug_prob=0,
            wavelets=True,
        )
    elif args.dataset_name in ["cave", "harvard"]:
        DatasetUsed = partial(
            HISRDataSets, normalize=False, aug_prob=0, wavelets=True
        )
    else:
        raise NotImplementedError(f"不支持的数据集: {args.dataset_name}")

    ds_train = DatasetUsed(d_train)
    ds_valid = DatasetUsed(d_valid)

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.num_workers, drop_last=False,
    )
    dl_valid = DataLoader(
        ds_valid, batch_size=min(10, len(ds_valid)), shuffle=False,
        pin_memory=True, num_workers=0, drop_last=False,
    )

    # =================== 扩散过程 ===================
    diffusion = ShiftDiffusion(
        denoise_fn,
        channels=image_n_channel,
        pred_mode="x_start",
        loss_type=args.loss_type,
        device=device,
        clamp_range=(0, 1),
        penalty_weight=args.penalty_weight,
    )
    diffusion.set_new_noise_schedule(
        sqrt_etas=make_sqrt_etas_schedule(
            schedule=args.schedule_type, n_timestep=args.n_steps
        )
    )
    diffusion = diffusion.to(device)

    # =================== EMA ===================
    ema_updater = EmaUpdater(
        diffusion, deepcopy(diffusion), decay=0.995, start_iter=20_000
    )

    # =================== 优化器与调度器 ===================
    optimizer = torch.optim.AdamW(
        denoise_fn.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100_000, 200_000, 350_000, 500_000, 750_000],
        gamma=0.2,
    )
    schedulers = StepsAll(scheduler)

    # =================== 训练循环 ===================
    iterations = args.start_iter
    best_sam = float("inf")

    logger.print(f"开始训练, 起始迭代: {iterations}, 最大迭代: {args.max_iterations}")
    logger.print(f"扩散步数: {args.n_steps}, 调度类型: {args.schedule_type}")
    logger.print(f"批量大小: {args.batch_size}, 学习率: {args.lr}")

    while iterations <= args.max_iterations:
        for pan, lms, hr, wavelets in dl_train:
            pan, lms, hr, wavelets = map(lambda x: x.to(device), (pan, lms, hr, wavelets))

            # 打包条件: [lms, pan, wavelets_upsampled]
            cond, _ = einops.pack(
                [
                    lms,
                    pan,
                    F.interpolate(wavelets, size=lms.shape[-1], mode="bilinear"),
                ],
                "b * h w",
            )

            # 训练一步
            optimizer.zero_grad()
            loss = diffusion(x=hr, y=lms, cond=cond, mode="train", current_iter=iterations)
            loss.backward()

            grad_clip(denoise_fn.parameters(), mode="norm", value=0.5)
            optimizer.step()
            ema_updater.update(iterations)
            schedulers.step()
            iterations += 1

            # 打印训练损失 (每100步)
            if iterations % 100 == 0:
                logger.print(
                    f"[Iter {iterations}/{args.max_iterations}] "
                    f"Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

            # =================== 验证 ===================
            if iterations % args.val_interval == 0:
                logger.print(f"=== 验证 (Iter {iterations}) ===")
                diffusion.model.eval()
                ema_updater.ema_model.model.eval()
                analysis_d = AnalysisPanAcc()

                with torch.no_grad():
                    for val_batch in dl_valid:
                        torch.cuda.empty_cache()
                        pan_v, lms_v, hr_v, wav_v = map(
                            lambda x: x.to(device), val_batch
                        )
                        cond_v, _ = einops.pack(
                            [
                                lms_v,
                                pan_v,
                                F.interpolate(wav_v, size=lms_v.shape[-1], mode="bilinear"),
                            ],
                            "b * h w",
                        )

                        # 使用EMA模型采样
                        sr = ema_updater.ema_model(
                            y=lms_v, cond=cond_v, mode="ddpm_sample"
                        )
                        sr = sr + lms_v
                        sr = sr.clip(0, 1)

                        analysis_d(hr_v.to(sr.device), sr)

                score = analysis_d.acc_ave
                logger.print(f"验证指标: {analysis_d.print_str()}")
                logger.log_scalars("diffusion_perf", score, iterations)

                # 保存模型
                save_path = os.path.join(
                    args.save_dir,
                    f"ema_{args.dataset_name}_SAM_{score['SAM']:.3f}_iter_{iterations}.pth",
                )
                path_legal_checker(save_path)
                torch.save(ema_updater.ema_model_state_dict, save_path)
                logger.print(f"保存模型: {save_path}")

                if score["SAM"] < best_sam:
                    best_sam = score["SAM"]
                    best_path = os.path.join(
                        args.save_dir,
                        f"best_{args.dataset_name}.pth",
                    )
                    torch.save(ema_updater.ema_model_state_dict, best_path)
                    logger.print(f"更新最优模型 (SAM={best_sam:.4f})")

                diffusion.model.train()

            if iterations > args.max_iterations:
                break

    logger.print(f"训练完成! 最优SAM: {best_sam:.4f}")


# =============================================================================
#  命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DiTPan 训练脚本")

    # 数据集
    parser.add_argument("--train_path", type=str, required=True,
                        help="训练数据集路径 (.h5)")
    parser.add_argument("--valid_path", type=str, required=True,
                        help="验证数据集路径 (.h5)")
    parser.add_argument("--dataset_name", type=str, default="wv3",
                        choices=["wv3", "wv2", "gf2", "qb", "cave", "harvard"],
                        help="数据集名称")

    # 模型
    parser.add_argument("--model_size", type=str, default="S",
                        choices=["S", "B"], help="模型大小: S(Small) 或 B(Base)")
    parser.add_argument("--image_size", type=int, default=64,
                        help="输入图像尺寸")
    parser.add_argument("--pretrain_weight", type=str, default=None,
                        help="预训练权重路径")

    # 扩散
    parser.add_argument("--n_steps", type=int, default=15,
                        help="扩散步数")
    parser.add_argument("--schedule_type", type=str, default="cosine",
                        choices=["cosine", "resshift"],
                        help="噪声调度类型")

    # 训练
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="初始学习率")
    parser.add_argument("--max_iterations", type=int, default=500_000,
                        help="最大训练迭代次数")
    parser.add_argument("--start_iter", type=int, default=0,
                        help="起始迭代次数 (断点续训)")
    parser.add_argument("--loss_type", type=str, default="l2",
                        choices=["l1", "l2", "smoothl1"],
                        help="损失函数类型")
    parser.add_argument("--penalty_weight", type=float, default=100.0,
                        help="范围惩罚权重")
    parser.add_argument("--val_interval", type=int, default=5000,
                        help="验证间隔 (迭代数)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")

    # 设备
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练设备")

    # 保存路径
    parser.add_argument("--save_dir", type=str, default="./weights",
                        help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default=".",
                        help="日志保存目录")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
