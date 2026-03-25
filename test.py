# =============================================================================
# DiTPan 推理/测试脚本
# =============================================================================
#
# 推理流程:
#   1. 加载训练好的DiTPan模型权重
#   2. 加载测试数据 (reduced resolution 或 full resolution)
#   3. 对每个batch执行完整DDPM采样
#   4. 计算全面指标: SAM, ERGAS, PSNR, CC, SSIM
#   5. 保存结果图像和.mat文件
#
# 使用方式:
#   python test.py --test_path data/wv3/test_wv3_multiExm1.h5 \
#                  --weight_path weights/best_wv3.pth \
#                  --dataset_name wv3
# =============================================================================

import argparse
import os
import time

import einops
import h5py
import numpy as np
import torch
import torchvision as tv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import savemat
from torch.utils.data import DataLoader

from models.dit_pan import DiTPan_S, DiTPan_B
from diffusion.shift_diffusion import ShiftDiffusion, make_sqrt_etas_schedule
from dataset.pan_dataset import PanDataset
from dataset.hisr import HISRDataSets
from utils.metric import AnalysisPanAcc, NonAnalysisPanAcc
from utils.misc import model_load, path_legal_checker


# =============================================================================
#  数据集配置 (与train.py保持一致)
# =============================================================================

DIVISION_DICT = {
    "wv3": 2047.0, "wv2": 2047.0, "gf2": 1023.0,
    "qb": 2047.0, "cave": 1.0, "harvard": 1.0,
}

RGB_CHANNEL_DICT = {
    "wv3": [4, 2, 0], "wv2": [4, 2, 0], "gf2": [0, 1, 2],
    "qb": [0, 1, 2], "cave": [29, 19, 9], "harvard": [29, 19, 9],
}


def get_dataset_config(dataset_name):
    if dataset_name in ["harvard", "cave"]:
        return 31, 3
    elif dataset_name == "wv3":
        return 8, 1
    elif dataset_name in ["gf2", "qb", "wv2"]:
        return 4, 1
    else:
        raise NotImplementedError(f"不支持的数据集: {dataset_name}")


# =============================================================================
#  推理主函数
# =============================================================================

@torch.no_grad()
def test(args):
    """
    DiTPan 推理主函数

    支持两种模式:
      1. Reduced Resolution (合成数据, 有GT, 可计算参考指标)
      2. Full Resolution (真实数据, 无GT, 仅保存结果)

    自检:
      1. 模型加载: 正确加载EMA权重 ✓
      2. 条件打包: 与训练时一致 [lms, pan, wavelets_upsampled] ✓
      3. 采样输出: sr = model_output + lms, 裁剪到[0,1] ✓
      4. 指标计算: SAM/ERGAS/PSNR/CC/SSIM 完整覆盖 ✓
      5. 结果保存: .mat格式 + 可视化图像 ✓
    """
    device = args.device
    if device != "cpu" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(device)
        except Exception:
            print(f"警告: 无法设置CUDA设备 {device}, 回退到CPU")
            device = "cpu"
    elif device != "cpu":
        print("警告: CUDA不可用, 回退到CPU")
        device = "cpu"

    # =================== 配置 ===================
    image_n_channel, pan_channel = get_dataset_config(args.dataset_name)
    division = DIVISION_DICT[args.dataset_name]
    rgb_channels = RGB_CHANNEL_DICT[args.dataset_name]

    print(f"数据集: {args.dataset_name}, 通道数: {image_n_channel}, PAN通道: {pan_channel}")
    print(f"归一化除数: {division}")

    # =================== 模型初始化 ===================
    model_fn = DiTPan_S if args.model_size == "S" else DiTPan_B
    denoise_fn = model_fn(
        input_size=args.image_size,
        in_channels=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=pan_channel,
        self_condition=True,
    ).to(device)

    # 加载权重
    denoise_fn = model_load(args.weight_path, denoise_fn, device=device)
    print(f"加载权重: {args.weight_path}")

    n_params = sum(p.numel() for p in denoise_fn.parameters()) / 1e6
    print(f"模型参数量: {n_params:.2f}M")

    # =================== 扩散过程 ===================
    diffusion = ShiftDiffusion(
        denoise_fn,
        channels=image_n_channel,
        pred_mode="x_start",
        loss_type="l1",  # 推理时不影响
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        sqrt_etas=make_sqrt_etas_schedule(
            schedule=args.schedule_type, n_timestep=args.n_steps
        ),
        device=device,
    )
    diffusion = diffusion.to(device)
    diffusion.model.eval()

    # =================== 数据集 ===================
    d_test = h5py.File(args.test_path, "r")

    if args.dataset_name in ["wv3", "wv2", "gf2", "qb"]:
        ds_test = PanDataset(
            d_test, full_res=args.full_res, norm_range=False,
            division=division, wavelets=True,
        )
    else:
        ds_test = HISRDataSets(
            d_test, normalize=False, aug_prob=0.0, wavelets=True
        )

    dl_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=0,
    )

    # =================== 采样与评估 ===================
    saved_name = "reduced" if not args.full_res else "full"
    analysis = AnalysisPanAcc() if not args.full_res else NonAnalysisPanAcc()

    preds = []
    e0_preds = []
    sample_times = len(dl_test)
    times_record = []

    for i, batch in enumerate(dl_test):
        if args.full_res:
            pan, lms, wavelets = batch
            gt = None
        else:
            pan, lms, gt, wavelets = batch

        print(f"采样 [{i + 1}/{sample_times}]")
        pan, lms, wavelets = map(lambda x: x.to(device), (pan, lms, wavelets))

        # 打包条件
        cond, _ = einops.pack(
            [lms, pan, F.interpolate(wavelets, size=lms.shape[-1], mode="bilinear")],
            "b * h w",
        )

        # DDPM采样
        start_time = time.time()
        sr = diffusion(y=lms, cond=cond, mode="ddpm_sample")
        elapsed = time.time() - start_time
        times_record.append(elapsed)

        e0_pred = sr.clone()
        sr = sr + lms
        sr = sr.clip(0, 1)
        e0_pred = e0_pred.clip(0, 1)

        # 计算指标 (仅reduced resolution)
        if not args.full_res and gt is not None:
            analysis(sr.detach().cpu(), gt)
            print(f"  当前batch指标: {analysis.print_str(analysis.last_acc)}")

        # 可视化 (可选)
        if args.show and sr.shape[0] > 0:
            s = tv.utils.make_grid(sr[:16], nrow=4, padding=0).cpu()
            s.clip_(0, 1)
            fig, ax = plt.subplots(
                figsize=(s.shape[2] // 100, s.shape[1] // 100), dpi=200
            )
            vis_channels = rgb_channels if sr.shape[1] > 3 else [0, 1, 2]
            ax.imshow(s.permute(1, 2, 0).numpy()[..., vis_channels])
            ax.set_axis_off()
            save_fig_path = path_legal_checker(
                f"./samples/ditpan_{args.dataset_name}_test/"
                f"test_{saved_name}_part_{i}.png"
            )
            fig.savefig(save_fig_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        # 收集预测结果
        sr_np = sr.detach().cpu().numpy() * division
        e0_np = e0_pred.detach().cpu().numpy() * division
        preds.append(sr_np.clip(0, division))
        e0_preds.append(e0_np.clip(0, division))

        if not args.full_res:
            print(f"  累计指标: {analysis.print_str()}")

    # =================== 性能统计 ===================
    if times_record:
        mean_time = np.mean(times_record)
        print(f"\n推理时间统计: 平均 {mean_time:.4f}s/batch, "
              f"FPS: {args.batch_size / mean_time:.2f}")

    # =================== 保存结果 ===================
    if not args.full_res:
        score = analysis.acc_ave
        sam_score = score["SAM"]
        print(f"\n===== 最终测试指标 =====")
        print(f"  SAM:   {score['SAM']:.4f}")
        print(f"  ERGAS: {score['ERGAS']:.4f}")
        print(f"  PSNR:  {score['PSNR']:.4f}")
        print(f"  CC:    {score['CC']:.4f}")
        print(f"  SSIM:  {score['SSIM']:.4f}")

        d = dict(
            gt=d_test["gt"][:],
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
            sr_e0=np.concatenate(e0_preds, axis=0),
            res_gt=d_test["gt"][:] - d_test["lms"][:],
        )
    else:
        sam_score = -1
        d = dict(
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )

    # 保存.mat文件
    mat_path = path_legal_checker(
        f"./samples/mat/DiTPan_SAM_{sam_score:.4f}_{saved_name}"
        f"_{args.dataset_name}_nstep{args.n_steps}.mat"
    )
    savemat(mat_path, d)
    print(f"保存结果至: {mat_path}")


# =============================================================================
#  命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DiTPan 推理/测试脚本")

    # 数据
    parser.add_argument("--test_path", type=str, required=True,
                        help="测试数据集路径 (.h5)")
    parser.add_argument("--weight_path", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--dataset_name", type=str, default="wv3",
                        choices=["wv3", "wv2", "gf2", "qb", "cave", "harvard"])

    # 模型
    parser.add_argument("--model_size", type=str, default="S",
                        choices=["S", "B"])
    parser.add_argument("--image_size", type=int, default=64)

    # 扩散
    parser.add_argument("--n_steps", type=int, default=15)
    parser.add_argument("--schedule_type", type=str, default="cosine")

    # 测试
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--full_res", action="store_true",
                        help="Full resolution测试 (无GT)")
    parser.add_argument("--show", action="store_true",
                        help="保存可视化结果")

    # 设备
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)
