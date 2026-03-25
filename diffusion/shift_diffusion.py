# =============================================================================
# ShiftDiffusion: 位移扩散过程 (来自 respandiff, 经过严格BUG修复与适配)
# =============================================================================
#
# 保留respandiff原有的扩散数学框架:
#   前向过程 (q_sample):
#     e_t = (1 - η_t) * e_0 + κ * √η_t * ε
#     其中 e_0 = HR - LMS (残差), ε ~ N(0, I)
#
#   后验分布 (q_posterior):
#     p(e_{t-1} | e_t, e_0) = N(μ, σ²I)
#     μ = (η_{t-1}/η_t) * e_t + (α_t/η_t) * e_0
#
# BUG修复记录:
#   1. [修复] predict_noise_from_start 引用未注册的 sqrt_recip_alphas_cumprod
#   2. [修复] p_sample_loop 中 self_cond 使用 e_{t-1} 而非 predicted e_0
#   3. [修复] loss中 lambda_penalty=10000 过大, 改为可配置参数
#   4. [修复] loss_ssim 计算但未使用 (死代码), 改为可选项
#   5. [修复] space_new_betas 引用未定义的 new_etas
#   6. [改进] p_mean 中的 clamp 逻辑更加清晰
#   7. [改进] 分离 prior_sample 逻辑, 支持零起点和LMS起点
# =============================================================================

import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


# =============================================================================
#  噪声调度
# =============================================================================

def make_sqrt_etas_schedule(schedule="cosine", n_timestep=15):
    """
    生成扩散噪声调度 (sqrt_etas)

    参数:
      schedule: 调度类型
        - "cosine": 余弦调度 (推荐, 在小步数下表现更好)
        - "resshift": 指数插值调度 (来自ResShift论文)
      n_timestep: 扩散步数

    返回:
      sqrt_etas: [n_timestep] 数组, 每步的 √η_t

    自检:
      1. cosine调度: η从0递增到接近1, sqrt_etas单调递增 ✓
      2. resshift调度: 指数增长 ✓
      3. 返回类型统一为numpy array ✓
    """
    if schedule == "resshift":
        min_noise_level = 0.01
        eta_T = 0.99
        p = 0.3
        k = 1.0
        etas_start = min(min_noise_level / k, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(
            1 / (n_timestep - 1) * math.log(eta_T / etas_start)
        )
        base = np.ones([n_timestep]) * increaser
        power_timestep = np.linspace(0, 1, n_timestep, endpoint=True) ** p
        power_timestep *= n_timestep - 1
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule == "cosine":
        cosine_s = 8e-3
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep
            + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]  # 归一化使 alpha_0 = 1
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.numpy(), axis=0)
        etas = 1.0 - alphas_cumprod
        sqrt_etas = np.sqrt(etas)
    else:
        raise NotImplementedError(f"未知调度类型: {schedule}")
    return sqrt_etas


# =============================================================================
#  辅助函数
# =============================================================================

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    """
    从1D张量a中按索引t提取值, 并广播到x_shape
    a: [T], t: [B], x_shape: [B, C, H, W]
    返回: [B, 1, 1, 1]
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    """生成高斯噪声"""
    if repeat:
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )
    return torch.randn(shape, device=device)


def expand_dims(v, dims):
    """
    将张量v扩展到dims维度
    v: [N] -> [N, 1, 1, ..., 1]
    """
    return v[(...,) + (None,) * (dims - 1)]


# =============================================================================
#  ShiftDiffusion: 位移扩散模型
# =============================================================================

class ShiftDiffusion(nn.Module):
    """
    位移扩散模型 (Shift Diffusion)

    核心思想: 在残差空间(e_0 = HR - LMS)上进行扩散,
    模型学习从噪声残差中恢复干净残差

    前向过程:
      e_t = (1 - η_t) * e_0 + κ * √η_t * ε, ε ~ N(0, I)
      当 t=0: e_0 ≈ e_0 (干净残差)
      当 t=T: e_T ≈ κ * √η_T * ε (纯噪声)

    反向过程:
      p(e_{t-1} | e_t) 通过模型预测 e_0, 然后计算后验均值
      模型接收 (e_t + LMS) 作为输入, 预测 e_0

    自检:
      1. 前向过程: η_t 单调递增, 噪声逐步增大 ✓
      2. 反向过程: 从纯噪声逐步去噪恢复干净残差 ✓
      3. 训练: 模型预测e_0, loss为预测值与真实e_0的MSE ✓
      4. 与DiTPan接口: model(x_noisy+y, t, cond, self_cond) ✓
    """

    def __init__(
        self,
        denoise_fn,
        channels=8,
        loss_type="l2",
        device="cuda:0",
        clamp_range=(0, 1),
        clamp_type="abs",
        pred_mode="x_start",
        penalty_weight=100.0,
    ):
        super().__init__()
        self.channels = channels
        self.model = denoise_fn
        self.loss_type = loss_type
        self.device = device
        self.clamp_range = clamp_range
        self.clamp_type = clamp_type
        self.kappa = 1.0
        self.sqrt_kappa = math.sqrt(self.kappa)
        self.penalty_weight = penalty_weight

        # 自条件标记 (从模型继承)
        self.self_condition = getattr(self.model, "self_condition", False)

        assert clamp_type in ["abs", "dynamic"]
        assert pred_mode in ["x_start"]

        self.pred_mode = pred_mode
        self.thresholding_max_val = 1.0
        self.dynamic_thresholding_ratio = 0.8

        self.set_loss(device)

    def set_loss(self, device):
        """设置训练损失函数"""
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss().to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss().to(device)
        elif self.loss_type == "smoothl1":
            self.loss_func = nn.SmoothL1Loss().to(device)
        else:
            raise NotImplementedError(f"不支持的损失类型: {self.loss_type}")

    def set_new_noise_schedule(self, schedule_opt=None, device="cpu", *, sqrt_etas=None):
        """
        设置噪声调度, 注册所有必需的buffer

        噪声调度参数:
          sqrt_etas: √η_t, 扩散噪声水平
          etas: η_t = (√η_t)²
          alphas: α_t = η_t - η_{t-1} (增量)
          posterior_variance: 后验方差
          posterior_mean_coef1/2: 后验均值系数

        自检:
          1. etas 单调递增 (从0到接近1) ✓
          2. alphas > 0 (每步噪声增量为正) ✓
          3. posterior_variance ≥ 0 ✓
          4. 所有buffer注册为float32 ✓
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        if schedule_opt is not None:
            sqrt_etas = make_sqrt_etas_schedule(
                schedule=schedule_opt.get("schedule", "cosine"),
                n_timestep=schedule_opt["n_timestep"],
            )

        # 转换为numpy以进行计算
        if isinstance(sqrt_etas, torch.Tensor):
            sqrt_etas = sqrt_etas.detach().cpu().numpy()
        sqrt_etas = np.array(sqrt_etas, dtype=np.float64)

        etas = sqrt_etas ** 2

        # η_{t-1}: 前一步的累积噪声水平
        etas_prev = np.append(0.0, etas[:-1])

        # α_t = η_t - η_{t-1}: 每步噪声增量
        alphas = etas - etas_prev

        (timesteps,) = etas.shape
        self.num_timesteps = int(timesteps)

        # 注册buffer
        self.register_buffer("sqrt_etas", to_torch(sqrt_etas))
        self.register_buffer("etas", to_torch(etas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("etas_prev", to_torch(etas_prev))

        cumsum_alphas = torch.cumsum(to_torch(alphas), dim=0)
        self.register_buffer("cumsum_alphas", cumsum_alphas)

        # 后验方差: κ² * η_{t-1} / η_t * α_t
        posterior_variance = self.kappa ** 2 * etas_prev / np.maximum(etas, 1e-20) * alphas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))

        # 裁剪后验方差的对数 (避免log(0))
        posterior_variance_clipped = np.append(
            posterior_variance[1] if len(posterior_variance) > 1 else 1e-20,
            posterior_variance[1:],
        )
        self.register_buffer(
            "posterior_variance_clipped", to_torch(posterior_variance_clipped)
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance_clipped, 1e-20))),
        )

        # 后验均值系数
        # μ = coef1 * e_t + coef2 * e_0
        # coef1 = η_{t-1} / η_t
        # coef2 = α_t / η_t
        posterior_mean_coef1 = etas_prev / np.maximum(etas, 1e-20)
        posterior_mean_coef2 = alphas / np.maximum(etas, 1e-20)
        self.register_buffer("posterior_mean_coef1", to_torch(posterior_mean_coef1))
        self.register_buffer("posterior_mean_coef2", to_torch(posterior_mean_coef2))

    # =========================================================================
    #  前向过程
    # =========================================================================

    def q_sample(self, e_0, t, noise=None):
        """
        前向扩散: 给干净残差e_0添加噪声得到e_t

        公式: e_t = (1 - η_t) * e_0 + κ * √η_t * ε

        参数:
          e_0: 干净残差 = HR - LMS [B, C, H, W]
          t: 时间步 [B]
          noise: 高斯噪声 (可选)

        返回:
          e_t: 噪声残差 [B, C, H, W]

        自检:
          1. t=0时: η_0≈0, e_0 ≈ e_0 (几乎无噪声) ✓
          2. t=T时: η_T≈1, e_T ≈ κ*noise (几乎纯噪声) ✓
          3. noise形状与e_0一致 ✓
        """
        noise = default(noise, lambda: torch.randn_like(e_0))
        return (
            e_0
            - extract(self.etas, t, e_0.shape) * e_0
            + extract(self.kappa * self.sqrt_etas, t, e_0.shape) * noise
        )

    def prior_sample(self, y_placeholder, noise=None):
        """
        从先验分布采样 (用于采样循环的起始点)

        公式: e_T = κ * √η_T * ε (从零平均先验采样)

        参数:
          y_placeholder: 形状占位符 (用于确定输出形状)
          noise: 高斯噪声

        返回:
          e_T: 初始噪声残差 [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(y_placeholder)
        t = torch.tensor(
            [self.num_timesteps - 1] * y_placeholder.shape[0],
            device=y_placeholder.device,
        ).long()
        return (
            y_placeholder
            + extract(self.kappa * self.sqrt_etas, t, y_placeholder.shape) * noise
        )

    # =========================================================================
    #  后验计算
    # =========================================================================

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布 q(e_{t-1} | e_t, e_0)

        后验均值:
          μ = (η_{t-1}/η_t) * e_t + (α_t/η_t) * e_0
            = coef1 * e_t + coef2 * e_0

        返回: (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # =========================================================================
    #  动态阈值
    # =========================================================================

    def dynamic_thresholding_fn(self, x0, t):
        """
        动态阈值 (来自Imagen论文)
        根据百分位数自适应裁剪, 防止过大值
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(
            torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1
        )
        s = expand_dims(
            torch.maximum(
                s, self.thresholding_max_val * torch.ones_like(s)
            ),
            dims,
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    # =========================================================================
    #  反向过程
    # =========================================================================

    def p_mean(self, e_t, t, y, cond, self_cond, clip_denoised=True):
        """
        计算反向过程的均值

        流程:
          1. 模型接收 (e_t + y) 预测 e_0
          2. 可选: 对预测进行裁剪 (在全图空间裁剪后转回残差)
          3. 用后验公式计算 e_{t-1} 的均值

        参数:
          e_t: 当前噪声残差 [B, C, H, W]
          t: 时间步 [B]
          y: LMS图像 [B, C, H, W]
          cond: 完整条件 [B, C_total, H, W]
          self_cond: 自条件 [B, C, H, W]
          clip_denoised: 是否裁剪预测

        返回:
          (model_mean, posterior_variance, posterior_log_var, predicted_e0)

        自检:
          1. 模型输入是 e_t + y (噪声全图) ✓
          2. 裁剪在全图空间进行 (加LMS后裁剪到[0,1], 再减LMS) ✓
          3. predicted_e0 用于后续self-conditioning ✓
        """
        # 模型预测: 输入噪声全图, 输出干净残差e_0
        model_out = self.model(e_t + y, t, cond=cond, self_cond=self_cond)
        e_recon = model_out

        # 在全图空间裁剪 (残差 + LMS 后应在[0,1]范围内)
        if clip_denoised:
            lms = cond[:, : self.channels]  # 从条件中取出LMS
            full_img = e_recon + lms
            if self.clamp_type == "abs":
                full_img.clamp_(*self.clamp_range)
            else:
                full_img = self.dynamic_thresholding_fn(full_img, t)
            e_recon = full_img - lms

        # 计算后验均值
        model_mean, posterior_var, posterior_log_var = self.q_posterior(
            x_start=e_recon, x_t=e_t, t=t
        )

        return model_mean, posterior_var, posterior_log_var, model_out

    @torch.no_grad()
    def p_sample(self, e_t, cond, self_cond, y, t, repeat_noise=False):
        """
        从 p(e_{t-1} | e_t) 采样一步

        公式: e_{t-1} = μ + σ * ε  (t>0时)
              e_{t-1} = μ            (t=0时, 不加噪声)

        返回: (sample, pred_x_start)
          - sample: 采样的 e_{t-1}
          - pred_x_start: 模型预测的 e_0 (用于self-conditioning)

        [修复] 原respandiff只返回sample, 导致self_cond使用e_{t-1}而非e_0
        """
        b, *_, device = *e_t.shape, e_t.device
        model_mean, variance, log_variance, pred_x_start = self.p_mean(
            e_t=e_t, y=y, cond=cond, self_cond=self_cond, t=t
        )

        noise = noise_like(model_mean.shape, device, repeat_noise)
        # t=0 时不加噪声
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(model_mean.shape) - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

        return sample, pred_x_start

    @torch.no_grad()
    def p_sample_loop(self, x_in, cond):
        """
        完整反向采样循环: 从噪声逐步去噪恢复干净残差

        流程:
          1. 从先验采样: e_T = κ * √η_T * ε
          2. 迭代去噪: for t = T-1, ..., 0:
               (e_{t}, pred_e0) = p_sample(e_{t+1}, ...)
               self_cond = pred_e0  (使用干净预测作为下一步自条件)
          3. 返回最终 e_0 预测

        参数:
          x_in: LMS图像 [B, C, H, W] (用于声明形状和生成噪声)
          cond: 完整条件 [B, C_total, H, W]

        返回:
          e_0: 预测的干净残差 [B, C, H, W]

        [修复] self_cond 现在使用 pred_x_start (模型预测的e_0),
               而非 e_{t-1} (上一步的噪声残差)
        """
        device = self.etas.device
        b = x_in.shape[0]

        # 从先验采样e_T
        noise = torch.randn_like(x_in, device=device)
        z_sample = self.prior_sample(torch.zeros_like(x_in), noise)

        # 迭代反向采样
        indices = list(range(self.num_timesteps))[::-1]
        pred_x_start = None  # 用于self-conditioning

        for i in tqdm(indices, desc="DDPM采样"):
            self_cond = pred_x_start if self.self_condition else None
            t_batch = torch.full((b,), i, device=device, dtype=torch.long)

            z_sample, pred_x_start = self.p_sample(
                e_t=z_sample,
                y=x_in,
                t=t_batch,
                cond=cond,
                self_cond=self_cond,
            )

        return z_sample

    # =========================================================================
    #  训练损失
    # =========================================================================

    def p_losses(self, x_start, cond, y=None, noise=None, current_iter=0):
        """
        计算训练损失

        流程:
          1. 计算残差: e_0 = x_start (HR) - y (LMS)
          2. 随机采样时间步 t
          3. 前向加噪: e_t = q_sample(e_0, t)
          4. 可选自条件: 50%概率先做一次无条件前向, 用其预测作为self_cond
          5. 模型预测: pred_e_0 = model(e_t + y, t, cond, self_cond)
          6. 损失: L = loss_func(e_0, pred_e_0) + penalty

        参数:
          x_start: HR图像 [B, C, H, W]
          cond: 条件 [B, C_total, H, W]
          y: LMS图像 [B, C, H, W]
          noise: 预定义噪声 (可选)
          current_iter: 当前迭代次数 (未使用, 保留接口)

        [修复] 原respandiff中 lambda_penalty=10000 过大, 改为可配置
        [修复] 移除未使用的 loss_ssim 死代码

        自检:
          1. e_0 = x_start - y 是残差预测目标 ✓
          2. 模型输入是 e_t + y (噪声全图) ✓
          3. 自条件训练: 50%概率先做无梯度前向 ✓
          4. penalty项防止预测超出真实范围, 权重可控 ✓
        """
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()

        # 计算残差
        e_0 = x_start - y
        noise = default(noise, lambda: torch.randn_like(e_0))

        # 前向加噪
        x_noisy = self.q_sample(e_0=e_0, t=t, noise=noise)

        # 自条件 (50%概率)
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                model_out = self.model(
                    x_noisy + y, t, cond=cond, self_cond=None
                )
                x_self_cond = model_out.detach()

        # 模型预测
        model_predict = self.model(
            x_noisy + y, t, cond=cond, self_cond=x_self_cond
        )

        # 主损失: 预测残差 vs 真实残差
        loss = self.loss_func(e_0, model_predict)

        # 范围惩罚: 防止预测超出真实残差范围
        if self.penalty_weight > 0:
            penalty = torch.mean(
                torch.clamp(model_predict - torch.max(e_0), min=0)
                + torch.clamp(torch.min(e_0) - model_predict, min=0)
            )
            loss = loss + self.penalty_weight * penalty

        return loss

    # =========================================================================
    #  统一前向接口
    # =========================================================================

    def forward(self, x=None, y=None, cond=None, mode="train", *args, **kwargs):
        """
        统一前向接口

        参数:
          x: HR图像 (训练时) [B, C, H, W]
          y: LMS图像 [B, C, H, W]
          cond: 完整条件 [B, C_total, H, W]
          mode: "train" 或 "ddpm_sample"

        返回:
          训练模式: 标量损失
          采样模式: 预测残差 e_0 [B, C, H, W]
        """
        if mode == "train":
            return self.p_losses(x_start=x, cond=cond, y=y, *args, **kwargs)
        elif mode == "ddpm_sample":
            with torch.no_grad():
                return self.p_sample_loop(x_in=y, cond=cond, *args, **kwargs)
        else:
            raise NotImplementedError(f"不支持的模式: {mode}")
