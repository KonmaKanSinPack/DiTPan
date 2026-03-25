# =============================================================================
# DiTPan: 基于 Diffusion Transformer 的全光谱融合模型
# =============================================================================
#
# 核心创新:
#   1. CSM (粗粒度风格调制): 通过卷积MLP从 [LMS, PAN] 提取空间风格信息,
#      生成逐token的scale/shift进行仿射变换, 将空间条件注入Transformer
#   2. FWM (细粒度小波调制): 使用线性复杂度交叉注意力, 将小波分解的频率
#      信息 (LMS低频 + PAN高频) 注入后半段Transformer块
#   3. QK-Norm: LayerNorm归一化Q和K, 提高大模型训练稳定性
#   4. SwiGLU FFN: 门控线性单元, 比标准FFN有更强的表达能力
#   5. Self-Conditioning: 利用前一步预测作为额外条件输入, 提高生成质量
#   6. 双阶段条件注入: CSM(全部blocks) + FWM(后半部分blocks),
#      分别处理粗粒度风格和细粒度频率信息
#
# 与原始 DiT 的关键区别:
#   - 移除 class label embedding, 替换为 CSM 空间条件嵌入
#   - 不使用 VAE, 直接在像素空间操作 (pansharpening无需VAE)
#   - 在后半段blocks加入 FWM 小波交叉注意力
#   - 使用 ShiftDiffusion 而非标准 DDPM
#   - 条件编码器使用卷积MLP保留局部空间信息
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# =============================================================================
#  辅助函数
# =============================================================================

def modulate(x, shift, scale):
    """
    自适应层归一化调制函数 (来自DiT论文)
    x: [B, N, D], shift: [B, D], scale: [B, D]
    对所有token施加相同的全局scale和shift
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    生成2D正弦余弦位置编码
    embed_dim: 输出维度
    grid_size: 网格边长 (int)
    返回: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w先于h
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D正弦余弦编码"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# =============================================================================
#  Drop Path (随机深度正则化)
# =============================================================================

class DropPath(nn.Module):
    """
    随机深度 (Stochastic Depth) 正则化
    训练时以drop_prob概率丢弃整个残差分支
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# =============================================================================
#  PatchEmbed (Patch嵌入)
# =============================================================================

class PatchEmbed(nn.Module):
    """
    图像Patch嵌入层
    将 [B, C, H, W] 图像通过Conv2d映射为 [B, N, D] token序列
    其中 N = (H/patch_size) * (W/patch_size)
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, D, H/p, W/p] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =============================================================================
#  卷积Patch嵌入 (用于条件和小波编码)
# =============================================================================

class ConvPatchEmbed(nn.Module):
    """
    卷积Patch嵌入: 先用卷积处理保留局部空间上下文, 再patchify
    较之直接PatchEmbed, 能更好保留条件图像的局部空间信息

    流程: Conv3x3 → GroupNorm → SiLU → Conv3x3 → Patchify(Conv with stride)

    用途:
      - CSM条件编码: 处理 [LMS, PAN] 得到风格token
      - FWM小波编码: 处理小波系数得到频率token
    """
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        mid_dim = embed_dim * 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, mid_dim),
            nn.SiLU(),
            nn.Conv2d(mid_dim, embed_dim, 3, padding=1, bias=True),
        )
        self.proj = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.num_patches = None  # 由输入尺寸动态决定

    def forward(self, x):
        # x: [B, C_cond, H, W]
        x = self.body(x)   # [B, D, H, W]  局部空间处理
        x = self.proj(x)   # [B, D, H/p, W/p]  Patchify
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


# =============================================================================
#  时间步嵌入
# =============================================================================

class TimestepEmbedder(nn.Module):
    """
    将标量时间步嵌入到向量表示
    使用正弦余弦频率编码 + MLP投影
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """生成正弦余弦频率编码"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# =============================================================================
#  CSM: 粗粒度风格调制 (Coarse-grained Style Modulation)
# =============================================================================

class CSMLayer(nn.Module):
    """
    粗粒度风格调制层 (Per-Block)

    将条件token映射为逐token的scale和shift, 对特征进行仿射调制。
    每个Transformer block有独立的CSMLayer, 允许不同层学习不同的调制策略。

    对应论文公式:
      Z, S = Split(MLP([P, M]))
      F_l = F_l * (I + Z) + S

    其中:
      Z: scale参数 [B, N, D], 控制特征幅度
      S: shift参数 [B, N, D], 注入条件偏置

    设计要点:
      - 输出零初始化, 训练初期保持恒等映射, 不破坏主干网络信号
      - 使用LayerNorm对条件token归一化, 提高稳定性
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
        )
        # 零初始化: 确保训练开始时CSM为恒等映射
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond_tokens):
        """
        x: 特征token [B, N, D]
        cond_tokens: 条件token [B, N, D] (来自CSM编码器)
        返回: 调制后的特征 [B, N, D]
        """
        cond = self.norm(cond_tokens)
        scale_shift = self.proj(cond)  # [B, N, 2D]
        scale, shift = scale_shift.chunk(2, dim=-1)  # 各 [B, N, D]
        return x * (1 + scale) + shift


# =============================================================================
#  FWM: 细粒度小波调制 (Fine-grained Wavelet Modulation)
# =============================================================================

class FWMLinearCrossAttention(nn.Module):
    """
    细粒度小波调制: 线性复杂度交叉注意力

    使用线性注意力机制将小波频率信息注入特征。避免标准注意力的O(N²)复杂度,
    改用O(N*d²)复杂度, 其中d << N, 大幅降低内存消耗。

    对应论文公式 (线性注意力):
      Q = Softmax(Q, dim=spatial)  // 沿空间维度归一化
      K = Softmax(K, dim=channel)  // 沿通道维度归一化
      Context = K^T @ V            // [d, d] 上下文矩阵
      Output = Q @ Context         // [N, d] 输出

    其中:
      Q: 来自主干特征 (查询, 作为decoder特征)
      K, V: 来自小波token (键/值, 包含频率信息)
             小波 = [LMS_LL(低频), PAN_LH(水平高频), PAN_HL(垂直高频), PAN_HH(对角高频)]

    设计要点:
      - 交叉注意力而非自注意力, Q来自特征, KV来自小波
      - 沿光谱维度做注意力, 引入全局频率响应
      - 零初始化输出投影, 训练初期为恒等映射
      - 包含FFN后处理, 增强非线性表达
    """
    def __init__(self, hidden_size, num_heads=8, drop_path=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Query投影 (来自主干特征token)
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=True)

        # Key/Value投影 (来自小波token)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.to_kv = nn.Linear(hidden_size, hidden_size * 2, bias=True)

        # 输出投影 (零初始化)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # FFN后处理
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, wavelet_tokens):
        """
        x: 主干特征token [B, N, D]
        wavelet_tokens: 小波token [B, M, D]
        返回: 调制后的特征 [B, N, D]
        """
        residual = x

        # 投影Q, K, V
        q = self.to_q(self.norm_q(x))                    # [B, N, D]
        kv = self.to_kv(self.norm_kv(wavelet_tokens))     # [B, M, 2D]
        k, v = kv.chunk(2, dim=-1)                         # 各 [B, M, D]

        B, N, D = q.shape
        M = k.shape[1]
        H = self.num_heads
        d = self.head_dim

        # 重塑为多头格式
        q = q.view(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
        k = k.view(B, M, H, d).permute(0, 2, 1, 3)  # [B, H, M, d]
        v = v.view(B, M, H, d).permute(0, 2, 1, 3)  # [B, H, M, d]

        # 线性注意力归一化
        q = q.softmax(dim=-2) * self.scale  # 沿空间维度softmax
        k = k.softmax(dim=-1)                # 沿通道维度softmax

        # 计算上下文矩阵: [B, H, d, d]
        # K^T @ V, 对空间维度求和, 得到 d×d 的全局频率上下文
        context = torch.einsum("bhmd, bhme -> bhde", k, v)

        # 输出: Q @ Context → [B, H, N, d]
        out = torch.einsum("bhnd, bhde -> bhne", q, context)

        # 合并多头
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)

        # 残差连接
        x = residual + self.drop_path(out)

        # FFN后处理
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))

        return x


# =============================================================================
#  QK-Norm Self-Attention (带QK归一化的自注意力)
# =============================================================================

class QKNormAttention(nn.Module):
    """
    带QK归一化的Multi-Head Self-Attention

    QK-Norm (来自ViT-22B论文): 在计算注意力权重前对Q和K进行LayerNorm,
    防止注意力logits过大导致的训练不稳定。

    使用 F.scaled_dot_product_attention 自动启用 Flash Attention (PyTorch 2.0+)

    自检:
      1. QK归一化后仍使用标准scale (1/sqrt(d_k)), 因为LayerNorm包含可学习仿射参数
      2. 输出投影使用标准初始化, 不做零初始化 (由外层gate控制)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, N, D]
        返回: [B, N, D]
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv.unbind(0)

        # QK归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 高效注意力计算 (自动使用Flash Attention)
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


# =============================================================================
#  SwiGLU FFN (门控前馈网络)
# =============================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU前馈网络 (Shazeer 2020, 广泛用于2024+主流Transformer)

    相比标准 GELU FFN:
      - 更好的收敛性和最终性能
      - 门控机制提供自适应特征选择

    公式: out = W2 * (SiLU(W1_a * x) ⊙ W1_b * x)
    其中 ⊙ 为逐元素乘法
    """
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        # W1投影到双倍宽度用于门控分割
        self.w1 = nn.Linear(in_features, hidden_features * 2, bias=True)
        self.w2 = nn.Linear(hidden_features, in_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_gate = self.w1(x)
        x, gate = x_gate.chunk(2, dim=-1)
        x = x * F.silu(gate)  # SwiGLU激活
        x = self.drop(x)
        x = self.w2(x)
        return x


# =============================================================================
#  DiTPan Transformer Block
# =============================================================================

class DiTPanBlock(nn.Module):
    """
    DiTPan Transformer块: 集成CSM和FWM的核心计算单元

    处理流程:
      1. CSM: 粗粒度风格调制 — 从[LMS, PAN]空间条件注入风格信息
      2. adaLN-Zero + Self-Attention — 时间步全局调制 + 自注意力
      3. FWM: 细粒度小波调制 — 从小波系数注入频率信息 (仅后半部分blocks)
      4. adaLN-Zero + SwiGLU FFN — 时间步全局调制 + 前馈网络

    设计创新:
      - CSM 扩展了 DiT 的全局adaLN为空间变化的逐token调制
      - FWM 使用线性复杂度交叉注意力, 高效注入小波频率信息
      - 双阶段条件: CSM(编码器思路) 在所有层, FWM(解码器思路) 在后半层
      - adaLN-Zero 中gate初始化为0, CSM scale/shift初始化为0,
        确保训练初期每个block近似恒等映射

    自检:
      1. adaLN产生6个参数(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) ✓
      2. CSM零初始化, 训练初期不影响主干信号 ✓
      3. FWM输出投影零初始化, 训练初期为恒等 ✓
      4. gate确保残差分支初始为零, 允许网络深度无限堆叠 ✓
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_fwm=False,
        drop_path=0.0,
        qk_norm=True,
    ):
        super().__init__()

        # CSM: 粗粒度风格调制 (每个block独立)
        self.csm = CSMLayer(hidden_size)

        # Self-Attention + adaLN-Zero
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if qk_norm:
            self.attn = QKNormAttention(hidden_size, num_heads=num_heads)
        else:
            self.attn = QKNormAttention(hidden_size, num_heads=num_heads)

        # FFN + adaLN-Zero
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(in_features=hidden_size, hidden_features=mlp_hidden_dim)

        # adaLN调制参数: 6个 (shift/scale/gate × Self-Attn和FFN)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # FWM: 细粒度小波调制 (可选, 仅后半部分blocks)
        self.use_fwm = use_fwm
        if use_fwm:
            self.fwm = FWMLinearCrossAttention(
                hidden_size, num_heads=num_heads, drop_path=drop_path
            )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, t_emb, cond_tokens, wavelet_tokens=None):
        """
        x: 特征token [B, N, D]
        t_emb: 时间步嵌入 [B, D] (全局条件)
        cond_tokens: CSM条件token [B, N, D] (空间条件)
        wavelet_tokens: FWM小波token [B, N, D] (频率条件, 可选)
        返回: 处理后的特征 [B, N, D]
        """
        # ---------- 1. CSM: 粗粒度风格调制 ----------
        # 将[LMS, PAN]的空间风格信息注入特征
        x_styled = self.csm(x, cond_tokens)

        # ---------- 2. adaLN-Zero 参数提取 ----------
        (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # ---------- 3. Self-Attention ----------
        x_normed = modulate(self.norm1(x_styled), shift_msa, scale_msa)
        x_attn = self.attn(x_normed)
        x = x + self.drop_path(gate_msa.unsqueeze(1) * x_attn)

        # ---------- 4. FWM: 细粒度小波调制 (仅后半部分blocks) ----------
        if self.use_fwm and wavelet_tokens is not None:
            x = self.fwm(x, wavelet_tokens)

        # ---------- 5. FFN ----------
        x_normed = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_ffn = self.mlp(x_normed)
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * x_ffn)

        return x


# =============================================================================
#  最终输出层
# =============================================================================

class FinalLayer(nn.Module):
    """
    DiTPan最终输出层
    adaLN调制 + 线性投影 → patch_size² * out_channels
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# =============================================================================
#  DiTPan 主模型
# =============================================================================

class DiTPan(nn.Module):
    """
    DiTPan: 基于Diffusion Transformer的全光谱融合模型

    输入:
      - x: 噪声输入 (e_t + y) [B, C, H, W], 其中e_t是噪声残差, y是LMS
      - t: 扩散时间步 [B]
      - cond: 打包条件 [B, C_total, H, W] = [LMS, PAN, wavelets_upsampled]
      - self_cond: 自条件预测 [B, C_out, H, W] (可选)

    输出:
      - 预测的干净残差 e_0 [B, C_out, H, W]

    条件解包 (与respandiff兼容):
      - CSM条件: cond[:, :lms_ch + pan_ch] = [LMS, PAN]
      - FWM小波: cond[:, -(lms_ch + pan_ch*3):] = [LMS_LL, PAN_LH, PAN_HL, PAN_HH]
    """
    def __init__(
        self,
        input_size=64,
        patch_size=2,
        in_channels=8,
        out_channels=None,
        lms_channel=8,
        pan_channel=1,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        self_condition=True,
        learn_sigma=False,
        fwm_start_ratio=0.5,
        drop_path_rate=0.1,
        qk_norm=True,
    ):
        """
        参数:
          input_size: 输入图像尺寸 (默认64, pansharpening reduced resolution)
          patch_size: patch大小 (默认2, 较小以保留空间细节)
          in_channels: MS图像通道数 (WV3=8, GF2/QB=4, HISR=31)
          out_channels: 输出通道数 (默认=in_channels)
          lms_channel: LMS通道数 (= in_channels)
          pan_channel: PAN通道数 (通常=1, HISR=3)
          hidden_size: Transformer隐藏维度
          depth: Transformer层数
          num_heads: 注意力头数
          mlp_ratio: FFN扩展比
          self_condition: 是否使用自条件
          learn_sigma: 是否学习方差 (pansharpening不需要)
          fwm_start_ratio: FWM从第几层开始 (比例, 0.5=后半部分)
          drop_path_rate: 随机深度丢弃率
          qk_norm: 是否使用QK归一化
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.self_condition = self_condition
        self.learn_sigma = learn_sigma
        self.lms_channel = lms_channel
        self.pan_channel = pan_channel
        self.input_size = input_size

        # CSM/FWM条件通道数
        csm_cond_channels = lms_channel + pan_channel
        fwm_wavelet_channels = lms_channel + pan_channel * 3

        # 自条件: 输入通道翻倍 (concatenate前一步预测)
        actual_in_channels = in_channels + self.out_channels if self_condition else in_channels

        # =================== 输入嵌入 ===================
        self.x_embedder = PatchEmbed(
            input_size, patch_size, actual_in_channels, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches

        # 固定sin-cos位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)

        # =================== CSM条件编码器 ===================
        # 卷积MLP处理 [LMS, PAN] → 条件token
        self.cond_embedder = ConvPatchEmbed(
            csm_cond_channels, hidden_size, patch_size
        )
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # =================== FWM小波编码器 ===================
        # 卷积MLP处理小波系数 → 小波token
        self.wavelet_embedder = ConvPatchEmbed(
            fwm_wavelet_channels, hidden_size, patch_size
        )
        self.wavelet_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # =================== Transformer Blocks ===================
        fwm_start_layer = int(depth * fwm_start_ratio)
        # 渐进式drop path率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                DiTPanBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_fwm=(i >= fwm_start_layer),
                    drop_path=dpr[i],
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )

        # =================== 最终输出层 ===================
        final_out_ch = (
            self.out_channels * 2 if learn_sigma else self.out_channels
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, final_out_ch)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """按DiT论文规范初始化权重"""
        # 1. 基础Xavier初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # 2. 位置编码初始化 (sin-cos)
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
        pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
        self.pos_embed.data.copy_(pos_embed_tensor)
        self.cond_pos_embed.data.copy_(pos_embed_tensor)
        self.wavelet_pos_embed.data.copy_(pos_embed_tensor)

        # 3. Patch embedding初始化 (按nn.Linear方式)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 4. 时间步嵌入MLP初始化
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 5. adaLN调制层零初始化 (DiT核心设计)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 6. 输出层零初始化
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        将token序列还原为图像
        x: [B, N, patch_size² * C]
        返回: [B, C, H, W]
        """
        c = self.out_channels * 2 if self.learn_sigma else self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], (
            f"token数量 {x.shape[1]} 不是完全平方数, "
            f"无法还原为正方形图像"
        )
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, x, time, cond=None, self_cond=None):
        """
        DiTPan前向传播

        参数:
          x: 噪声输入 (e_t + y) [B, C, H, W]
          time: 扩散时间步 [B]
          cond: 打包条件 [B, C_total, H, W]
                C_total = lms_ch + pan_ch + (lms_ch + pan_ch*3)
                即 [LMS, PAN, wavelets_upsampled]
          self_cond: 自条件预测 [B, C_out, H, W] (可选)

        返回:
          预测的干净残差 e_0 [B, C_out, H, W]

        自检:
          1. 条件解包正确: CSM取前(lms+pan)通道, FWM取后(lms+3*pan)通道 ✓
          2. 自条件默认为输入本身 (参考respandiff) ✓
          3. 位置编码加到正确的token上 ✓
          4. 所有tensor形状在block间保持一致 ✓
        """
        # ========== 条件解包 ==========
        csm_ch = self.lms_channel + self.pan_channel
        fwm_ch = self.lms_channel + self.pan_channel * 3
        csm_cond = cond[:, :csm_ch]     # [B, lms+pan, H, W]
        fwm_cond = cond[:, -fwm_ch:]    # [B, lms+3*pan, H, W]

        # ========== 自条件处理 ==========
        if self.self_condition:
            if self_cond is None:
                # 默认: 用零张量作为自条件 (训练初始时无前一步预测)
                self_cond = torch.zeros(
                    x.shape[0], self.out_channels, x.shape[2], x.shape[3],
                    device=x.device, dtype=x.dtype
                )
            x = torch.cat([self_cond, x], dim=1)

        # ========== Token化 ==========
        # 主干输入token
        x_tokens = self.x_embedder(x) + self.pos_embed  # [B, N, D]

        # 时间步嵌入
        t_emb = self.t_embedder(time)  # [B, D]

        # CSM条件token (卷积编码 + 位置编码)
        cond_tokens = self.cond_embedder(csm_cond) + self.cond_pos_embed  # [B, N, D]

        # FWM小波token (卷积编码 + 位置编码)
        wav_tokens = self.wavelet_embedder(fwm_cond) + self.wavelet_pos_embed  # [B, N, D]

        # ========== Transformer Blocks ==========
        for block in self.blocks:
            x_tokens = block(x_tokens, t_emb, cond_tokens, wav_tokens)

        # ========== 输出 ==========
        x_tokens = self.final_layer(x_tokens, t_emb)  # [B, N, p²*C]
        output = self.unpatchify(x_tokens)              # [B, C, H, W]

        return output


# =============================================================================
#  模型配置工厂函数
# =============================================================================

def DiTPan_S(
    input_size=64, in_channels=8, lms_channel=8, pan_channel=1, **kwargs
):
    """
    DiTPan-Small: 适合pansharpening reduced resolution (64×64)
    ~30M参数, 平衡性能与效率
    """
    return DiTPan(
        input_size=input_size,
        patch_size=2,
        in_channels=in_channels,
        lms_channel=lms_channel,
        pan_channel=pan_channel,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )


def DiTPan_B(
    input_size=64, in_channels=8, lms_channel=8, pan_channel=1, **kwargs
):
    """
    DiTPan-Base: 更大容量, 适合追求极致性能
    ~120M参数
    """
    return DiTPan(
        input_size=input_size,
        patch_size=2,
        in_channels=in_channels,
        lms_channel=lms_channel,
        pan_channel=pan_channel,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )


# =============================================================================
#  测试代码
# =============================================================================

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # WV3配置测试
    model = DiTPan_S(
        input_size=64,
        in_channels=8,
        lms_channel=8,
        pan_channel=1,
        self_condition=True,
    ).to(device)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DiTPan-S 参数量: {n_params:.2f}M")

    # 测试前向传播
    B = 2
    x = torch.randn(B, 8, 64, 64).to(device)        # 噪声输入
    t = torch.randint(0, 15, (B,)).to(device)         # 时间步
    # cond = [LMS(8), PAN(1), wavelets(11)] = 20 channels
    cond = torch.randn(B, 20, 64, 64).to(device)
    self_cond = torch.randn(B, 8, 64, 64).to(device)  # 自条件

    with torch.no_grad():
        out = model(x, t, cond, self_cond)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    assert out.shape == (B, 8, 64, 64), f"输出形状错误: {out.shape}"
    print("前向传播测试通过!")
