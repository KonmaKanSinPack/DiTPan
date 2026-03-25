# DiTPan: Diffusion Transformer for Pansharpening

基于 Diffusion Transformer 的全光谱融合模型，将 DiT 架构适配到 pansharpening 任务。

## 核心创新

1. **CSM (粗粒度风格调制)**: 卷积MLP从 [LMS, PAN] 提取空间风格信息，生成逐token的 scale/shift 进行仿射变换
2. **FWM (细粒度小波调制)**: 线性复杂度交叉注意力，将小波分解的频率信息注入后半段 Transformer blocks
3. **ShiftDiffusion**: 残差空间扩散 (e₀ = HR - LMS)，而非标准 DDPM
4. **Self-Conditioning**: 利用前一步预测作为额外条件输入
5. **QK-Norm + SwiGLU FFN**: 提高训练稳定性和表达能力

## 环境配置

```bash
# 推荐 Python 3.10+, PyTorch 2.0+
pip install torch torchvision einops h5py pywt scikit-image scipy matplotlib tqdm
```

## 项目结构

```
DiTPan/
├── train.py                    # 训练脚本
├── test.py                     # 推理/测试脚本
├── models/
│   └── dit_pan.py              # DiTPan 模型 (CSM + FWM)
├── diffusion/
│   └── shift_diffusion.py      # ShiftDiffusion 扩散过程
├── dataset/
│   ├── pan_dataset.py          # 遥感数据集 (WV3/GF2/QB)
│   └── hisr.py                 # 高光谱数据集 (CAVE/Harvard)
├── utils/
│   ├── metric.py               # SAM/ERGAS/PSNR/CC/SSIM 指标
│   ├── optim_utils.py          # EMA 更新器
│   ├── logger.py               # Tensorboard 日志
│   ├── lr_scheduler.py         # 学习率调度
│   └── misc.py                 # 工具函数
├── data/                       # 数据目录
│   └── wv3/
│       ├── train_wv3.h5
│       └── test_wv3_multiExm1.h5
└── weights/                    # 模型权重保存目录
```

## 数据格式

H5 文件需包含以下字段:

| 字段 | 形状 | 说明 |
|------|------|------|
| `gt`  | (N, C, H, H) | Ground Truth 多光谱图像 |
| `lms` | (N, C, H, H) | 上采样后的低分辨率多光谱图像 |
| `ms`  | (N, C, H/4, H/4) | 原始低分辨率多光谱图像 |
| `pan` | (N, 1, H, H) | 全色波段图像 |

WV3 数据集: C=8, 训练 H=64, 测试 H=256。像素值范围 [0, 2047]。

## 训练

```bash
python train.py \
    --dataset_name wv3 \
    --train_path data/wv3/train_wv3.h5 \
    --valid_path data/wv3/test_wv3_multiExm1.h5 \
    --device cuda:0
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_name` | `wv3` | 数据集: wv3, wv2, gf2, qb, cave, harvard |
| `--train_path` | (必填) | 训练 H5 文件路径 |
| `--valid_path` | (必填) | 验证 H5 文件路径 |
| `--model_size` | `S` | 模型大小: S (79M) 或 B |
| `--image_size` | `64` | 训练图像尺寸 |
| `--n_steps` | `15` | 扩散步数 |
| `--schedule_type` | `cosine` | 噪声调度: cosine 或 resshift |
| `--batch_size` | `32` | 批量大小 |
| `--lr` | `1e-4` | 初始学习率 |
| `--max_iterations` | `500000` | 最大训练迭代次数 |
| `--val_interval` | `5000` | 验证间隔 |
| `--loss_type` | `l2` | 损失函数: l1, l2, smoothl1 |
| `--penalty_weight` | `100.0` | 范围惩罚权重 |
| `--device` | `cuda:0` | 训练设备 |
| `--save_dir` | `./weights` | 权重保存目录 |
| `--pretrain_weight` | None | 预训练权重路径 (可选) |

训练过程会自动:
- 每 5000 步验证并保存模型 (按 SAM 指标选择最优)
- 使用 EMA 模型进行验证和保存
- 记录 Tensorboard 日志到 `./runs/`

## 测试

```bash
python test.py \
    --test_path data/wv3/test_wv3_multiExm1.h5 \
    --weight_path weights/best_wv3.pth \
    --dataset_name wv3 \
    --device cuda:0
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test_path` | (必填) | 测试 H5 文件路径 |
| `--weight_path` | (必填) | 模型权重路径 |
| `--dataset_name` | `wv3` | 数据集名称 |
| `--model_size` | `S` | 与训练时一致 |
| `--image_size` | `64` | 与训练时一致 |
| `--n_steps` | `15` | 扩散步数 |
| `--schedule_type` | `cosine` | 噪声调度 |
| `--batch_size` | `1` | 推理批量大小 |
| `--full_res` | (flag) | Full resolution 测试 (无GT) |
| `--show` | (flag) | 保存可视化结果 |
| `--device` | `cuda:0` | 推理设备 |

测试输出:
- 终端打印 SAM / ERGAS / PSNR / CC / SSIM 指标
- 保存 `.mat` 文件到 `./samples/mat/`

## 模型架构

**DiTPan-S** (默认): hidden_size=384, depth=12, num_heads=6, patch_size=2, **79.33M** 参数

条件注入方式:
- 输入条件: `cond = [LMS(C), PAN(1), wavelets_upsampled(C+3)]` → 共 2C+4 通道 (WV3: 20通道)
- **CSM** (全部 12 个 block): 卷积MLP提取空间风格 → 逐token仿射变换
- **FWM** (后 6 个 block): 小波频率信息 → 线性交叉注意力
- **AdaLN**: 时间步 + 全局条件 → 全局调制

支持可变分辨率推理 (通过位置编码双三次插值)。
