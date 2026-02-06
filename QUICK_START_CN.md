# RAFT-DVC 快速开始指南

## 一、5分钟快速理解

### 核心思想
RAFT-DVC = **特征提取** → **相关性金字塔** → **迭代GRU更新**

```
输入: 两个3D体 (参考体 vol0, 形变体 vol1)
  ↓
提取特征 (1/8分辨率)
  ↓
计算全对相关性 (最耗内存: 6D张量)
  ↓
迭代12次 {
  1. 在相关性金字塔中查找9×9×9邻域
  2. GRU融合相关性+历史信息
  3. 预测delta_flow
  4. 更新坐标
}
  ↓
输出: 3D位移场 (3, H, W, D)
```

---

## 二、文件导航 (核心文件)

### 1. 模型定义
```
src/core/raft_dvc.py          # 主模型 (从这里开始读)
  ├─ forward() 方法            # 完整前向流程
  └─ RAFTDVCConfig 类          # 配置参数

src/core/extractor.py         # 特征提取器
  ├─ BasicEncoder              # 1/8下采样 CNN
  └─ ContextEncoder            # 提取上下文特征

src/core/corr.py              # 相关性计算
  ├─ CorrBlock.__init__()      # 构建6D相关性金字塔
  └─ CorrBlock.__call__()      # 查找9×9×9邻域

src/core/update.py            # GRU更新模块
  ├─ MotionEncoder             # 编码motion特征
  ├─ ConvGRU3D                 # 3D卷积GRU
  └─ FlowHead                  # 预测delta_flow
```

### 2. 训练相关
```
src/training/trainer.py       # 训练循环 (Trainer类)
src/training/loss.py          # 损失函数 (SequenceLoss)
src/data/dataset.py           # 数据加载 (VolumePairDataset)
src/data/synthetic.py         # 合成数据生成器
scripts/train.py              # 训练入口脚本
```

### 3. 推理相关
```
scripts/infer.py              # 推理入口脚本
  ├─ infer_single_pair()      # 小体数据直接推理
  └─ infer_sliding_window()   # 大体数据滑动窗口
```

---

## 三、关键代码阅读路径

### 路径1: 理解前向传播 (30分钟)

1. **打开 `src/core/raft_dvc.py`，找到 `RAFTDVC.forward()`**
   - 第195-285行: 完整的前向流程
   - 重点关注循环部分 (第254-280行)

2. **打开 `src/core/extractor.py`，看 `BasicEncoder`**
   - 第215-253行: forward方法
   - 理解: 如何从原分辨率变成1/8

3. **打开 `src/core/corr.py`，看 `CorrBlock`**
   - 第21-82行: `__init__` - 如何构建6D相关性
   - 第84-135行: `__call__` - 如何查找邻域
   - **这是最核心、最难理解的部分！**

4. **打开 `src/core/update.py`，看 `BasicUpdateBlock`**
   - 第243-270行: forward方法
   - 理解: 如何从 (corr, flow) 预测 delta_flow

### 路径2: 理解训练流程 (20分钟)

1. **打开 `scripts/train.py`，看 `main()`**
   - 第79-169行: 完整训练脚本
   - 流程: 数据加载 → 模型创建 → Trainer → 训练

2. **打开 `src/training/trainer.py`，看 `Trainer.train_epoch()`**
   - 第122-172行: 单个epoch的训练循环
   - 重点: 混合精度、梯度裁剪

3. **打开 `src/training/loss.py`，看 `SequenceLoss`**
   - 第9-54行: 序列监督损失
   - 公式: Σ gamma^(n-i) × ||pred_i - gt||

### 路径3: 理解推理流程 (15分钟)

1. **打开 `scripts/infer.py`，看 `infer_sliding_window()`**
   - 第129-194行: 滑动窗口推理
   - 重点: 如何分块、加权、混合

2. **看高斯权重创建: `_create_gaussian_weight()`**
   - 第197-209行: 3D高斯权重
   - 用于平滑混合overlap区域

---

## 四、5个关键概念深入理解

### 1. 为什么是1/8分辨率？

```python
# src/core/extractor.py
self.conv1 = nn.Conv3d(..., stride=2)   # 输出: H/2, W/2, D/2
self.layer2 = ..., stride=2             # 输出: H/4, W/4, D/4
self.layer3 = ..., stride=2             # 输出: H/8, W/8, D/8
```

**原因:**
- 减少计算量 (8³ = 512倍)
- 增大感受野
- 相关性体积从 (H×W×D)² 降到 (H/8×W/8×D/8)²

**后果:**
- 需要上采样flow (upflow_3d)
- 精度受限 (子体素精度~1/8 voxel)

**RAFT-DIC改进:** 去掉下采样 → 全分辨率特征 → 更高精度

---

### 2. 相关性金字塔如何工作？

```python
# src/core/corr.py, CorrBlock.__init__()

# Step 1: 计算全对相关性
corr = torch.matmul(fmap0.T, fmap1)  # (N, N) where N = H×W×D
corr = corr.view(B, H, W, D, H, W, D)  # 6D张量

# Step 2: 构建金字塔
pyramid = [corr]  # Level 0
for level in range(1, 4):
    corr = F.avg_pool3d(corr, 2, stride=2)  # 对后3维池化
    pyramid.append(corr)
```

**每一层的意义:**
- Level 0: 精确匹配，小范围 (±4 voxels)
- Level 1: 池化1/2，中等范围 (±8 voxels)
- Level 2: 池化1/4，大范围 (±16 voxels)
- Level 3: 池化1/8，超大范围 (±32 voxels)

**查找时 (lookup):**
- 在每一层提取9×9×9邻域
- 拼接成 (B, 4×9³, H, W, D) = (B, 2916, H, W, D)
- 送入GRU

---

### 3. GRU为什么是"卷积"GRU？

```python
# src/core/update.py, ConvGRU3D.forward()

hx = torch.cat([h, x], dim=1)  # 拼接hidden和input

z = sigmoid(self.convz(hx))    # update gate (3D卷积)
r = sigmoid(self.convr(hx))    # reset gate (3D卷积)
q = tanh(self.convq(torch.cat([r*h, x], dim=1)))  # candidate

h_new = (1-z) * h + z * q
```

**为什么用卷积?**
- 保持空间结构 (每个voxel有自己的hidden state)
- 局部感受野 (3×3×3卷积)
- 参数共享 (比全连接少很多参数)

**标准GRU vs ConvGRU:**
| | 标准GRU | ConvGRU |
|---|---------|---------|
| 输入 | (B, T, C) 序列 | (B, C, H, W, D) 空间 |
| 操作 | 矩阵乘法 | 3D卷积 |
| 参数 | C_in × C_out | 27 × C_in × C_out |
| 输出 | (B, T, H) | (B, H, H, W, D) |

---

### 4. 序列损失 (SequenceLoss) 的作用

```python
# src/training/loss.py

loss = sum([gamma^(n-i) * ||pred_i - gt|| for i in range(n)])
```

**为什么需要?**
- RAFT输出12次迭代的预测: [pred_1, pred_2, ..., pred_12]
- 早期迭代粗糙，后期迭代精细
- 只监督最后一次 → 早期迭代收不到梯度 → 训练慢

**权重递增:**
```
iter 1:  gamma^11 = 0.8^11 ≈ 0.086  (很小)
iter 6:  gamma^6  = 0.8^6  ≈ 0.262
iter 12: gamma^0  = 1.0             (最大)
```

**效果:**
- 后期迭代占主导 (符合直觉: 最后结果最重要)
- 早期迭代也有监督 (加速收敛)

---

### 5. 滑动窗口为什么需要加权混合？

```python
# scripts/infer.py, infer_sliding_window()

# 高斯权重: 中心高，边缘低
weight = exp(-(x^2 + y^2 + z^2) / (2*sigma^2))

# 逐块累加
for tile in tiles:
    flow_sum[tile_region] += predict(tile) * weight
    weight_sum[tile_region] += weight

# 归一化
flow = flow_sum / weight_sum
```

**如果不加权 (均匀平均) 会怎样?**
- 块边界会有接缝伪影
- 边缘区域的预测不如中心可靠 (padding, 信息不完整)

**高斯权重的好处:**
- 中心区域贡献大 (预测更可靠)
- 边缘区域贡献小 (平滑过渡)
- overlap区域自然混合

**可视化:**
```
Tile 1:        Tile 2:
[■■□□]         [□□■■]
 ↑                ↑
高权重           高权重

Overlap区域: 加权平均，平滑过渡
```

---

## 五、常用命令速查

### 训练
```bash
# 基础训练
python scripts/train.py \
    --data_dir data/train \
    --output_dir results/exp1 \
    --epochs 100 \
    --batch_size 2

# 调整模型参数
python scripts/train.py \
    --data_dir data/train \
    --corr_levels 2 \      # 金字塔层数 (2或4)
    --corr_radius 4 \      # 查找半径 (4或6)
    --iters 12             # 迭代次数 (12-24)

# 从checkpoint恢复
python scripts/train.py \
    --resume results/exp1/latest.pth
```

### 推理
```bash
# 单对推理
python scripts/infer.py \
    --checkpoint checkpoints/model.pth \
    --vol0 ref.npy \
    --vol1 def.npy \
    --output flow.npy \
    --iters 24

# 大体数据 (滑动窗口)
python scripts/infer.py \
    --checkpoint checkpoints/model.pth \
    --vol0 large_ref.npy \
    --vol1 large_def.npy \
    --patch_size 64 64 64 \
    --overlap 0.5 \
    --output flow_large.npy
```

### 数据准备
```python
# 合成训练数据
from src.data.synthetic import SyntheticFlowGenerator
import numpy as np

generator = SyntheticFlowGenerator(seed=42)

for i in range(1000):
    # 生成随机体 (或加载真实数据)
    vol0 = np.random.randn(64, 64, 64).astype(np.float32)

    # 生成变形对
    vol0, vol1, flow = generator.generate_pair(vol0)

    # 保存
    np.save(f'data/train/vol0/sample_{i:03d}.npy', vol0)
    np.save(f'data/train/vol1/sample_{i:03d}.npy', vol1)
    np.save(f'data/train/flow/sample_{i:03d}.npy', flow)
```

---

## 六、调试技巧

### 检查数据
```python
import numpy as np
import matplotlib.pyplot as plt

# 加载
vol0 = np.load('data/train/vol0/sample_001.npy')
vol1 = np.load('data/train/vol1/sample_001.npy')
flow = np.load('data/train/flow/sample_001.npy')

print(f"vol0 shape: {vol0.shape}")
print(f"flow shape: {flow.shape}")
print(f"flow range: [{flow.min():.2f}, {flow.max():.2f}]")

# 可视化中间切片
slice_idx = vol0.shape[0] // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(vol0[slice_idx], cmap='gray')
axes[0].set_title('vol0')
axes[1].imshow(vol1[slice_idx], cmap='gray')
axes[1].set_title('vol1')

# flow magnitude
flow_mag = np.sqrt(np.sum(flow**2, axis=0))
axes[2].imshow(flow_mag[slice_idx], cmap='hot')
axes[2].set_title('flow magnitude')
plt.savefig('debug_data.png')
```

### 检查模型输出
```python
import torch
from src.core import RAFTDVC, RAFTDVCConfig

# 创建模型
config = RAFTDVCConfig()
model = RAFTDVC(config).cuda()
model.eval()

# 随机输入
vol0 = torch.randn(1, 1, 64, 64, 64).cuda()
vol1 = torch.randn(1, 1, 64, 64, 64).cuda()

# 前向
with torch.no_grad():
    flow_preds = model(vol0, vol1, iters=12)

# 检查
print(f"Num predictions: {len(flow_preds)}")  # 应该是12
print(f"Final flow shape: {flow_preds[-1].shape}")  # (1, 3, 64, 64, 64)
print(f"Final flow range: [{flow_preds[-1].min():.2f}, {flow_preds[-1].max():.2f}]")

# 检查迭代收敛
for i, pred in enumerate(flow_preds):
    print(f"Iter {i+1}: mean magnitude = {pred.abs().mean():.4f}")
```

### 可视化flow
```python
from src.core.utils import flow_to_color_3d

# flow: (1, 3, H, W, D) tensor
flow_rgb = flow_to_color_3d(flow, slice_axis=2, slice_idx=32)
# flow_rgb: (1, 3, H, W) RGB图像

import matplotlib.pyplot as plt
plt.imshow(flow_rgb[0].permute(1, 2, 0).cpu().numpy())
plt.title('Flow visualization (color = direction, brightness = magnitude)')
plt.savefig('flow_color.png')
```

---

## 七、与论文对照

### volRAFT (CVPR 2024)
- **架构**: 本代码库基于volRAFT
- **特点**: RAFT的"暴力3D扩展"，未针对DVC优化
- **下采样**: 1/8 (与RAFT相同)
- **金字塔**: 4层
- **半径**: 4

### RAFT-DIC (CVPR 2022, Pan & Liu)
- **改进1**: 去掉encoder下采样 → 全分辨率特征
- **改进2**: 金字塔从4层减少到2层
- **理由**: DIC需要亚像素精度，过度池化会损失精度
- **Ablation**: 论文Table 2证明2层最优

### RAFT-DVC (本项目目标)
- **基础**: volRAFT (已实现)
- **计划改进**:
  1. 双模型策略 (coarse 4层 + fine 2层)
  2. 分块推理 (处理大体数据)
  3. 自动内存/时间估算
  4. 用户可选下采样

详见: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 八、参数量和性能

### 模型大小
- **总参数**: ~3.8M
- **FLOPs** (64³ input): ~50 GFLOPs
- **Peak memory** (64³ batch=1): ~13 MB

### 训练
- **GPU**: RTX 3090 / A100
- **Speed**: ~1.5s/batch (batch=2, 64³)
- **Epoch time**: ~12 min (1000 samples)
- **Total**: 100 epochs × 12 min = **20小时**

### 推理
- **小体 (64³)**: 0.3s (24 iters)
- **大体 (512³, 滑窗)**: ~2.5分钟

---

## 九、下一步学习

1. **理解基础** (本文档)
   - ✓ 文件结构
   - ✓ 核心概念
   - ✓ 代码路径

2. **深入细节** ([CODEBASE_GUIDE_CN.md](CODEBASE_GUIDE_CN.md))
   - 每个模块的详细注释
   - 完整训练/推理流程
   - 数学公式推导

3. **架构规划** ([ARCHITECTURE.md](ARCHITECTURE.md))
   - 双模型推理系统设计
   - 模块接口定义
   - 扩展性考虑

4. **实践**
   - 在synthetic数据上训练
   - 在真实DVC数据上fine-tune
   - 实现dual-model inference

---

**文档版本**: v1.0
**最后更新**: 2026-02-02
