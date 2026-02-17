# SEA-RAFT 架构创新在 RAFT-DVC 中的应用评估与未来计划

## 概述

除了 Mixture-of-Laplace (MoL) 不确定性估计之外（见 [MOL_UNCERTAINTY_EXPERIMENTS.md](MOL_UNCERTAINTY_EXPERIMENTS.md)），SEA-RAFT (ECCV 2024) 还提出了多项架构改进。本文档评估这些创新对 **3D 体积 DVC** 场景的适用性，分析实现复杂度，并规划实验路线。

**核心判断**：这些改进属于 **独立的研究贡献**，不应与当前的稀疏特征处理工作（Cutout + Uncertainty）混在一篇论文中。建议作为后续工作或第二篇论文的内容。

---

## 1. Direct Initial Flow Regression（直接回归初始流）

### 1.1 SEA-RAFT 中的做法

传统 RAFT 将位移场初始化为零（`coords1 = coords0`），然后通过 GRU 迭代逐步修正。SEA-RAFT 改为：

1. 将两幅图像拼接为 2 通道输入
2. 送入 Context Encoder（复用已有网络）
3. 直接预测一个粗略的初始位移场
4. GRU 从这个非零初始值开始迭代修正

**效果**：训练时仅需 **4 次迭代**（vs RAFT 的 12 次），推理时 12 次即达到 SOTA。

### 1.2 对 DVC 的价值评估

| 维度 | 评估 |
|------|------|
| **潜在收益** | ★★★★★ 极高 — 3D 迭代的计算/内存成本远高于 2D |
| **实现复杂度** | ★★★☆☆ 中等 — 修改 context encoder 输入和 forward loop |
| **风险** | ★★★☆☆ 中等 — 3D encoder 无预训练权重，初始预测质量未知 |

**关键差异与挑战**：
- SEA-RAFT 使用预训练 ResNet 作为 backbone，天然具备良好的特征表示能力。而 RAFT-DVC 的 3D encoder 从零训练，初始流的预测质量可能较差
- 但即使粗略的初始流（如只捕捉到大致方向和量级），也比零初始化好得多
- 对大变形（>10 voxels）的改善尤其显著——零初始化时需要多次迭代才能"够到"

### 1.3 实现方案

**当前架构**（[src/core/raft_dvc.py](../src/core/raft_dvc.py)）：
```
vol0 (B,1,D,H,W) → cnet → split → (hidden_state, context)  # 1通道输入
vol0, vol1       → fnet → fmap0, fmap1                       # 各1通道输入
coords1 = coords0  # 零初始化
```

**改进方案 A — 复用 Context Encoder（SEA-RAFT 风格）**：
```
cat(vol0, vol1) (B,2,D,H,W) → cnet_init → initial_flow (B,3,D/8,H/8,W/8)
vol0 (B,1,D,H,W) → cnet → split → (hidden_state, context)  # 不变
coords1 = coords0 + initial_flow  # 非零初始化
```

- 需要一个接受 2 通道输入的 encoder（或修改现有 cnet 的第一层）
- 加一个 `InitialFlowHead`（类似 FlowHead，3 通道输出）
- Config toggle: `initial_flow: true/false`

**改进方案 B — 轻量级 Cost Volume 回归**：
```
fmap0, fmap1 → global_corr → lightweight_decoder → initial_flow
```

- 利用已有的 feature maps 计算全局相关性
- 用小型 decoder 从相关性体积回归初始流
- 优点：不增加 encoder 参数；缺点：全局相关性在 3D 中内存开销大

**推荐方案 A**：更简单，与 SEA-RAFT 一致，易于对比。

### 1.4 代码改动清单

| 文件 | 改动 |
|------|------|
| `src/core/raft_dvc.py` | 添加 `use_initial_flow` config；在 `forward()` 中添加初始流预测分支 |
| `src/core/extractor.py` | 添加 `InitialFlowEncoder`（2通道输入版本）或修改 ContextEncoder 支持可变输入通道 |
| `src/core/update.py` | 添加 `InitialFlowHead`（3通道输出） |
| `scripts/train_confocal.py` | 无需改动（初始流在 model forward 内部处理） |
| Config | 添加 `initial_flow: true` toggle |

### 1.5 实验设计

| 实验 | 设置 | 目的 |
|------|------|------|
| Baseline | 12 iterations, zero init | 对照组 |
| InitFlow-12 | 12 iterations, learned init | 初始流是否改善收敛 |
| InitFlow-4 | 4 iterations, learned init | 是否能大幅减少迭代 |
| InitFlow-8 | 8 iterations, learned init | 精度-效率平衡点 |

**评估指标**：
- EPE vs iteration count（每次迭代后的中间EPE，不只是最终）
- GPU内存 vs iteration count
- 训练速度（samples/sec）

---

## 2. Architecture Simplification（架构现代化）

### 2.1 SEA-RAFT 的改进

SEA-RAFT 做了两项重要的架构简化：

**A. Backbone 升级**：自定义 encoder → 标准预训练 ResNet
- 原始 RAFT 的 encoder 是自定义的小型 ResNet 变体
- SEA-RAFT 替换为标准 ResNet-18/34，利用 ImageNet 预训练权重
- 更强的特征提取能力，更稳定的训练

**B. GRU 替换为 ConvNeXt Update Block**：
- 原始 RAFT 使用 ConvGRU（卷积门控循环单元）
- SEA-RAFT 替换为包含 ConvNeXt 模块的简单 RNN
- ConvNeXt 使用 depthwise separable conv + LayerNorm + GELU，更高效

### 2.2 对 DVC 的价值评估

| 维度 | Backbone 升级 | GRU 替换 |
|------|--------------|---------|
| **潜在收益** | ★★★☆☆ | ★★★★☆ |
| **实现复杂度** | ★★★★☆ 高 | ★★★★☆ 高 |
| **风险** | ★★★★☆ 高 | ★★★☆☆ 中 |

**Backbone 升级的挑战**：
- 3D 预训练 ResNet 存在（如 video classification 的 R3D），但：
  - Domain gap 巨大（RGB 视频 vs 灰度共焦显微图像）
  - 通道数不匹配（3ch RGB vs 1ch grayscale）
  - 时空分辨率差异大
- 结论：**预训练权重可能无用**，但标准化的 ResNet-3D 架构本身仍值得尝试

**GRU → ConvNeXt Update 的机会**：
- 3D ConvGRU 是计算瓶颈之一（12次迭代×全3D卷积）
- ConvNeXt 的 depthwise conv 在 3D 中效率提升更显著
- 我们已有 `SepConvGRU3D`（沿 H/W/D 分离），可以作为向 ConvNeXt 过渡的中间步骤

### 2.3 实现方案

**阶段 A — GRU 现代化（推荐先做）**：

```python
class ConvNeXtUpdateBlock(nn.Module):
    """SEA-RAFT style update block with ConvNeXt instead of GRU."""

    def __init__(self, hidden_dim=96, context_dim=64):
        # ConvNeXt block: depthwise_conv → layernorm → pointwise → GELU → pointwise
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, 7, padding=3, groups=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pwconv1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * hidden_dim, hidden_dim)

        # Input projection (motion + context → hidden)
        self.input_proj = nn.Conv3d(motion_dim + context_dim, hidden_dim, 1)

        # Recurrent gate (simple gating mechanism)
        self.gate = nn.Conv3d(hidden_dim * 2, hidden_dim, 1)
```

- Config toggle: `update_block: "gru" | "convnext"`
- 与现有 FlowHead/UncertaintyHead 完全兼容（输入输出维度不变）

**阶段 B — Backbone 标准化（可选）**：

```python
class ResNet3DEncoder(nn.Module):
    """Standard 3D ResNet as feature encoder."""

    def __init__(self, input_dim=1, output_dim=128, layers=[2, 2, 2, 2]):
        # Standard ResNet-18 3D structure
        # Output at 1/8 resolution, matches current BasicEncoder
```

- 与现有 encoder_type 系统集成：`encoder_type: "resnet3d"`

### 2.4 代码改动清单

| 阶段 | 文件 | 改动 |
|------|------|------|
| A | `src/core/update.py` | 添加 `ConvNeXtUpdateBlock` |
| A | `src/core/raft_dvc.py` | 添加 `update_block` config 选项 |
| B | `src/core/extractor.py` | 添加 `ResNet3DEncoder` |
| B | `src/core/raft_dvc.py` | 添加到 encoder_map |

### 2.5 实验设计

| 实验 | Update Block | Encoder | 目的 |
|------|-------------|---------|------|
| Baseline | ConvGRU | BasicEncoder | 对照 |
| ConvNeXt-GRU | ConvNeXt | BasicEncoder | GRU替换的独立效果 |
| ResNet3D | ConvGRU | ResNet3D | Backbone替换的独立效果 |
| Full-Modern | ConvNeXt | ResNet3D | 完全现代化 |

---

## 3. Rigid-Motion Pre-training（刚性运动预训练）

### 3.1 SEA-RAFT 的做法

SEA-RAFT 在 TartanAir 数据集（静态场景+相机运动 → 纯刚性光流）上预训练，然后在目标数据集上微调。

### 3.2 对 DVC 的价值评估

| 维度 | 评估 |
|------|------|
| **潜在收益** | ★★★★☆ 高 — 刚性变换是所有变形的基础 |
| **实现复杂度** | ★☆☆☆☆ 极低 — 已有所有基础设施 |
| **风险** | ★★☆☆☆ 低 — 最差情况是预训练没帮助 |

**DVC 场景下的对应物**：
- TartanAir (刚性运动) → 纯仿射变形（平移 + 旋转 + 缩放）
- 目标数据集 → 复杂的 bspline/localized/combined 变形
- 我们已有完整的数据生成 pipeline（[configs/data_generation/](../configs/data_generation/)）

### 3.3 实现方案

**Step 1 — 生成刚性运动预训练数据集**：
```yaml
# configs/data_generation/confocal_128_rigid_pretrain.yaml
deformation:
  types: ["affine"]      # 仅仿射变形
  affine:
    translation: [-15, 15]  # 较大范围的平移
    rotation: [-10, 10]     # 度
    scale: [0.95, 1.05]
# 粒子参数与 v1 相同
```

**Step 2 — 两阶段训练**：
```
阶段1: 在 rigid 数据上从零训练 300 epochs → checkpoint_pretrained.pth
阶段2: 在 complex deformation 数据上 finetune 200 epochs (低LR)
```

**代码改动：零**。现有的 `finetune` checkpoint 机制已支持此工作流。

### 3.4 实验设计

| 实验 | 预训练 | 微调 | 目的 |
|------|--------|------|------|
| Baseline | 无 | complex 300ep | 对照 |
| Rigid-PT | rigid 300ep | complex 200ep | 预训练效果 |
| Rigid-PT-short | rigid 100ep | complex 200ep | 预训练时长影响 |

**评估重点**：
- 微调阶段的收敛速度（EPE vs epoch 曲线）
- 最终 EPE 是否优于从零训练
- 对大变形样本的改善（预训练应帮助 large displacement）

---

## 4. Iteration Count vs Accuracy 分析（前置实验）

### 4.1 动机

在实现 Initial Flow Regression 之前，需要先建立**当前 baseline 的迭代效率曲线**作为对照：
- 不同迭代次数下 EPE 如何变化？
- 收敛的"拐点"在哪里？（边际收益递减点）
- 推理时间如何随迭代次数线性增长？

这个分析**不需要任何代码改动**，只需要在推理时指定不同的 `--iters` 值。

### 4.2 当前代码现状

**已支持但未暴露**：

| 文件 | `--iters` 参数 | 默认值 | 状态 |
|------|---------------|--------|------|
| [inference_test.py](../inference_test.py) | `--iters` | 20 | 有，但 run.bat 未传递 |
| [scripts/infer.py](../scripts/infer.py) | `--iters` | 24 | 有，但 run.bat 未传递 |
| [run.bat](../run.bat) | — | — | 完全没有迭代次数选项 |
| 模型 forward() | `iters` 参数 | config.iters (12) | 支持任意值 |

### 4.3 实施方案：修改 run.bat

在推理流程中（`:SELECT_INFERENCE_MODE` 之后、`:RUN_INFERENCE` 之前），添加一步迭代次数选择：

```batch
:SELECT_ITERS
cls
echo ======================================================================
echo Select Iteration Count
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Split:         %split%
echo.
echo Options:
echo  1. 4 iterations  (fast, lower accuracy)
echo  2. 8 iterations  (balanced)
echo  3. 12 iterations (training default)
echo  4. 20 iterations (inference default)
echo  5. 24 iterations (high accuracy)
echo  6. Custom number
echo  7. Sweep mode: run 2,4,6,8,12,16,20,24 and compare
echo  8. Back
echo.
echo ======================================================================
set /p iters_choice="Enter your choice (1-8): "

if "%iters_choice%"=="7" goto RUN_ITERS_SWEEP
REM ... (normal single-iters flow)
```

**Sweep 模式**是关键：自动对同一数据集用多个迭代次数推理，输出汇总表。

### 4.4 Sweep 分析脚本

建议新建一个轻量分析脚本 `scripts/analyze_iterations.py`：

```python
"""Analyze EPE vs iteration count for a trained model.

Usage:
    python scripts/analyze_iterations.py \
        --checkpoint outputs/training/.../checkpoint_best.pth \
        --data_dir data/synthetic_confocal_128_v1 \
        --split val \
        --iters_list 2 4 6 8 12 16 20 24

Output:
    - Table: iters | EPE_mean | EPE_median | time_per_sample
    - Plot:  EPE vs iters curve (saved as PNG)
    - CSV:   raw data for further analysis
"""
```

核心逻辑（伪代码）：
```python
for iters in iters_list:
    for sample in dataset:
        t0 = time.time()
        flow_pred = model(vol0, vol1, iters=iters, test_mode=True)
        t1 = time.time()
        epe = compute_epe(flow_pred, flow_gt)
        # 收集 per-sample, per-iters 结果
    # 汇总统计

# 输出表格和曲线图
```

**更进一步**：可以记录每次迭代的**中间 EPE**（不需要 test_mode，直接用 flow_predictions 列表的每个元素计算 EPE），这样一次 24-iteration 推理就能生成完整的 EPE vs iteration 曲线：

```python
# 在 model.forward() 中，不用 test_mode，获取所有中间预测
flow_predictions = model(vol0, vol1, iters=24, test_mode=False)
# flow_predictions[i] 是第 i+1 次迭代后的 flow

for i, flow_pred in enumerate(flow_predictions):
    epe_at_iter_i = (flow_pred - flow_gt).norm(dim=1).mean()
```

这比 sweep 模式更高效——只需要一次 forward pass。

### 4.5 实验设计

| 实验 | 模型 | 数据 | 目的 |
|------|------|------|------|
| **Baseline 收敛曲线** | 当前 best checkpoint | val set (100 samples) | 建立 EPE vs iter 基准曲线 |
| **Cutout 模型对比** | cutout_v2 checkpoint | val set | cutout 是否改变收敛速度 |
| **大变形 vs 小变形** | baseline | 按 GT displacement 分组 | 大变形是否需要更多迭代 |
| *（未来）Initial Flow* | initial flow 模型 | val set | 与 baseline 曲线对比 |

**关键指标**：
- **EPE@iter_k**：第 k 次迭代后的 EPE
- **收敛率**：`(EPE@iter_k - EPE@iter_{k-1}) / EPE@iter_{k-1}`
- **"足够好"的迭代次数**：EPE 相对于最终值的 95% 处的迭代次数
- **推理时间**：每个样本的 wall-clock time vs iters

### 4.6 预期结果

基于 RAFT 论文和 3D 体积的特性，预期：

```
EPE vs Iterations (示意):

EPE
 │
 │  *
 │    *
 │      *
 │        *  *
 │              *  *  *  *  *  *  *  *  ← 收敛（~12+ iters）
 │
 └──────────────────────────────────── iters
    2  4  6  8  10 12 14 16 18 20 22 24
              ↑
          拐点（边际收益递减）
```

- **iter 1-4**：EPE 快速下降（最大的修正在前几步）
- **iter 4-12**：EPE 继续下降但速度放缓
- **iter 12+**：EPE 基本不变（额外迭代是浪费）
- **大变形样本**：收敛更慢，拐点可能在 iter 8-16

这为 Initial Flow Regression 提供量化的改进目标：如果 initial flow 能在 iter 0 就达到 baseline iter 4 的水平，则总迭代次数可以从 12 降到 8。

### 4.7 实施优先级

这个分析应该在 **P0（当前工作）完成后、P1 之前或并行** 进行：
- 不需要任何架构改动
- 只需要新建一个分析脚本 + run.bat 小改动
- 结果直接为论文一提供 "model efficiency" 的数据
- 同时为未来 Initial Flow 工作建立对照基准

---

## 5. 综合优先级排序

### 4.1 实现优先级

| 优先级 | 改进 | 理由 |
|--------|------|------|
| **P0（当前工作）** | Cutout + Uncertainty (MoL/NLL) | 解决核心的稀疏特征问题 |
| **P1（第一个后续实验）** | Rigid-Motion Pre-training | 实现成本最低，风险最小 |
| **P2（第二个后续实验）** | Direct Initial Flow | 对 3D DVC 价值最高，但需要架构改动 |
| **P3（长期）** | ConvNeXt Update Block | 效率提升显著，但验证工作量大 |
| **P4（可选）** | ResNet3D Backbone | 收益不确定（无可用预训练权重） |

### 4.2 推荐的论文组织

**论文一（当前）**：RAFT-DVC + 稀疏特征处理
- Contribution 1: 3D RAFT-DVC baseline（合成数据训练，实验数据验证）
- Contribution 2: Cutout augmentation for sparse features
- Contribution 3: Uncertainty estimation (NLL or MoL+Cutout)
- Discussion: 提及 SEA-RAFT 其他创新作为 future work

**论文二（后续）**：RAFT-DVC 架构优化
- Contribution 1: Direct initial flow regression → 减少迭代次数
- Contribution 2: Rigid-motion pre-training → 更好的收敛
- Contribution 3: ConvNeXt update block → 计算效率
- 如果效果显著，可以组合为 "RAFT-DVC v2"

### 4.3 代码架构兼容性

当前的策略系统（`strategies` config section）已支持良好的模块化。上述改进的集成方式：

```yaml
# 未来完整的 config 可能如下：
model:
  encoder_type: "1/8"           # 或 "resnet3d"
  update_block: "gru"           # 或 "convnext"
  initial_flow: false           # 或 true
  uncertainty_mode: "mol"       # "none" / "nll" / "mol"

strategies:
  cutout:
    enabled: true
  rigid_pretrain:
    checkpoint: "outputs/pretrained/rigid_300ep.pth"  # finetune from

training:
  epochs: 200
  # ...
```

所有改进都是 **独立 toggle**，可以自由组合实验。

---

## 6. 与当前工作的关系

### 6.1 不阻塞当前工作

以上所有改进都**不依赖**当前的 Cutout+MoL 实验结果。当前应聚焦于：

1. 运行 `confocal_128_v1_1_8_p4_r4_cutout_uncertainty_mol_v4` 实验
2. 如果 Cutout+MoL 仍 alpha collapse → fallback 到 plain NLL + `log_b.clamp(min=-2)`
3. 验证 uncertainty map 与实际误差的相关性
4. 在实验数据集上评估泛化能力

### 6.2 可能的交叉收益

- **Initial Flow + Uncertainty**：初始流减少了迭代次数，uncertainty head 也需要更少的迭代来收敛
- **Rigid Pre-training + Cutout**：预训练建立基础匹配能力 → Cutout 微调增强对稀疏区域的鲁棒性
- **ConvNeXt + Initial Flow**：两者结合可能使 4 次迭代在 3D 中成为可能（目前 12 次是必需的）

---

## 附录：SEA-RAFT 关键参考

- **论文**: "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow" (ECCV 2024)
- **代码**: https://github.com/princeton-vl/SEA-RAFT
- **关键配置**: `var_min=0, var_max=10` (β₂ clamp range)
- **训练设置**: TartanAir pretrain → FlyingThings3D → Sintel/KITTI finetune

---

*文档创建时间: 2026-02-11*
*状态: 规划阶段，待当前 Cutout+MoL 实验完成后开始实施*
