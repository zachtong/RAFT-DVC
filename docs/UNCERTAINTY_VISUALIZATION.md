# Uncertainty Visualization Guide

本文档介绍 RAFT-DVC 中的不确定性可视化功能，包括 2D 切片和 3D 体渲染。

---

## 功能概述

新的可视化模块 (`src/visualization/`) 提供以下功能：

1. **Feature Density Map**：基于输入体积计算特征密度图（无需训练）
2. **3D Volume Rendering**：高质量的不确定性三维体渲染
3. **2D Slice Visualization**：增强的 2D 切片对比图（包含 feature density）
4. **多后端支持**：自动选择最佳可用后端（PyVista > Matplotlib）

---

## 安装

### 基础依赖（必需）

已包含在 `requirements.txt` 中，安装 RAFT-DVC 时自动安装：

```bash
matplotlib>=3.3.0
scipy>=1.7.0
```

### 高质量 3D 渲染（可选）

为了获得最佳的 3D 体渲染质量，推荐安装 PyVista：

```bash
pip install pyvista
```

**说明**：
- 如果**不安装** PyVista，系统会自动回退到 Matplotlib 后端（质量略低，但功能完整）
- PyVista 需要额外的系统依赖（VTK），在某些环境可能难以安装
- 对于小体积（≤64³），Matplotlib 后端已足够；大体积（≥128³）建议使用 PyVista

---

## 使用方式

### 1. 训练时自动可视化

训练脚本 (`scripts/train_confocal.py`) 已集成 3D 可视化，无需额外配置。

#### 行为

- **2D 可视化**：每个 validation epoch 都会生成
- **3D 可视化**：每 **10 个 epoch** 生成一次（可通过 `render_3d_freq` 参数调整）

#### 输出位置

TensorBoard 日志中：
- `val/visualization_2d`：2D 切片对比图（包含 feature density、scatter plot 等）
- `val/visualization_3d`：3D 体渲染图

#### 查看

```bash
tensorboard --logdir outputs/training/<experiment_name> --port 6006
```

在浏览器中打开 `http://localhost:6006` → **IMAGES** 标签页。

---

### 2. 推理时生成可视化

推理脚本 (`inference_test.py`) 会自动保存不确定性可视化（如果模型有 uncertainty head）。

#### 运行推理

```bash
python inference_test.py \
    --checkpoint outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty/checkpoint_best.pth \
    --data_dir data/synthetic_confocal_128_v1 \
    --split test
```

#### 输出文件

对于每个样本，生成以下文件：
- `uncertainty_<sample_id>.png`：2D 可视化（4×4 面板，包含所有统计信息）
- `uncertainty_<sample_id>_3d.png`：**新增** 3D 体渲染

示例路径：
```
outputs/inference/confocal_128_v1_1_8_p4_r4_uncertainty/checkpoint_best/test/
├── uncertainty_000.png       # 2D 可视化
├── uncertainty_000_3d.png    # 3D 渲染
├── uncertainty_001.png
├── uncertainty_001_3d.png
└── ...
```

---

### 3. 独立测试脚本

提供了一个独立测试脚本，无需运行完整训练即可测试可视化功能。

#### 运行测试

```bash
# 自动选择最佳后端
python test_3d_viz.py

# 指定 PyVista 后端
python test_3d_viz.py --backend pyvista

# 指定 Matplotlib 后端（无需 PyVista）
python test_3d_viz.py --backend matplotlib

# 自定义体积大小
python test_3d_viz.py --size 128 --backend pyvista
```

#### 输出

在 `test_outputs/` 目录下生成：
- `test_3d_<backend>.png`：3D 体渲染
- `test_2d_comparison.png`：2D 对比图

---

## 可视化内容详解

### Feature Density Map

**定义**：基于输入体积的局部特征密度（无需训练，纯几何计算）

**计算方法**：
1. 二值化：`volume > threshold`（默认 0.1）
2. 高斯模糊：sigma=3.0 voxels
3. 归一化到 [0, 1]

**物理意义**：
- **高密度区域**（绿色）：有 beads 存在，测量可靠
- **低密度区域**（暗/透明）：无 beads，预测是外推，不确定性高

---

### 3D Volume Rendering

#### 视觉编码

- **灰色半透明**：输入体积（beads）
- **红色不透明渐变**：不确定性
  - 低不确定性区域：完全透明（不显示）
  - 高不确定性区域：红色强不透明（醒目）
- **绿色半透明**（可选）：feature density

#### 相机角度

默认 45° 俯视角：
- Elevation = 30°
- Azimuth = 45°
- 可通过参数调整

#### 渲染模式（PyVista）

**Adaptive Opacity**（默认）：
- 50th percentile 以下：完全透明
- 50-90th percentile：渐变
- 90th percentile 以上：强不透明

**Linear Opacity**：
- 线性映射不确定性到透明度

---

### 2D Slice Visualization

增强的 2D 可视化包含以下面板（如果有 feature density）：

```
┌──────────────┬──────────────┬──────────────┐
│ Input Volume │ Error Heatmap│ Uncertainty  │
│              │              │  Heatmap     │
├──────────────┼──────────────┼──────────────┤
│ Feature      │ Error vs     │ Correlation  │
│ Density Map  │ Uncertainty  │ by Density   │
│              │ Scatter Plot │ Quartiles    │
└──────────────┴──────────────┴──────────────┘
```

**新增**：
- **Feature Density Map**：显示输入的局部特征密度
- **Correlation by Density Quartiles**：按密度分组的 Spearman ρ
  - 验证假设：高密度区域的 uncertainty 更准确（correlation 更高）

---

## API 参考

### 1. 计算 Feature Density

```python
from src.visualization import compute_feature_density

density = compute_feature_density(
    volume,              # (D, H, W) numpy array
    threshold=0.1,       # intensity threshold for "feature"
    sigma=3.0,           # Gaussian blur sigma (voxels)
    min_density=0.0,     # output minimum
    max_density=1.0      # output maximum
)
# Returns: (D, H, W) density map in [min_density, max_density]
```

---

### 2. 3D 体渲染

```python
from src.visualization import auto_render_uncertainty_volume

# 自动选择后端
fig_or_img = auto_render_uncertainty_volume(
    volume=vol0,              # (D, H, W) input volume
    uncertainty=unc_map,      # (D, H, W) uncertainty
    feature_density=density,  # (D, H, W) optional
    backend='auto',           # 'auto', 'matplotlib', or 'pyvista'
    view_elev=30.0,           # camera elevation
    view_azim=45.0,           # camera azimuth
    figsize=(12, 10),         # matplotlib figure size
    title='Uncertainty 3D'
)

# 如果是 matplotlib Figure，可以保存或添加到 TensorBoard
if hasattr(fig_or_img, 'savefig'):
    fig_or_img.savefig('output.png', dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig_or_img)
```

#### PyVista 后端专用参数

```python
from src.visualization import render_uncertainty_volume_pyvista

render_uncertainty_volume_pyvista(
    volume=vol0,
    uncertainty=unc_map,
    feature_density=density,
    save_path='uncertainty_3d.png',  # 直接保存
    window_size=(1920, 1080),        # 输出分辨率
    opacity_mode='adaptive',         # 'adaptive' 或 'linear'
    camera_position=None,            # 自定义相机位置（可选）
)
```

---

### 3. 2D 对比图

```python
from src.visualization import create_uncertainty_comparison_figure

fig = create_uncertainty_comparison_figure(
    volume=vol0,              # (D, H, W)
    flow_pred=flow_pred,      # (3, D, H, W)
    flow_gt=flow_gt,          # (3, D, H, W)
    uncertainty=unc_avg,      # (D, H, W) averaged over components
    feature_density=density,  # (D, H, W) optional
    slice_idx=64,             # Z-slice index
    figsize=(18, 10)          # with density: (18, 10), else (12, 10)
)
# Returns: matplotlib Figure
```

---

## 技术细节

### 为什么要分离 Learned Uncertainty 和 Feature Density？

在 DVC 问题中：

| 指标 | 来源 | 优势 | 局限性 |
|------|------|------|--------|
| **Learned Uncertainty (b)** | 模型预测 | 捕捉模型内部的"难度感知" | 与实际误差相关性有限（ρ≈0.5） |
| **Feature Density** | 几何计算 | 确定性，直观，高相关性 | 无法反映变形复杂度 |

**两者互补**：
- **Density**：告诉用户"这里有没有测量点"
- **Uncertainty**：告诉用户"模型对这个预测的信心"

在论文中，建议**同时提供**两者，而非将 narrative 定位为"精确的 uncertainty estimation"。

---

### 性能考虑

#### 3D 渲染频率

训练时默认每 10 个 epoch 渲染一次（可调整 `render_3d_freq`）。原因：
- PyVista 渲染：~2-5 秒/volume（128³）
- Matplotlib 渲染：~10-30 秒/volume（64³，128³ 会更慢）

#### 体积大小建议

| 体积大小 | 推荐后端 | 渲染时间（估计） |
|----------|----------|-----------------|
| ≤64³     | Matplotlib | 5-10 秒 |
| 128³     | PyVista | 2-5 秒 |
| 128³     | Matplotlib | 20-40 秒（不推荐）|
| ≥256³    | PyVista | 5-15 秒 |

---

## 故障排查

### Q1: ImportError: No module named 'pyvista'

**解决**：系统会自动回退到 Matplotlib 后端，功能完整但质量略低。

若需要 PyVista：
```bash
pip install pyvista
```

---

### Q2: PyVista 安装失败

**原因**：PyVista 依赖 VTK，在某些系统（Windows、某些 Linux 发行版）可能有兼容性问题。

**解决方案**：
1. 使用 Conda（推荐）：
   ```bash
   conda install -c conda-forge pyvista
   ```

2. 或继续使用 Matplotlib 后端（功能完整，仅渲染质量略低）

---

### Q3: 3D 渲染图为空或全黑

**可能原因**：
- 不确定性值全部很低（所有 voxel 都被阈值过滤掉了）
- Opacity mode 设置不当

**解决**：
- 检查不确定性值分布：`print(uncertainty.min(), uncertainty.max(), uncertainty.mean())`
- 尝试 `opacity_mode='linear'`（而非 `'adaptive'`）
- 调低 `threshold_percentile`（Matplotlib 后端）

---

### Q4: TensorBoard 中看不到 3D 图

**检查**：
1. 确认当前 epoch 是 `render_3d_freq` 的倍数（默认每 10 个 epoch）
2. 确认模型有 uncertainty head（`predict_uncertainty=True`）
3. 刷新 TensorBoard 页面

---

## 示例结果

### 2D Slice Visualization

![2D Example](../assets/uncertainty_2d_example.png) *(示例图，需要实际生成)*

- 左上：输入体积（sparse beads）
- 右上：误差热图（红色=高误差）
- 左下：Feature density（绿色=高密度）
- 右下：Error vs Uncertainty scatter + Spearman ρ

---

### 3D Volume Rendering

![3D Example](../assets/uncertainty_3d_example.png) *(示例图，需要实际生成)*

- 灰色半透明球体：Beads
- 红色云状区域：高不确定性区域（远离 beads）
- 绿色半透明：高 feature density 区域（可选）

---

## 论文中的建议展示

### Figure 布局建议

```
┌─────────────────────────────────────────────────┐
│  Figure X: Uncertainty Quantification           │
├───────────────────┬─────────────────────────────┤
│  (a) 2D Slice     │  (b) 3D Volume Rendering    │
│  - Input          │                              │
│  - Error heatmap  │  45° isometric view          │
│  - Uncertainty    │  Red: high uncertainty       │
│  - Feature density│  Green: high feature density │
├───────────────────┴─────────────────────────────┤
│  (c) Quantitative Analysis                      │
│  - Scatter plot (Error vs Uncertainty)          │
│  - Spearman ρ = 0.52 (full vol)                 │
│  - Correlation by density quartiles             │
└─────────────────────────────────────────────────┘
```

### Caption 建议

> "**Figure X: Dual Confidence Indicators for DVC Predictions.**
> We provide two complementary confidence metrics:
> (a) 2D mid-slice showing input beads, prediction error, learned uncertainty, and geometric feature density.
> (b) 3D volume rendering: red regions indicate high learned uncertainty; green regions indicate high feature density (reliable measurements).
> (c) While learned uncertainty (b) shows moderate correlation with actual error (Spearman ρ=0.52), the combination of learned uncertainty and feature density provides users with a comprehensive assessment of prediction reliability. High-density regions (Q4) show stronger correlation (ρ=0.68), confirming that uncertainty estimation is more accurate near measurement points."

---

## 扩展功能

### 未来可能的增强

1. **交互式 3D 渲染**：使用 Plotly/Dash 生成可旋转的 Web 界面
2. **Uncertainty 热图动画**：逐层切片的视频（GIF/MP4）
3. **多样本对比**：并排显示多个样本的 3D 渲染
4. **VR/AR 支持**：导出为 VTK/STL 格式供 3D Slicer 等工具使用

---

## 相关文档

- [SEA-RAFT Architecture Innovations](SEA_RAFT_ARCHITECTURE_INNOVATIONS.md)：未来的架构改进计划
- [Sparse Features Strategy](sparse_features_strategy.md)：稀疏特征处理的总体策略
- [MOL Uncertainty Experiments](MOL_UNCERTAINTY_EXPERIMENTS.md)：Mixture-of-Laplace 实验记录

---

**版本**：v1.0
**最后更新**：2026-02-11
**维护者**：RAFT-DVC Team
