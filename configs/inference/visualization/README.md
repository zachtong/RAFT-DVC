# 3D Visualization Configuration Guide

## 概述

从硬编码参数迁移到配置文件系统，允许用户通过YAML配置文件完全控制3D volume rendering的所有参数。

## 快速开始

### 使用预定义配置

```bash
# 使用默认配置（与之前的硬编码参数相同）
python inference_test.py \
    --checkpoint outputs/training/exp_uncertainty/checkpoint_best.pth \
    --data_dir data/synthetic_confocal \
    --split test \
    --sample 90 \
    --enable_3d_rendering \
    --viz-config configs/inference/visualization/default.yaml

# 只显示uncertainty，不显示input volume
python inference_test.py \
    --checkpoint outputs/training/exp_uncertainty/checkpoint_best.pth \
    --data_dir data/synthetic_confocal \
    --split test \
    --sample 90 \
    --enable_3d_rendering \
    --viz-config configs/inference/visualization/uncertainty_only.yaml

# 高质量输出（5K分辨率 + slice可视化）
python inference_test.py \
    --checkpoint outputs/training/exp_uncertainty/checkpoint_best.pth \
    --data_dir data/synthetic_confocal \
    --split test \
    --sample 90 \
    --enable_3d_rendering \
    --viz-config configs/inference/visualization/high_quality.yaml

# 暗色主题
python inference_test.py \
    --checkpoint outputs/training/exp_uncertainty/checkpoint_best.pth \
    --data_dir data/synthetic_confocal \
    --split test \
    --sample 90 \
    --enable_3d_rendering \
    --viz-config configs/inference/visualization/dark_theme.yaml
```

### 不使用配置文件（保持向后兼容）

仍然可以使用原有的命令行参数（不提供 `--viz-config`）：

```bash
python inference_test.py \
    --checkpoint outputs/training/exp_uncertainty/checkpoint_best.pth \
    --data_dir data/synthetic_confocal \
    --split test \
    --sample 90 \
    --enable_3d_rendering \
    --render_camera_elevation 25 \
    --render_camera_azimuth 135 \
    --render_zoom 0.4 \
    --render_show_slice \
    --render_resolution 3840 2120
```

**重要**: 如果同时提供 `--viz-config` 和命令行参数，配置文件的参数优先级更高（命令行参数会被忽略）。

---

## 预定义配置文件

### 1. `default.yaml`

- **用途**: 标准可视化，与之前的硬编码参数完全一致
- **特点**:
  - 相机角度: elevation=25°, azimuth=135°
  - Input volume opacity: 0.3（半透明）
  - Uncertainty opacity: 1.0（完全不透明）
  - Density opacity: 0.7
  - 分辨率: 3840×2120
  - 白色背景
  - 显示边界框和color bars

### 2. `uncertainty_only.yaml`

- **用途**: 只显示uncertainty map，不叠加input volume
- **特点**:
  - **Input volume opacity: 0.0**（完全隐藏）
  - 不显示volume color bar（因为volume隐藏了）
  - 其他参数与default相同
- **适用场景**:
  - 专注于uncertainty分布
  - 避免input volume干扰视觉判断
  - 强调模型预测的不确定性区域

### 3. `high_quality.yaml`

- **用途**: 发表级高质量图像
- **特点**:
  - 分辨率: **5120×2880**（5K）
  - 显示2D slice可视化
  - 更大的字体（65px）和更粗的边框（6px）
  - Color bar带outline（更清晰）
- **适用场景**:
  - 论文图片
  - 高分辨率presentation
  - 详细分析

### 4. `dark_theme.yaml`

- **用途**: 演示和presentation用暗色主题
- **特点**:
  - **黑色背景**
  - 热力图colormap（'hot' for uncertainty, 'viridis' for density）
  - 白色边框（适配暗色背景）
  - Cyan色slice frame
  - Input volume opacity稍高（0.4）
- **适用场景**:
  - 黑色背景的slide
  - 投影演示
  - 视觉对比强烈的场景

---

## 配置文件结构

### 完整的配置模板

```yaml
# Camera settings
camera:
  elevation: 25        # 相机仰角（度）
  azimuth: 135         # 相机方位角（度）
  zoom: 0.4            # 缩放因子

# Uncertainty volume (left panel) settings
uncertainty:
  opacity: 1.0         # 不确定性层透明度 (0-1)
  opacity_mode: 'adaptive'  # 'adaptive' (sigmoid) 或 'linear'
  colormap: 'Reds'     # Matplotlib colormap名称
  unit: 'vx'           # 单位标签

# Density volume (right panel) settings
density:
  opacity: 0.7         # Density层透明度 (0-1)
  opacity_mode: 'linear'    # 'adaptive' 或 'linear'
  colormap: 'Greens'   # Matplotlib colormap名称
  unit: ''             # 单位标签（空字符串=无单位）
  threshold: 0.1       # 特征检测阈值
  sigma: 3.0           # Gaussian模糊sigma

# Input volume (beads/particles) settings
input_volume:
  opacity: 0.3         # Input volume叠加透明度 (0-1)
                       # 设置为 0 可完全隐藏input volume
  unit: ''             # 单位标签

# Rendering quality
rendering:
  window_size: [3840, 2120]  # 输出分辨率 [宽, 高]
  background_color: 'white'  # 'white' 或 'black'

# Visual elements
display:
  # Scalar bars (color bars)
  show_scalar_bar: true           # 显示unc/density的color bars
  show_volume_scalar_bar: true    # 显示input volume的color bar
  scalar_bar_font_size: 55        # Color bar字体大小
  scalar_bar_font_family: 'arial' # Color bar字体
  scalar_bar_outline: false       # Color bar边框

  # Coordinate axes
  show_axes: false                # 显示坐标轴

  # Bounding box
  show_bounds: true               # 显示边界框
  bounds_color: 'black'           # 边界框颜色
  bounds_width: 5                 # 边界框线宽

# Slice visualization (optional)
slice:
  show_slice: false              # 在3D volumes下方显示2D slices
  slice_z: null                  # Z坐标（null=中间）
  frame_color: 'darkblue'        # Slice框线颜色
  frame_width: 3.0               # Slice框线宽度
```

---

## 创建自定义配置

### 步骤

1. **复制现有配置作为模板**：
   ```bash
   cp configs/inference/visualization/default.yaml configs/inference/visualization/my_custom.yaml
   ```

2. **编辑参数**：
   - 修改相机角度以获得不同视角
   - 调整透明度以突出不同的信息
   - 更改colormap以适应不同的审美需求
   - 调整分辨率以平衡质量和速度

3. **使用自定义配置**：
   ```bash
   python inference_test.py \
       --checkpoint ... \
       --enable_3d_rendering \
       --viz-config configs/inference/visualization/my_custom.yaml
   ```

### 常见自定义场景

#### 场景1: 隐藏input volume，只看uncertainty

```yaml
input_volume:
  opacity: 0.0         # 完全隐藏
```

#### 场景2: 更高的相机角度（俯视）

```yaml
camera:
  elevation: 60        # 更陡的角度
  azimuth: 135
  zoom: 0.5            # 稍微放大
```

#### 场景3: 使用不同的colormap

```yaml
uncertainty:
  colormap: 'hot'      # 热力图

density:
  colormap: 'Blues'    # 蓝色系
```

#### 场景4: 超高分辨率输出

```yaml
rendering:
  window_size: [7680, 4320]  # 8K分辨率

display:
  scalar_bar_font_size: 80   # 更大字体
  bounds_width: 8            # 更粗边框
```

#### 场景5: 启用slice可视化

```yaml
slice:
  show_slice: true
  slice_z: 32              # 指定z坐标（或null=自动）
  frame_color: 'red'       # 红色框线更醒目
  frame_width: 4.0
```

---

## 参数说明

### Camera参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `elevation` | float | -90 to 90 | 相机仰角，0=水平，正值=俯视 |
| `azimuth` | float | 0 to 360 | 相机方位角，0=正北，顺时针 |
| `zoom` | float | >0 | 缩放因子，<1=缩小，>1=放大 |

### Opacity参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `uncertainty.opacity` | float | 0-1 | Uncertainty层透明度 |
| `density.opacity` | float | 0-1 | Density层透明度 |
| `input_volume.opacity` | float | 0-1 | **Input volume透明度（0=完全隐藏）** |

**重要**: `input_volume.opacity = 0` 可以完全隐藏input volume的显示。

### Opacity Mode

- **`adaptive`**: 使用sigmoid函数，低值更透明，高值快速变不透明
  - 适合uncertainty（突出高不确定性区域）
- **`linear`**: 线性映射，所有值等比例透明
  - 适合density（均匀显示密度分布）

### Colormap选项

常用的matplotlib colormap：

| Colormap | 适用场景 |
|----------|---------|
| `'Reds'` | Uncertainty（默认） |
| `'Greens'` | Density（默认） |
| `'hot'` | 热力图风格 |
| `'viridis'` | 感知均匀，适合暗背景 |
| `'plasma'` | 高对比度 |
| `'Blues'` | 冷色调 |
| `'RdYlBu'` | 发散colormap |

完整列表: https://matplotlib.org/stable/tutorials/colors/colormaps.html

### 分辨率建议

| 用途 | 分辨率 | 说明 |
|------|--------|------|
| 快速预览 | 1920×1080 | 最快 |
| 标准输出 | 3840×2120 | 默认（4K） |
| 高质量 | 5120×2880 | 5K，发表级 |
| 超高清 | 7680×4320 | 8K，打印用 |

**注意**: 更高的分辨率需要更多渲染时间和内存。

---

## 与run.bat集成（可选）

如果希望在`run.bat`中添加配置文件选择，可以添加类似菜单：

```batch
:SELECT_VIZ_CONFIG
echo Select visualization config:
echo  1. Default (standard quality)
echo  2. Uncertainty Only (hide input volume)
echo  3. High Quality (5K + slice)
echo  4. Dark Theme
echo  5. None (use command-line args)
set /p viz_choice="Choice (1-5): "

if "%viz_choice%"=="1" set viz_config=--viz-config configs/inference/visualization/default.yaml
if "%viz_choice%"=="2" set viz_config=--viz-config configs/inference/visualization/uncertainty_only.yaml
if "%viz_choice%"=="3" set viz_config=--viz-config configs/inference/visualization/high_quality.yaml
if "%viz_choice%"=="4" set viz_config=--viz-config configs/inference/visualization/dark_theme.yaml
if "%viz_choice%"=="5" set viz_config=

REM Then in inference command:
python inference_test.py ... %viz_config% %enable_3d_rendering% ...
```

---

## 故障排除

### 问题1: 配置文件未找到

**错误**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决**:
- 确认配置文件路径正确
- 使用绝对路径或相对于项目根目录的路径
- 检查文件名拼写

### 问题2: YAML解析错误

**错误**: `yaml.scanner.ScannerError`

**解决**:
- 检查YAML语法（缩进必须用空格，不能用tab）
- 确认所有字符串正确引号（尤其是包含特殊字符的）
- 使用在线YAML验证器检查

### 问题3: 参数不生效

**可能原因**:
- 配置文件参数被命令行参数覆盖
- 检查是否同时使用了 `--viz-config` 和 `--render_*` 参数

**解决**:
- 只使用配置文件：移除所有 `--render_*` 参数
- 或者不使用配置文件：移除 `--viz-config`

### 问题4: Input volume仍然可见

即使设置了 `input_volume.opacity: 0.0`，可能是因为：

**检查**:
1. 确认配置文件正确加载（查看终端输出）
2. 确认没有其他配置覆盖
3. 尝试手动指定非常小的值（例如 `0.001`）

---

## 最佳实践

### 1. 版本控制

将自定义配置文件加入git版本控制：

```bash
git add configs/inference/visualization/my_custom.yaml
git commit -m "Add custom visualization config for uncertainty analysis"
```

### 2. 配置文件命名

建议命名规范：
- `{purpose}_{variant}.yaml`
- 例如: `uncertainty_high_res.yaml`, `density_dark_bg.yaml`

### 3. 文档注释

在自定义配置文件顶部添加注释说明用途：

```yaml
# Custom visualization for paper Figure 3
# Shows uncertainty-only with high contrast colormap
# Created: 2026-02-11
# Author: Your Name

camera:
  elevation: 30
  ...
```

### 4. 参数实验

创建多个配置文件进行A/B测试：

```bash
# 测试不同的相机角度
python inference_test.py ... --viz-config configs/inference/visualization/camera_view1.yaml
python inference_test.py ... --viz-config configs/inference/visualization/camera_view2.yaml

# 比较输出图像，选择最佳视角
```

---

## 技术细节

### 配置文件加载流程

1. 用户提供 `--viz-config path/to/config.yaml`
2. `load_visualization_config()` 函数读取YAML
3. 嵌套字典被展平为单层参数字典
4. 参数字典通过 `**viz_params` 传递给 `render_3d_volume_comparison()`
5. 渲染函数使用配置参数替代默认值

### 优先级规则

1. **配置文件参数** > 命令行参数 > 函数默认值
2. 如果 `--viz-config` 存在，所有 `--render_*` 参数被忽略
3. 如果 `--viz-config` 不存在，使用命令行参数和函数默认值

### 向后兼容性

- 所有现有的命令行参数仍然有效
- 不使用 `--viz-config` 时，行为与之前完全一致
- 配置文件是可选功能，不影响现有工作流

---

## 相关文档

- [INFERENCE_3D_RENDERING_GUIDE.md](../../../INFERENCE_3D_RENDERING_GUIDE.md) - 3D rendering功能总览
- [viz_helpers.py](../../../viz_helpers.py) - 底层可视化函数
- [inference_test.py](../../../inference_test.py) - 推理脚本

---

**最后更新**: 2026-02-11

**关键特性**:
✅ 完全可配置的3D rendering参数
✅ 预定义的4种配置模板
✅ 支持隐藏input volume（`input_volume.opacity: 0.0`）
✅ 向后兼容所有现有功能
✅ 灵活的自定义配置支持
