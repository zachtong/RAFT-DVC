# 3D可视化参数调试指南

## 概述

本指南介绍如何使用扩展的3D可视化参数系统来调试和优化 uncertainty 和 density 的体渲染效果。

**新增功能**（2026-02-11）：
- ✅ 相机角度和缩放的精细控制
- ✅ 颜色映射（colormap）和颜色范围（clim）自定义
- ✅ 每层独立的透明度控制
- ✅ 颜色条（scalar bar）显示
- ✅ 背景色、坐标轴等视觉元素控制
- ✅ 交互式调试脚本
- ✅ YAML配置文件支持

---

## 快速开始

### 1. 交互式调试

使用 `tune_3d_viz.py` 快速测试不同的参数组合：

```bash
# 基础测试（使用默认参数）
python tune_3d_viz.py

# 调整相机角度
python tune_3d_viz.py --elevation 60 --azimuth 120

# 放大视图
python tune_3d_viz.py --zoom 1.5

# 更改颜色映射
python tune_3d_viz.py --unc_cmap hot --density_cmap Blues

# 手动设置颜色范围（增强对比度）
python tune_3d_viz.py --unc_clim 0.8 2.0

# 调整透明度
python tune_3d_viz.py --unc_opacity 0.8 --vol_opacity 0.2

# 深色主题
python tune_3d_viz.py --background black --no-axes

# 显示颜色条
python tune_3d_viz.py --scalar_bar --scalar_bar_title "Uncertainty (voxels)"

# 高分辨率输出
python tune_3d_viz.py --resolution 3840 2160

# 使用真实数据
python tune_3d_viz.py --data path/to/volume.npy --unc path/to/uncertainty.npy

# 组合多个参数
python tune_3d_viz.py \
    --elevation 45 --azimuth 135 --zoom 1.2 \
    --unc_cmap plasma --unc_clim 0.5 2.5 \
    --unc_opacity 0.9 --background black \
    --scalar_bar --resolution 2560 1440
```

### 2. 保存最佳配置

找到最佳参数后，保存到 `viz_config_template.yaml`：

```yaml
camera:
  elevation: 45.0
  azimuth: 135.0
  zoom: 1.2

colormaps:
  uncertainty:
    name: 'plasma'
    clim: [0.5, 2.5]
  density:
    name: 'Greens'
    clim: null

opacity:
  mode: 'adaptive'
  volume: 0.2
  uncertainty: 0.9
  density: 0.8

visual:
  background_color: 'black'
  show_axes: false
  title: 'Flow Uncertainty'
  scalar_bar:
    enabled: true
    title: 'Uncertainty (voxels)'

output:
  window_size: [2560, 1440]
  backend: 'auto'
```

### 3. 在推理中使用

方式 A：直接在代码中指定参数

```python
from viz_helpers import visualize_all

# 在推理循环中
for i, batch in enumerate(test_loader):
    # ... 运行模型 ...

    visualize_all(
        vol0, flow_pred, flow_gt, uncertainty,
        output_dir, f'sample_{i:03d}',
        # 自定义参数
        camera_elevation=45,
        camera_azimuth=135,
        zoom=1.2,
        uncertainty_cmap='plasma',
        uncertainty_clim=(0.5, 2.5),
        uncertainty_opacity=0.9,
        background_color='black',
        show_scalar_bar=True
    )
```

方式 B：从配置文件加载

```python
import yaml
from pathlib import Path
from viz_helpers import visualize_all

# 加载配置
config = yaml.safe_load(Path('viz_config_template.yaml').read_text())

# 提取渲染参数（参考 example_inference_with_viz_config.py）
render_kwargs = load_viz_config('viz_config_template.yaml')

# 在推理中使用
for i, batch in enumerate(test_loader):
    # ... 运行模型 ...

    visualize_all(
        vol0, flow_pred, flow_gt, uncertainty,
        output_dir, f'sample_{i:03d}',
        **render_kwargs  # 使用配置文件中的所有参数
    )
```

---

## 参数详解

### 相机控制

#### 简单模式（推荐）

```python
camera_elevation=45.0   # 仰角（0-90度）
                        # 0 = 水平视角，90 = 俯视

camera_azimuth=135.0    # 方位角（0-360度）
                        # 0/360 = 正前方，90 = 右侧，180 = 后方

zoom=1.5                # 缩放（>1 放大，<1 缩小）
```

#### 高级模式

```python
camera_position=[
    (cam_x, cam_y, cam_z),    # 相机位置
    (focal_x, focal_y, focal_z),  # 焦点（通常是体积中心）
    (up_x, up_y, up_z)        # 向上向量
]
```

注意：如果指定 `camera_position`，会覆盖 `elevation/azimuth/zoom`。

### 颜色映射

#### Colormap 选择

常用选项：
- **顺序型**（单一颜色渐变）：`'Reds'`, `'Blues'`, `'Greens'`, `'Purples'`, `'Oranges'`, `'Greys'`
- **热力图**：`'hot'`, `'afmhot'`, `'gist_heat'`
- **感知均匀**：`'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, `'cividis'`
- **发散型**：`'coolwarm'`, `'seismic'`, `'RdBu'`, `'RdYlBu'`

```python
uncertainty_cmap='plasma'  # Uncertainty 层的颜色映射
density_cmap='Greens'      # Density 层的颜色映射
```

完整列表：https://matplotlib.org/stable/gallery/color/colormap_reference.html

#### 颜色范围（clim）

手动设置颜色映射的数值范围以增强对比度：

```python
uncertainty_clim=(0.5, 2.0)  # 将 0.5 映射到最小颜色，2.0 映射到最大颜色
                              # 超出范围的值会被截断

density_clim=(0.0, 1.0)      # Density 的颜色范围

# 不指定（None）= 自动使用数据的 min/max
```

**使用建议**：
- 如果渲染颜色对比度太低，手动缩小 clim 范围
- 例如：数据范围是 [0.3, 3.5]，但大部分数据在 [0.5, 2.0]，则设置 `clim=(0.5, 2.0)` 可以更好地显示主要变化

### 透明度控制

#### 模式选择

```python
opacity_mode='adaptive'  # 自适应模式（sigmoid），强调高值区域
opacity_mode='linear'    # 线性模式，均匀透明度
```

#### 分层透明度

```python
volume_opacity=0.3       # 输入体积（beads）的透明度（0-1）
                         # 建议：0.2-0.4，太高会遮挡 uncertainty/density

uncertainty_opacity=1.0  # Uncertainty 层的透明度缩放（0-1）
                         # 1.0 = 完全不透明（最醒目）

density_opacity=0.8      # Density 层的透明度缩放（0-1）
```

**调试技巧**：
- 如果 uncertainty 层太淡，增加 `uncertainty_opacity`
- 如果 beads 太亮遮挡了 uncertainty，降低 `volume_opacity`
- 如果多层混在一起看不清，分别调整各层透明度

### 视觉元素

```python
background_color='white'  # 背景颜色
                          # 可选：'white', 'black', 或 RGB 元组 (0.5, 0.5, 0.5)

show_axes=True            # 是否显示坐标轴（X, Y, Z）

title='Uncertainty Volume'  # 标题文字
                            # 空字符串 '' 表示不显示标题

show_scalar_bar=True      # 显示颜色条（color bar）
scalar_bar_args={         # 颜色条详细参数（PyVista only）
    'title': 'Uncertainty (voxels)',
    'position_x': 0.85,
    'position_y': 0.05,
    'width': 0.1,
    'height': 0.9
}
```

### 输出设置

```python
window_size=(1920, 1080)  # 输出分辨率（宽, 高）
                          # 常用分辨率：
                          # - Full HD: (1920, 1080)
                          # - 2K: (2560, 1440)
                          # - 4K: (3840, 2160)

backend='auto'            # 渲染后端
                          # 'auto' = 自动选择（优先 PyVista）
                          # 'pyvista' = 高质量（需要安装）
                          # 'matplotlib' = 降级后端（始终可用）
```

---

## 常见可视化场景

### 场景 1：论文高质量图（纯白背景，无干扰元素）

```python
visualize_uncertainty_3d(
    volume, uncertainty, 'uncertainty.png',
    camera_elevation=30,
    camera_azimuth=45,
    zoom=1.3,
    uncertainty_cmap='Reds',
    background_color='white',
    show_axes=False,  # 隐藏坐标轴
    title='',         # 无标题
    window_size=(2560, 1440)  # 高分辨率
)
```

### 场景 2：演示文稿（深色背景，高对比度）

```python
visualize_uncertainty_3d(
    volume, uncertainty, 'uncertainty.png',
    camera_elevation=45,
    camera_azimuth=135,
    zoom=1.2,
    uncertainty_cmap='hot',  # 热力图颜色
    background_color='black',
    show_axes=True,
    title='Flow Uncertainty Estimation',
    uncertainty_opacity=1.0,
    volume_opacity=0.2
)
```

### 场景 3：定量分析（带颜色条和数值范围）

```python
visualize_uncertainty_3d(
    volume, uncertainty, 'uncertainty.png',
    camera_elevation=30,
    camera_azimuth=60,
    uncertainty_cmap='viridis',
    uncertainty_clim=(0.5, 2.0),  # 固定颜色范围
    show_scalar_bar=True,
    scalar_bar_args={'title': 'Uncertainty (voxels)'},
    background_color='white'
)
```

### 场景 4：交互探索（多角度对比）

```bash
# 生成 4 个不同角度的视图
for azim in 0 90 180 270; do
    python tune_3d_viz.py \
        --elevation 30 --azimuth $azim --zoom 1.2 \
        --output uncertainty_azim_${azim}.png
done
```

### 场景 5：动画帧序列（用于制作旋转动画）

```python
import numpy as np
from pathlib import Path

output_dir = Path('animation_frames')
output_dir.mkdir(exist_ok=True)

# 生成 360 度旋转序列（每 5 度一帧）
for i, azimuth in enumerate(np.arange(0, 360, 5)):
    visualize_uncertainty_3d(
        volume, uncertainty,
        output_dir / f'frame_{i:03d}.png',
        camera_elevation=30,
        camera_azimuth=azimuth,
        zoom=1.2,
        uncertainty_cmap='Reds',
        background_color='white',
        show_axes=False,
        title=''
    )

# 使用 ffmpeg 合成视频：
# ffmpeg -framerate 30 -i frame_%03d.png -c:v libx264 -pix_fmt yuv420p rotation.mp4
```

---

## 参数调试工作流

### 步骤 1：确定基础视角

```bash
# 尝试不同的相机角度
python tune_3d_viz.py --elevation 30 --azimuth 45
python tune_3d_viz.py --elevation 45 --azimuth 135
python tune_3d_viz.py --elevation 60 --azimuth 225
```

### 步骤 2：调整缩放

```bash
# 找到合适的缩放级别
python tune_3d_viz.py --elevation 45 --azimuth 135 --zoom 0.8
python tune_3d_viz.py --elevation 45 --azimuth 135 --zoom 1.0
python tune_3d_viz.py --elevation 45 --azimuth 135 --zoom 1.2
python tune_3d_viz.py --elevation 45 --azimuth 135 --zoom 1.5
```

### 步骤 3：优化颜色映射

```bash
# 比较不同的 colormap
python tune_3d_viz.py ... --unc_cmap Reds
python tune_3d_viz.py ... --unc_cmap hot
python tune_3d_viz.py ... --unc_cmap plasma
python tune_3d_viz.py ... --unc_cmap viridis
```

### 步骤 4：设置颜色范围

```bash
# 先不指定，看自动范围
python tune_3d_viz.py ... --show-params
# 查看输出中的 "Uncertainty range: [0.3, 3.5]"

# 根据数据分布手动调整
python tune_3d_viz.py ... --unc_clim 0.5 2.0
python tune_3d_viz.py ... --unc_clim 0.8 2.5
```

### 步骤 5：微调透明度

```bash
# 调整各层透明度
python tune_3d_viz.py ... --vol_opacity 0.2 --unc_opacity 1.0
python tune_3d_viz.py ... --vol_opacity 0.3 --unc_opacity 0.8
```

### 步骤 6：设置输出样式

```bash
# 白色背景，无坐标轴（论文风格）
python tune_3d_viz.py ... --background white --no-axes

# 黑色背景，有颜色条（演示风格）
python tune_3d_viz.py ... --background black --scalar_bar
```

### 步骤 7：保存最佳配置

将最终参数写入 `viz_config_template.yaml`，然后在所有推理中使用。

---

## 完整 API 参考

### PyVista 后端支持的参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `camera_position` | tuple | None | 高级相机位置 `[(x,y,z), (fx,fy,fz), (ux,uy,uz)]` |
| `camera_elevation` | float | 30.0 | 仰角（度，0-90） |
| `camera_azimuth` | float | 45.0 | 方位角（度，0-360） |
| `zoom` | float | 1.0 | 缩放因子 |
| `uncertainty_cmap` | str | 'Reds' | Uncertainty 的颜色映射 |
| `density_cmap` | str | 'Greens' | Density 的颜色映射 |
| `uncertainty_clim` | tuple | None | Uncertainty 颜色范围 `(vmin, vmax)` |
| `density_clim` | tuple | None | Density 颜色范围 `(vmin, vmax)` |
| `opacity_mode` | str | 'adaptive' | 透明度模式（'adaptive' 或 'linear'） |
| `volume_opacity` | float | 0.3 | 输入体积透明度（0-1） |
| `uncertainty_opacity` | float | 1.0 | Uncertainty 层透明度（0-1） |
| `density_opacity` | float | 0.8 | Density 层透明度（0-1） |
| `show_scalar_bar` | bool | False | 显示颜色条 |
| `scalar_bar_args` | dict | None | 颜色条详细参数 |
| `background_color` | str | 'white' | 背景颜色 |
| `show_axes` | bool | True | 显示坐标轴 |
| `title` | str | '...' | 标题文字 |
| `window_size` | tuple | (1920, 1080) | 输出分辨率 `(width, height)` |

### Helper 函数签名

```python
# 单独渲染 uncertainty
visualize_uncertainty_3d(
    volume, uncertainty, output_path,
    backend='auto', colormap='Reds',
    **render_kwargs  # 上述所有参数
)

# 单独渲染 density
visualize_density_3d(
    volume, output_path,
    threshold=0.1, sigma=3.0,
    backend='auto', colormap='Greens',
    **render_kwargs
)

# 一次生成全部 4 张图（2D + 3×3D）
visualize_all(
    vol0, flow_pred, flow_gt, uncertainty,
    output_dir, sample_name='sample',
    backend='auto',
    **render_kwargs
)
```

---

## 故障排除

### 问题 1：渲染的图片太暗

**原因**：Uncertainty 或 density 值范围太大，导致大部分区域映射到颜色条的低端。

**解决方案**：
```python
# 手动设置颜色范围
uncertainty_clim=(percentile_50, percentile_95)
```

或者增加透明度：
```python
uncertainty_opacity=1.0
volume_opacity=0.2  # 降低 beads 亮度
```

### 问题 2：看不到 uncertainty/density 层

**原因**：透明度太低或被 beads 遮挡。

**解决方案**：
```python
uncertainty_opacity=1.0  # 增加到最大
volume_opacity=0.2       # 降低 beads 透明度
```

### 问题 3：颜色条数值范围不对

**原因**：自动 clim 使用了数据的 min/max，但异常值导致范围过大。

**解决方案**：
```python
# 查看数据分布
print(f"Min: {uncertainty.min()}, Max: {uncertainty.max()}")
print(f"Median: {np.median(uncertainty)}")
print(f"95th percentile: {np.percentile(uncertainty, 95)}")

# 使用百分位数设置 clim
uncertainty_clim=(np.percentile(uncertainty, 5), np.percentile(uncertainty, 95))
```

### 问题 4：图片分辨率模糊

**解决方案**：
```python
window_size=(3840, 2160)  # 使用 4K 分辨率
```

### 问题 5：PyVista 报错

**后备方案**：使用 Matplotlib 后端
```python
backend='matplotlib'
```

---

## 最佳实践

1. **先用小数据调试**：使用 `--size 64` 快速迭代参数
2. **固定颜色范围**：对比多个样本时使用相同的 `clim`，确保颜色一致性
3. **保存配置文件**：调试好的参数写入 YAML，避免每次手动指定
4. **批量生成**：使用循环生成多角度或多参数对比图
5. **命名规范**：输出文件名包含参数信息，如 `unc_elev30_azim45_zoom1.2.png`

---

## 相关文件

- `tune_3d_viz.py` - 交互式调试脚本
- `viz_helpers.py` - Helper 函数（用于 inference）
- `viz_config_template.yaml` - 配置文件模板
- `example_inference_with_viz_config.py` - 配置文件使用示例
- `src/visualization/volume_render.py` - 底层渲染实现

---

**最后更新**：2026-02-11
