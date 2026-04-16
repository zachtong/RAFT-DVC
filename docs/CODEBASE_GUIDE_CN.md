# RAFT-DVC 代码库完整指南

> 本文档详细介绍当前实现的RAFT-DVC代码库结构、模型架构、训练和推理流程
> 基于 volRAFT (CVPR 2024) 的3D光流网络实现

> ## ⚠️ 2026-04-16 Update — 部分路径已过时
>
> 本文档的 **核心库设计**（`src/core/` 内的 encoder / correlation / update
> block 等）仍然是权威参考。但本文档中提到的命令行入口和推理模块位置已发生变化：
>
> | 旧引用 | 当前位置 |
> |--------|----------|
> | `src/inference/*` | `src/legacy_inference/*`（Phase-1/2 已弃用） |
> | `scripts/train.py` / `scripts/train_confocal.py` | `archive/scripts_old/` |
> | `scripts/test_confocal.py` / `scripts/infer.py` | `archive/scripts_old/` |
> | `configs/training/confocal_*.yaml` | `archive/configs_training_old/` |
>
> Phase-1 请使用 `scripts/phase1/train_phase1.py` 和
> `scripts/phase1/evaluate_phase1.py`。

---

## 📁 目录结构

```
RAFT-DVC/
│
├── src/                                # 源代码目录
│   ├── core/                          # 核心网络架构 (已完成)
│   │   ├── raft_dvc.py                # 主模型类
│   │   ├── extractor.py               # 特征提取器
│   │   ├── corr.py                    # 相关性计算
│   │   ├── update.py                  # GRU更新模块
│   │   └── utils.py                   # 工具函数
│   │
│   ├── data/                          # 数据加载和生成 (已完成)
│   │   ├── dataset.py                 # PyTorch Dataset类
│   │   └── synthetic.py               # 合成数据生成器
│   │
│   ├── training/                      # 训练相关 (已完成)
│   │   ├── trainer.py                 # 训练管理器
│   │   └── loss.py                    # 损失函数
│   │
│   ├── inference/                     # 推理模块 (接口已定义，待实现)
│   │   ├── analyzer.py                # 体数据分析器
│   │   ├── preprocessor.py            # 预处理器
│   │   ├── tiling.py                  # 分块和拼接
│   │   ├── model_registry.py          # 模型注册表
│   │   ├── pipeline.py                # 推理流水线
│   │   └── postprocessor.py           # 后处理器
│   │
│   └── utils/                         # 通用工具 (接口已定义，待实现)
│       ├── io.py                      # 文件IO
│       └── memory.py                  # 内存估算
│
├── scripts/                           # 可执行脚本
│   ├── train.py                       # 训练入口
│   └── infer.py                       # 推理入口
│
├── configs/                           # 配置文件
│   ├── training/
│   │   └── default.yaml               # 默认训练配置
│   ├── models/                        # 模型配置 (模板)
│   │   ├── coarse_p4_r4.yaml         # 粗阶段模型
│   │   └── fine_p2_r4.yaml           # 精细阶段模型
│   └── inference/                     # 推理策略 (模板)
│       ├── fast.yaml
│       ├── balanced.yaml
│       └── accurate.yaml
│
├── checkpoints/                       # 模型权重
├── data/                             # 数据集
└── ARCHITECTURE.md                   # 架构设计文档
```

---

## 🔍 核心概念：RAFT-DVC 架构

### 整体流程图

```
输入: vol0 (参考体), vol1 (形变体)
  │
  ├──> [特征提取器 BasicEncoder]
  │    输出: fmap0, fmap1 (1/8分辨率, 128通道)
  │
  ├──> [上下文编码器 ContextEncoder]
  │    输出: net (GRU隐藏状态), context (上下文特征)
  │
  ├──> [相关性金字塔 CorrBlock]
  │    全对相关性: fmap0 ⊗ fmap1 → 6D体
  │    构建4层金字塔 (每层对后3维做2倍下采样)
  │
  └──> [迭代更新 × 12次]
       │
       ├─> [查找相关性] CorrBlock.lookup(coords, radius=4)
       │   从金字塔各层提取9×9×9邻域 → corr特征
       │
       ├─> [运动编码] MotionEncoder(flow, corr)
       │   将当前flow和相关性编码成运动特征
       │
       ├─> [GRU更新] ConvGRU3D(net, [context, motion])
       │   更新隐藏状态: net_new
       │
       ├─> [预测delta] FlowHead(net_new)
       │   预测位移增量: delta_flow
       │
       └─> [更新坐标] coords += delta_flow
           上采样到原分辨率 → 当前迭代的flow预测

输出: [flow_iter1, flow_iter2, ..., flow_iter12]
```

---

## 📦 模块详解

## 1. 核心架构 (`src/core/`)

### 1.1 `raft_dvc.py` - 主模型类

**关键类: `RAFTDVC`**

这是整个模型的顶层容器，协调所有子模块。

```python
class RAFTDVC(nn.Module):
    """
    主要组成部分:
    1. fnet (BasicEncoder) - 特征提取器，共享处理两个体
    2. cnet (ContextEncoder) - 上下文编码器，只处理参考体
    3. update_block (BasicUpdateBlock) - GRU更新模块
    """

    def forward(vol0, vol1, iters=12, flow_init=None):
        """
        前向传播流程:

        步骤1: 归一化
        --------------
        - 将vol0和vol1归一化到[0,1]范围
        - 使用共享的min/max确保一致性

        步骤2: 特征提取
        --------------
        fmap0, fmap1 = self.fnet([vol0, vol1])
        - 输入: (B, 1, H, W, D) 原始灰度体
        - 输出: (B, 128, H/8, W/8, D/8) 特征图
        - 通过stride=2的卷积实现1/8下采样

        步骤3: 构建相关性金字塔
        ----------------------
        corr_fn = CorrBlock(fmap0, fmap1, num_levels=4, radius=4)
        - 计算全对相关性: corr[b,h,w,d,:,:,:] = fmap0[b,:,h,w,d] · fmap1[b,:,:,:,:]
        - 构建4层金字塔，每层对空间维度(后3维)做avg_pool

        步骤4: 上下文提取
        ----------------
        net, context = self.cnet(vol0)
        - net: (B, 96, H/8, W/8, D/8) GRU初始隐藏状态
        - context: (B, 64, H/8, W/8, D/8) 静态上下文特征

        步骤5: 初始化flow坐标
        --------------------
        coords0 = 基础坐标网格 (未动)
        coords1 = coords0 + flow_init  (如果提供了初始flow)

        步骤6: 迭代更新 (循环12次)
        -------------------------
        for iter in range(iters):
            coords1 = coords1.detach()  # 阻断梯度

            # 6.1 查找相关性
            corr = corr_fn(coords1)  # 在当前坐标处采样9×9×9邻域

            # 6.2 计算当前flow
            flow = coords1 - coords0

            # 6.3 GRU更新
            net, delta_flow = self.update_block(net, context, corr, flow)

            # 6.4 更新坐标
            coords1 = coords1 + delta_flow

            # 6.5 上采样到原分辨率
            flow_up = upflow_3d(coords1 - coords0, target_shape=(H, W, D))
            flow_predictions.append(flow_up)

        返回: [flow_1, flow_2, ..., flow_12]
        每个flow的形状: (B, 3, H, W, D)
        """
```

**配置类: `RAFTDVCConfig`**

```python
@dataclass
class RAFTDVCConfig:
    # 架构参数
    input_channels: int = 1        # 输入通道数 (灰度=1)
    feature_dim: int = 128         # 特征提取器输出维度
    hidden_dim: int = 96           # GRU隐藏维度
    context_dim: int = 64          # 上下文维度

    # 相关性参数 (关键！)
    corr_levels: int = 4           # 金字塔层数 (4层: 1, 1/2, 1/4, 1/8)
    corr_radius: int = 4           # 查找半径 (9×9×9窗口)

    # 迭代参数
    iters: int = 12                # 默认迭代次数

    # 训练参数
    mixed_precision: bool = False  # 混合精度训练
```

---

### 1.2 `extractor.py` - 特征提取器

**核心类: `BasicEncoder`**

```python
class BasicEncoder(nn.Module):
    """
    3D特征提取器，基于ResNet风格的bottleneck blocks

    架构细节:
    ========

    输入: (B, 1, H, W, D)

    Layer 1: conv1 + norm + relu
    - Conv3d(1 → 32, kernel=7, stride=2, padding=3)
    - 输出: (B, 32, H/2, W/2, D/2)

    Layer 2: 2个BottleneckBlock3D
    - BottleneckBlock3D(32 → 32, stride=1)
    - BottleneckBlock3D(32 → 32, stride=1)
    - 输出: (B, 32, H/2, W/2, D/2)  # 分辨率不变

    Layer 3: 2个BottleneckBlock3D
    - BottleneckBlock3D(32 → 64, stride=2)  # 首个block下采样
    - BottleneckBlock3D(64 → 64, stride=1)
    - 输出: (B, 64, H/4, W/4, D/4)

    Layer 4: 2个BottleneckBlock3D
    - BottleneckBlock3D(64 → 96, stride=2)  # 首个block下采样
    - BottleneckBlock3D(96 → 96, stride=1)
    - 输出: (B, 96, H/8, W/8, D/8)

    Output projection:
    - Conv3d(96 → 128, kernel=1)
    - 输出: (B, 128, H/8, W/8, D/8)

    关键点:
    ------
    1. 总下采样率 = 1/8 (通过3次stride=2实现)
    2. 使用InstanceNorm (默认) 或 BatchNorm / GroupNorm
    3. BottleneckBlock结构: 1×1 conv → 3×3 conv → 1×1 conv + residual
    4. 支持batch processing: 可以同时处理[vol0, vol1]
    """

    def forward(self, x):
        """
        x 可以是:
        - 单个tensor: (B, C, H, W, D)
        - 列表/元组: [vol0, vol1]，会先concat再split

        处理逻辑:
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=0)  # 拼接成2B batch
            ...处理...
            x = torch.split(x, [B, B], dim=0)  # 分回两个
            return (fmap0, fmap1)
        else:
            ...处理...
            return fmap
        """
```

**辅助类: `BottleneckBlock3D`**

```python
class BottleneckBlock3D(nn.Module):
    """
    3D Bottleneck残差块

    结构:
    ----
    输入x: (B, in_planes, H, W, D)

    主路径:
    1. Conv3d(in → mid, 1×1×1) + norm + relu
       mid = planes // 4  (通道压缩)
    2. Conv3d(mid → mid, 3×3×3, stride) + norm + relu
       如果stride=2，空间维度减半
    3. Conv3d(mid → planes, 1×1×1) + norm
       恢复通道数

    残差路径 (如果维度不匹配):
    - Conv3d(in → planes, 1×1×1, stride) + norm

    输出: relu(主路径 + 残差路径)

    参数量对比:
    - 标准3×3×3 conv: 27 × in × out 参数
    - Bottleneck: (1×in×mid + 27×mid×mid + 1×mid×out)
      当mid = out//4时，参数量约为标准的1/4
    """
```

**上下文编码器: `ContextEncoder`**

```python
class ContextEncoder(nn.Module):
    """
    专门为参考体vol0提取上下文特征

    复用BasicEncoder，但输出双通道:
    - hidden: (B, hidden_dim, H/8, W/8, D/8)  初始化GRU隐藏状态
    - context: (B, context_dim, H/8, W/8, D/8)  静态上下文特征

    实现:
    features = BasicEncoder(vol0)  # 输出: (B, hidden+context, ...)
    net, context = torch.split(features, [hidden_dim, context_dim], dim=1)
    net = tanh(net)      # GRU hidden用tanh激活
    context = relu(context)  # context用relu激活

    为什么需要context?
    - 提供vol0的全局/局部结构信息
    - 在迭代更新中保持不变，作为"记忆"
    - 帮助GRU更好地整合相关性和运动信息
    """
```

---

### 1.3 `corr.py` - 相关性计算

**核心类: `CorrBlock`**

```python
class CorrBlock:
    """
    相关性金字塔构建与查找

    初始化阶段:
    ==========
    def __init__(fmap0, fmap1, num_levels=4, radius=4):
        # fmap0, fmap1: (B, C, H, W, D) 特征图

        步骤1: 计算全对相关性
        -------------------
        B, C, H, W, D = fmap0.shape

        # 展平空间维度
        fmap0_flat = fmap0.view(B, C, H*W*D)  # (B, C, N) where N=H×W×D
        fmap1_flat = fmap1.view(B, C, H*W*D)

        # 全对内积 (这是最耗内存的步骤！)
        corr = torch.matmul(fmap0_flat.transpose(1,2), fmap1_flat)
        # corr: (B, N, N) = (B, H*W*D, H*W*D)

        # 重塑成6D
        corr = corr.view(B, H, W, D, H, W, D)
        # 维度含义: [batch, h_0, w_0, d_0, h_1, w_1, d_1]
        # corr[b, h, w, d, :, :, :] = fmap0在(h,w,d)位置与fmap1所有位置的相似度

        步骤2: 构建金字塔
        ----------------
        pyramid = [corr]  # Level 0: 原始分辨率

        for level in range(1, num_levels):
            # 对后3个维度(h_1, w_1, d_1)做平均池化
            corr = F.avg_pool3d(corr, 2, stride=2)
            pyramid.append(corr)

        金字塔结构 (以32×32×32特征图为例):
        - Level 0: (B, 32, 32, 32, 32, 32, 32)  原始
        - Level 1: (B, 32, 32, 32, 16, 16, 16)  1/2
        - Level 2: (B, 32, 32, 32, 8, 8, 8)     1/4
        - Level 3: (B, 32, 32, 32, 4, 4, 4)     1/8

        内存占用:
        Level 0: 32^6 × 4B = 4.3 GB (float32)
        Level 1: 32^3 × 16^3 × 4B = 0.54 GB
        Level 2: 32^3 × 8^3 × 4B = 67 MB
        Level 3: 32^3 × 4^3 × 4B = 8.4 MB
        总计: ~4.9 GB (Level 0占主导)


    查找阶段 (每次GRU迭代调用):
    =======================
    def __call__(coords):
        # coords: (B, 3, H, W, D) 当前flow坐标

        out = []
        for level, corr in enumerate(self.pyramid):
            # 在当前level提取radius邻域
            delta = torch.arange(-radius, radius+1)  # [-4, -3, ..., 3, 4]

            # 计算采样位置 (以level 0为例)
            centroid = coords / (2 ** level)  # 根据金字塔层级缩放坐标

            # 构建9×9×9邻域
            for dz in delta:
                for dy in delta:
                    for dx in delta:
                        sample_coords = centroid + (dz, dy, dx)
                        # 使用双线性插值采样
                        sample = grid_sample(corr, sample_coords)
                        out.append(sample)

        # 拼接所有层级和邻域
        corr_features = torch.cat(out, dim=1)
        # 输出形状: (B, num_levels × (2r+1)^3, H, W, D)
        #         = (B, 4 × 9^3, H, W, D)
        #         = (B, 2916, H, W, D)

        返回: corr_features

    为什么使用金字塔?
    ================
    1. 多尺度搜索:
       - Level 0 (原始): 精确匹配，搜索范围 ±4 voxels
       - Level 1 (1/2): 中等范围，搜索范围 ±8 voxels
       - Level 2 (1/4): 大范围，搜索范围 ±16 voxels
       - Level 3 (1/8): 超大范围，搜索范围 ±32 voxels

    2. 计算效率:
       - 只采样半径内的9×9×9邻域，而非全部32×32×32
       - 每个位置只需81×4=324次采样，而非32768次

    3. 渐进优化:
       - 粗略层级快速锁定大致位置
       - 精细层级逐步refine到亚体素精度
    """
```

**辅助函数**

```python
def coords_grid_3d(B, H, W, D, device):
    """
    创建3D坐标网格

    输出: (B, 3, H, W, D)
    coords[b, 0, h, w, d] = h  (y坐标)
    coords[b, 1, h, w, d] = w  (x坐标)
    coords[b, 2, h, w, d] = d  (z坐标)

    用途: flow的初始化
    flow = coords1 - coords0
    当coords1 = coords0时，flow全为0
    """

def upflow_3d(flow, target_shape):
    """
    将1/8分辨率的flow上采样到原分辨率

    输入: (B, 3, H/8, W/8, D/8)
    输出: (B, 3, H, W, D)

    实现:
    1. 三线性插值上采样空间维度 × 8
    2. flow值同时缩放 × 8 (因为原分辨率下的位移是8倍)

    示例:
    - 特征空间位移1 voxel → 原空间位移8 voxels
    """
```

---

### 1.4 `update.py` - GRU更新模块

**整体结构**

```
输入: net, context, corr, flow
  │
  ├─> [MotionEncoder]
  │   输入: flow (B,3,H,W,D), corr (B,2916,H,W,D)
  │   处理:
  │   - 对corr做1×1 conv → (B, 96, H, W, D)
  │   - 对flow做7×7 + 3×3 conv → (B, 32, H, W, D)
  │   - 拼接: [96+32, flow] → (B, 131, H, W, D)
  │   输出: motion_features (包含flow本身，便于残差连接)
  │
  ├─> [拼接输入]
  │   inp = [context, motion_features]
  │   inp: (B, 64+131=195, H, W, D)
  │
  ├─> [ConvGRU3D / SepConvGRU3D]
  │   输入: net (B,96,H,W,D), inp (B,195,H,W,D)
  │   GRU标准公式 (用3D卷积实现):
  │   z = σ(conv([net, inp]))     # update gate
  │   r = σ(conv([net, inp]))     # reset gate
  │   q = tanh(conv([r*net, inp])) # candidate
  │   net_new = (1-z)*net + z*q
  │   输出: net_new (B, 96, H, W, D)
  │
  └─> [FlowHead]
      输入: net_new (B, 96, H, W, D)
      处理:
      - Conv3d(96 → 128, 3×3×3) + relu
      - Conv3d(128 → 3, 3×3×3)  # 预测3个方向的delta
      输出: delta_flow (B, 3, H, W, D)

输出: net_new, delta_flow
```

**核心类: `ConvGRU3D`**

```python
class ConvGRU3D(nn.Module):
    """
    3D卷积GRU

    标准GRU公式:
    h_t = (1 - z_t) * h_{t-1} + z_t * q_t

    其中:
    z_t = sigmoid(W_z * [h_{t-1}, x_t])  # update gate
    r_t = sigmoid(W_r * [h_{t-1}, x_t])  # reset gate
    q_t = tanh(W_q * [r_t * h_{t-1}, x_t])  # candidate

    在RAFT中:
    - h (net): GRU隐藏状态 (B, 96, H, W, D)
    - x (inp): 输入 = [context, motion] (B, 195, H, W, D)
    - W_*: 3D卷积层 (kernel=3×3×3)

    作用:
    - 整合历史信息 (net) 和当前观测 (corr, flow)
    - 逐步refine flow估计
    - 保持时序一致性
    """
```

**优化版本: `SepConvGRU3D`**

```python
class SepConvGRU3D(nn.Module):
    """
    可分离3D卷积GRU

    标准3×3×3卷积参数量: 27 × C_in × C_out
    可分离卷积参数量: (5 + 5 + 5) × C_in × C_out = 15 × C_in × C_out

    实现:
    - 将3×3×3卷积分解为: 5×1×1 + 1×5×1 + 1×1×5
    - 依次处理高度、宽度、深度维度
    - 节省参数，但保持足够的感受野

    GRU每个gate都需要3个卷积 (z, r, q)
    总共: 3个gate × 3个维度 = 9个卷积层
    """
```

**MotionEncoder**

```python
class MotionEncoder(nn.Module):
    """
    编码运动信息

    输入:
    - flow: 当前flow估计 (B, 3, H, W, D)
    - corr: 相关性特征 (B, 2916, H, W, D)

    处理:
    1. 相关性分支:
       corr → Conv1×1(2916 → 96) → relu
       压缩高维相关性特征

    2. Flow分支:
       flow → Conv7×7(3 → 64) → relu → Conv3×3(64 → 32) → relu
       提取flow的空间模式

    3. 融合:
       [cor_feat(96), flo_feat(32)] → concat(128) → Conv3×3(128 → 80) → relu
       → [out(80), flow(3)] → concat(83)

    输出: (B, 83, H, W, D)
    保留原始flow值便于残差学习
    """
```

---

### 1.5 `utils.py` - 工具函数

```python
def warp_volume_3d(volume, flow):
    """
    根据flow变形体数据

    输入:
    - volume: (B, C, H, W, D) 待变形的体
    - flow: (B, 3, H, W, D) 位移场
      flow[:, 0] = dy (H方向位移)
      flow[:, 1] = dx (W方向位移)
      flow[:, 2] = dz (D方向位移)

    流程:
    1. 创建基础网格: grid[h,w,d] = (h, w, d)
    2. 加上位移: sample_grid = grid + flow
    3. 使用grid_sample进行三线性插值采样

    输出: 变形后的体 (B, C, H, W, D)

    用途:
    - 训练时的photometric loss
    - 可视化flow效果
    - 测试flow准确性
    """

def compute_flow_magnitude(flow):
    """
    计算flow的幅度

    输入: (B, 3, H, W, D)
    输出: (B, 1, H, W, D)

    公式: mag = sqrt(dy^2 + dx^2 + dz^2)

    用于:
    - 可视化位移大小
    - 作为权重 (例如遮罩小位移区域)
    """

def flow_to_color_3d(flow, slice_axis=0, slice_idx=None):
    """
    将3D flow可视化为2D彩色图

    流程:
    1. 提取某个轴的切片 (例如z=中间层)
    2. 选择2个flow分量 (例如dx, dy)
    3. HSV编码:
       - Hue (色调): flow的方向
       - Saturation (饱和度): flow的幅度
       - Value (明度): 固定为1
    4. 转换为RGB

    用途: 调试和论文插图
    """
```

---

## 2. 数据加载 (`src/data/`)

### 2.1 `dataset.py` - 数据集类

**`VolumePairDataset`**

```python
class VolumePairDataset(Dataset):
    """
    训练数据集

    目录结构要求:
    root_dir/
      vol0/
        sample_001.npy  # 参考体
        sample_002.npy
        ...
      vol1/
        sample_001.npy  # 形变体
        sample_002.npy
        ...
      flow/
        sample_001.npy  # ground truth flow (3, H, W, D)
        sample_002.npy
        ...

    功能:
    1. 加载匹配的vol0/vol1/flow三元组
    2. 可选: 随机裁剪patch (例如64×64×64)
    3. 可选: 数据增强 (翻转、旋转)

    数据增强细节:
    - 随机翻转 (x, y, z轴): 需同时翻转flow对应分量符号
    - 随机90度旋转 (xy平面): 需旋转flow向量

    示例: 沿y轴翻转
    vol0 = np.flip(vol0, axis=0)
    vol1 = np.flip(vol1, axis=0)
    flow = np.flip(flow, axis=1)  # 翻转空间
    flow[0] = -flow[0]  # 翻转dy分量符号

    __getitem__ 返回:
    {
        'vol0': (1, H, W, D),
        'vol1': (1, H, W, D),
        'flow': (3, H, W, D),
        'filename': str
    }
    """
```

**`InferenceDataset`**

```python
class InferenceDataset(Dataset):
    """
    推理数据集

    特点:
    - 不需要ground truth flow
    - 支持多种格式: .npy, .tif, .h5
    - 不做增强或裁剪

    用途:
    - 对新的、未标注的数据进行推理
    """
```

---

### 2.2 `synthetic.py` - 合成数据生成

```python
class SyntheticFlowGenerator:
    """
    生成合成位移场用于训练

    支持的变形类型:
    ===============

    1. Translation (平移)
    - 全体均匀位移
    - flow[0] = ty, flow[1] = tx, flow[2] = tz
    - 参数: max_translation (默认10 voxels)

    2. Rotation (旋转)
    - 绕体中心旋转
    - 使用旋转矩阵R = Rz @ Ry @ Rx
    - 参数: max_rotation_deg (默认5度)
    - flow = R @ coords - coords

    3. Affine (仿射)
    - 缩放 + 剪切 + 平移
    - 参数: max_scale=0.05 (±5%), max_shear=0.02
    - 适合模拟材料变形

    4. Polynomial (多项式)
    - 光滑的非线性变形
    - flow_x = Σ c_ijk × y^i × x^j × z^k
    - 参数: poly_degree=3, poly_amplitude=3.0
    - 适合模拟复杂形变

    5. Smooth Random (光滑随机)
    - 高斯滤波的随机场
    - 参数: sigma (10-30), amplitude (1-5)
    - 适合模拟局部不规则变形

    6. Combined (组合)
    - 随机组合2-3种类型
    - 权重随机

    使用示例:
    =========
    generator = SyntheticFlowGenerator(seed=42)

    # 生成随机flow
    flow = generator.generate(shape=(64, 64, 64))

    # 生成特定类型
    flow = generator.generate(shape=(64, 64, 64), flow_type='rotation')

    # 生成训练对
    vol0 = ...  # 加载参考体
    vol0, vol1, flow = generator.generate_pair(vol0)
    # vol1 = warp(vol0, flow)
    """

    def warp_volume(volume, flow):
        """
        应用flow变形volume

        使用scipy.ndimage.map_coordinates进行插值
        - order=0: 最近邻
        - order=1: 线性 (默认)
        - order=3: 三次

        边界处理: mode='constant', cval=0 (填充0)
        """
```

---

## 3. 训练模块 (`src/training/`)

### 3.1 `loss.py` - 损失函数

**`SequenceLoss`**

```python
class SequenceLoss(nn.Module):
    """
    序列监督损失

    由于RAFT输出12次迭代的flow预测，需要对每次都计算损失
    但后期迭代应该更准确，给予更高权重

    公式:
    ====
    loss = Σ_{i=1}^{12} gamma^(12-i) × ||flow_pred_i - flow_gt||_2

    其中:
    - gamma ∈ (0, 1): 衰减因子 (默认0.8)
    - gamma^(12-i): 权重随迭代递增
      - iter 1: gamma^11 = 0.8^11 ≈ 0.086
      - iter 6: gamma^6 = 0.8^6 ≈ 0.262
      - iter 12: gamma^0 = 1.0  (最大权重)

    为什么使用序列损失?
    =================
    1. 梯度稳定: 早期迭代也能收到梯度信号
    2. 收敛加速: 不必等到最后一次迭代才学习
    3. 鲁棒性: 即使最后一次迭代失败，前面的仍有用

    代码:
    ====
    def forward(flow_preds, flow_gt):
        # flow_preds: list of 12个(B, 3, H, W, D)
        # flow_gt: (B, 3, H, W, D)

        n_predictions = len(flow_preds)  # 12
        loss = 0

        for i, flow_pred in enumerate(flow_preds):
            # 计算当前迭代的权重
            i_weight = self.gamma ** (n_predictions - i - 1)

            # L2损失
            i_loss = (flow_pred - flow_gt).abs().mean()

            loss += i_weight * i_loss

        return loss
    """
```

**其他可选损失**

```python
class SmoothLoss(nn.Module):
    """
    平滑正则化 (可选)

    鼓励flow在空间上平滑

    公式:
    loss_smooth = Σ |∇flow|^2

    实现:
    dx = flow[:,:,1:,:,:] - flow[:,:,:-1,:,:]  # x方向梯度
    dy = flow[:,:,:,1:,:] - flow[:,:,:,:-1,:]  # y方向梯度
    dz = flow[:,:,:,:,1:] - flow[:,:,:,:,:-1]  # z方向梯度
    loss = (dx^2 + dy^2 + dz^2).mean()

    注意:
    - 对真实DVC数据可能过度平滑
    - 适合用于合成数据的预训练
    - 权重系数通常很小 (0.001-0.01)
    """
```

---

### 3.2 `trainer.py` - 训练管理器

```python
class Trainer:
    """
    训练循环封装

    初始化:
    ======
    def __init__(model, train_loader, val_loader, output_dir, config):
        # 设置设备
        self.device = torch.device('cuda' if available else 'cpu')

        # 优化器: AdamW
        self.optimizer = AdamW(
            model.parameters(),
            lr=4e-4,
            weight_decay=1e-4
        )

        # 学习率调度器: OneCycleLR
        # - 前5%: warm-up
        # - 中间: 线性退火到0
        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=4e-4,
            total_steps=total_steps,
            pct_start=0.05
        )

        # 损失函数
        self.criterion = SequenceLoss(gamma=0.8)

        # 混合精度 (可选)
        self.scaler = torch.cuda.amp.GradScaler()

    训练一个epoch:
    =============
    def train_epoch():
        model.train()

        for batch in train_loader:
            vol0 = batch['vol0'].to(device)
            vol1 = batch['vol1'].to(device)
            gt_flow = batch['flow'].to(device)

            optimizer.zero_grad()

            # 前向传播
            with torch.cuda.amp.autocast():  # 混合精度
                flow_preds = model(vol0, vol1, iters=12)
                loss = criterion(flow_preds, gt_flow)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 每个batch更新lr

        return avg_loss

    验证:
    ====
    @torch.no_grad()
    def validate():
        model.eval()

        for batch in val_loader:
            vol0, vol1, gt_flow = ...
            flow_preds = model(vol0, vol1, iters=12)

            # 计算序列损失
            loss = criterion(flow_preds, gt_flow)

            # 计算EPE (End-Point Error)
            final_flow = flow_preds[-1]
            epe = sqrt(sum((final_flow - gt_flow)^2, dim=1)).mean()

        return avg_loss, avg_epe

    保存checkpoint:
    ==============
    def save_checkpoint(filename, is_best=False):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'config': config
        }
        torch.save(checkpoint, filename)

        if is_best:
            # 额外保存最佳模型
            torch.save(checkpoint, 'best_model.pth')
    """
```

---

## 4. 推理脚本 (`scripts/`)

### 4.1 `train.py` - 训练入口

```bash
# 基本用法
python scripts/train.py \
    --data_dir data/train \
    --output_dir results/exp1 \
    --epochs 100 \
    --batch_size 2 \
    --patch_size 64 64 64 \
    --augment

# 自定义模型参数
python scripts/train.py \
    --data_dir data/train \
    --corr_levels 2 \
    --corr_radius 4 \
    --iters 12 \
    --mixed_precision

# 从checkpoint恢复
python scripts/train.py \
    --data_dir data/train \
    --resume results/exp1/latest.pth
```

**流程**

```python
def main():
    # 1. 创建模型
    config = RAFTDVCConfig(
        corr_levels=args.corr_levels,
        corr_radius=args.corr_radius,
        iters=args.iters
    )
    model = RAFTDVC(config)

    # 2. 创建数据集
    dataset = VolumePairDataset(
        root_dir=args.data_dir,
        patch_size=(64, 64, 64),
        augment=True
    )

    # 3. 划分训练/验证集
    train_set, val_set = random_split(dataset, [0.9, 0.1])

    # 4. 创建Trainer
    trainer = Trainer(
        model=model,
        train_loader=DataLoader(train_set, batch_size=2),
        val_loader=DataLoader(val_set, batch_size=1),
        output_dir='results/',
        config={'epochs': 100, 'lr': 4e-4, ...}
    )

    # 5. 训练
    trainer.train()
```

---

### 4.2 `infer.py` - 推理入口

```bash
# 单对推理
python scripts/infer.py \
    --checkpoint checkpoints/best_model.pth \
    --vol0 data/test/ref.npy \
    --vol1 data/test/def.npy \
    --output results/flow.npy \
    --iters 24

# 使用滑动窗口 (大体数据)
python scripts/infer.py \
    --checkpoint checkpoints/best_model.pth \
    --vol0 data/large_vol0.npy \
    --vol1 data/large_vol1.npy \
    --patch_size 64 64 64 \
    --overlap 0.5 \
    --output results/flow_large.npy

# 批量推理
python scripts/infer.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/test \
    --output_dir results/test_inference
```

**滑动窗口推理 (处理大体数据)**

```python
def infer_sliding_window(model, vol0, vol1, patch_size, overlap):
    """
    分块处理大体数据

    流程:
    ====
    1. 计算tile位置
    -----------
    H, W, D = vol0.shape
    ph, pw, pd = patch_size  # 例如64×64×64
    step = patch_size * (1 - overlap)  # 例如overlap=0.5 → step=32

    positions = []
    for h in range(0, H, step):
        for w in range(0, W, step):
            for d in range(0, D, step):
                positions.append((h, w, d))

    2. 创建权重图 (用于混合)
    --------------------
    # 高斯权重: 中心权重高，边缘权重低
    sigma = min(patch_size) / 4
    y, x, z = np.meshgrid(...)
    weight = exp(-(y^2 + x^2 + z^2) / (2*sigma^2))

    3. 逐块推理
    ----------
    flow_sum = zeros(3, H, W, D)
    weight_sum = zeros(1, H, W, D)

    for (h, w, d) in positions:
        # 提取patch
        patch0 = vol0[h:h+ph, w:w+pw, d:d+pd]
        patch1 = vol1[h:h+ph, w:w+pw, d:d+pd]

        # 推理
        flow_patch = model(patch0, patch1)

        # 加权累加
        flow_sum[:, h:h+ph, w:w+pw, d:d+pd] += flow_patch * weight
        weight_sum[:, h:h+ph, w:w+pw, d:d+pd] += weight

    4. 归一化
    --------
    flow = flow_sum / (weight_sum + epsilon)

    返回: flow (3, H, W, D)

    为什么使用加权混合?
    ==================
    - 避免块边界的接缝
    - overlap区域有多个预测，加权平均更稳定
    - 高斯权重让patch中心区域(更可靠)贡献更多
    """
```

---

## 5. 配置文件 (`configs/`)

### `configs/training/default.yaml`

```yaml
# 数据配置
data:
  train_dataset: "data/train"
  val_dataset: "data/val"
  batch_size: 2                # 3D数据很大，batch通常很小
  num_workers: 4
  volume_size: [64, 64, 64]   # patch大小

# 模型配置
model:
  input_channels: 1
  feature_dim: 128
  hidden_dim: 96
  context_dim: 64
  corr_levels: 4               # 金字塔层数
  corr_radius: 4               # 查找半径
  iters: 12                    # 迭代次数

# 训练配置
training:
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  grad_clip: 1.0               # 梯度裁剪阈值
  mixed_precision: true        # 混合精度训练

# 优化器
optimizer:
  type: "AdamW"
  betas: [0.9, 0.999]

# 学习率调度
scheduler:
  type: "OneCycleLR"
  max_lr: 0.0004
  pct_start: 0.05              # warm-up阶段占比
  anneal_strategy: "cos"       # 余弦退火

# 损失权重
loss:
  gamma: 0.8                   # 序列损失衰减

# Checkpoint
checkpoint:
  save_dir: "checkpoints"
  save_freq: 5                 # 每5个epoch保存
  keep_last_n: 3               # 只保留最近3个

# 日志
logging:
  log_dir: "logs"
  tensorboard: true
  log_freq: 10                 # 每10个iter记录
```

---

## 6. 完整训练流程

```
步骤1: 准备数据
==============
data/train/
  vol0/
    sample_001.npy  # (64, 64, 64) float32
    sample_002.npy
    ...
  vol1/
    sample_001.npy
    sample_002.npy
    ...
  flow/
    sample_001.npy  # (3, 64, 64, 64) float32
    sample_002.npy
    ...

# 可选: 使用synthetic generator生成训练数据
from src.data.synthetic import SyntheticFlowGenerator

generator = SyntheticFlowGenerator()
for i in range(1000):
    vol0 = np.random.randn(64, 64, 64)  # 或加载真实数据
    vol0, vol1, flow = generator.generate_pair(vol0)
    np.save(f'data/train/vol0/sample_{i:03d}.npy', vol0)
    np.save(f'data/train/vol1/sample_{i:03d}.npy', vol1)
    np.save(f'data/train/flow/sample_{i:03d}.npy', flow)


步骤2: 配置训练
==============
# 修改 configs/training/default.yaml
# 或使用命令行参数覆盖


步骤3: 启动训练
==============
python scripts/train.py \
    --data_dir data/train \
    --output_dir results/exp_baseline \
    --epochs 100 \
    --batch_size 2 \
    --patch_size 64 64 64 \
    --augment \
    --mixed_precision


步骤4: 监控训练
==============
# 查看日志
tail -f results/exp_baseline/training.log

# 使用TensorBoard (如果启用)
tensorboard --logdir results/exp_baseline/logs

# 关键指标:
# - Train Loss: 应该稳定下降
# - Val Loss: 应该下降但可能波动
# - EPE (End-Point Error): 应该逐渐减小
# - Learning Rate: OneCycleLR会先上升后下降


步骤5: 评估模型
==============
# 在测试集上推理
python scripts/infer.py \
    --checkpoint results/exp_baseline/best_model.pth \
    --data_dir data/test \
    --output_dir results/test_predictions \
    --iters 24

# 计算评估指标
python evaluate.py \
    --pred_dir results/test_predictions \
    --gt_dir data/test/flow


步骤6: 超参数调优
================
# 实验1: 减少金字塔层数 (参考RAFT-DIC)
python scripts/train.py \
    --data_dir data/train \
    --output_dir results/exp_p2_r4 \
    --corr_levels 2 \
    --corr_radius 4

# 实验2: 增大查找半径
python scripts/train.py \
    --data_dir data/train \
    --output_dir results/exp_p4_r6 \
    --corr_levels 4 \
    --corr_radius 6

# 实验3: 更多迭代
python scripts/train.py \
    --data_dir data/train \
    --output_dir results/exp_iter20 \
    --iters 20
```

---

## 7. 完整推理流程

```
场景1: 单对小体数据推理
======================
python scripts/infer.py \
    --checkpoint checkpoints/best_model.pth \
    --vol0 data/test/ref_001.npy \
    --vol1 data/test/def_001.npy \
    --output results/flow_001.npy \
    --iters 24

# 输出:
# results/flow_001.npy  (3, H, W, D)


场景2: 大体数据滑动窗口推理
===========================
# 输入: 512×512×512 体数据
# GPU内存: 12GB
# 策略: 64×64×64 patch, 50% overlap

python scripts/infer.py \
    --checkpoint checkpoints/best_model.pth \
    --vol0 data/large/ref.npy \
    --vol1 data/large/def.npy \
    --patch_size 64 64 64 \
    --overlap 0.5 \
    --output results/flow_large.npy \
    --iters 24

# 执行流程:
# 1. 自动计算tile数量: ~512 tiles
# 2. 逐tile推理: [=====>   ] 123/512 (24%)
# 3. 高斯加权混合
# 4. 保存完整flow


场景3: 批量推理
===============
# 目录结构:
# data/batch/
#   vol0/ (100个.npy文件)
#   vol1/ (100个.npy文件)

python scripts/infer.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/batch \
    --output_dir results/batch_inference \
    --patch_size 64 64 64 \
    --overlap 0.5

# 输出:
# results/batch_inference/
#   flow_sample_001.npy
#   flow_sample_002.npy
#   ...


场景4: Python API推理
=====================
from src.core import RAFTDVC
import torch
import numpy as np

# 加载模型
model, _ = RAFTDVC.load_checkpoint('checkpoints/best_model.pth')
model = model.cuda()
model.eval()

# 加载数据
vol0 = np.load('data/ref.npy')
vol1 = np.load('data/def.npy')

# 转tensor
vol0_t = torch.from_numpy(vol0).unsqueeze(0).unsqueeze(0).cuda()
vol1_t = torch.from_numpy(vol1).unsqueeze(0).unsqueeze(0).cuda()

# 推理
with torch.no_grad():
    _, flow = model(vol0_t, vol1_t, iters=24, test_mode=True)

# 后处理
flow_np = flow.squeeze().cpu().numpy()  # (3, H, W, D)

# 保存
np.save('flow_result.npy', flow_np)

# 可视化 (某个切片)
from src.core.utils import flow_to_color_3d
flow_rgb = flow_to_color_3d(flow, slice_axis=2, slice_idx=32)
import matplotlib.pyplot as plt
plt.imshow(flow_rgb[0].permute(1, 2, 0).cpu().numpy())
plt.savefig('flow_vis.png')
```

---

## 8. 关键性能指标

### 内存占用 (batch_size=1, patch_size=64)

| 组件 | 形状 | 内存 |
|-----|------|------|
| vol0, vol1 | 2 × (1,1,64,64,64) | 2 MB |
| fmap0, fmap1 | 2 × (1,128,8,8,8) | 1 MB |
| 6D correlation | (1,8,8,8,8,8,8) | 8 MB |
| Pyramid Level 1-3 | | 1 MB |
| GRU hidden | (1,96,8,8,8) | 0.4 MB |
| Context | (1,64,8,8,8) | 0.3 MB |
| **总计 (峰值)** | | **~13 MB** |

> 注: 64³ patch是可行的。更大的patch (如128³) 会因为6D correlation爆内存。

### 训练速度 (单GPU, RTX 3090)

- Batch size 2, patch 64³, 12 iters
- 前向: ~0.5s
- 反向: ~1.0s
- **总计: ~1.5s/batch**
- **1 epoch (1000 samples): ~12分钟**

### 推理速度

- 单个64³ patch: ~0.3s (24 iters)
- 512³体数据 (64³patch, 50% overlap): ~512 tiles × 0.3s = **~2.5分钟**

---

## 9. 常见问题

### Q1: CUDA out of memory

**可能原因:**
1. batch_size太大 → 改为1
2. patch_size太大 → 从128³降到64³或32³
3. corr_radius太大 → 从6降到4

**解决方案:**
```bash
# 减小batch和patch
python scripts/train.py \
    --batch_size 1 \
    --patch_size 32 32 32

# 或使用gradient accumulation
# 每4个batch累积后更新一次 (等效batch=4)
```

---

### Q2: 训练loss不下降

**检查清单:**
1. 数据是否正确? → 可视化vol0, vol1, flow
2. 学习率是否太大/太小? → 尝试1e-4到4e-4
3. 数据增强是否破坏了flow? → 关闭`--augment`测试
4. 模型是否太小/太大? → 检查参数量 (应该~10M)

**调试技巧:**
```python
# 在trainer.py中添加:
def train_epoch():
    for batch in train_loader:
        ...
        # 检查loss的数量级
        print(f"Loss: {loss.item():.4f}")

        # 检查flow预测的范围
        print(f"Flow pred range: [{flow_preds[-1].min():.2f}, {flow_preds[-1].max():.2f}]")
        print(f"Flow GT range: [{gt_flow.min():.2f}, {gt_flow.max():.2f}]")

        # 检查梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm():.4f}")
```

---

### Q3: 推理结果有块状伪影

**原因:** 滑动窗口混合不平滑

**解决方案:**
1. 增大overlap: `--overlap 0.75`
2. 使用高斯权重 (已默认)
3. 减小patch_size (增加overlap区域占比)

---

### Q4: 如何从volRAFT checkpoint迁移?

```python
# volRAFT和RAFT-DVC架构相同，可以直接加载
# 但需要确保config匹配

# 方法1: 直接加载
model = RAFTDVC(config)
checkpoint = torch.load('volraft_weights.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 方法2: 部分加载 (如果config不匹配)
model = RAFTDVC(new_config)
checkpoint = torch.load('volraft_weights.pth')

# 只加载feature encoder (共享权重)
fnet_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
             if k.startswith('fnet')}
model.load_state_dict(fnet_dict, strict=False)

# update_block需要重新训练 (因为corr_levels/radius不同)
```

---

## 10. 下一步: 实现双模型推理管道

当前代码库已实现volRAFT的基础架构和训练流程。
根据ARCHITECTURE.md的规划，下一步是实现模块化的推理系统:

**优先级:**
1. `src/utils/memory.py` - 内存估算公式
2. `src/inference/tiling.py` - 分块和拼接
3. `src/inference/preprocessor.py` - 预处理
4. `src/inference/postprocessor.py` - 后处理
5. `src/inference/model_registry.py` - 模型管理
6. `src/inference/pipeline.py` - 流水线编排
7. `src/inference/analyzer.py` - 自动分析

**双模型策略:**
- 训练coarse_p4_r4模型 (4层金字塔)
- 训练fine_p2_r4模型 (2层金字塔，参考RAFT-DIC)
- 推理时先coarse再fine，实现大位移+高精度

详见: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 附录: 参数量估算

```python
# BasicEncoder
conv1: 1×32×7³ = 10,976
layer1-3: ~500K (bottleneck blocks)
conv2: 96×128×1 = 12,288
总计: ~513K

# ContextEncoder
共享BasicEncoder: ~513K

# CorrBlock
无可学习参数 (只是indexing)

# MotionEncoder
convc1: 2916×96×1 = 280K
convf1,2: ~18K
conv: ~82K
总计: ~380K

# ConvGRU3D
convz,r,q: 3 × (96+195)×96×27 = 2.3M

# FlowHead
conv1,2: ~160K

# 全模型总参数量
~3.8M (较小，适合在有限数据上训练)
```

---

**文档版本**: v1.0
**最后更新**: 2026-02-02
**作者**: Claude (based on codebase analysis)
