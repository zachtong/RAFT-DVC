# RAFT-DVC 工具集

本目录包含一些有用的工具脚本，用于环境设置、验证和示例代码。

## 文件说明

### GPU验证工具

- **`test_cuda.py`** - CUDA功能测试脚本
  ```bash
  python tools/test_cuda.py
  ```
  测试内容：
  - PyTorch版本和CUDA可用性
  - 创建GPU tensor
  - 矩阵运算
  - 3D卷积
  - 反向传播

- **`test_cuda.bat`** - Windows批处理启动器
  ```
  双击运行或: tools\test_cuda.bat
  ```

### 环境配置

- **`install_pytorch_nightly.bat`** - PyTorch Nightly安装脚本（RTX 5090必需）

  **重要**: RTX 5090显卡需要CUDA 12.8+支持，必须使用PyTorch Nightly版本。

  用法：
  ```
  双击运行或: tools\install_pytorch_nightly.bat
  ```

  此脚本会：
  1. 卸载现有PyTorch
  2. 安装PyTorch Nightly (CUDA 12.8)
  3. 验证安装是否成功

### 推理示例

- **`example_inference.py`** - 完整的推理示例代码，包含可视化

  功能：
  - 加载训练好的模型
  - 运行推理
  - 生成详细的可视化结果（体积切片、flow对比、误差分析）
  - 保存统计信息

  用法：
  ```bash
  # 需要准备测试数据（vol0, vol1, flow ground truth）
  python tools/example_inference.py
  ```

  输出：
  - 4张可视化PNG图片
  - 预测flow的.npy文件
  - 统计信息文本

  **注意**: 此脚本硬编码了数据路径，需要根据实际情况修改：
  - Line 21: `data_dir` 路径
  - Line 257: checkpoint路径

## 使用场景

### 1. 首次环境设置（RTX 5090用户）

```bash
# 1. 安装PyTorch Nightly
tools\install_pytorch_nightly.bat

# 2. 验证CUDA功能
tools\test_cuda.bat
```

### 2. 环境问题排查

如果训练或推理时遇到CUDA错误：
```bash
python tools/test_cuda.py
```

查看输出，确认：
- ✓ CUDA可用
- ✓ GPU型号正确识别
- ✓ 所有测试通过

### 3. 学习推理流程

查看 `example_inference.py` 了解完整的推理pipeline：
- 如何加载模型
- 如何准备输入数据
- 如何运行推理
- 如何可视化结果

可以将其作为模板，修改为自己的推理脚本。

## 其他说明

- 这些工具是独立的，不是核心代码的一部分
- 核心训练和推理脚本在 `scripts/` 目录
- 详细文档参见项目根目录的 `CODEBASE_GUIDE_CN.md` 和 `QUICK_START_CN.md`
