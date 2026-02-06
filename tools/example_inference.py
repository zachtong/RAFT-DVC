"""
测试推理脚本 - 使用训练的模型进行推理并可视化结果

用法:
    python scripts/test_inference.py
"""

import os
import sys
from pathlib import Path

# 解决OpenMP库冲突问题（Windows上常见）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from src.core import RAFTDVC


def load_test_data():
    """加载一个测试样本"""
    data_dir = Path("data/test_dataset_merged")

    # 使用验证集样本（未用于训练）
    vol0 = np.load(data_dir / "vol0" / "val_sample_0000.npy")
    vol1 = np.load(data_dir / "vol1" / "val_sample_0000.npy")
    flow_gt = np.load(data_dir / "flow" / "val_sample_0000.npy")

    return vol0, vol1, flow_gt


def run_inference(model, vol0, vol1, device, iters=12):
    """运行推理"""
    # 准备输入
    vol0_t = torch.from_numpy(vol0.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    vol1_t = torch.from_numpy(vol1.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    vol0_t = vol0_t.to(device)
    vol1_t = vol1_t.to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        _, flow = model(vol0_t, vol1_t, iters=iters, test_mode=True)

    # 转换为numpy
    flow = flow.squeeze(0).cpu().numpy()

    return flow


def visualize_flow(vol0, vol1, flow_pred, flow_gt, output_dir):
    """
    可视化流场结果

    保存以下内容：
    1. 中间切片的体积对比
    2. 预测flow vs 真实flow的magnitude对比
    3. 每个分量的误差分布
    4. 统计信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W, D = vol0.shape
    mid_h, mid_w, mid_d = H // 2, W // 2, D // 2

    # 计算flow magnitude
    flow_pred_mag = np.linalg.norm(flow_pred, axis=0)
    flow_gt_mag = np.linalg.norm(flow_gt, axis=0)

    # 计算误差
    flow_error = flow_pred - flow_gt
    epe = np.linalg.norm(flow_error, axis=0)  # End-Point Error

    # ========== 1. 体积中间切片对比 ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Volume Slices (Middle)', fontsize=16, fontweight='bold')

    # XY平面 (Z=mid_d)
    axes[0, 0].imshow(vol0[:, :, mid_d], cmap='gray')
    axes[0, 0].set_title('Vol0 - XY Slice')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(vol1[:, :, mid_d], cmap='gray')
    axes[1, 0].set_title('Vol1 - XY Slice')
    axes[1, 0].axis('off')

    # XZ平面 (Y=mid_w)
    axes[0, 1].imshow(vol0[:, mid_w, :], cmap='gray')
    axes[0, 1].set_title('Vol0 - XZ Slice')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(vol1[:, mid_w, :], cmap='gray')
    axes[1, 1].set_title('Vol1 - XZ Slice')
    axes[1, 1].axis('off')

    # YZ平面 (X=mid_h)
    axes[0, 2].imshow(vol0[mid_h, :, :], cmap='gray')
    axes[0, 2].set_title('Vol0 - YZ Slice')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(vol1[mid_h, :, :], cmap='gray')
    axes[1, 2].set_title('Vol1 - YZ Slice')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / '1_volume_slices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 2. Flow Magnitude 对比 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Flow Magnitude Comparison', fontsize=16, fontweight='bold')

    vmin = min(flow_pred_mag.min(), flow_gt_mag.min())
    vmax = max(flow_pred_mag.max(), flow_gt_mag.max())

    # 预测的magnitude
    im0 = axes[0, 0].imshow(flow_pred_mag[:, :, mid_d], cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Predicted Flow Mag - XY')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(flow_pred_mag[:, mid_w, :], cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Predicted Flow Mag - XZ')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(flow_pred_mag[mid_h, :, :], cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Predicted Flow Mag - YZ')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])

    # 真实的magnitude
    im3 = axes[1, 0].imshow(flow_gt_mag[:, :, mid_d], cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Ground Truth Flow Mag - XY')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(flow_gt_mag[:, mid_w, :], cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Ground Truth Flow Mag - XZ')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(flow_gt_mag[mid_h, :, :], cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title('Ground Truth Flow Mag - YZ')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(output_dir / '2_flow_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 3. Flow 分量对比 ==========
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    fig.suptitle('Flow Components (XY Slice at Z=mid)', fontsize=16, fontweight='bold')

    components = ['X', 'Y', 'Z']
    for i in range(3):
        vmin = min(flow_pred[i, :, :, mid_d].min(), flow_gt[i, :, :, mid_d].min())
        vmax = max(flow_pred[i, :, :, mid_d].max(), flow_gt[i, :, :, mid_d].max())

        # 预测
        im_pred = axes[i, 0].imshow(flow_pred[i, :, :, mid_d], cmap='seismic', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Predicted Flow {components[i]}')
        axes[i, 0].axis('off')
        plt.colorbar(im_pred, ax=axes[i, 0])

        # 真实
        im_gt = axes[i, 1].imshow(flow_gt[i, :, :, mid_d], cmap='seismic', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Ground Truth Flow {components[i]}')
        axes[i, 1].axis('off')
        plt.colorbar(im_gt, ax=axes[i, 1])

    plt.tight_layout()
    plt.savefig(output_dir / '3_flow_components.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 4. 误差分析 ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')

    # EPE分布
    im_epe = axes[0, 0].imshow(epe[:, :, mid_d], cmap='hot')
    axes[0, 0].set_title('End-Point Error (EPE) - XY Slice')
    axes[0, 0].axis('off')
    plt.colorbar(im_epe, ax=axes[0, 0])

    # EPE直方图
    axes[0, 1].hist(epe.flatten(), bins=50, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('EPE Distribution')
    axes[0, 1].set_xlabel('Error (voxels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # 分量误差直方图
    for i, comp in enumerate(['X', 'Y', 'Z']):
        axes[1, 0].hist(flow_error[i].flatten(), bins=50, alpha=0.5, label=f'{comp} Error')
    axes[1, 0].set_title('Component-wise Error Distribution')
    axes[1, 0].set_xlabel('Error (voxels)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 统计文本
    stats_text = (
        f"Flow Statistics:\n"
        f"{'='*40}\n"
        f"Mean EPE: {epe.mean():.4f} voxels\n"
        f"Median EPE: {np.median(epe):.4f} voxels\n"
        f"Max EPE: {epe.max():.4f} voxels\n"
        f"\n"
        f"Predicted Flow Magnitude:\n"
        f"  Mean: {flow_pred_mag.mean():.4f}\n"
        f"  Max: {flow_pred_mag.max():.4f}\n"
        f"\n"
        f"Ground Truth Flow Magnitude:\n"
        f"  Mean: {flow_gt_mag.mean():.4f}\n"
        f"  Max: {flow_gt_mag.max():.4f}\n"
        f"\n"
        f"Component-wise MAE:\n"
        f"  X: {np.abs(flow_error[0]).mean():.4f}\n"
        f"  Y: {np.abs(flow_error[1]).mean():.4f}\n"
        f"  Z: {np.abs(flow_error[2]).mean():.4f}\n"
    )

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / '4_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 保存统计信息到文本文件
    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write(stats_text)

    print(f"\n✓ 可视化结果已保存到: {output_dir}")
    print(stats_text)


def main():
    print("="*60)
    print("RAFT-DVC 推理测试")
    print("="*60)
    print()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    checkpoint_path = "results/test/best_model.pth"
    print(f"加载模型: {checkpoint_path}")
    model, train_info = RAFTDVC.load_checkpoint(checkpoint_path, device=device)
    model.eval()
    print(f"模型参数量: {model.get_num_parameters():,}")
    if train_info:
        epoch = train_info.get('epoch', 'N/A')
        val_loss = train_info.get('val_loss', None)
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else 'N/A'
        print(f"训练信息: epoch={epoch}, val_loss={val_loss_str}")
    print()

    # 加载测试数据
    print("加载测试数据...")
    vol0, vol1, flow_gt = load_test_data()
    print(f"Volume形状: {vol0.shape}")
    print(f"Flow GT形状: {flow_gt.shape}")
    print()

    # 运行推理
    print("运行推理...")
    iters = 12  # 使用更多迭代次数获得更好的结果
    flow_pred = run_inference(model, vol0, vol1, device, iters=iters)
    print(f"预测Flow形状: {flow_pred.shape}")
    print()

    # 可视化
    output_dir = Path("results/test/inference_visualization")
    print("生成可视化...")
    visualize_flow(vol0, vol1, flow_pred, flow_gt, output_dir)

    # 保存预测结果
    np.save(output_dir / 'flow_predicted.npy', flow_pred)
    print(f"\n✓ 预测flow已保存到: {output_dir / 'flow_predicted.npy'}")

    print()
    print("="*60)
    print("推理测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()
