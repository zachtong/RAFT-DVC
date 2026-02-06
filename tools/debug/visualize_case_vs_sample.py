"""
Visualize the relationship between CASE and SAMPLE data.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_case_sample_relationship():
    """Create visualization showing case-to-sample extraction."""

    # Load case data
    case_dir = Path('data/Lehu\'s_data/VolRAFT_Synthetic_Dataset/case_0000')
    v0_case = np.load(case_dir / 'v0_256.npy')
    centers = np.load(case_dir / 'centers.npy')

    # Load sample data
    sample_dir = Path('data/Lehu\'s_data/VolRAFT_Synthetic_Dataset/samples/sample_0000000')
    v0_sample = np.load(sample_dir / 'v0.npy')[0]
    with open(sample_dir / 'meta.json', 'r') as f:
        meta = json.load(f)

    # Get patch info
    ox, oy, oz = meta['patch_origin_xyz']
    ps = meta['patch_size']

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Case volume with patch location highlighted (3D view)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # Draw case volume boundary
    case_size = v0_case.shape[0]
    ax1.plot([0, case_size], [0, 0], [0, 0], 'k-', linewidth=2)
    ax1.plot([0, case_size], [case_size, case_size], [0, 0], 'k-', linewidth=2)
    ax1.plot([0, case_size], [0, 0], [case_size, case_size], 'k-', linewidth=2)
    ax1.plot([0, 0], [0, case_size], [0, 0], 'k-', linewidth=2)
    ax1.plot([case_size, case_size], [0, case_size], [0, 0], 'k-', linewidth=2)
    ax1.plot([0, 0], [0, 0], [0, case_size], 'k-', linewidth=2)

    # Draw sample patch boundary (highlighted)
    corners_x = [ox, ox+ps, ox+ps, ox, ox, ox+ps, ox+ps, ox]
    corners_y = [oy, oy, oy+ps, oy+ps, oy, oy, oy+ps, oy+ps]
    corners_z = [oz, oz, oz, oz, oz+ps, oz+ps, oz+ps, oz+ps]

    for i in range(4):
        ax1.plot([corners_x[i], corners_x[(i+1)%4]],
                [corners_y[i], corners_y[(i+1)%4]],
                [corners_z[i], corners_z[(i+1)%4]], 'r-', linewidth=3)
        ax1.plot([corners_x[i+4], corners_x[(i+1)%4+4]],
                [corners_y[i+4], corners_y[(i+1)%4+4]],
                [corners_z[i+4], corners_z[(i+1)%4+4]], 'r-', linewidth=3)
        ax1.plot([corners_x[i], corners_x[i+4]],
                [corners_y[i], corners_y[i+4]],
                [corners_z[i], corners_z[i+4]], 'r-', linewidth=3)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'CASE Volume ({case_size}³)\nwith SAMPLE patch ({ps}³) highlighted',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, case_size)
    ax1.set_ylim(0, case_size)
    ax1.set_zlim(0, case_size)

    # Subplot 2: Case middle slice
    ax2 = fig.add_subplot(2, 3, 2)
    mid = case_size // 2
    ax2.imshow(v0_case[mid], cmap='gray')

    # Highlight sample region if it intersects this slice
    if oz <= mid < oz + ps:
        from matplotlib.patches import Rectangle
        rect = Rectangle((oy, ox), ps, ps, linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(oy + ps/2, ox - 2, 'Sample Region', color='r',
                ha='center', fontsize=10, fontweight='bold')

    ax2.set_title(f'CASE Z-slice (z={mid})', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Subplot 3: Sample middle slice
    ax3 = fig.add_subplot(2, 3, 3)
    sample_mid = ps // 2
    ax3.imshow(v0_sample[sample_mid], cmap='gray')
    ax3.set_title(f'SAMPLE Z-slice (z={sample_mid})', fontsize=12, fontweight='bold', color='red')
    ax3.axis('off')

    # Subplot 4: Particle distribution in case
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(centers[:, 0], centers[:, 1], s=0.1, alpha=0.5, c='blue')

    # Highlight sample region
    from matplotlib.patches import Rectangle
    rect = Rectangle((oy, ox), ps, ps, linewidth=2, edgecolor='r', facecolor='red', alpha=0.2)
    ax4.add_patch(rect)

    ax4.set_xlim(0, case_size)
    ax4.set_ylim(0, case_size)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('X')
    ax4.set_title(f'Particle Distribution (XY projection)\n{centers.shape[0]} beads',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Text explanation
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')

    explanation = f"""
CASE vs SAMPLE Explanation

CASE (case_0000/):
• Full synthetic volume: {case_size}³ voxels
• Contains {meta['n_beads_case']} randomly positioned beads
• Files:
  - v0_256.npy: Reference volume
  - flow_256.npy: Ground truth flow field
  - centers.npy: Bead positions (X,Y,Z)
• Represents one physical simulation

SAMPLE (sample_0000000/):
• Cropped patch: {ps}³ voxels
• Extracted from CASE at position [{ox}, {oy}, {oz}]
• Files:
  - v0.npy: Reference volume (cropped)
  - v1.npy: Deformed volume (warped by flow)
  - flow.npy: Ground truth flow (cropped)
  - meta.json: Metadata linking to case
• Used for training/testing

Relationship:
Sample = CASE[{ox}:{ox+ps}, {oy}:{oy+ps}, {oz}:{oz+ps}]

Multiple samples can be extracted from
one case at different positions.
"""

    ax5.text(0.05, 0.95, explanation, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Subplot 6: Dataset statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    with open('data/Lehu\'s_data/VolRAFT_Synthetic_Dataset/dataset_index.json', 'r') as f:
        dataset_index = json.load(f)

    case_counts = {}
    for sample in dataset_index:
        case_id = sample['case_id']
        case_counts[case_id] = case_counts.get(case_id, 0) + 1

    stats = f"""
Dataset Statistics

Total Samples: {len(dataset_index)}
Total Cases: {len(case_counts)}

Samples per case:
"""
    for case_id, count in sorted(case_counts.items()):
        stats += f"  Case {case_id}: {count} sample(s)\n"

    stats += f"""
Current sample metadata:
  global_sample_id: {meta['global_sample_id']}
  case_id: {meta['case_id']}
  max_disp_case: {meta['max_disp_case']:.4f} voxels
  patch_size: {meta['patch_size']}³
  patch_origin: [{ox}, {oy}, {oz}]
"""

    ax6.text(0.05, 0.95, stats, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    output_path = 'case_vs_sample_explanation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    visualize_case_sample_relationship()
