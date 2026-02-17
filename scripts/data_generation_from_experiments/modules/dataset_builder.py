"""
Dataset builder - applies deformations and saves final dataset.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

# Import existing deformation and quality control modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_generation.modules.deformation import DeformationGenerator, compute_flow_statistics
from scripts.data_generation.modules.quality_control import QualityChecker


class DatasetBuilder:
    """Build final dataset with deformations."""

    def __init__(self, config: dict):
        """
        Args:
            config: Full configuration dict
        """
        self.config = config
        self.volume_size = tuple(config['extraction']['volume_size'])

        # Initialize deformation generator
        self.deform_gen = DeformationGenerator(
            self.volume_size,
            config['deformations']
        )

        # Initialize quality checker
        qc_config = config.get('quality_check', {
            'min_visible_beads_ratio': 0.0,  # Disable for real data
            'max_displacement_ratio': 0.5
        })
        self.qc = QualityChecker(qc_config)

        # Output settings
        self.output_config = config['output']
        self.base_dir = Path(self.output_config['base_dir'])
        self.save_metadata = self.output_config.get('save_metadata', True)
        self.save_visualizations = self.output_config.get('save_visualizations', True)
        self.num_visualize = self.output_config.get('num_visualize', 5)

        # Random seed
        np.random.seed(config.get('random_seed', 42))

    def build_dataset(
        self,
        volumes: List[np.ndarray],
        extraction_metadata: List[Dict],
        split_allocation: Dict[str, int]
    ):
        """
        Build complete dataset with deformations.

        Args:
            volumes: List of preprocessed vol0 volumes
            extraction_metadata: Metadata from extraction (source file, position, etc.)
            split_allocation: Dict mapping split name to number of samples
                             e.g., {'train': 0, 'val': 0, 'test': 50}
        """
        print("\n" + "=" * 70)
        print("Building Dataset from Experimental Data")
        print("=" * 70)

        # Allocate volumes to splits
        split_volumes = self._allocate_to_splits(
            volumes,
            extraction_metadata,
            split_allocation
        )

        # Generate each split
        for split_name, split_data in split_volumes.items():
            if split_data['num_samples'] > 0:
                self._generate_split(
                    split_name,
                    split_data['volumes'],
                    split_data['metadata']
                )

        print("\n" + "=" * 70)
        print("✓ Dataset generation complete!")
        print("=" * 70)

    def _allocate_to_splits(
        self,
        volumes: List[np.ndarray],
        extraction_metadata: List[Dict],
        split_allocation: Dict[str, int]
    ) -> Dict:
        """
        Allocate volumes to train/val/test splits.

        Args:
            volumes: All extracted volumes
            extraction_metadata: Metadata for each volume
            split_allocation: Desired number of samples per split

        Returns:
            Dict with split data
        """
        # Check total
        total_requested = sum(split_allocation.values())
        total_available = len(volumes)

        if total_requested > total_available:
            raise ValueError(
                f"Requested {total_requested} samples but only "
                f"{total_available} volumes extracted"
            )

        print(f"\nAllocating {total_requested} samples to splits:")
        for split, count in split_allocation.items():
            print(f"  {split}: {count}")

        # Shuffle volumes
        indices = np.random.permutation(len(volumes))

        # Allocate
        split_volumes = {}
        idx = 0

        for split_name in ['train', 'val', 'test']:
            num_samples = split_allocation.get(split_name, 0)

            split_indices = indices[idx:idx + num_samples]
            split_volumes[split_name] = {
                'num_samples': num_samples,
                'volumes': [volumes[i] for i in split_indices],
                'metadata': [extraction_metadata[i] for i in split_indices]
            }

            idx += num_samples

        return split_volumes

    def _generate_split(
        self,
        split: str,
        vol0_volumes: List[np.ndarray],
        extraction_metadata: List[Dict]
    ):
        """
        Generate a split (train/val/test) with deformations.

        Args:
            split: Split name ('train', 'val', or 'test')
            vol0_volumes: List of vol0 volumes
            extraction_metadata: Extraction metadata for each volume
        """
        num_samples = len(vol0_volumes)
        output_dir = self.base_dir / split

        # Create directories
        for subdir in ['vol0', 'vol1', 'flow', 'metadata']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        if self.save_visualizations:
            (output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {split} set ({num_samples} samples)...")

        # Generate samples
        #
        # Warping strategy (backward warp + swap):
        #   1. Generate flow field F, treat it as backward flow
        #   2. Backward warp: vol_warped[q] = vol_original[q - F(q)]
        #      (exact, using grid_sample with bilinear interpolation)
        #   3. Swap roles: vol0 = vol_warped, vol1 = vol_original
        #   4. Forward flow from vol0 to vol1 = -F
        #      (defined at vol0 positions, mathematically exact)
        #
        # This avoids the error of naively using forward flow as backward flow,
        # and avoids the need for iterative flow inversion.
        #
        for idx in tqdm(range(num_samples), desc=f"Generating {split}"):
            vol_original = vol0_volumes[idx]
            extract_meta = extraction_metadata[idx]

            # Generate deformation field (treated as backward flow)
            backward_flow, deform_meta = self._generate_deformation()

            # Backward warp: vol_warped[q] = vol_original[q - F(q)]
            vol_warped = self._backward_warp(vol_original, backward_flow)

            # Swap: vol0 = warped (reference), vol1 = original (deformed)
            # Forward flow from vol0 to vol1 = -backward_flow
            vol0 = vol_warped
            vol1 = vol_original
            forward_flow = -backward_flow

            # Compute statistics on the saved forward flow
            flow_stats = compute_flow_statistics(forward_flow)

            # Compile metadata
            metadata = {
                'sample_idx': idx,
                'split': split,
                'source': 'experimental',
                'extraction': extract_meta,
                'deform_type': deform_meta['type'],
                'deform_params': deform_meta,
                'flow_stats': flow_stats,
                'volume_stats': {
                    'vol0_mean': float(np.mean(vol0)),
                    'vol0_std': float(np.std(vol0)),
                    'vol0_min': float(np.min(vol0)),
                    'vol0_max': float(np.max(vol0)),
                    'vol1_mean': float(np.mean(vol1)),
                    'vol1_std': float(np.std(vol1)),
                }
            }

            # Save data (vol0 --apply flow--> vol1)
            self._save_sample(
                output_dir,
                idx,
                vol0,
                vol1,
                forward_flow,
                metadata,
                visualize=(idx < self.num_visualize)
            )

        print(f"✓ {split} set complete: {num_samples} samples")

    def _generate_deformation(self) -> Tuple[np.ndarray, Dict]:
        """
        Generate a deformation field.

        Returns:
            flow: Deformation field (3, D, H, W)
            metadata: Deformation metadata
        """
        return self.deform_gen.generate()

    def _backward_warp(self, volume: np.ndarray, backward_flow: np.ndarray) -> np.ndarray:
        """
        Backward warp a volume using a backward flow field.

        For each output position q, sample from input at position (q - flow(q)):
            output[q] = input[q - backward_flow(q)]

        This is exact (no iterative approximation needed) because the flow
        is defined at the output grid positions.

        Args:
            volume: Input volume (D, H, W)
            backward_flow: Backward flow field (3, D, H, W), defined at output positions

        Returns:
            warped: Warped volume (D, H, W)
        """
        import torch
        import torch.nn.functional as F

        vol_torch = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        flow_torch = torch.from_numpy(backward_flow).float()

        D, H, W = volume.shape
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.arange(D, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )

        # Sample from (q - flow(q))
        sample_z = grid_z - flow_torch[0]
        sample_y = grid_y - flow_torch[1]
        sample_x = grid_x - flow_torch[2]

        # Normalize to [-1, 1] for grid_sample
        sample_z = 2.0 * sample_z / (D - 1) - 1.0
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        sample_x = 2.0 * sample_x / (W - 1) - 1.0

        # grid_sample expects (1, D, H, W, 3) in (x, y, z) order
        grid = torch.stack([sample_x, sample_y, sample_z], dim=-1).unsqueeze(0)

        warped_torch = F.grid_sample(
            vol_torch, grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )

        return warped_torch[0, 0].numpy()

    def _save_sample(
        self,
        output_dir: Path,
        sample_idx: int,
        vol0: np.ndarray,
        vol1: np.ndarray,
        flow: np.ndarray,
        metadata: Dict,
        visualize: bool = False
    ):
        """
        Save a generated sample to disk.

        Args:
            output_dir: Output directory for this split
            sample_idx: Sample index
            vol0: Original volume
            vol1: Warped volume
            flow: Flow field
            metadata: Sample metadata
            visualize: Whether to create visualization
        """
        # Save volumes and flow
        np.save(
            output_dir / 'vol0' / f'sample_{sample_idx:04d}.npy',
            vol0
        )
        np.save(
            output_dir / 'vol1' / f'sample_{sample_idx:04d}.npy',
            vol1
        )
        np.save(
            output_dir / 'flow' / f'sample_{sample_idx:04d}.npy',
            flow
        )

        # Save metadata (JSON format, same as synthetic data)
        if self.save_metadata:
            metadata_path = output_dir / 'metadata' / f'sample_{sample_idx:04d}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Create visualization
        if visualize and self.save_visualizations:
            sample_data = {
                'vol0': vol0,
                'vol1': vol1,
                'flow': flow,
                'metadata': metadata
            }
            vis_path = output_dir / 'visualizations' / f'sample_{sample_idx:04d}.png'
            self.qc.visualize_sample(sample_data, vis_path, sample_idx)
