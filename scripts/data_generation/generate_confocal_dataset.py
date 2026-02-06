"""
Main script to generate synthetic confocal particle dataset.

Usage:
    python scripts/data_generation/generate_confocal_dataset.py
    python scripts/data_generation/generate_confocal_dataset.py --config configs/data_generation/confocal_particles_small.yaml
"""

import os
import sys
from pathlib import Path
import time
import json
import yaml
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import generation modules
from scripts.data_generation.modules.deformation import DeformationGenerator, compute_flow_statistics
from scripts.data_generation.modules.beads import BeadGenerator, BeadRenderer, compute_bead_statistics
from scripts.data_generation.modules.warping import ForwardWarper, compute_warp_statistics
from scripts.data_generation.modules.imaging import ImagingSimulator, compute_imaging_statistics
from scripts.data_generation.modules.quality_control import QualityChecker


class DatasetGenerator:
    """Main dataset generation orchestrator."""

    def __init__(self, config_path):
        """
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.volume_shape = tuple(self.config['dataset']['volume_shape'])

        # Initialize modules
        self.deform_gen = DeformationGenerator(self.volume_shape,
                                               self.config['deformation'])
        self.bead_gen = BeadGenerator(self.volume_shape,
                                      self.config['particles'])
        self.renderer = BeadRenderer(self.volume_shape,
                                     self.config['particles'])
        self.warper = ForwardWarper(self.volume_shape)
        self.imaging = ImagingSimulator(self.config['imaging'])
        self.qc = QualityChecker(self.config['generation']['quality_check'])

        # Set random seed
        np.random.seed(self.config['dataset']['random_seed'])

    def generate_single_sample(self, sample_idx, split='train'):
        """
        Generate a single training sample.

        Args:
            sample_idx: Sample index
            split: 'train', 'val', or 'test'

        Returns:
            sample_data: dict with vol0, vol1, flow, metadata
            passed: bool - whether sample passed quality check
        """
        # 1. Generate deformation field
        flow, deform_meta = self.deform_gen.generate()

        # 2. Generate beads for vol0
        beads_vol0 = self.bead_gen.generate_beads()

        # 3. Render vol0 (clean)
        vol0_clean = self.renderer.render(beads_vol0)

        # 4. Warp beads to create vol1
        beads_vol1, num_visible = self.warper.warp_beads(beads_vol0, flow)

        # 5. Render vol1 (clean)
        vol1_clean = self.renderer.render(beads_vol1)

        # 6. Apply imaging effects
        vol0, imaging_meta0 = self.imaging.apply_imaging_effects(vol0_clean,
                                                                  apply_photobleaching=False)
        vol1, imaging_meta1 = self.imaging.apply_imaging_effects(vol1_clean,
                                                                  apply_photobleaching=True)

        # 7. Compute statistics
        flow_stats = compute_flow_statistics(flow)
        bead_stats = compute_bead_statistics(beads_vol0)
        warp_stats = compute_warp_statistics(beads_vol0, beads_vol1)

        # 8. Compile metadata
        metadata = {
            'sample_idx': sample_idx,
            'split': split,
            'deform_type': deform_meta['type'],
            'deform_params': deform_meta,
            'bead_stats': bead_stats,
            'warp_stats': warp_stats,
            'flow_stats': flow_stats,
            'imaging_vol0': imaging_meta0,
            'imaging_vol1': imaging_meta1
        }

        # 9. Prepare sample data
        sample_data = {
            'vol0': vol0,
            'vol1': vol1,
            'flow': flow,
            'metadata': metadata
        }

        # 10. Quality check
        passed, issues = self.qc.check_sample(sample_data)

        if not passed:
            print(f"Warning: Sample {sample_idx} failed quality check: {issues}")

        return sample_data, passed

    def save_sample(self, sample_data, output_dir, sample_idx):
        """
        Save a generated sample to disk.

        Args:
            sample_data: dict with vol0, vol1, flow, metadata
            output_dir: Output directory for this split
            sample_idx: Sample index
        """
        # Save volumes and flow
        np.save(output_dir / 'vol0' / f'sample_{sample_idx:04d}.npy',
                sample_data['vol0'])
        np.save(output_dir / 'vol1' / f'sample_{sample_idx:04d}.npy',
                sample_data['vol1'])
        np.save(output_dir / 'flow' / f'sample_{sample_idx:04d}.npy',
                sample_data['flow'])

        # Save metadata
        if self.config['generation']['save_metadata']:
            metadata_dir = output_dir / 'metadata'
            metadata_dir.mkdir(exist_ok=True)

            with open(metadata_dir / f'sample_{sample_idx:04d}.json', 'w') as f:
                json.dump(sample_data['metadata'], f, indent=2)

    def generate_split(self, split='train'):
        """
        Generate all samples for a split (train/val/test).

        Args:
            split: 'train', 'val', or 'test'
        """
        num_samples = self.config['dataset']['num_samples'][split]
        output_dir = Path(self.config['dataset']['output_dir']) / split

        # Create directories
        for subdir in ['vol0', 'vol1', 'flow']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {split} set ({num_samples} samples)...")

        # Generate samples
        passed_count = 0
        failed_count = 0

        if self.config['generation']['parallel']:
            # Parallel generation
            num_workers = min(self.config['generation']['num_workers'],
                            cpu_count())
            print(f"Using {num_workers} parallel workers")

            # Use a wrapper function for multiprocessing
            def generate_wrapper(idx):
                return self.generate_single_sample(idx, split)

            with Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(generate_wrapper, range(num_samples)),
                    total=num_samples,
                    desc=f"Generating {split}"
                ))

            # Save all samples
            for idx, (sample_data, passed) in enumerate(results):
                self.save_sample(sample_data, output_dir, idx)

                if passed:
                    passed_count += 1
                else:
                    failed_count += 1

                # Visualize first few samples
                if (self.config['generation']['visualize_samples'] and
                    idx < self.config['generation']['num_visualize']):
                    vis_dir = output_dir / 'visualizations'
                    vis_dir.mkdir(exist_ok=True)
                    self.qc.visualize_sample(
                        sample_data,
                        vis_dir / f'sample_{idx:04d}.png',
                        idx
                    )

        else:
            # Sequential generation (easier for debugging)
            for idx in tqdm(range(num_samples), desc=f"Generating {split}"):
                sample_data, passed = self.generate_single_sample(idx, split)

                self.save_sample(sample_data, output_dir, idx)

                if passed:
                    passed_count += 1
                else:
                    failed_count += 1

                # Visualize first few samples
                if (self.config['generation']['visualize_samples'] and
                    idx < self.config['generation']['num_visualize']):
                    vis_dir = output_dir / 'visualizations'
                    vis_dir.mkdir(exist_ok=True)
                    self.qc.visualize_sample(
                        sample_data,
                        vis_dir / f'sample_{idx:04d}.png',
                        idx
                    )

        print(f"✓ {split} set complete: {passed_count} passed, {failed_count} failed")

    def generate_all(self):
        """Generate complete dataset (train + val + test)."""
        start_time = time.time()

        print("="*70)
        print("RAFT-DVC Synthetic Confocal Dataset Generation")
        print("="*70)

        # Generate each split
        for split in ['train', 'val', 'test']:
            self.generate_split(split)

        # Save dataset metadata
        output_dir = Path(self.config['dataset']['output_dir'])
        metadata = {
            'version': '1.0',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'generation_time_seconds': time.time() - start_time
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save config for reproducibility
        with open(output_dir / 'generation_config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ Dataset generation complete!")
        print(f"  Total time: {elapsed/60:.1f} minutes")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic confocal particle dataset for RAFT-DVC'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_generation/confocal_particles.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    # Configuration file path
    config_path = project_root / args.config

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(f"Available configs:")
        config_dir = project_root / 'configs' / 'data_generation'
        for cfg in sorted(config_dir.glob('*.yaml')):
            print(f"  - {cfg.relative_to(project_root)}")
        return

    print(f"Using configuration: {config_path.relative_to(project_root)}")

    # Create generator and run
    generator = DatasetGenerator(config_path)
    generator.generate_all()

    print("\n✓ All done! You can now train on the generated dataset.")
    print("\nNext steps:")
    print("  1. Inspect visualizations in the output directory")
    print("  2. Update training config to point to this dataset")
    print("  3. Run training: run.bat -> Train Custom Model")


if __name__ == '__main__':
    main()
