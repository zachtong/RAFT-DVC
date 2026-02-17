"""
Generate dataset from experimental TIF images.

Usage:
    python scripts/data_generation_from_experiments/generate_from_tif.py --config configs/data_generation_from_experiments/experiment_name.yaml
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from scripts.data_generation_from_experiments.modules import (
    TifLoader,
    VolumeExtractor,
    Preprocessor,
    DatasetBuilder
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate dataset from experimental TIF images'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("RAFT-DVC Experimental Data Generation")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Dataset: {config['dataset_name']}")
    print()

    start_time = time.time()

    try:
        # Step 1: Load TIF files
        print("Step 1: Loading TIF files...")
        print("-" * 70)
        tif_loader = TifLoader(config)
        summary = tif_loader.get_summary()

        print(f"Total files: {summary['num_total_files']}")
        print(f"Valid files: {summary['num_valid_files']}")
        print(f"Shape range: {summary['min_shape']} - {summary['max_shape']}")
        print(f"Mean shape: {summary['mean_shape']}")
        print()

        # Load valid files into memory
        valid_files = tif_loader.filter_valid_files()
        print(f"Loading {len(valid_files)} TIF files into memory...")
        tif_volumes = []
        tif_file_names = []

        for file_path in valid_files:
            vol = tif_loader.load_file(file_path)
            tif_volumes.append(vol)
            tif_file_names.append(file_path.name)
            print(f"  Loaded: {file_path.name} - shape {vol.shape}")

        print(f"✓ Loaded {len(tif_volumes)} volumes")
        print()

        # Step 2: Extract volumes
        print("Step 2: Extracting sub-volumes...")
        print("-" * 70)
        extractor = VolumeExtractor(config)
        extracted_volumes, extraction_metadata = extractor.extract_volumes(
            tif_volumes,
            tif_file_names
        )
        print()

        # Step 3: Preprocess volumes
        print("Step 3: Preprocessing volumes...")
        print("-" * 70)
        preprocessor = Preprocessor(config)
        processed_volumes = preprocessor.process_batch(extracted_volumes)
        print(f"✓ Preprocessed {len(processed_volumes)} volumes")
        print()

        # Step 4: Build dataset with deformations
        print("Step 4: Building dataset with deformations...")
        print("-" * 70)
        builder = DatasetBuilder(config)

        # Get split allocation from config
        splits = config['splits']
        split_allocation = {
            'train': splits['train']['num_samples'],
            'val': splits['val']['num_samples'],
            'test': splits['test']['num_samples']
        }

        builder.build_dataset(
            processed_volumes,
            extraction_metadata,
            split_allocation
        )

        # Save generation summary
        output_dir = Path(config['output']['base_dir'])
        summary_path = output_dir / 'generation_summary.yaml'

        generation_summary = {
            'dataset_name': config['dataset_name'],
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': str(config_path),
            'source_files': summary['valid_file_names'],
            'num_source_files': summary['num_valid_files'],
            'num_extracted_volumes': len(extracted_volumes),
            'splits': split_allocation,
            'generation_time_seconds': time.time() - start_time
        }

        with open(summary_path, 'w') as f:
            yaml.dump(generation_summary, f, default_flow_style=False)

        # Save config for reproducibility
        config_copy_path = output_dir / 'generation_config.yaml'
        with open(config_copy_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("✓ Dataset generation complete!")
        print(f"  Total time: {elapsed / 60:.1f} minutes")
        print(f"  Output: {output_dir}")
        print(f"  Summary: {summary_path}")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Inspect visualizations in the output directory")
        print("  2. Run inference using run.bat")
        print(f"     - Select dataset: {config['dataset_name']}")
        print("     - Choose appropriate checkpoint")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR during generation:")
        print(str(e))
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
