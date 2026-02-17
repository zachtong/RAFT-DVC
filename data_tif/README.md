# Raw Experimental TIF Files Directory

This directory stores raw experimental TIF/TIFF files for dataset generation.

## Structure

```
data_tif/
├── experiment_name_1/
│   ├── scan_001.tif
│   ├── scan_002.tif
│   └── ...
├── experiment_name_2/
│   └── ...
└── README.md (this file)
```

## Usage

1. **Create experiment directory**:
   ```bash
   mkdir -p data_tif/your_experiment_name
   ```

2. **Add TIF files**:
   - Copy your experimental TIF files to the directory
   - Files can have any naming convention
   - Supported: `.tif`, `.tiff`
   - Minimum size: 128×128×128 voxels

3. **Create config**:
   - Copy example config from `configs/data_generation_from_experiments/`
   - Set `raw_data_dir: "data_tif/your_experiment_name"`

4. **Generate dataset**:
   ```bash
   run.bat
   → 3. Dataset Generation
   → 2. Generate Dataset from Experimental TIF Images
   ```

## Example

```bash
# 1. Create directory for Jin Yang's experiment
mkdir -p data_tif/JinYang_confocal_beads_indentation

# 2. Copy TIF files
cp /path/to/experimental/data/*.tif data_tif/JinYang_confocal_beads_indentation/

# 3. Config already exists at:
# configs/data_generation_from_experiments/JinYang_confocal_beads_indentation.yaml

# 4. Generate using run.bat or:
python scripts/data_generation_from_experiments/generate_from_tif.py \
    --config configs/data_generation_from_experiments/JinYang_confocal_beads_indentation.yaml
```

## Notes

- **Storage**: TIF files stay here, processed datasets go to `data/experiment_name/`
- **Size**: Large TIF files (>1GB) are supported via memory-mapped loading
- **Reuse**: Same raw data can be used with multiple configs (different deformation parameters)
- **Version control**: Add `data_tif/` to `.gitignore` if files are very large

## See Also

- Full documentation: `docs/EXPERIMENTAL_DATA_GENERATION.md`
- Example config: `configs/data_generation_from_experiments/JinYang_confocal_beads_indentation.yaml`
