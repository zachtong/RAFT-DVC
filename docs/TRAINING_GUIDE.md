# RAFT-DVC Training and Testing Guide

Complete guide for training and testing RAFT-DVC on the synthetic confocal dataset.

---

## Quick Start

### 1. Generate Dataset (if not done yet)

```cmd
generate_dataset.bat
```

**Output:** `data/synthetic_confocal/` (800 train + 100 val + 100 test samples)

**Time:** ~1-1.5 hours

---

### 2. Train Model

**Option A: Using batch file (easiest)**
```cmd
train_confocal.bat
```

**Option B: Command line**
```cmd
conda activate raft-dvc
python scripts/train_confocal.py
```

**What happens:**
- Loads configuration from `configs/training/confocal_baseline.yaml`
- Trains for 100 epochs with validation every 5 epochs
- Saves checkpoints to `results/confocal_baseline/`
- Creates TensorBoard logs in `results/confocal_baseline/logs/`

**Estimated time:** 8-12 hours on RTX 5090 (depends on GPU)

**Monitor progress:**
```cmd
# In another terminal
conda activate raft-dvc
tensorboard --logdir results/confocal_baseline/logs
```
Then open browser at `http://localhost:6006`

---

### 3. Test Model

**After training completes:**

**Option A: Using batch file**
```cmd
test_confocal.bat
```

**Option B: Command line**
```cmd
python scripts/test_confocal.py --checkpoint results/confocal_baseline/checkpoint_best.pth --save_visualizations --save_predictions
```

**Output:**
- `results/confocal_baseline/test_results/test_results.json` - Metrics (EPE, accuracy, etc.)
- `results/confocal_baseline/test_results/visualizations/` - PNG visualizations
- `results/confocal_baseline/test_results/predictions/` - Predicted flow fields (.npy)

---

## Configuration Details

### Training Configuration (`configs/training/confocal_baseline.yaml`)

Key parameters you can modify:

**Data Settings:**
```yaml
data:
  augment: true  # Random flips/rotations
  patch_size: null  # null=full volume, [64,64,64]=patches
```

**Model Architecture:**
```yaml
model:
  feature_dim: 128  # Encoder feature dimension
  hidden_dim: 96    # GRU hidden dimension
  iters: 12         # Refinement iterations
```

**Training Settings:**
```yaml
training:
  epochs: 100
  batch_size: 2     # Reduce if OOM (Out of Memory)
  lr: 0.0001
  use_amp: true     # Mixed precision (saves memory)
```

---

## Expected Results

### Baseline Performance (after 100 epochs)

Based on similar synthetic datasets:

- **EPE (End-Point Error):** ~0.5-1.0 voxels
- **1px accuracy:** >80% (percentage of voxels with EPE < 1)
- **3px accuracy:** >95%

**Note:** Exact numbers depend on:
1. Deformation complexity in your generated data
2. Bead density
3. Noise level

---

## Training Tips

### If Training is Too Slow

**Option 1: Reduce data size**
```yaml
# In configs/training/confocal_baseline.yaml
training:
  batch_size: 1  # Reduce from 2 to 1
```

**Option 2: Use patches instead of full volumes**
```yaml
data:
  patch_size: [64, 64, 64]  # Train on patches
```
This will be 8x faster but may reduce accuracy slightly.

**Option 3: Reduce model size**
```yaml
model:
  feature_dim: 64   # Reduce from 128
  hidden_dim: 48    # Reduce from 96
```

---

### If Out of Memory (OOM)

1. **Enable mixed precision** (should already be on):
   ```yaml
   training:
     use_amp: true
   ```

2. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 1
   ```

3. **Use patches**:
   ```yaml
   data:
     patch_size: [64, 64, 64]
   ```

4. **Reduce model size** (see above)

---

## Resume Training

If training is interrupted:

```cmd
python scripts/train_confocal.py --resume results/confocal_baseline/checkpoint_epoch_050.pth
```

Or modify config:
```yaml
checkpoint:
  resume: "results/confocal_baseline/checkpoint_epoch_050.pth"
```

---

## Advanced Usage

### Train with Different Configuration

1. Copy baseline config:
   ```cmd
   copy configs\training\confocal_baseline.yaml configs\training\my_experiment.yaml
   ```

2. Modify parameters in `my_experiment.yaml`

3. Train with new config:
   ```cmd
   python scripts/train_confocal.py --config configs/training/my_experiment.yaml
   ```

---

### Test with More Iterations

More iterations = better accuracy but slower:

```cmd
python scripts/test_confocal.py --checkpoint results/confocal_baseline/checkpoint_best.pth --iters 32
```

Default is 24 iterations. Try 32 or 48 for better results.

---

### Visualize Specific Samples

```cmd
python tools/visualize_confocal.py --dataset data/synthetic_confocal --split test --sample 5
```

Compare with prediction:
```cmd
python scripts/test_confocal.py --checkpoint results/confocal_baseline/checkpoint_best.pth --save_visualizations
```

Then check `results/confocal_baseline/test_results/visualizations/`

---

## Understanding Metrics

### EPE (End-Point Error)
- Average Euclidean distance between predicted and ground truth flow
- Lower is better
- **Good:** < 1.0 voxel
- **Acceptable:** 1-3 voxels
- **Poor:** > 5 voxels

### Accuracy (1px, 3px, 5px)
- Percentage of voxels with EPE below threshold
- Higher is better
- **Good 1px accuracy:** > 80%
- **Good 3px accuracy:** > 95%

---

## Troubleshooting

### Issue: NaN loss during training

**Solution:**
1. Reduce learning rate:
   ```yaml
   training:
     lr: 0.00005  # Half of default
   ```

2. Check data:
   ```cmd
   python tools/visualize_confocal.py --random 5
   ```
   Make sure flow fields look reasonable.

---

### Issue: No improvement after many epochs

**Possible causes:**
1. Learning rate too low → Increase `max_lr`
2. Model too small → Increase `feature_dim` and `hidden_dim`
3. Data too noisy → Regenerate with higher SNR

---

### Issue: Training works but test EPE is high

This means **overfitting** to training data.

**Solutions:**
1. More data augmentation
2. Add regularization (increase `weight_decay`)
3. Early stopping (use checkpoint from earlier epoch)

---

## File Structure After Training

```
RAFT-DVC/
├── data/
│   └── synthetic_confocal/
│       ├── train/  (800 samples)
│       ├── val/    (100 samples)
│       └── test/   (100 samples)
│
├── results/
│   └── confocal_baseline/
│       ├── checkpoint_best.pth        # Best model (lowest val EPE)
│       ├── checkpoint_epoch_100.pth   # Final epoch
│       ├── config.yaml                # Training config (copy)
│       ├── logs/                      # TensorBoard logs
│       └── test_results/              # Test outputs
│           ├── test_results.json
│           ├── visualizations/
│           └── predictions/
│
├── configs/
│   └── training/
│       └── confocal_baseline.yaml
│
└── scripts/
    ├── train_confocal.py
    └── test_confocal.py
```

---

## Next Steps After Training

### 1. Analyze Results

Check test metrics:
```cmd
type results\confocal_baseline\test_results\test_results.json
```

Look at visualizations to understand failure cases.

---

### 2. Improve Model (Optional)

If results are not satisfactory:

**Option A: Increase model capacity**
- Larger `feature_dim` and `hidden_dim`
- More correlation levels

**Option B: Train longer**
- Increase `epochs` to 200

**Option C: Better data**
- Regenerate with more samples
- Different deformation types

---

### 3. Apply to Real Data

Once satisfied with synthetic results, test on real confocal data:

1. Organize real data in same format:
   ```
   real_data/
   ├── vol0/
   └── vol1/
   ```

2. Run inference:
   ```cmd
   python scripts/infer.py --checkpoint results/confocal_baseline/checkpoint_best.pth --data_dir real_data/
   ```

---

## Performance Benchmarks

On RTX 5090 (32GB), 128³ volumes:

| Batch Size | Memory Usage | Speed (samples/sec) |
|------------|--------------|---------------------|
| 1          | ~15 GB       | ~0.3                |
| 2          | ~28 GB       | ~0.5                |

**Training time:**
- 800 samples × 100 epochs ÷ 0.5 samples/sec ÷ 3600 sec/hr ≈ **44 hours** (batch_size=2)

**With validation (every 5 epochs):**
- Add ~2-3 hours
- **Total: ~50 hours** for complete training

---

## Questions?

Check:
1. Main project README: `README.md`
2. Codebase guide: `CODEBASE_GUIDE_CN.md`
3. Data generation guide: `tools/README.md`

Or review training logs and TensorBoard.
