# SimVP Velocity Prediction Training Guide

This setup adapts SimVP for dual-satellite velocity prediction using your NPZ data format.

## 📋 File Structure

```
SimVP/
├── API/
│   ├── dataloader_velocity.py       ← Custom velocity data loader (NEW)
│   ├── dataloader.py                ← Updated to support velocity (MODIFIED)
│   ├── dataloader_moving_mnist.py
│   └── dataloader_taxibj.py
├── train_velocity.py                ← Training script for velocity (NEW)
├── train_velocity.sh                ← Bash training script (NEW)
├── main.py                          ← Original entry point
├── exp.py
├── model.py
└── modules.py
```

## 🚀 Quick Start

### 1. Prepare Your Data

Ensure your data is organized as:
```
../data/
├── wacv_data/
│   └── 1300m_kfold_w_sensor_noise_both/
│       └── fold_01_val_r5-6_c0-2/
│           ├── train_w.npz
│           ├── val_w.npz
│           └── test_w.npz
```

Or directly in `../data/`:
```
../data/
├── train_w.npz
├── val_w.npz
└── test_w.npz
```

### 2. Install Dependencies

```bash
# Ensure you're in the SimVP environment
conda activate SimVP

# Or if you haven't created it yet:
cd SimVP
conda env create -f environment.yml
conda activate SimVP
```

### 3. Run Training

**Option A: Using bash script (easiest)**
```bash
cd SimVP
bash train_velocity.sh
```

**Option B: Direct Python command**
```bash
cd SimVP
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/ \
  --batch_size 16 \
  --epochs 100 \
  --lr 0.001 \
  --ex_name velocity_simvp_baseline
```

**Option C: With specific parameters**
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/ \
  --batch_size 32 \
  --hid_S 256 \
  --hid_T 1024 \
  --N_S 5 \
  --N_T 8 \
  --epochs 200 \
  --lr 0.0005 \
  --ex_name velocity_simvp_large
```

## 📊 Configuration Parameters

### Data Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataname` | `velocity` | Dataset name (use `velocity` for your data) |
| `--data_root` | `../data/` | Root directory containing train/val/test NPZ files |
| `--batch_size` | `16` | Training batch size |
| `--val_batch_size` | `16` | Validation/test batch size |
| `--num_workers` | `4` | Dataloader workers |

### Model Architecture Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--in_shape` | `[12, 2, 128, 128]` | Input shape: [T, C, H, W] (fixed for velocity) |
| `--hid_S` | `128` | Spatial hidden channels (64-256) |
| `--hid_T` | `512` | Temporal hidden channels (256-1024) |
| `--N_S` | `4` | Spatial encoder/decoder depth (3-6) |
| `--N_T` | `8` | Temporal modeling depth (4-8) |
| `--groups` | `8` | Channel groups (4-16) |

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--log_step` | `1` | Logging frequency (epochs) |
| `--seed` | `1` | Random seed |

## 🎯 Recommended Configurations

### Fast Training (Validation Only)
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/ \
  --epochs 50 \
  --batch_size 32 \
  --hid_S 64 \
  --hid_T 256 \
  --N_S 3 \
  --N_T 6 \
  --ex_name velocity_simvp_small
```

### Balanced (Default)
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/ \
  --epochs 100 \
  --batch_size 16 \
  --hid_S 128 \
  --hid_T 512 \
  --N_S 4 \
  --N_T 8 \
  --ex_name velocity_simvp_baseline
```

### High Precision (Best Accuracy)
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/ \
  --epochs 200 \
  --batch_size 8 \
  --hid_S 256 \
  --hid_T 1024 \
  --N_S 5 \
  --N_T 8 \
  --lr 0.0003 \
  --ex_name velocity_simvp_large
```

## 📈 Output and Results

Results will be saved in:
```
results/
└── {ex_name}/
    ├── log.log                  # Training logs
    ├── model_param.json         # Configuration snapshot
    └── checkpoints/
        └── latest.pth           # Best model checkpoint
```

Example output during training:
```
==================================================
SimVP Configuration for Velocity Prediction
==================================================
  batch_size........................ 16
  data_root........................ ../data/
  dataname......................... velocity
  epochs........................... 100
  ex_name.......................... velocity_simvp_baseline
  hid_S............................ 128
  hid_T............................ 512
  ...
==================================================

> loading data...
[VelocityDataset] Loaded 1000 sequences from train_w.npz (dual-sat mode)
  X shape: (1000, 12, 2, 128, 128), Y shape: (1000, 12, 1, 128, 128)
  X normalization: max=1.2345, norm_const=1.2345
  Y velocity range: [-1.5432, 3.3210]
  Y scale factor: 3.8210 (maps ±3.3210 -> ±0.95)

Epoch 1/100: Loss=0.0234 | MAE=0.5432
Epoch 2/100: Loss=0.0187 | MAE=0.4932
...
```

## 🔧 Custom Data Loader Details

The custom `dataloader_velocity.py`:

1. **Loads NPZ files** with keys `X` (input images) and `Y` (target velocity)
   - X shape: `[N, T, 2, 128, 128]` - N sequences, 12 timesteps, 2 satellites, 128×128 spatial
   - Y shape: `[N, T, 1, 128, 128]` - N sequences, 12 timesteps, 1 velocity channel, 128×128

2. **Normalizes consistently**:
   - X: Divided by max value (preserves relative magnitudes)
   - Y: Symmetric normalization (maps ±max_abs_value → ±0.95)

3. **Supports augmentation**:
   - Rotations (0°, 90°, 180°, 270°)
   - Horizontal/vertical flips

4. **Provides denormalization**:
   - Convert predictions back to physical velocity (m/s)

## 📝 Example: Running Multiple Models for Comparison

Create `run_comparison.sh`:

```bash
#!/bin/bash

# Run multiple configurations for comparison
for name in small baseline large
do
    case $name in
        small)
            python train_velocity.py --dataname velocity --data_root ../data/ \
                --hid_S 64 --hid_T 256 --epochs 100 --ex_name velocity_simvp_small
            ;;
        baseline)
            python train_velocity.py --dataname velocity --data_root ../data/ \
                --hid_S 128 --hid_T 512 --epochs 100 --ex_name velocity_simvp_baseline
            ;;
        large)
            python train_velocity.py --dataname velocity --data_root ../data/ \
                --hid_S 256 --hid_T 1024 --epochs 100 --ex_name velocity_simvp_large
            ;;
    esac
done

echo "All models trained! Check results/ directory"
```

## ⚠️ Common Issues

**Issue: "Cannot find train_w.npz"**
- Ensure your NPZ files are in the correct location
- Check `--data_root` parameter
- The script searches recursively in data_root

**Issue: Out of memory (OOM)**
- Reduce `--batch_size` (try 8 instead of 16)
- Reduce `--hid_S` and `--hid_T` (try 64 and 256)
- Reduce `--N_S` and `--N_T` (try 3 and 6)

**Issue: Very slow training**
- Increase `--num_workers` (use more CPU cores for data loading)
- Reduce `--epochs` for testing
- Use smaller model (`--hid_S 64 --hid_T 256`)

## 🤖 Integration with Your Main Project

After training SimVP, compare with your model:

```python
import json
import torch

# Load SimVP metrics
with open('SimVP/results/velocity_simvp_baseline/log.log') as f:
    simvp_metrics = f.read()

# Load your model metrics from main.py
your_metrics = {...}  # From your training

# Compare
print("=" * 70)
print("COMPARISON: SimVP vs Your MiT-B1 Model")
print("=" * 70)
print(f"SimVP MSE: {simvp_mse}")
print(f"Your Model MSE: {your_mse}")
print(f"Improvement: {(simvp_mse - your_mse) / simvp_mse * 100:.2f}%")
```

## 📚 References

- Original SimVP paper: [arXiv:2206.05099](https://arxiv.org/abs/2206.05099)
- Your velocity prediction project: Main training script at `../main.py`
- Dataset format: `../train/dataset.py`

## ✅ Next Steps

1. ✓ Custom velocity data loader created
2. ✓ Training script ready
3. → Run baseline training: `bash train_velocity.sh`
4. → Compare with your model
5. → Prepare results for WACV 2027 rebuttal
