# SimVP Training on Envelope K-Fold Data

This guide shows how to train SimVP on your envelope k-fold dataset.

## 📁 Data Structure

```
data/wacv_data/envelop_kfold_w_sensor_noise_both/
├── fold_01_val_r0-1_c0-2/
│   ├── train_w.npz
│   └── val_w.npz
├── fold_02_val_r0-1_c4-6/
│   ├── train_w.npz
│   └── val_w.npz
├── fold_03_val_r2-3_c0-6/
│   ├── train_w.npz
│   └── val_w.npz
├── fold_04_val_r5-6_c0-2/
│   ├── train_w.npz
│   └── val_w.npz
└── test_w.npz          ← Shared test set (same for all folds)
```

## 🚀 Quick Start

### Test Data Loader First
```bash
cd SimVP
python test_velocity_loader.py \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/
```

### Train on Specific Fold (Fold 1)
```bash
bash train_envelope_kfold.sh 1
# or with fold number 01-04
bash train_envelope_kfold.sh 1  # trains on fold_01_val_r0-1_c0-2
bash train_envelope_kfold.sh 2  # trains on fold_02_val_r0-1_c4-6
bash train_envelope_kfold.sh 3  # trains on fold_03_val_r2-3_c0-6
bash train_envelope_kfold.sh 4  # trains on fold_04_val_r5-6_c0-2
```

### Train All Folds (Ensemble)
```bash
bash train_velocity_kfold.sh
# This will:
# 1. Train on fold_01_val_r0-1_c0-2 
# 2. Train on fold_02_val_r0-1_c4-6
# 3. Train on fold_03_val_r2-3_c0-6
# 4. Train on fold_04_val_r5-6_c0-2
# All using the shared test_w.npz for evaluation
```

### Manual Training (Full Control)
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 16 \
  --epochs 100 \
  --lr 0.001 \
  --ex_name velocity_simvp_fold_01
```

## 📊 Understanding the Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `--dataname velocity` | Dataset type (must be 'velocity') | - |
| `--data_root` | Path to fold directory (contains train_w.npz, val_w.npz) | - |
| `--test_root` | Path to shared test set directory (contains test_w.npz) | Same as data_root |
| `--batch_size` | Training batch size | 16 |
| `--epochs` | Number of epochs | 100 |
| `--lr` | Learning rate | 0.001 |
| `--ex_name` | Experiment name (results saved here) | `velocity_simvp` |

## 📈 Results Location

Each training creates results in:
```
results/
└── velocity_simvp_fold_XX/
    ├── log.log                  ← Training logs
    ├── model_param.json         ← Config snapshot
    └── checkpoints/
        └── latest.pth           ← Best model
```

## 🔄 K-Fold Ensemble Evaluation

After training all 4 folds, evaluate ensemble:

```python
import numpy as np
import torch
from API.dataloader_velocity import VelocityPredictionDataset

# Load test set
test_set = VelocityPredictionDataset(
    '../data/wacv_data/envelop_kfold_w_sensor_noise_both/test_w.npz',
    is_train=False
)

# Load 4 fold models and ensemble
ensemble_preds = []
for fold_num in [1, 2, 3, 4]:
    checkpoint_path = f'results/velocity_simvp_fold_{fold_num:02d}/checkpoints/latest.pth'
    ckpt = torch.load(checkpoint_path)
    # ... load model and get predictions
    ensemble_preds.append(pred)

# Average predictions
ensemble_pred = np.mean(ensemble_preds, axis=0)
```

## 💡 Recommended Commands

### For Quick Testing (validation)
```bash
bash train_envelope_kfold.sh 1  # Train just fold 1, 100 epochs
```

### For Production (all folds)
```bash
bash train_velocity_kfold.sh    # Train all 4 folds for ensemble
```

### For High Precision
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 8 \
  --epochs 200 \
  --hid_S 256 \
  --hid_T 1024 \
  --lr 0.0003 \
  --ex_name velocity_simvp_fold_01_large
```

## ⚠️ Troubleshooting

**Issue: "Cannot find train_w.npz"**
```bash
# Check fold structure
ls -la ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/
# Should show: train_w.npz, val_w.npz
```

**Issue: Out of memory**
```bash
# Reduce batch size and model size
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 8 \
  --hid_S 64 \
  --hid_T 256
```

**Issue: Very slow training**
```bash
# Check GPU usage
nvidia-smi

# Increase workers
python train_velocity.py ... --num_workers 8
```

## 📝 Next Steps

1. ✓ Test data loader: `python test_velocity_loader.py --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/`
2. → Train fold 1: `bash train_envelope_kfold.sh 1`
3. → Train all folds: `bash train_velocity_kfold.sh`
4. → Compare with your MiT-B1 model
5. → Prepare results for WACV 2027 rebuttal
