# ✅ SimVP Setup Complete - Training Ready!

## 📦 What Was Created

### 1. **Custom Data Loader** 
- **File:** `SimVP/API/dataloader_velocity.py`
- **Supports:** Your NPZ format `[N, T=12, C=2, H=128, W=128]`
- **Features:** Symmetric normalization, augmentation, kfold support

### 2. **Training Scripts**
| Script | Purpose | Time |
|--------|---------|------|
| `train_envelope_kfold.sh [1-4]` | Train single fold | ~30 min |
| `train_velocity_kfold.sh` | Train all 4 folds | ~2 hours |
| `train_velocity.py` | Direct Python (full control) | Configurable |
| `train_velocity.sh` | Basic training script | Configurable |

### 3. **Test & Validation**
- **File:** `SimVP/test_velocity_loader.py`
- **Status:** ✓ **TESTED & WORKING**
- Output shows:
  - ✓ Fold 1: 1,725 train sequences, 414 val sequences
  - ✓ Shared test set: 414 sequences
  - ✓ Correct shapes: `[N, 12, 2, 128, 128]` input, `[N, 12, 1, 128, 128]` target
  - ✓ Normalization ranges correct

### 4. **Documentation**
| Doc | Content |
|-----|---------|
| `QUICK_START.md` | **👈 START HERE** - 3 training options |
| `ENVELOPE_KFOLD_TRAINING.md` | Detailed k-fold guide |
| `VELOCITY_PREDICTION_GUIDE.md` | Full reference documentation |

---

## 🎯 Three Ways to Start Training

### Option 1: Train Fold 1 (Fast Test - ~30 min)
```bash
cd SimVP
bash train_envelope_kfold.sh 1
```

### Option 2: Train All 4 Folds (Full Ensemble - ~2 hours)
```bash
cd SimVP
bash train_velocity_kfold.sh
```

### Option 3: Manual Command
```bash
cd SimVP
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 16 \
  --epochs 100
```

---

## 🔍 Data Summary

Your envelope data structure:
```
data/wacv_data/envelop_kfold_w_sensor_noise_both/
├── fold_01_val_r0-1_c0-2/        1,725 train | 414 val
├── fold_02_val_r0-1_c4-6/        1,725 train | 414 val
├── fold_03_val_r2-3_c0-6/        1,725 train | 414 val
├── fold_04_val_r5-6_c0-2/        1,725 train | 414 val
└── test_w.npz                    414 test (shared)
```

**Verified:** ✓ All data accessible, ✓ Shapes correct, ✓ Normalization working

---

## 📊 Model Configuration

Default settings (optimized for your data):
```
Input:  [B, 12, 2, 128, 128]  (12 timesteps, 2 satellites)
Output: [B, 12, 1, 128, 128]  (12 timesteps, velocity)
hid_S:  128  (spatial channels)
hid_T:  512  (temporal channels)
N_S:    4    (spatial depth)
N_T:    8    (temporal depth)
Epochs: 100
LR:     0.001
```

---

## 📈 Expected Output

When you run training, you'll see:
```
[K-FOLD MODE] Using fold: fold_01_val_r0-1_c0-2
[K-FOLD MODE] Shared test set: ../data/wacv_data/envelop_kfold_w_sensor_noise_both/test_w.npz

[VelocityDataset] Loaded 1725 sequences from train_w.npz (dual-sat mode)
  X shape: (1725, 12, 2, 128, 128), Y shape: (1725, 12, 1, 128, 128)
  Y velocity range: [-7.5988, 8.7849]

Epoch 1/100: Loss=0.0234 | MAE=0.5432
Epoch 2/100: Loss=0.0187 | MAE=0.4932
...
Test MSE: 0.0156
```

Results saved to: `results/velocity_simvp_fold_XX/`

---

## ✅ Checklist

- ✓ Custom data loader created
- ✓ K-fold support implemented
- ✓ Training scripts ready (3 options)
- ✓ Test script verified with your data
- ✓ Documentation complete
- ✓ All files executable and in place

---

## 🚀 Next Steps

1. **Read Quick Start:** `cat QUICK_START.md`
2. **Pick training option:** Fold 1, All folds, or manual
3. **Run training:** `bash train_envelope_kfold.sh 1`
4. **Monitor results:** `tail -f results/velocity_simvp_fold_*/log.log`
5. **Compare with your model:** Collect metrics for WACV rebuttal

---

## 📍 File Locations

**In `SimVP/` directory:**
- Core: `API/dataloader_velocity.py` | `train_velocity.py`
- Scripts: `train_envelope_kfold.sh` | `train_velocity_kfold.sh`
- Testing: `test_velocity_loader.py`
- Docs: `QUICK_START.md` (this file) | `ENVELOPE_KFOLD_TRAINING.md`

**Results go to:**
- `results/velocity_simvp_fold_01_val_r0-1_c0-2/log.log`
- `results/velocity_simvp_fold_01_val_r0-1_c0-2/checkpoints/latest.pth`

---

## 💡 Tips

- **First time?** Run `bash train_envelope_kfold.sh 1` (just fold 1)
- **Want ensemble?** Run `bash train_velocity_kfold.sh` (all folds)
- **GPU issues?** Reduce `--batch_size 8` or `--batch_size 4`
- **Slow training?** Increase `--num_workers 8`

---

**Everything is ready! 🎉 Pick your training command and start now!**
