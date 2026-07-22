# 🚀 Quick Start: SimVP Training on Envelope K-Fold Data

## Your Data Location
```
data/wacv_data/envelop_kfold_w_sensor_noise_both/
├── fold_01_val_r0-1_c0-2/     (1725 train, 414 val sequences)
├── fold_02_val_r0-1_c4-6/     (1725 train, 414 val sequences)
├── fold_03_val_r2-3_c0-6/     (1725 train, 414 val sequences)
├── fold_04_val_r5-6_c0-2/     (1725 train, 414 val sequences)
└── test_w.npz                  (414 test sequences - shared)
```

✓ **Data loader tested and working!**

---

## ⚡ Three Ways to Train

### Option 1: Train ONE Fold (Fastest - for testing)
```bash
cd SimVP
bash train_envelope_kfold.sh 1
```
**Time:** ~30 min | **Result:** `results/velocity_simvp_fold_01_val_r0-1_c0-2/`

### Option 2: Train ALL Folds (Full ensemble)
```bash
cd SimVP
bash train_velocity_kfold.sh
```
**Time:** ~2 hours | **Result:** 4 separate models for ensemble

### Option 3: Manual Control
```bash
cd SimVP
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 16 \
  --epochs 100 \
  --ex_name my_experiment_name
```

---

## 📊 Training Configurations

### Small Model (Fast validation)
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 32 \
  --epochs 50 \
  --hid_S 64 \
  --hid_T 256 \
  --ex_name envelope_simvp_small
```

### Default Model (Balanced)
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 16 \
  --epochs 100 \
  --hid_S 128 \
  --hid_T 512 \
  --ex_name envelope_simvp_baseline
```

### Large Model (Best accuracy)
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
  --ex_name envelope_simvp_large
```

---

## 📈 Expected Output

When you run training:
```
========================================
SimVP Configuration for Velocity Prediction
========================================
[K-FOLD MODE] Using fold: fold_01_val_r0-1_c0-2
[K-FOLD MODE] Shared test set: ../data/wacv_data/envelop_kfold_w_sensor_noise_both/test_w.npz
========================================

> loading data...
[DataLoader] Train: .../fold_01_val_r0-1_c0-2/train_w.npz
[DataLoader] Val:   .../fold_01_val_r0-1_c0-2/val_w.npz
[DataLoader] Test:  .../envelop_kfold_w_sensor_noise_both/test_w.npz

[VelocityDataset] Loaded 1725 sequences from train_w.npz (dual-sat mode)
  X shape: (1725, 12, 2, 128, 128), Y shape: (1725, 12, 1, 128, 128)
  X normalization: max=30.3656, norm_const=30.3656
  Y velocity range: [-7.5988, 8.7849]
  Y scale factor: 9.2473 (maps ±8.7849 -> ±0.95)

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Epoch 1/100: Loss=0.0234 | MAE=0.5432
Epoch 2/100: Loss=0.0187 | MAE=0.4932
...
>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

---

## 🎯 Recommended First Steps

**Step 1:** Run ONE fold (fastest feedback)
```bash
cd SimVP
bash train_envelope_kfold.sh 1
```

**Step 2:** Check results
```bash
cat results/velocity_simvp_fold_01_val_r0-1_c0-2/log.log
```

**Step 3:** If good, train all folds for ensemble
```bash
bash train_velocity_kfold.sh
```

**Step 4:** Compare with your MiT-B1 model for WACV rebuttal

---

## 🔗 Key Files

- **Training script:** `train_velocity.py`
- **Data loader:** `API/dataloader_velocity.py`
- **K-fold ensemble script:** `train_velocity_kfold.sh`
- **Single fold script:** `train_envelope_kfold.sh`
- **Test script:** `test_velocity_loader.py`
- **Full documentation:** `ENVELOPE_KFOLD_TRAINING.md`

---

## ⚠️ If You Get Errors

**Error: "Cannot find train_w.npz"**
→ Make sure you're in the SimVP directory: `cd SimVP`

**Error: Out of memory**
→ Reduce batch_size: `--batch_size 8` or `--batch_size 4`

**Error: Very slow training**
→ Increase workers: `--num_workers 8`

**Error: GPU not detected**
→ Check: `nvidia-smi`

---

## 📞 Quick Commands Reference

| Command | What it does |
|---------|------------|
| `bash train_envelope_kfold.sh 1` | Train fold 1 only |
| `bash train_envelope_kfold.sh 2` | Train fold 2 only |
| `bash train_velocity_kfold.sh` | Train all 4 folds |
| `python test_velocity_loader.py --data_root ... --test_root ...` | Test data loader |
| `ls results/` | See all training results |
| `cat results/velocity_simvp_fold_01_val_r0-1_c0-2/log.log` | View training logs |

---

## ✅ You're Ready!

Everything is set up. Pick your preferred training option above and go! 🚀
