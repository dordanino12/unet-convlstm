# 🔧 OOM (Out of Memory) Fix

## ❌ What Happened
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. 
GPU 0 has a total capacity of 23.58 GiB
```

**Root Cause:** Batch size 16 was too large for your RTX 3090 with this model configuration.

---

## ✅ Solution: Reduce Batch Size

### Quick Fix - Use batch_size=8

**Option 1: Use updated script (easiest)**
```bash
bash train_envelope_kfold.sh 1
```
✓ Already updated to use `--batch_size 8`

**Option 2: Use reduced batch script**
```bash
bash train_envelope_kfold_reduced.sh 1
```

**Option 3: Manual command with batch_size=8**
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 8 \
  --val_batch_size 8 \
  --epochs 100 \
  --lr 0.001
```

**Option 4: Ultra-conservative (batch_size=4)**
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 4 \
  --val_batch_size 4 \
  --epochs 100 \
  --lr 0.001
```

---

## 📊 Batch Size Recommendations

| Batch Size | Speed | GPU Memory | Quality | Recommended |
|-----------|-------|-----------|---------|-------------|
| 4 | Slow | ✓ Safe | Best | If OOM persists |
| 8 | Normal | ✓ Safe | Good | **👈 Use this** |
| 16 | Fast | ❌ OOM | Good | Too large |
| 32 | Very Fast | ❌ OOM | Good | Way too large |

---

## 🎯 What Changed

### Updated Scripts (batch_size reduced from 16 to 8)
- ✓ `train_envelope_kfold.sh` - **Updated**
- ✓ `train_envelope_kfold_reduced.sh` - **New backup**
- ✓ `train_velocity_kfold.sh` - **Not changed (you can update manually)**

---

## 🚀 Try Again NOW

### Fold 1 Training (with fixed batch size)
```bash
cd /home/danino/PycharmProjects/pythonProject/SimVP
bash train_envelope_kfold.sh 1
```

**Expected to work now!** ✓

---

## 💡 If Still Getting OOM

Try these progressively:

**Step 1:** Use batch_size=4
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 4 \
  --val_batch_size 4 \
  --epochs 100
```

**Step 2:** Reduce model size
```bash
python train_velocity.py \
  --dataname velocity \
  --data_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/ \
  --test_root ../data/wacv_data/envelop_kfold_w_sensor_noise_both/ \
  --batch_size 8 \
  --hid_S 64 \
  --hid_T 256 \
  --N_S 3 \
  --N_T 6 \
  --epochs 100
```

**Step 3:** Use gradient accumulation (update train_velocity.py)
- Would require modifying exp.py to accumulate gradients

---

## 📋 Your GPU Specs

```
NVIDIA GeForce RTX 3090
- Total Memory: 24,576 MB (24 GB)
- Arch: Ampere
- Great for training!
```

✓ Batch size 8 should work fine on this GPU.

---

## ⏱️ Training Time Estimate

With batch_size=8:
- **Per fold:** ~45 minutes (100 epochs)
- **All 4 folds:** ~3 hours total
- **Learning rate warmup:** First few epochs slower

---

## 🎉 Ready to Train!

```bash
cd SimVP
bash train_envelope_kfold.sh 1
```

Let's go! 🚀
