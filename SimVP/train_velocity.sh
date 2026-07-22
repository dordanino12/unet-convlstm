#!/bin/bash
# Training scripts for SimVP with velocity prediction data

DATA_ROOT="../data"
EPOCHS=100
LR=0.001

echo "=========================================="
echo "SimVP Training for Velocity Prediction"
echo "=========================================="

# Create results directory
mkdir -p results

# Basic training with default parameters
echo ""
echo "[1] Training SimVP - Default Configuration"
echo "  Input: [B, 12, 2, 128, 128]"
echo "  hid_S: 128, hid_T: 512, N_S: 4, N_T: 8"
echo ""
python train_velocity.py \
  --dataname velocity \
  --data_root $DATA_ROOT \
  --batch_size 16 \
  --val_batch_size 16 \
  --epochs $EPOCHS \
  --lr $LR \
  --ex_name velocity_simvp_baseline

echo ""
echo "Training complete! Results saved to: results/velocity_simvp_baseline/"
