#!/bin/bash
# Simple training script for envelope kfold data
# Usage: bash train_envelope_kfold.sh [fold_number] [--use_bin_loss | --no_bin_loss] [--use_mask | --no_mask]

KFOLD_ROOT="../data/wacv_data/1000m_kfold_w_sensor_noise_both"

# Parse fold number if provided
if [ -z "$1" ]; then
    FOLD_NUM="01"
    echo "No fold specified, using fold_01"
else
    FOLD_NUM=$(printf "%02d" $1)
fi

# Check for loss argument (Argument #2), default to --use_bin_loss if empty
LOSS_FLAG=${2:---use_bin_loss}

# Check for mask argument (Argument #3), default to --use_mask if empty
MASK_FLAG=${3:---use_mask}

FOLD_DIR="$KFOLD_ROOT/fold_${FOLD_NUM}_val_r*_c*"
FOLD_PATH=$(ls -d $FOLD_DIR 2>/dev/null | head -1)

if [ -z "$FOLD_PATH" ]; then
    echo "❌ Error: Could not find fold_${FOLD_NUM} in $KFOLD_ROOT"
    echo "Available folds:"
    ls -d $KFOLD_ROOT/fold_* 2>/dev/null
    exit 1
fi

FOLD_NAME=$(basename "$FOLD_PATH")
echo "=========================================="
echo "Training on: $FOLD_NAME"
echo "Using loss flag: $LOSS_FLAG"
echo "Using mask flag: $MASK_FLAG"
echo "=========================================="
echo ""

python train_velocity.py \
  --dataname velocity \
  --data_root "$FOLD_PATH" \
  --test_root "$KFOLD_ROOT" \
  --batch_size 8 \
  --val_batch_size 8 \
  --epochs 100 \
  --lr 0.001 \
  --ex_name "velocity_simvp_binlos_1000m_${FOLD_NAME}" \
  $LOSS_FLAG \
  $MASK_FLAG

echo ""
echo "Training complete!"
echo "Results: results/velocity_simvp_binlos_1000m_${FOLD_NAME}/"