#!/bin/bash
# K-Fold Training for SimVP with Velocity Prediction Data
# Trains on each fold and aggregates results

DATA_ROOT="../data/wacv_data/envelop_kfold_w_sensor_noise_both"
EPOCHS=100
LR=0.001
BATCH_SIZE=16

echo "=========================================="
echo "SimVP K-Fold Training for Velocity"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Epochs: $EPOCHS"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Error: Data directory not found: $DATA_ROOT"
    exit 1
fi

# Check for test set
if [ ! -f "$DATA_ROOT/test_w.npz" ]; then
    echo "❌ Error: Shared test set not found: $DATA_ROOT/test_w.npz"
    exit 1
fi

# Create results directory
mkdir -p results

# Find all fold directories
FOLDS=$(ls -d $DATA_ROOT/fold_* 2>/dev/null | sort)

if [ -z "$FOLDS" ]; then
    echo "❌ Error: No fold directories found in $DATA_ROOT"
    exit 1
fi

FOLD_COUNT=$(echo "$FOLDS" | wc -w)
echo "Found $FOLD_COUNT folds"
echo ""

# Train on each fold
FOLD_NUM=1
for fold_path in $FOLDS; do
    fold_name=$(basename "$fold_path")
    
    echo "[$FOLD_NUM/$FOLD_COUNT] Training on $fold_name..."
    echo "  Input: $fold_path"
    
    python train_velocity.py \
        --dataname velocity \
        --data_root "$fold_path" \
        --test_root "$DATA_ROOT" \
        --batch_size $BATCH_SIZE \
        --val_batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --ex_name "velocity_simvp_${fold_name}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Fold $fold_name completed"
    else
        echo "❌ Fold $fold_name failed"
    fi
    
    echo ""
    FOLD_NUM=$((FOLD_NUM + 1))
done

echo "=========================================="
echo "All folds completed!"
echo "Results saved to: results/"
echo "=========================================="
