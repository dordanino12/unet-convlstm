"""
Main Training Script
--------------------
Dual-Satellite Velocity Estimation
Supports two architectures:
1. Custom Temporal U-Net (ConvLSTM-based)
2. Pre-trained ResNet18 U-Net (Frozen Encoder)
"""

from __future__ import annotations
import os
import copy
import sys
import json
import argparse
import subprocess
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import torch.fft
import math
import torch.nn.functional as F
# --- Local Imports ---

from train.dataset import NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet, PretrainedTemporalUNetMitB1, PretrainedTemporalUNetMitB2, PretrainedTemporalUNetMitB3

# --- Exponent Loss Function ---
def exponent_loss(y_pred, y, mask=None, use_mask=True):
    """
    Computes weighted exponent loss and spatial gradient loss.
    - Penalizes high-velocity errors more heavily (cubic weighting).
    - Uses exponent on L1 error.
    - Ensures numerical stability with epsilon.
    """
    # 1. Weighted Exponent Loss
    abs_diff = (y_pred - y).abs()
    weight = 1.0 + 4.0 * (y.abs() ** 8)

    if use_mask and mask is not None:
        numerator = (abs_diff * mask * weight).sum()
        denominator = (mask * weight).sum() + 1e-8
        weighted_exp = numerator / denominator
    else:
        weighted_exp = (abs_diff * weight).mean()

    # 2. Gradient Loss (Spatial smoothness and edge preservation)
    def spatial_gradients(tensor):
        dx = tensor[..., :, 1:] - tensor[..., :, :-1]
        dy = tensor[..., 1:, :] - tensor[..., :-1, :]
        return dx, dy

    dx_pred, dy_pred = spatial_gradients(y_pred)
    dx_gt, dy_gt = spatial_gradients(y)

    # Crop to smallest spatial dim to avoid shape mismatch
    H_min = min(dx_pred.shape[-2], dy_pred.shape[-2])
    W_min = min(dx_pred.shape[-1], dy_pred.shape[-1])

    grad_diff = (dx_pred[..., :H_min, :W_min] - dx_gt[..., :H_min, :W_min]).abs() + \
                (dy_pred[..., :H_min, :W_min] - dy_gt[..., :H_min, :W_min]).abs()

    if use_mask and mask is not None:
        mask_c = mask[..., :H_min, :W_min]
        grad_loss = (grad_diff * mask_c).sum() / (mask_c.sum() + 1e-8)
    else:
        grad_loss = grad_diff.mean()

    # Combine losses (0.005 weight for gradients)
    total_loss = weighted_exp #+ 0.005 * grad_loss
    return total_loss

#------------
# Loss Function: Weighted L1 + Gradient Loss
# -----------------------------------------------------
def compute_loss(y_pred, y, mask=None, use_mask=True, dataset_obj=None, unmasked_weight_factor=0.1, debug_bins=False, bin_min=None, bin_max=None):

    abs_diff = (y_pred - y).abs()
    # Dynamically determine bin min/max from training data
    if bin_min is None or bin_max is None:
        raise ValueError("bin_min and bin_max must be provided from the training dataset")
    BIN_MIN = float(bin_min)
    BIN_MAX = float(bin_max)
    BIN_WIDTH = 0.1
    NUM_BINS = int(math.ceil((BIN_MAX - BIN_MIN) / BIN_WIDTH))

    # Prepare mask for binning (masked pixels only)
    mask_broadcasted = None
    spatial_mask = None
    mask_for_bins = None
    if use_mask == "slice_mask" and mask is not None:
        
        # Select the 5th time step. Using '5' instead of '5:6' drops the time dimension.
        # Shape goes from [32, 12, 1, 128, 128] -> [32, 1, 128, 128]
        mask_slice_5 = mask[:, 5, :, :, :] 
        
        EXPAND_KERNEL = 5 
        padding = EXPAND_KERNEL // 2
        
        # max_pool2d is now happy because it is receiving a 4D tensor
        expanded_mask_slice = F.max_pool2d(
            mask_slice_5, 
            kernel_size=EXPAND_KERNEL, 
            stride=1, 
            padding=padding
        )
        
        # Add the time dimension back: [32, 1, 128, 128] -> [32, 1, 1, 128, 128]
        expanded_mask_slice = expanded_mask_slice.unsqueeze(1)
        
        # Now it safely expands to [32, 12, 1, 128, 128]
        mask_broadcasted = expanded_mask_slice.expand_as(mask)
        mask_for_bins = mask_broadcasted > 0.5

    if mask_for_bins is not None and mask_for_bins.sum() == 0:
        # No masked pixels in this batch: fall back to all pixels for bin stats
        mask_for_bins = None

    # Calculate bin weights based on GT velocity distribution (denormalize y first)
    y_denorm = dataset_obj.denormalize(y)  # Convert to physical velocity (m/s)
    y_denorm = y_denorm.clamp(BIN_MIN, BIN_MAX - 1e-6)
    if mask_for_bins is not None:
        y_flat = y_denorm[mask_for_bins]
    else:
        y_flat = y_denorm.flatten()

    # Count pixels in each bin
    bin_counts = torch.zeros(NUM_BINS, device=y.device)
    for i in range(NUM_BINS):
        bin_start = BIN_MIN + i * BIN_WIDTH
        bin_end = bin_start + BIN_WIDTH
        bin_mask = (y_flat >= bin_start) & (y_flat < bin_end)
        bin_counts[i] = bin_mask.sum().float()

    # Calculate inverse frequency weights (only for non-empty bins)
    total_pixels = y_flat.numel()
    bin_weights = torch.zeros(NUM_BINS, device=y.device)
    non_empty = bin_counts > 0
    bin_weights[non_empty] = total_pixels / (bin_counts[non_empty] + 1e-8)

    # Normalize bin weights to average of 1 over non-empty bins
    if non_empty.any():
        bin_weights[non_empty] = bin_weights[non_empty] / (bin_weights[non_empty].mean() + 1e-8)

    # Cap extreme weights to avoid instability
    bin_weights = torch.clamp(bin_weights, max=100.0)

    # Print bin stats only when requested (e.g., once per epoch)
    if debug_bins:
        print("[BIN DEBUG] bin_counts:")
        for idx in range(NUM_BINS):
            b_start = BIN_MIN + idx * BIN_WIDTH
            b_end = b_start + BIN_WIDTH
            print(f"  [{b_start:.2f}, {b_end:.2f}): {int(bin_counts[idx].item())}")

    # Assign weights to each pixel based on its bin
    pixel_bin_weights = torch.zeros_like(y_denorm)
    for i in range(NUM_BINS):
        bin_start = BIN_MIN + i * BIN_WIDTH
        bin_end = bin_start + BIN_WIDTH
        bin_mask = (y_denorm >= bin_start) & (y_denorm < bin_end)
        pixel_bin_weights[bin_mask] = bin_weights[i]

    # --- Apply spatial mask if needed ---
    if use_mask == "slice_mask" and mask is not None:
        # Extract mask from time step 5 and broadcast
        if mask_broadcasted is None:
            mask_slice_5 = mask[:, 5, :, :, :] 
            EXPAND_KERNEL = 5 
            padding = EXPAND_KERNEL // 2
            
            expanded_mask_slice = F.max_pool2d(
                mask_slice_5, 
                kernel_size=EXPAND_KERNEL, 
                stride=1, 
                padding=padding
            )
            expanded_mask_slice = expanded_mask_slice.unsqueeze(1)

            mask_broadcasted = expanded_mask_slice.expand_as(mask)

        spatial_mask = torch.ones_like(mask_broadcasted)
        spatial_mask[mask_broadcasted > 0.5] = 1.0
        spatial_mask[mask_broadcasted <= 0.5] = unmasked_weight_factor

        combined_weight = pixel_bin_weights * spatial_mask
        denom = combined_weight.sum()
        if denom < 1e-8:
            weighted_l1 = torch.zeros((), device=y.device)
        else:
            numerator = (abs_diff * combined_weight).sum()
            weighted_l1 = numerator / (denom + 1e-8)

    elif use_mask is True and mask is not None:
        combined_weight = pixel_bin_weights * mask
        denom = combined_weight.sum()
        if denom < 1e-8:
            weighted_l1 = torch.zeros((), device=y.device)
        else:
            numerator = (abs_diff * combined_weight).sum()
            weighted_l1 = numerator / (denom + 1e-8)
    else:
        weighted_l1 = (abs_diff * pixel_bin_weights).sum() / (pixel_bin_weights.sum() + 1e-8)
    # ...existing code...
    def spatial_gradients(tensor):
        dx = tensor[..., :, 1:] - tensor[..., :, :-1]
        dy = tensor[..., 1:, :] - tensor[..., :-1, :]
        return dx, dy

    dx_pred, dy_pred = spatial_gradients(y_pred)
    dx_gt, dy_gt = spatial_gradients(y)

    # Crop to smallest spatial dim
    H_min = min(dx_pred.shape[2], dy_pred.shape[2])
    W_min = min(dx_pred.shape[3], dy_pred.shape[3])

    grad_diff = (dx_pred[..., :H_min, :W_min] - dx_gt[..., :H_min, :W_min]).abs() + \
                (dy_pred[..., :H_min, :W_min] - dy_gt[..., :H_min, :W_min]).abs()

    if use_mask == "slice_mask" and mask is not None:
        mask_c = mask_broadcasted[..., :H_min, :W_min]
        spatial_mask_c = spatial_mask[..., :H_min, :W_min]
        denom = spatial_mask_c.sum()
        if denom < 1e-8:
            grad_loss = torch.zeros((), device=y.device)
        else:
            grad_loss = (grad_diff * spatial_mask_c).sum() / (denom + 1e-8)
    elif use_mask is True and mask is not None:
        mask_c = mask[..., :H_min, :W_min]
        denom = mask_c.sum()
        if denom < 1e-8:
            grad_loss = torch.zeros((), device=y.device)
        else:
            grad_loss = (grad_diff * mask_c).sum() / (denom + 1e-8)
    else:
        grad_loss = grad_diff.mean()

    # Combine losses: L1 + 0.005 * Gradient Loss
    total_loss = weighted_l1 #+ 0.005 * grad_loss
    return total_loss

# -----------------------------------------------------
# Training Loop
# -----------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, dataset_obj, scaler, use_mask=True, unmasked_weight_factor=0.1, debug_bins_once=False, loss_mode='bins', loss_interp=0.0, bin_min=None, bin_max=None):
    model.train()
    total_loss, n = 0.0, 0

    # Accumulators for training metrics
    all_mae = []
    all_sq_err = [] 
    all_err = []

    for batch_idx, (x, y, mask) in enumerate(loader):

        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass (AMP)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output, _ = model(x)

        # Compatibility handling
        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output

        debug_bins = debug_bins_once and batch_idx == 0
        # Loss selection logic
        if loss_mode == 'exponent':
            loss = exponent_loss(y_pred, y, mask, use_mask)
        elif loss_mode == 'bins':
            loss = compute_loss(
                y_pred,
                y,
                mask,
                use_mask,
                dataset_obj,
                unmasked_weight_factor,
                debug_bins=debug_bins,
                bin_min=bin_min,
                bin_max=bin_max
            )
        elif loss_mode == 'interp':
            # Interpolate between exponent and bins loss
            loss_exp = exponent_loss(y_pred, y, mask, use_mask)
            loss_bins = compute_loss(
                y_pred,
                y,
                mask,
                use_mask,
                dataset_obj,
                unmasked_weight_factor,
                debug_bins=debug_bins,
                bin_min=bin_min,
                bin_max=bin_max
            )
            loss = (1 - loss_interp) * loss_exp + loss_interp * loss_bins
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item() * x.size(0)
        n += x.size(0)

        # --- Metric Calculation (No Grad) ---
        with torch.no_grad():
            y_denorm = dataset_obj.denormalize(y).cpu().numpy()
            pred_denorm = dataset_obj.denormalize(y_pred).cpu().numpy()
            mask_np = mask.cpu().numpy()

            diff = pred_denorm - y_denorm

            if use_mask:
                # Use the mask directly (boolean mask or 0/1 mask)
                valid_mask = (mask_np.astype(bool))

                if np.any(valid_mask):
                    valid_diff = diff[valid_mask]
                    all_mae.extend(np.abs(valid_diff))
                    all_sq_err.extend(valid_diff ** 2)
                    all_err.extend(valid_diff)
            else:
                all_mae.extend(np.abs(diff).flatten())
                all_sq_err.extend((diff ** 2).flatten())
                all_err.extend(diff.flatten())

    # Aggregate Metrics
    avg_loss = total_loss / n
    if len(all_mae) > 0:
        avg_mae = np.mean(all_mae)
        avg_rmse = np.sqrt(np.mean(all_sq_err))
        avg_me = np.mean(all_err)
    else:
        avg_mae, avg_rmse, avg_me = 0.0, 0.0, 0.0

    return avg_loss, avg_mae, avg_rmse, avg_me


# -----------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, dataset_obj, use_mask=True, unmasked_weight_factor=0.1, loss_mode='bins', loss_interp=0.0, bin_min=None, bin_max=None):
    model.eval()
    total_loss, n = 0.0, 0
    
    # Accumulators for metrics
    all_mae = []
    all_sq_err = [] 
    all_err = []    
    
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output, _ = model(x)

        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output

        # Loss selection logic (same as training)
        if loss_mode == 'exponent':
            loss = exponent_loss(y_pred, y, mask, use_mask)
        elif loss_mode == 'bins':
            loss = compute_loss(y_pred, y, mask, use_mask, dataset_obj, unmasked_weight_factor, bin_min=bin_min, bin_max=bin_max)
        elif loss_mode == 'interp':
            loss_exp = exponent_loss(y_pred, y, mask, use_mask)
            loss_bins = compute_loss(y_pred, y, mask, use_mask, dataset_obj, unmasked_weight_factor, bin_min=bin_min, bin_max=bin_max)
            loss = (1 - loss_interp) * loss_exp + loss_interp * loss_bins
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")
        total_loss += loss.detach().item() * x.size(0)
        n += x.size(0)
        
        # 2. Calc Real Metrics (Denormalized space)
        y_denorm = dataset_obj.denormalize(y).cpu().numpy()
        pred_denorm = dataset_obj.denormalize(y_pred).cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        diff = pred_denorm - y_denorm
        
        if use_mask:
            # Use the mask directly
            valid_mask = (mask_np.astype(bool))
            if np.any(valid_mask):
                valid_diff = diff[valid_mask]
                all_mae.extend(np.abs(valid_diff))
                all_sq_err.extend(valid_diff ** 2)
                all_err.extend(valid_diff)
        else:
            all_mae.extend(np.abs(diff).flatten())
            all_sq_err.extend((diff ** 2).flatten())
            all_err.extend(diff.flatten())

    # Aggregate Metrics
    avg_loss = total_loss / n
    
    if len(all_mae) > 0:
        avg_mae = np.mean(all_mae)
        avg_rmse = np.sqrt(np.mean(all_sq_err))
        avg_me = np.mean(all_err)
    else:
        avg_mae, avg_rmse, avg_me = 0.0, 0.0, 0.0
        
    return avg_loss, avg_mae, avg_rmse, avg_me


# -----------------------------------------------------
# Main Execution
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--kfold-dir",
        type=str,
        default=None,
        help="Optional k-fold root directory (contains fold_* subfolders). If provided, runs each fold.",
    )
    cli_args, _ = parser.parse_known_args()

        # --- 1. Global Configuration ---
    BATCH_SIZE_START = 32
    BATCH_SIZE_FINETUNE = 16

    # 3-Stage Training Configuration
    EPOCHS_STAGE1 = 20 # Stage 1: Frozen encoder, no refiner (train decoder/LSTM/head only)
    EPOCHS_STAGE2 = 80   # Stage 2: Unfreeze encoder, no refiner (train full model except refiner)
    EPOCHS_STAGE3 = 0  # Stage 3: Freeze full model, train only refiner (fine-tune predictions)
    EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2 + EPOCHS_STAGE3
    # --- Loss schedule parameters ---
    EXPONENT_EPOCHS = 10
    INTERP_EPOCHS = 5  # Number of epochs to interpolate between exponent and bins loss (increased for smoother transition)

    LR_STAGE1 = 1e-3
    LR_STAGE2 = 3e-4
    LR_STAGE3 = 1e-4
    WEIGHT_DECAY = 1e-4

    BACKBONE = "mit_b1"  # "resnet18", "mit_b1", "mit_b2", or "mit_b3"
    USE_MASK = "slice_mask"  # True, False, or "slice_mask"
    USE_ENVELOP_AS_A_INPUT = False  # Whether to feed GT envelope velocity as an extra input channel
    # Use only one satellite image (first channel) instead of two
    USE_ONE_SATELLITE = False
    UNMASKED_WEIGHT_FACTOR = 0.9  # Weight multiplier for unmasked areas in slice_mask mode
    TRAIN_AUGMENT = False
    NPZ_TRAIN_PATH = os.getenv("NPZ_TRAIN_PATH_OVERRIDE", "data/wacv_data/1300m_kfold_w_sensor_noise_both/fold_01_val_r5-6_c0-2/train_w.npz")
    NPZ_VAL_PATH = os.getenv("NPZ_VAL_PATH_OVERRIDE", "data/wacv_data/1300m_kfold_w_sensor_noise_both/fold_01_val_r5-6_c0-2/val_w.npz")
    NPZ_TEST_PATH = os.getenv("NPZ_TEST_PATH_OVERRIDE", "data/wacv_data/1300m_kfold_w_sensor_noise_both/test_w.npz")
    GT_ENVELOPE_NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/data_orit_envelop_train_w.npz"
    model_name = f"{BACKBONE}_1300m"
    TYPE_VEL = "1300m"
    model_name_suffix = os.getenv("MODEL_NAME_SUFFIX", "")
    if model_name_suffix:
        model_name = model_name + model_name_suffix
    if USE_ONE_SATELLITE:
        model_name = model_name + "_one_sat"
    USE_CONV_LSTM = True  # Whether to use ConvLSTM layers in the decoder

    # Refiner config
    USE_REFINER = False
    REFINER_HIDDEN_CHANNELS = 32

    # Bin debug logging
    DEBUG_BINS_ONCE_PER_EPOCH = False  # Set to False to disable bin count logging

    # Checkpoint loading
    LOAD_CHECKPOINT = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    def _pick_existing(base_dir, names):
        for name in names:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                return p
        return None

    # Parent k-fold runner: iterate folds and call this script in single mode per fold.
    if cli_args.kfold_dir:
        kfold_root = cli_args.kfold_dir
        if not os.path.isdir(kfold_root):
            print(f"ERROR: --kfold-dir is not a directory: {kfold_root}")
            sys.exit(1)

        fold_dirs = sorted(
            d for d in os.listdir(kfold_root)
            if d.startswith("fold_") and os.path.isdir(os.path.join(kfold_root, d))
        )
        if not fold_dirs:
            print(f"ERROR: No fold_* directories found in: {kfold_root}")
            sys.exit(1)

        test_path = _pick_existing(kfold_root, ["test_w.npz", "test_uvw.npz", "test.npz"])
        if test_path is None:
            print(f"ERROR: Could not find shared test file under {kfold_root} (expected test_w.npz or test_uvw.npz)")
            sys.exit(1)

        print(f"[K-FOLD MODE] Root: {kfold_root}")
        print(f"[K-FOLD MODE] Found {len(fold_dirs)} folds")
        print(f"[K-FOLD MODE] Shared test set: {test_path}")

        metrics_dir = os.path.join("models", "wacv", "kfold_metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        fold_metrics = []

        for fold_name in fold_dirs:
            fold_path = os.path.join(kfold_root, fold_name)
            train_path = _pick_existing(fold_path, ["train_w.npz", "train_uvw.npz", "train.npz"])
            val_path = _pick_existing(fold_path, ["val_w.npz", "val_uvw.npz", "val.npz"])

            if train_path is None or val_path is None:
                print(f"[WARN] Skipping {fold_name}: missing train/val npz")
                continue

            out_metrics = os.path.join(metrics_dir, f"{fold_name}.json")
            env = os.environ.copy()
            env["NPZ_TRAIN_PATH_OVERRIDE"] = train_path
            env["NPZ_VAL_PATH_OVERRIDE"] = val_path
            env["NPZ_TEST_PATH_OVERRIDE"] = test_path
            env["MODEL_NAME_SUFFIX"] = f"_{fold_name}"
            env["METRICS_OUT_PATH"] = out_metrics

            print(f"\n[K-FOLD] Running {fold_name}")
            print(f"  train={train_path}")
            print(f"  val={val_path}")
            proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)], env=env)
            if proc.returncode != 0:
                print(f"[WARN] Fold {fold_name} failed with code {proc.returncode}")
                continue

            if os.path.exists(out_metrics):
                try:
                    with open(out_metrics, "r", encoding="utf-8") as f:
                        fold_metrics.append(json.load(f))
                except Exception as e:
                    print(f"[WARN] Could not read metrics for {fold_name}: {e}")

        if not fold_metrics:
            print("ERROR: No successful fold metrics collected.")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("K-FOLD SUMMARY")
        print("=" * 70)
        for m in fold_metrics:
            print(
                f"{m.get('fold_name', 'fold')}: "
                f"Loss={m.get('test_loss', 0.0):.4f} | "
                f"MAE={m.get('test_mae', 0.0):.4f} | "
                f"RMSE={m.get('test_rmse', 0.0):.4f} | "
                f"ME={m.get('test_me', 0.0):.4f}"
            )

        mean_loss = float(np.mean([m.get("test_loss", 0.0) for m in fold_metrics]))
        mean_mae = float(np.mean([m.get("test_mae", 0.0) for m in fold_metrics]))
        mean_rmse = float(np.mean([m.get("test_rmse", 0.0) for m in fold_metrics]))
        mean_me = float(np.mean([m.get("test_me", 0.0) for m in fold_metrics]))

        print("-" * 70)
        print(f"MEAN ACROSS FOLDS: Loss={mean_loss:.4f} | MAE={mean_mae:.4f} | RMSE={mean_rmse:.4f} | ME={mean_me:.4f}")

        # Ensemble evaluation on the shared test set.
        print("\n[ENSEMBLE] Evaluating mean prediction across fold models...")
        test_dataset = NPZSequenceDataset(
            test_path,
            use_gt_envelope_as_input=USE_ENVELOP_AS_A_INPUT,
            gt_envelope_npz_path=GT_ENVELOPE_NPZ_PATH,
            augment=False,
            use_one_satellite=USE_ONE_SATELLITE,
        )
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)

        ensemble_sum = None
        models_used = 0

        for m in fold_metrics:
            fold_name = m.get("fold_name", "")
            train_path = m.get("npz_train")
            checkpoint_path = os.path.join("models", "wacv", f"{BACKBONE}_{TYPE_VEL}_{fold_name}_best_bin_loss.pt")
            if not train_path or not os.path.exists(train_path):
                print(f"[ENSEMBLE] Skipping {fold_name}: missing train path")
                continue
            if not os.path.exists(checkpoint_path):
                print(f"[ENSEMBLE] Skipping {fold_name}: missing checkpoint {checkpoint_path}")
                continue

            train_dataset_for_norm = NPZSequenceDataset(
                train_path,
                use_gt_envelope_as_input=USE_ENVELOP_AS_A_INPUT,
                gt_envelope_npz_path=GT_ENVELOPE_NPZ_PATH,
                augment=False,
                use_one_satellite=USE_ONE_SATELLITE,
            )

            fold_model = None
            if BACKBONE == "resnet18":
                fold_model = PretrainedTemporalUNet(
                    out_channels=1,
                    lstm_layers=1 if USE_CONV_LSTM else 0,
                    freeze_encoder=True,
                    in_channels=train_dataset_for_norm.X.shape[2],
                    use_conv_lstm=USE_CONV_LSTM,
                    use_refiner=False,
                    refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
                ).to(device)
            elif BACKBONE == "mit_b1":
                fold_model = PretrainedTemporalUNetMitB1(
                    out_channels=1,
                    lstm_layers=1 if USE_CONV_LSTM else 0,
                    freeze_encoder=True,
                    in_channels=train_dataset_for_norm.X.shape[2],
                    use_conv_lstm=USE_CONV_LSTM,
                    use_refiner=False,
                    refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
                ).to(device)
            elif BACKBONE == "mit_b2":
                fold_model = PretrainedTemporalUNetMitB2(
                    out_channels=1,
                    lstm_layers=1 if USE_CONV_LSTM else 0,
                    freeze_encoder=True,
                    in_channels=train_dataset_for_norm.X.shape[2],
                    use_conv_lstm=USE_CONV_LSTM,
                    use_refiner=False,
                    refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
                ).to(device)
            elif BACKBONE == "mit_b3":
                fold_model = PretrainedTemporalUNetMitB3(
                    out_channels=1,
                    lstm_layers=1 if USE_CONV_LSTM else 0,
                    freeze_encoder=True,
                    in_channels=train_dataset_for_norm.X.shape[2],
                    use_conv_lstm=USE_CONV_LSTM,
                    use_refiner=False,
                    refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
                ).to(device)

            if fold_model is None:
                continue

            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get('model_state', ckpt)
            fold_model.load_state_dict(state, strict=False)
            fold_model.eval()

            fold_preds = []
            with torch.no_grad():
                for x, _, _ in test_loader:
                    x = x.to(device)
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        out, _ = fold_model(x)
                    if isinstance(out, list):
                        out = torch.stack(out, dim=1)
                    pred_denorm = train_dataset_for_norm.denormalize(out).cpu().numpy().astype(np.float64)
                    fold_preds.append(pred_denorm)

            if not fold_preds:
                continue

            fold_pred = np.concatenate(fold_preds, axis=0)

            if ensemble_sum is None:
                ensemble_sum = fold_pred
            else:
                ensemble_sum += fold_pred
            models_used += 1

        if models_used > 0 and ensemble_sum is not None:
            ensemble_pred = (ensemble_sum / float(models_used)).astype(np.float32)
            gt_all = np.asarray(test_dataset.Y, dtype=np.float32)

            diff = ensemble_pred - gt_all
            # Use the same metric convention as evaluate(): mask-based if USE_MASK is enabled.
            if USE_MASK:
                all_mask = []
                for _, _, mask in test_loader:
                    all_mask.append(mask.cpu().numpy())
                mask_np = np.concatenate(all_mask, axis=0)
                valid = mask_np.astype(bool)
                if np.any(valid):
                    valid_diff = diff[valid]
                    ens_mae = float(np.mean(np.abs(valid_diff)))
                    ens_rmse = float(np.sqrt(np.mean(valid_diff ** 2)))
                    ens_me = float(np.mean(valid_diff))
                else:
                    ens_mae = ens_rmse = ens_me = 0.0
            else:
                ens_mae = float(np.mean(np.abs(diff)))
                ens_rmse = float(np.sqrt(np.mean(diff ** 2)))
                ens_me = float(np.mean(diff))

            print(f"[ENSEMBLE RESULT] Models used: {models_used}")
            print(f"[ENSEMBLE RESULT] Loss is not recomputed for ensemble; metrics are on mean denormalized predictions.")
            print(f"[ENSEMBLE RESULT] MAE={ens_mae:.4f} | RMSE={ens_rmse:.4f} | ME={ens_me:.4f}")
        else:
            print("[ENSEMBLE] No valid fold checkpoints were available for ensemble evaluation.")

        print("=" * 70)
        sys.exit(0)



    # --- 2. Data Loading ---
    required_npz_paths = [NPZ_TRAIN_PATH, NPZ_VAL_PATH, NPZ_TEST_PATH]
    missing_paths = [p for p in required_npz_paths if not os.path.exists(p)]
    if missing_paths:
        print("ERROR: Missing required split datasets:")
        for p in missing_paths:
            print(f"  - {p}")
        exit(1)

    train_dataset = NPZSequenceDataset(
        NPZ_TRAIN_PATH,
        use_gt_envelope_as_input=USE_ENVELOP_AS_A_INPUT,
        gt_envelope_npz_path=GT_ENVELOPE_NPZ_PATH,
        augment=TRAIN_AUGMENT,
        augment_repeats=3,
        deterministic_aug=True,
        use_one_satellite=USE_ONE_SATELLITE,
    )
    val_dataset = NPZSequenceDataset(
        NPZ_VAL_PATH,
        use_gt_envelope_as_input=USE_ENVELOP_AS_A_INPUT,
        gt_envelope_npz_path=GT_ENVELOPE_NPZ_PATH,
        augment=False,
        use_one_satellite=USE_ONE_SATELLITE
    )
    test_dataset = NPZSequenceDataset(
        NPZ_TEST_PATH,
        use_gt_envelope_as_input=USE_ENVELOP_AS_A_INPUT,
        gt_envelope_npz_path=GT_ENVELOPE_NPZ_PATH,
        augment=False,
        use_one_satellite=USE_ONE_SATELLITE
    )

    # Force train-derived normalization for val/test for consistent denormalized metrics.
    shared_scale = train_dataset.scale
    shared_norm_const = train_dataset.norm_const
    val_dataset.scale = shared_scale
    val_dataset.norm_const = shared_norm_const
    test_dataset.scale = shared_scale
    test_dataset.norm_const = shared_norm_const

    y_raw = np.asarray(train_dataset.Y, dtype=np.float32)
    bin_min = float(np.min(y_raw))
    bin_max = float(np.max(y_raw))
    print(f"[INFO] Derived bin bounds from training data: BIN_MIN={bin_min:.4f}, BIN_MAX={bin_max:.4f}")

    train_ds = train_dataset
    val_ds = val_dataset
    test_ds = test_dataset

    print(f"Train sequences (base): {train_dataset.N}")
    print(f"Train sequences (augmented): {len(train_ds)}")
    print(f"Val sequences: {len(val_ds)}")
    print(f"Test sequences: {len(test_ds)}")
    print(f"[INFO] Shared normalization from train split: norm_const={shared_norm_const:.4f}, scale={shared_scale:.4f}")
    _, in_channels, _, _ = train_ds[0][0].shape

    def make_loaders(batch_size):
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
        return train_loader, val_loader, test_loader

    train_loader, val_loader, test_loader = make_loaders(BATCH_SIZE_START)

    # --- 3. Model Initialization ---
    # Start with Stage 1: No refiner
    if BACKBONE == "resnet18":
        print("[INFO] Initializing Pre-trained ResNet18 Model...")
        model = PretrainedTemporalUNet(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=True,
            in_channels=in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    elif BACKBONE == "mit_b1":
        print("[INFO] Initializing Pre-trained MiT-B1 Model...")
        model = PretrainedTemporalUNetMitB1(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=True,
            in_channels=in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    elif BACKBONE == "mit_b2":
        print("[INFO] Initializing Pre-trained MiT-B2 Model...")
        model = PretrainedTemporalUNetMitB2(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=True,
            in_channels=in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    elif BACKBONE == "mit_b3":
        print("[INFO] Initializing Pre-trained MiT-B3 Model...")
        model = PretrainedTemporalUNetMitB3(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=True,
            in_channels=in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    else:
        raise ValueError(f"Unsupported BACKBONE: {BACKBONE}")

    # Load checkpoint if provided
    if LOAD_CHECKPOINT is not None:
        print(f"[INFO] Loading model weights from {LOAD_CHECKPOINT}")
        checkpoint = torch.load(LOAD_CHECKPOINT, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("[INFO] Model weights loaded.")

    refiner_enabled = False  # Start with no refiner
    current_stage = 1

    def set_encoder_trainable(model_obj, trainable):
        if hasattr(model_obj, "encoder"):
            for param in model_obj.encoder.parameters():
                param.requires_grad = trainable

    set_encoder_trainable(model, False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1,
        weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    print("[INFO] STAGE 1: Encoder FROZEN, Refiner DISABLED")
    print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    best_state = None  # Best for the current stage only
    best_stage1_state = None
    best_stage2_state = None
    best_stage3_state = None
    best_stage1_val_loss = float('inf')
    best_stage2_val_loss = float('inf')
    best_stage3_val_loss = float('inf')
    save_dir = "models/wacv"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting 3-stage training for {EPOCHS} total epochs...")
    print(f"  Stage 1 (epochs 1-{EPOCHS_STAGE1}): Frozen encoder, no refiner (train decoder/LSTM/head)")
    print(f"  Stage 2 (epochs {EPOCHS_STAGE1+1}-{EPOCHS_STAGE1+EPOCHS_STAGE2}): Unfreeze encoder, no refiner (train full model)")
    print(f"  Stage 3 (epochs {EPOCHS_STAGE1+EPOCHS_STAGE2+1}-{EPOCHS}): Refiner enabled, freeze rest of the model\n")

    for epoch in range(1, EPOCHS + 1):
        # Stage 1 -> Stage 2 transition
        if epoch == EPOCHS_STAGE1 + 1:
            print("\n" + "="*70)
            print("[INFO] STAGE 1 -> STAGE 2 TRANSITION")
            print("="*70)
            if best_stage1_state is not None:
                print("[INFO] Loading best Stage 1 weights before Stage 2.")
                model.load_state_dict(best_stage1_state)
            elif best_state is not None:
                print("[INFO] Loading best Stage 1 weights before Stage 2.")
                model.load_state_dict(best_state)
                best_stage1_state = copy.deepcopy(best_state)

            current_stage = 2
            set_encoder_trainable(model, True)
            train_loader, val_loader, test_loader = make_loaders(BATCH_SIZE_FINETUNE)

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_STAGE2,
                weight_decay=WEIGHT_DECAY
            )
            scaler = GradScaler(enabled=(device.type == "cuda"))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            best_val_loss = float('inf')  # Reset for Stage 2
            best_state = None
            print("[INFO] STAGE 2: Encoder TRAINABLE, Refiner DISABLED")
            print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print("="*70 + "\n")

        # Stage 2 -> Stage 3 transition
        if epoch == EPOCHS_STAGE1 + EPOCHS_STAGE2 + 1:
            print("\n" + "="*70)
            print("[INFO] STAGE 2 -> STAGE 3 TRANSITION")
            print("="*70)
            if best_stage2_state is not None:
                print("[INFO] Loading best Stage 2 weights before Stage 3.")
                model.load_state_dict(best_stage2_state)
            elif best_state is not None:
                print("[INFO] Loading best Stage 2 weights before Stage 3.")
                model.load_state_dict(best_state)
                best_stage2_state = copy.deepcopy(best_state)

            current_stage = 3

            # Freeze ALL model parameters first
            print("[INFO] Freezing all model parameters...")
            for param in model.parameters():
                param.requires_grad = False

            # Enable refiner and unfreeze refiner parameters
            if USE_REFINER and not refiner_enabled:
                if hasattr(model, "enable_refiner"):
                    print("[INFO] Enabling refiner...")
                    model.enable_refiner(REFINER_HIDDEN_CHANNELS)
                    refiner_enabled = True
                    print("[INFO] ✓ Refiner ENABLED")

            # Unfreeze only refiner parameters
            if refiner_enabled and hasattr(model, 'refiner') and model.refiner is not None:
                print("[INFO] Unfreezing refiner parameters...")
                for param in model.refiner.parameters():
                    param.requires_grad = True

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_STAGE3,
                weight_decay=WEIGHT_DECAY
            )
            scaler = GradScaler(enabled=(device.type == "cuda"))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            # Do not reset best_val_loss here; keep tracking best across all stages
            best_state = None
            print("[INFO] STAGE 3: Full Model FROZEN, Only Refiner TRAINABLE")
            print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print("="*70 + "\n")


        # --- Loss mode selection ---
        if epoch <= EXPONENT_EPOCHS:
            loss_mode = 'exponent'
            loss_interp = 0.0
        elif epoch <= EXPONENT_EPOCHS + INTERP_EPOCHS:
            loss_mode = 'interp'
            # Linearly increase interpolation from 0 to 1
            loss_interp = (epoch - EXPONENT_EPOCHS) / INTERP_EPOCHS
        else:
            loss_mode = 'bins'
            loss_interp = 1.0

        tr_loss, tr_mae, tr_rmse, tr_me = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_dataset,
            scaler,
            use_mask=USE_MASK,
            unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR,
            debug_bins_once=DEBUG_BINS_ONCE_PER_EPOCH,
            loss_mode=loss_mode,
            loss_interp=loss_interp,
            bin_min=bin_min,
            bin_max=bin_max,
        )

        # --- Loss mode selection for evaluation ---
        if epoch <= EXPONENT_EPOCHS:
            eval_loss_mode = 'exponent'
            eval_loss_interp = 0.0
        elif epoch <= EXPONENT_EPOCHS + INTERP_EPOCHS:
            eval_loss_mode = 'interp'
            eval_loss_interp = (epoch - EXPONENT_EPOCHS) / INTERP_EPOCHS
        else:
            eval_loss_mode = 'bins'
            eval_loss_interp = 1.0

        val_loss, val_mae, val_rmse, val_me = evaluate(
            model,
            val_loader,
            device,
            train_dataset,
            use_mask=USE_MASK,
            unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR,
            loss_mode=eval_loss_mode,
            loss_interp=eval_loss_interp,
            bin_min=bin_min,
            bin_max=bin_max
        )

        # Update scheduler based on Val Loss
        scheduler.step(val_loss)

        # Print rich metrics for both Train and Val
        stage_label = f"Stage {current_stage}"
        print(f"Epoch {epoch}/{EPOCHS} ({stage_label}):")
        print(f"  Train: Loss={tr_loss:.4f} | MAE={tr_mae:.4f} | RMSE={tr_rmse:.4f} | ME={tr_me:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} | MAE={val_mae:.4f} | RMSE={val_rmse:.4f} | ME={val_me:.4f}")

        # Save Best Model (per stage)
        # Save best model only after switching to bin loss
        if loss_mode == 'bins':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                print(f"   -> New best model for bin loss stage! Saving...")
                save_path = os.path.join(save_dir, f"{model_name}_best_bin_loss.pt")
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {
                        'type': BACKBONE,
                        'in_channels': in_channels,
                        'use_one_satellite': USE_ONE_SATELLITE,
                        'stage': current_stage
                    },
                    'val_loss': best_val_loss,
                    'epoch': epoch,
                    'stage': current_stage
                }, save_path)

    if best_stage3_state is not None:
        model.load_state_dict(best_stage3_state)
        final_best_val_loss = best_stage3_val_loss
    # Load best model from bin loss stage for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        final_best_val_loss = best_val_loss
    else:
        final_best_val_loss = float('inf')

    print(f"Training complete. Best Validation Loss: {final_best_val_loss:.6f}")
    test_loss, test_mae, test_rmse, test_me = evaluate(
        model, test_loader, device, train_dataset, use_mask=USE_MASK, unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR, bin_min=bin_min, bin_max=bin_max
    )
    print(f"Test:  Loss={test_loss:.4f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | ME={test_me:.4f}")

    metrics_out_path = os.getenv("METRICS_OUT_PATH", "")
    if metrics_out_path:
        try:
            payload = {
                "fold_name": os.getenv("MODEL_NAME_SUFFIX", "").lstrip("_") or "single",
                "npz_train": NPZ_TRAIN_PATH,
                "npz_val": NPZ_VAL_PATH,
                "npz_test": NPZ_TEST_PATH,
                "test_loss": float(test_loss),
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "test_me": float(test_me),
            }
            out_dir = os.path.dirname(metrics_out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(metrics_out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[INFO] Wrote fold metrics to {metrics_out_path}")
        except Exception as e:
            print(f"[WARN] Failed writing metrics file: {e}")
