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
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
import numpy as np
import torch.fft
import math

# --- Local Imports ---
from train.dataset import NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet, PretrainedTemporalUNetMitB1, PretrainedTemporalUNetMitB2, PretrainedTemporalUNetMitB3

# -----------------------------------------------------
# Loss Function: Weighted L1 + Gradient Loss
# -----------------------------------------------------
def compute_loss(y_pred, y, mask=None, use_mask=True, dataset_obj=None, unmasked_weight_factor=0.1, debug_bins=False):

    abs_diff = (y_pred - y).abs()
    BIN_MIN = -7.60
    BIN_MAX = 8.78
    BIN_WIDTH = 0.5
    NUM_BINS = int(math.ceil((BIN_MAX - BIN_MIN) / BIN_WIDTH))

    # Prepare mask for binning (masked pixels only)
    mask_broadcasted = None
    spatial_mask = None
    mask_for_bins = None
    if use_mask == "slice_mask" and mask is not None:
        mask_slice_5 = mask[:, 5:6, :, :]
        mask_broadcasted = mask_slice_5.expand_as(mask)
        mask_for_bins = mask_broadcasted > 0.5
    elif use_mask is True and mask is not None:
        mask_for_bins = mask > 0.5

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
            mask_slice_5 = mask[:, 5:6, :, :]
            mask_broadcasted = mask_slice_5.expand_as(mask)

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
    total_loss = weighted_l1 + 0.005 * grad_loss
    return total_loss

# -----------------------------------------------------
# Training Loop
# -----------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, dataset_obj, scaler, use_mask=True, unmasked_weight_factor=0.1, debug_bins_once=False):
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
        loss = compute_loss(
            y_pred,
            y,
            mask,
            use_mask,
            dataset_obj,
            unmasked_weight_factor,
            debug_bins=debug_bins
        )
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
def evaluate(model, loader, device, dataset_obj, use_mask=True, unmasked_weight_factor=0.1):
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
            
        # 1. Calc Loss (Normalized space)
        loss = compute_loss(y_pred, y, mask, use_mask, dataset_obj, unmasked_weight_factor)
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
    # --- 1. Global Configuration ---
    BATCH_SIZE_START = 32
    BATCH_SIZE_FINETUNE = 16

    # 3-Stage Training Configuration
    EPOCHS_STAGE1 = 1   # Stage 1: Frozen encoder, no refiner (train decoder/LSTM/head only)
    EPOCHS_STAGE2 = 1   # Stage 2: Unfreeze encoder, no refiner (train full model except refiner)
    EPOCHS_STAGE3 = 1   # Stage 3: Freeze full model, train only refiner (fine-tune predictions)
    EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2 + EPOCHS_STAGE3

    LR_STAGE1 = 1e-3
    LR_STAGE2 = 3e-4
    LR_STAGE3 = 1e-4
    WEIGHT_DECAY = 1e-4

    BACKBONE = "mit_b2"  # "resnet18", "mit_b1", "mit_b2", or "mit_b3"
    USE_MASK = True  # True, False, or "slice_mask"
    USE_ENVELOP_AS_A_INPUT = False  # Whether to feed GT envelope velocity as an extra input channel
    UNMASKED_WEIGHT_FACTOR = 0.9  # Weight multiplier for unmasked areas in slice_mask mode
    TRAIN_AUGMENT = False
    NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_W_top_w.npz"
    GT_ENVELOPE_NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_W_top_w.npz"
    model_name = f"{BACKBONE}_envelop"

    # Refiner config
    USE_REFINER = True
    REFINER_HIDDEN_CHANNELS = 32

    # Bin debug logging
    DEBUG_BINS_ONCE_PER_EPOCH = False  # Set to False to disable bin count logging


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- 2. Data Loading ---
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: Dataset not found at {NPZ_PATH}")
        exit(1)

    dataset = NPZSequenceDataset(
        NPZ_PATH,
        use_gt_envelope_as_input=USE_ENVELOP_AS_A_INPUT,
        gt_envelope_npz_path=GT_ENVELOPE_NPZ_PATH,
        augment=False
    )
    train_dataset = NPZSequenceDataset(
        NPZ_PATH,
        use_gt_envelope_as_input=USE_ENVELOP_AS_A_INPUT,
        gt_envelope_npz_path=GT_ENVELOPE_NPZ_PATH,
        augment=TRAIN_AUGMENT,
        augment_repeats=3,
        deterministic_aug=True
    )
    print(f"Dataset length: {len(dataset)}")
    _, in_channels, _, _ = dataset[0][0].shape

    torch.manual_seed(42)  # Ensure reproducibility
    g = torch.Generator().manual_seed(42)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    print(f"Train sequences (base): {n_train}")

    perm = torch.randperm(n_total, generator=g).tolist()
    base_train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    if train_dataset.augment and train_dataset.deterministic_aug:
        train_idx = []
        for r in range(train_dataset.augment_repeats):
            train_idx.extend([i + r * n_total for i in base_train_idx])
    else:
        train_idx = list(base_train_idx)

    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    print(f"Train sequences (augmented): {len(train_ds)}")
    print(f"Val sequences: {len(val_ds)}")
    print(f"Test sequences: {len(test_ds)}")

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
            lstm_layers=1,
            freeze_encoder=True,
            in_channels=in_channels,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    elif BACKBONE == "mit_b1":
        print("[INFO] Initializing Pre-trained MiT-B1 Model...")
        model = PretrainedTemporalUNetMitB1(
            out_channels=1,
            lstm_layers=1,
            freeze_encoder=True,
            in_channels=in_channels,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    elif BACKBONE == "mit_b2":
        print("[INFO] Initializing Pre-trained MiT-B2 Model...")
        model = PretrainedTemporalUNetMitB2(
            out_channels=1,
            lstm_layers=1,
            freeze_encoder=True,
            in_channels=in_channels,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    elif BACKBONE == "mit_b3":
        print("[INFO] Initializing Pre-trained MiT-B3 Model...")
        model = PretrainedTemporalUNetMitB3(
            out_channels=1,
            lstm_layers=2,
            freeze_encoder=True,
            in_channels=in_channels,
            use_refiner=False,  # Stage 1: No refiner
            refiner_hidden_channels=REFINER_HIDDEN_CHANNELS
        ).to(device)
    else:
        raise ValueError(f"Unsupported BACKBONE: {BACKBONE}")

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
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
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
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting 3-stage training for {EPOCHS} total epochs...")
    print(f"  Stage 1 (epochs 1-{EPOCHS_STAGE1}): Frozen encoder, no refiner (train decoder/LSTM/head)")
    print(f"  Stage 2 (epochs {EPOCHS_STAGE1+1}-{EPOCHS_STAGE1+EPOCHS_STAGE2}): Unfreeze encoder, no refiner (train full model)")
    print(f"  Stage 3 (epochs {EPOCHS_STAGE1+EPOCHS_STAGE2+1}-{EPOCHS}): Freeze full model, train only refiner\n")

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
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
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

            # Enable refiner and unfreeze only refiner parameters
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
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            best_val_loss = float('inf')  # Reset for Stage 3
            best_state = None
            print("[INFO] STAGE 3: Full Model FROZEN, Only Refiner TRAINABLE")
            print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print("="*70 + "\n")

        tr_loss, tr_mae, tr_rmse, tr_me = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            dataset,
            scaler,
            use_mask=USE_MASK,
            unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR,
            debug_bins_once=DEBUG_BINS_ONCE_PER_EPOCH
        )

        val_loss, val_mae, val_rmse, val_me = evaluate(
            model, val_loader, device, dataset, use_mask=USE_MASK, unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR
        )

        # Update scheduler based on Val Loss
        scheduler.step(val_loss)

        # Print rich metrics for both Train and Val
        stage_label = f"Stage {current_stage}"
        print(f"Epoch {epoch}/{EPOCHS} ({stage_label}):")
        print(f"  Train: Loss={tr_loss:.4f} | MAE={tr_mae:.4f} | RMSE={tr_rmse:.4f} | ME={tr_me:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} | MAE={val_mae:.4f} | RMSE={val_rmse:.4f} | ME={val_me:.4f}")

        # Save Best Model (per stage)
        if val_loss < best_val_loss:
            should_save = True
            if current_stage == 2:
                should_save = val_loss < best_stage1_val_loss
            elif current_stage == 3:
                should_save = val_loss < best_stage2_val_loss

            if should_save:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                if current_stage == 1:
                    best_stage1_state = copy.deepcopy(best_state)
                    best_stage1_val_loss = best_val_loss
                elif current_stage == 2:
                    best_stage2_state = copy.deepcopy(best_state)
                    best_stage2_val_loss = best_val_loss
                elif current_stage == 3:
                    best_stage3_state = copy.deepcopy(best_state)
                    best_stage3_val_loss = best_val_loss
                print(f"   -> New best model for {stage_label}! Saving...")

                save_path = os.path.join(save_dir, f"{model_name}_best_skip.pt")

                torch.save({
                    'model_state': model.state_dict(),
                    'config': {'type': BACKBONE, 'in_channels': in_channels, 'stage': current_stage},
                    'val_loss': best_val_loss,
                    'epoch': epoch,
                    'stage': current_stage
                }, save_path)

    if best_stage3_state is not None:
        model.load_state_dict(best_stage3_state)
        final_best_val_loss = best_stage3_val_loss
    elif best_stage2_state is not None:
        model.load_state_dict(best_stage2_state)
        final_best_val_loss = best_stage2_val_loss
    elif best_stage1_state is not None:
        model.load_state_dict(best_stage1_state)
        final_best_val_loss = best_stage1_val_loss
    elif best_state is not None:
        model.load_state_dict(best_state)
        final_best_val_loss = best_val_loss
    else:
        final_best_val_loss = float('inf')

    test_loss, test_mae, test_rmse, test_me = evaluate(
        model, test_loader, device, dataset, use_mask=USE_MASK, unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR
    )

    print(f"Training complete. Best Validation Loss: {final_best_val_loss:.6f}")
    print(f"Test:  Loss={test_loss:.4f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | ME={test_me:.4f}")
