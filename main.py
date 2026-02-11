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
from torch.utils.data import DataLoader
import numpy as np
import torch.fft

# --- Local Imports ---
from train.dataset import NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet

# -----------------------------------------------------
# Loss Function: Weighted L1 + Gradient Loss
# -----------------------------------------------------
def compute_loss(y_pred, y, mask=None, use_mask=True, dataset_obj=None, unmasked_weight_factor=0.1):
    """
    Computes weighted L1 loss and spatial gradient loss.
    Supports three mask types:
    - True (binary mask): Use mask as-is
    - False (no mask): No masking applied
    - "slice_mask": Extract mask from time step 5 and broadcast to all frames
                    Apply full weight to masked areas, reduced weight to unmasked areas

    Args:
        y_pred: Predicted values (B, T, H, W) or (B, T, C, H, W)
        y: Ground truth values (same shape as y_pred)
        mask: Mask tensor (B, T, H, W) or string "slice_mask"
        use_mask: True/False/"slice_mask"
        dataset_obj: Dataset object for denormalization
        unmasked_weight_factor: Weight multiplier for unmasked areas (default 0.1)
    """
    abs_diff = (y_pred - y).abs()
    weight = torch.exp(8.0 * y.abs())

    # Initialize mask variables for gradient loss section
    mask_broadcasted = None
    spatial_mask = None

    if use_mask == "slice_mask" and mask is not None:
        # Extract mask from time step 5 and broadcast to all T frames
        mask_slice_5 = mask[:, 5:6, :, :]  # Shape: (B, 1, H, W)
        mask_broadcasted = mask_slice_5.expand_as(mask)  # Broadcast to (B, T, H, W)

        # Create weighted mask: full weight for masked areas, reduced weight for unmasked
        spatial_mask = torch.ones_like(mask_broadcasted)
        spatial_mask[mask_broadcasted > 0.5] = 1.0  # Full weight for masked areas
        spatial_mask[mask_broadcasted <= 0.5] = unmasked_weight_factor  # Reduced weight for unmasked

        # Combine with velocity-based weight
        combined_weight = weight * spatial_mask

        numerator = (abs_diff * combined_weight).sum()
        denominator = combined_weight.sum() + 1e-8
        weighted_l1 = numerator / denominator

    elif use_mask is True and mask is not None:
        # Binary mask mode: original behavior
        numerator = (abs_diff * mask * weight).sum()
        denominator = (mask * weight).sum() + 1e-8
        weighted_l1 = numerator / denominator
    else:
        # No mask mode
        weighted_l1 = (abs_diff * weight).sum() / (weight.sum() + 1e-8)

    # 2. Gradient Loss (Spatial smoothness and edge preservation)
    def spatial_gradients(tensor):
        dx = tensor[..., :, 1:] - tensor[..., :, :-1]
        dy = tensor[..., 1:, :] - tensor[..., :-1, :]
        return dx, dy

    dx_pred, dy_pred = spatial_gradients(y_pred)
    dx_gt, dy_gt = spatial_gradients(y)

    # Crop to smallest spatial dim to avoid shape mismatch
    H_min = min(dx_pred.shape[3], dy_pred.shape[3])
    W_min = min(dx_pred.shape[4], dy_pred.shape[4])

    # Calculate gradient differences
    grad_diff = (dx_pred[..., :H_min, :W_min] - dx_gt[..., :H_min, :W_min]).abs() + \
                (dy_pred[..., :H_min, :W_min] - dy_gt[..., :H_min, :W_min]).abs()

    if use_mask == "slice_mask" and mask is not None:
        # Use the broadcasted spatial mask for gradient loss
        mask_c = mask_broadcasted[..., :H_min, :W_min]
        spatial_mask_c = spatial_mask[..., :H_min, :W_min]
        grad_loss = (grad_diff * spatial_mask_c).sum() / (spatial_mask_c.sum() + 1e-8)
    elif use_mask is True and mask is not None:
        mask_c = mask[..., :H_min, :W_min]
        grad_loss = (grad_diff * mask_c).sum() / (mask_c.sum() + 1e-8)
    else:
        grad_loss = grad_diff.mean()

    # Combine losses (0.005 weight for gradients)
    total_loss = weighted_l1 + 0.005 * grad_loss
    return total_loss

# -----------------------------------------------------
# Training Loop
# -----------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, dataset_obj, use_mask=True, unmasked_weight_factor=0.1):
    model.train()
    total_loss, n = 0.0, 0
    
    # Accumulators for training metrics
    all_mae = []
    all_sq_err = [] 
    all_err = []

    #for x, y, mask in loader: #tqdm(loader, desc="Training", leave=False):
    for x, y, mask in loader:

        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        output, _ = model(x)
        
        # Compatibility handling
        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output
            
        loss = compute_loss(y_pred, y, mask, use_mask, dataset_obj, unmasked_weight_factor)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
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
        
        output, _ = model(x)
        
        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output
            
        # 1. Calc Loss (Normalized space)
        loss = compute_loss(y_pred, y, mask, use_mask, dataset_obj, unmasked_weight_factor)
        total_loss += loss.item() * x.size(0)
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
    BATCH_SIZE = 32
    EPOCHS = 250
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    FREEZE_ENCODER = False  # True: Freeze encoder (faster, less memory), False: Train encoder (slower, more capacity)
    USE_MASK = "slice_mask"  # True, False, or "slice_mask"
    UNMASKED_WEIGHT_FACTOR = 0.2  # Weight multiplier for unmasked areas in slice_mask mode
    NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_W_500m_w.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # --- 2. Data Loading ---
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: Dataset not found at {NPZ_PATH}")
        exit(1)

    dataset = NPZSequenceDataset(NPZ_PATH)
    print(f"Dataset length: {len(dataset)}")

    torch.manual_seed(42)  # Ensure reproducibility
    g = torch.Generator().manual_seed(42)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=g
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # --- 3. Model Initialization ---
    print("[INFO] Initializing Pre-trained ResNet18 Model...")
    model = PretrainedTemporalUNet(
        out_channels=1,
        lstm_layers=2,
        freeze_encoder=FREEZE_ENCODER
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Model name reflects encoder state
    encoder_state = "frozen" if FREEZE_ENCODER else "trainable"
    model_name = f"resnet18_{encoder_state}_2lstm_layers_500m"

    print(f"[INFO] Encoder is {'FROZEN' if FREEZE_ENCODER else 'TRAINABLE'}")
    print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    best_state = None
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_mae, tr_rmse, tr_me = train_one_epoch(
            model, train_loader, optimizer, device, dataset, use_mask=USE_MASK, unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR
        )
        
        val_loss, val_mae, val_rmse, val_me = evaluate(
            model, val_loader, device, dataset, use_mask=USE_MASK, unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR
        )
        
        # Update scheduler based on Val Loss
        scheduler.step(val_loss)
        
        # Print rich metrics for both Train and Val
        print(f"Epoch {epoch}/{EPOCHS}:")
        print(f"  Train: Loss={tr_loss:.4f} | MAE={tr_mae:.4f} | RMSE={tr_rmse:.4f} | ME={tr_me:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} | MAE={val_mae:.4f} | RMSE={val_rmse:.4f} | ME={val_me:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"   -> New best model! Saving...")
            
            save_path = os.path.join(save_dir, f"{model_name}_best_skip.pt")
            
            saved_cfg = {'type': 'resnet18', 'freeze_encoder': FREEZE_ENCODER}

            torch.save({
                'model_state': model.state_dict(),
                'config': saved_cfg,
                'val_loss': best_val_loss,
                'epoch': epoch
            }, save_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae, test_rmse, test_me = evaluate(
        model, test_loader, device, dataset, use_mask=USE_MASK, unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR
    )

    print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")
    print(f"Test:  Loss={test_loss:.4f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | ME={test_me:.4f}")