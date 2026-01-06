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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.fft

# --- Local Imports ---
# Ensure these files exist in the 'train' folder
from train.unet import TemporalUNetDualView, NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet

# -----------------------------------------------------
# Loss Function: Weighted L1 + Gradient Loss
# -----------------------------------------------------
def compute_loss(y_pred, y, mask=None, use_mask=True):
    """
    Computes weighted L1 loss and spatial gradient loss.
    - Penalizes high-velocity errors more heavily.
    - Ensures numerical stability with epsilon.
    """
    # 1. Weighted L1
    abs_diff = (y_pred - y).abs()

    # Weight scales linearly with velocity (Cubic increase)
    weight = 1.0 + 4.0 * (y.abs() ** 3)

    if use_mask and mask is not None:
        numerator = (abs_diff * mask * weight).sum()
        denominator = (mask * weight).sum() + 1e-8
        weighted_l1 = numerator / denominator
    else:
        weighted_l1 = (abs_diff * weight).mean()

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

    if use_mask and mask is not None:
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
def train_one_epoch(model, loader, optimizer, device, dataset_obj, use_mask=True):
    model.train()
    total_loss, n = 0.0, 0
    
    # Accumulators for training metrics
    all_mae = []
    all_sq_err = [] 
    all_err = []

    #for x, y, mask in loader: #tqdm(loader, desc="Training", leave=False):
    for x, y, mask in tqdm(loader, desc="Training", leave=False):

        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        output, _ = model(x)
        
        # Compatibility handling
        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output
            
        loss = compute_loss(y_pred, y, mask, use_mask)
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
def evaluate(model, loader, device, dataset_obj, use_mask=True):
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
        loss = compute_loss(y_pred, y, mask, use_mask)
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
    USE_PRETRAINED = True  # Set False for Custom Model, True for ResNet
    
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    USE_MASK = False
    min_y = None  # 7.5987958908081055
    max_y = None  # 8.784920692443848
    NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_500m_slices_w.npz"
    
    CUSTOM_CFG = {
        'base_ch': 64,
        'use_attention': False,
        'use_skip_lstm': True
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # --- 2. Data Loading ---
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: Dataset not found at {NPZ_PATH}")
        exit(1)

    dataset = NPZSequenceDataset(NPZ_PATH, min_y=min_y, max_y=max_y)
    print(f"Dataset length: {len(dataset)}")
    
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    
    torch.manual_seed(42)  # Ensure reproducibility
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # --- 3. Model Initialization ---
    if USE_PRETRAINED:
        print("[INFO] Initializing Pre-trained ResNet18 Model...")
        model = PretrainedTemporalUNet(
            out_channels=1,
            lstm_layers=2,
            freeze_encoder=True 
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=LR, 
            weight_decay=WEIGHT_DECAY
        )
        model_name = "resnet18_frozen_2lstm_layers_500m_slice"
        
    else:
        print("[INFO] Initializing Custom Temporal U-Net...")
        model = TemporalUNetDualView(
            in_channels_per_sat=1,
            out_channels=1,
            base_ch=CUSTOM_CFG['base_ch'],
            lstm_layers=1,
            use_skip_lstm=CUSTOM_CFG['use_skip_lstm'],
            use_attention=CUSTOM_CFG['use_attention']
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        model_name = "custom_unet_64ch"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_mae, tr_rmse, tr_me = train_one_epoch(
            model, train_loader, optimizer, device, dataset, use_mask=USE_MASK
        )
        
        val_loss, val_mae, val_rmse, val_me = evaluate(
            model, val_loader, device, dataset, use_mask=USE_MASK
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
            print(f"   -> New best model! Saving...")
            
            save_path = os.path.join(save_dir, f"{model_name}_best_skip.pt")
            
            if USE_PRETRAINED:
                saved_cfg = {'type': 'resnet18', 'freeze_encoder': True}
            else:
                saved_cfg = {'type': 'custom', **CUSTOM_CFG}

            torch.save({
                'model_state': model.state_dict(),
                'config': saved_cfg,
                'val_loss': best_val_loss,
                'epoch': epoch
            }, save_path)

    print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")