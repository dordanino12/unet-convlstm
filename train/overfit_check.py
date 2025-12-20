import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import both model classes
from train.unet import NPZSequenceDataset, TemporalUNetDualView
from train.resnet18 import PretrainedTemporalUNet

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- TOGGLE MODEL TYPE HERE ---
USE_PRETRAINED = True  # Set True for ResNet18, False for Custom UNet
# ------------------------------

npz_path = os.path.join(parent_dir, "data/dataset_trajectory_sequences_samples.npz")
if not os.path.exists(npz_path):
    npz_path = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples.npz"

# ---------------------------------------------------------
# LOGIC
# ---------------------------------------------------------
def run_overfit_test_and_save():
    # 1. Load Full Dataset
    dataset = NPZSequenceDataset(npz_path)
    total_samples = len(dataset)
    
    # 2. Select specific indices manually
    num_samples = 16
    selected_indices = np.random.choice(total_samples, num_samples, replace=False)
    print(f"\n[INFO] Selected Sequence Indices for Training: {selected_indices}")
    
    # Create a subset with only these indices
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
    
    # Get the single batch of data
    batch = next(iter(loader)) 
    x, y, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)

    print(f"Batch Shapes -> X: {x.shape}, Y: {y.shape}")

    # 3. Initialize Model based on Flag
    if USE_PRETRAINED:
        print("[INFO] Initializing Pre-trained ResNet18 Model...")
        model = PretrainedTemporalUNet(
            out_channels=1, 
            lstm_layers=1, 
            freeze_encoder=True 
        ).to(device)
        
        # Optimize only trainable parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=1e-3, weight_decay=1e-4
        )
        model_type_str = 'resnet18'
        save_cfg = {'type': 'resnet18', 'freeze_encoder': True, 'lstm_layers': 1}

    else:
        print("[INFO] Initializing Custom Temporal U-Net...")
        model = TemporalUNetDualView(
            in_channels_per_sat=1,
            out_channels=1,
            base_ch=64,
            lstm_layers=1,
            use_skip_lstm=True,
            use_attention=False
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model_type_str = 'custom'
        save_cfg = {'type': 'custom', 'base_ch': 64, 'use_skip_lstm': True, 'use_attention': False}

    print(f"\n--- Starting Overfit Test ({model_type_str}) ---")
    
    # Run for many iterations to force memorization
    for i in range(3001): 
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(x)
        
        # Compatibility handling:
        # Custom model returns a LIST of tensors -> needs stack
        # ResNet model returns a TENSOR -> no stack needed
        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output

        # Simple MSE Loss on mask for overfitting check
        diff = (y_pred - y) ** 2
        loss = (diff * mask).sum() / (mask.sum() + 1e-6)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Iter {i:04d} | Loss: {loss.item():.6f}")

            # If converged, save and exit
            if loss.item() < 0.0005:
                print("\n[SUCCESS] Loss is near zero!")
                
                # --- SAVE LOGIC ---
                filename = f"overfitted_{model_type_str}_100_model.pt"
                save_path = os.path.join(current_dir, filename)
                
                print(f"[INFO] Saving model to: {save_path}")
                print(f"[INFO] This model is overfitted on indices: {selected_indices}")
                
                torch.save({
                    'model_state': model.state_dict(),
                    'config': save_cfg,
                    'indices': selected_indices
                }, save_path)
                
                return

    print("\n[WARNING] Did not reach perfect convergence, but saving anyway.")
    torch.save({
        'model_state': model.state_dict(),
        'config': save_cfg,
        'train_indices': selected_indices
    }, os.path.join(current_dir, f"overfitted_{model_type_str}_failed.pt"))

if __name__ == "__main__":
    run_overfit_test_and_save()