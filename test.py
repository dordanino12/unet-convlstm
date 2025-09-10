import torch
import numpy as np
from main import TemporalUNetDualView
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# Load the model checkpoint
# -----------------------------
checkpoint = torch.load(
    "temporal_unet_convlstm_dualview_from_npz.pt",
    map_location="cuda" if torch.cuda.is_available() else "cpu",
    weights_only=True
)

cfg = checkpoint['cfg']

# Recreate the model and load its state
model = TemporalUNetDualView(
    in_channels_per_sat=cfg['in_channels_per_sat'],
    out_channels=cfg['out_channels'],
    use_skip_lstm=cfg['use_skip_lstm']
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# Load the NPZ dataset
# -----------------------------
class NPZSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, min_vel=-8.79623031616211, max_vel=10.423197746276855):
        data = np.load(npz_path)
        self.X = data["X"]  # [N, T, 2, H, W]
        self.Y = data["Y"]  # [N, T, 1, H, W]
        self.min_vel = min_vel
        self.max_vel = max_vel

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # [T, 2, H, W]
        y = torch.from_numpy(self.Y[idx]).float()  # [T, 1, H, W]
        # Normalize velocities to [-1,1]
        y = 2 * (y - self.min_vel) / (self.max_vel - self.min_vel) - 1
        mask = ((x[:, 0:1] > 0.12) | (x[:, 1:2] > 0.12)).float()
        return x, y, mask

npz_path = "dataset_sequences_original.npz"
dataset = NPZSequenceDataset(npz_path)
sequence_idx = 10
input_seq, gt_vel_seq, mask_seq = dataset[sequence_idx]
T, C, H, W = input_seq.shape

# -----------------------------
# Denormalize GT velocities for the whole sequence
# -----------------------------
gt_vel_denorm = 0.5 * (gt_vel_seq + 1) * (dataset.max_vel - dataset.min_vel) + dataset.min_vel

# -----------------------------
# Run inference incrementally
# -----------------------------
for t_len in range(1, T+1):
    x_input = input_seq[:t_len].unsqueeze(0).to(device)  # [1, t_len, 2, H, W]
    with torch.no_grad():
        pred_seq, _ = model(x_input)
    pred_vel = torch.stack(pred_seq, dim=1).squeeze(0).cpu().numpy()  # [T,1,H,W]

    # Denormalize predicted velocities
    pred_vel_denorm = 0.5 * (pred_vel + 1) * (dataset.max_vel - dataset.min_vel) + dataset.min_vel

    # Last frame for display
    last_input = input_seq[t_len-1,0].cpu().numpy()
    last_gt = gt_vel_denorm[t_len-1, 0].cpu().numpy()
    last_pred = pred_vel_denorm[t_len-1,0]
    last_mask = mask_seq[t_len-1,0].cpu().numpy()
    last_pred_masked = last_pred * last_mask

    # -----------------------------
    # Determine vmin/vmax for this sequence
    # -----------------------------
    vmin_seq = min(last_gt.min(), last_pred.min())
    vmax_seq = max(last_gt.max(), last_pred.max())

    # -----------------------------
    # Plot results with both input channels
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Sequence frames: {t_len}", fontsize=14)

    # Display both input channels
    axes[0, 0].imshow(input_seq[t_len - 1, 0].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("Last Input Frame (Sat 1)")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(input_seq[t_len - 1, 1].cpu().numpy(), cmap='gray')
    axes[1, 0].set_title("Last Input Frame (Sat 2)")
    axes[1, 0].axis('off')

    # Ground-truth velocity
    im1 = axes[0, 1].imshow(last_gt, cmap='jet', vmin=vmin_seq, vmax=vmax_seq)
    axes[0, 1].set_title("GT Velocity")
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # # Predicted velocity (raw)
    # im2 = axes[0, 2].imshow(last_pred, cmap='jet', vmin=vmin_seq, vmax=vmax_seq)
    # axes[0, 2].set_title("Predicted Velocity (Raw)")
    # axes[0, 2].axis('off')
    # fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Predicted velocity (masked)
    im3 = axes[0, 2].imshow(last_pred_masked, cmap='jet', vmin=vmin_seq, vmax=vmax_seq)
    axes[0, 2].set_title("Predicted Velocity (Masked)")
    axes[0, 2].axis('off')
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Leave second row for second input channel only (or replicate other plots if desired)
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    #axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()

