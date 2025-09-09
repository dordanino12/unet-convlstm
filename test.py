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
    "temporal_unet_convlstm_dualview.pt",
    map_location="cuda" if torch.cuda.is_available() else "cpu"
)
cfg = checkpoint['cfg']

# Recreate the model and load its state
model = TemporalUNetDualView(
    in_channels_per_sat=cfg['in_channels_per_sat'],
    out_channels=cfg['out_channels'],
    use_skip_lstm =cfg['use_skip_lstm']
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# -----------------------------
# Load the dataset
# -----------------------------
class MovingMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        npz_file = np.load(path)
        self.data = npz_file["data"]  # (N, T, 2, H, W)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx])  # (T, 2, H, W)
        x = seq[:, 0:1].repeat(1, 2, 1, 1)  # duplicate channel for two “satellites”
        y = seq[:, 1:2]  # velocity map
        return x, y


dataset = MovingMNISTDataset("moving_mnist_2.npz")

# -----------------------------
# Select a sequence
# -----------------------------
sequence_idx = 5 # index of the sequence to test
input_seq, gt_vel_seq = dataset[sequence_idx]  # (T, 2, H, W), (T, 1, H, W)
T = input_seq.shape[0]

max_vel = 5.0  # denormalization factor

# -----------------------------
# Run inference incrementally
# -----------------------------
for t_len in range(1, T + 1):
    # Take the first t_len frames
    x_input = input_seq[:t_len].unsqueeze(0).to(device)  # [1, t_len, 2, H, W]

    with torch.no_grad():
        predicted_seq, _ = model(x_input)   # predicted_seq: [T, B, C, H, W] or similar

    # -----------------------------
    # Extract mask from the last frame of the input
    # -----------------------------
    # Here, using the first channel of the last frame as mask (any pixel > 0 is "object")
    mask = (x_input[0, t_len-1, 0] > 0).float().cpu().numpy()   # shape [H, W]

    # Last frame in the input sequence (for display)
    last_frame_digit = input_seq[t_len - 1, 0].cpu().numpy()  # first channel
    last_frame_gt_vel = gt_vel_seq[t_len - 1, 0].cpu().numpy()

    # Predicted velocity for last frame
    pred_vel = predicted_seq[-1].squeeze(0).cpu().numpy()[0]  # [H, W]
    pred_vel = pred_vel * max_vel  # denormalize

    # -----------------------------
    # Apply mask on predicted velocity
    # -----------------------------
    pred_vel_masked = pred_vel * mask   # zero out the background

    # -----------------------------
    # Plot results with colorbar
    # -----------------------------
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f"Number of Frames in the Sequence: {t_len}", fontsize=14)

    # Last input frame
    axes[0].imshow(last_frame_digit, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Last Input Frame")
    axes[0].axis('off')

    # Ground-truth velocity
    im1 = axes[1].imshow(last_frame_gt_vel, cmap='hot', vmin=-max_vel, vmax=max_vel)
    axes[1].set_title("GT Velocity")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Predicted velocity (raw)
    im2 = axes[2].imshow(pred_vel, cmap='hot', vmin=-max_vel, vmax=max_vel)
    axes[2].set_title("Predicted Velocity (Raw)")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Predicted velocity (masked)
    im3 = axes[3].imshow(pred_vel_masked, cmap='hot', vmin=-max_vel, vmax=max_vel)
    axes[3].set_title("Predicted Velocity (Masked)")
    axes[3].axis('off')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()