import torch
import numpy as np
from main import TemporalUNetDualView
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    out_channels=cfg['out_channels']
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


dataset = MovingMNISTDataset("moving_mnist.npz")

# -----------------------------
# Select a sequence
# -----------------------------
sequence_idx = 105
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
        predicted_seq, _ = model(x_input)

    # Last frame in the input sequence
    last_frame_digit = input_seq[t_len - 1, 0].cpu().numpy()  # first channel
    last_frame_gt_vel = gt_vel_seq[t_len - 1, 0].cpu().numpy()

    # Predicted velocity for last frame
    pred_vel = predicted_seq[-1].squeeze(0).cpu().numpy()[0]  # [H, W]
    pred_vel = pred_vel * max_vel  # denormalize

    # -----------------------------
    # Plot results with colorbar
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
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

    # Predicted velocity
    im2 = axes[2].imshow(pred_vel, cmap='hot', vmin=-max_vel, vmax=max_vel)
    axes[2].set_title("Predicted Velocity")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
