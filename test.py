import torch
import numpy as np
from main import TemporalUNetDualView  # Ensure this matches your file structure
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the model checkpoint
checkpoint = torch.load(
    "temporal_unet_convlstm_dualview_l1_mask.pt",
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

# Load the dataset
class MovingMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        npz_file = np.load(path)
        self.data = npz_file["data"]  # (N, T, 2, H, W)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx])  # (T, 2, H, W)

        # Input: digit sequence (first channel)
        x = seq[:-1, 0:1]  # (T-1, 1, H, W)

        # duplicate channel to simulate two cameras
        x = x.repeat(1, 2, 1, 1)  # (T-1, 2, H, W)

        # Target: next digit frame (first channel at t+1)
        y = seq[1:, 0:1]  # (T-1, 1, H, W)

        return x, y

dataset = MovingMNISTDataset("moving_mnist.npz")

# Select a sequence
sequence_idx = 5
input_seq, gt_seq = dataset[sequence_idx]  # (T-1, 2, H, W), (T-1, 1, H, W)

# Use the first 15 frames as input
input_seq = input_seq[:17].unsqueeze(0)  # (1, 15, 2, H, W)
gt_next_frame = gt_seq[17].squeeze(0).numpy()  # The ground-truth next frame after those 15 frames → (H, W)

# Predict the next frame
with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_seq = input_seq.to(device)
    predicted_seq, _ = model(input_seq)  # list or tensor of predictions

# Extract the predicted next frame
predicted_next_frame = predicted_seq[-1].squeeze(0).cpu().numpy()  # (2, H, W)
predicted_next_frame = predicted_next_frame[0]  # take first channel

# Show GT and Prediction side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(gt_next_frame, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Frame")
axes[0].axis('off')

axes[1].imshow(predicted_next_frame, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("Predicted Frame velocity map")
axes[1].axis('off')

plt.tight_layout()
plt.show()
