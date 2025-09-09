import torch
from torch.utils.data import Dataset
import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -----------------------------
# Dataset for PKL sequences
# -----------------------------

class PKLSequenceDataset(Dataset):
    def __init__(self, folder, seq_len=20, overlap=10):
        self.seq_len = seq_len
        self.stride = seq_len - overlap
        self.files = sorted(glob.glob(os.path.join(folder, "*.pkl")))

        # organize by location and time index
        # loc_dict['0_0'] = {2000: 'full_path_to_file', 2020: 'full_path_to_file', ...}
        self.loc_dict = {}
        for f in self.files:
            base = os.path.basename(f)
            parts = base.split('_')
            time_idx = int(parts[-3])
            loc_idx = f"{parts[-2]}_{parts[-1].split('.')[0]}"  # e.g., '0_0'
            if loc_idx not in self.loc_dict:
                self.loc_dict[loc_idx] = {}
            self.loc_dict[loc_idx][time_idx] = f

        # sort timesteps for each location
        for loc in self.loc_dict:
            self.loc_dict[loc] = dict(sorted(self.loc_dict[loc].items()))

        # build sequences with stride for each location
        self.sequences = []  # list of dicts: {'loc': loc_idx, 'files': [paths]}
        self.seq_time_indices = []  # corresponding time indices
        for loc, time_file_dict in self.loc_dict.items():
            times_sorted = sorted(time_file_dict.keys())
            for i in range(0, len(times_sorted) - seq_len + 1, self.stride):
                seq_times = times_sorted[i:i + seq_len]
                seq_files = [time_file_dict[t] for t in seq_times]
                self.sequences.append({'loc': loc, 'files': seq_files})
                self.seq_time_indices.append(seq_times)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        loc_idx = seq_info['loc']
        seq_files = seq_info['files']
        seq_times = self.seq_time_indices[idx]

        inputs = []
        targets = []

        for f in seq_files:
            with open(f, 'rb') as pf:
                data = pickle.load(pf)
            tensors = data['tensors']  # (2,3,H,W)
            target = np.ma.getdata(data['target'])  # (H,W) or (T,H,W)

            # select cameras 0 & 2, first timestep
            x = tensors[0, [0, 2]]
            inputs.append(torch.from_numpy(x).float())

            # target at first timestep
            y = target[0] if target.ndim == 3 else target
            targets.append(torch.from_numpy(y).unsqueeze(0).float())

        x_seq = torch.stack(inputs, dim=0)  # [seq_len, 2, H, W]
        y_seq = torch.stack(targets, dim=0) # [seq_len, 1, H, W]

        return x_seq, y_seq, seq_times



# -----------------------------
# Load dataset
# -----------------------------
folder_path = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_mitsuba_transformed_iso/"
seq_len = 20
overlap = 10
dataset = PKLSequenceDataset(folder_path, seq_len=seq_len, overlap=overlap)

# Choose a sample for visualization
sample_idx = 100
x_seq, y_seq, time_seq = dataset[sample_idx]  # x_seq: [T,2,H,W], y_seq: [T,1,H,W]

sequence_cams = x_seq.numpy()
sequence_target = y_seq.squeeze(1).numpy()

# -----------------------------
# Visualization with animation
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
frame_cam0 = axes[0].imshow(sequence_cams[0, 0], cmap='gray')
frame_cam1 = axes[1].imshow(sequence_cams[0, 1], cmap='gray')
frame_target = axes[2].imshow(sequence_target[0], cmap='jet')

for ax in axes:
    ax.axis('off')

axes[0].set_title(f"Camera 0 | Seq {sample_idx}")
axes[1].set_title(f"Camera 2 | Seq {sample_idx}")
axes[2].set_title(f"Target | Seq {sample_idx}")

def update(t):
    frame_cam0.set_data(sequence_cams[t, 0])
    frame_cam1.set_data(sequence_cams[t, 1])
    frame_target.set_data(sequence_target[t])

    # show the original timestep, sequence length, and overlap
    fig.suptitle(
        f"Sequence {sample_idx} | First Frame original timestep: {time_seq[t]} | "
        f"Seq length: {seq_len} | Overlap: {overlap}"
    )
    return frame_cam0, frame_cam1, frame_target


ani = FuncAnimation(fig, update, frames=sequence_cams.shape[0], interval=200, blit=True)
plt.show()
