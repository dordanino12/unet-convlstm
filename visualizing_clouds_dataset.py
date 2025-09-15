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
        self.loc_dict = {}
        for f in self.files:
            base = os.path.basename(f)
            parts = base.split('_')
            time_idx = int(parts[-3])
            loc_idx = f"{parts[-2]}_{parts[-1].split('.')[0]}"
            if loc_idx not in self.loc_dict:
                self.loc_dict[loc_idx] = {}
            self.loc_dict[loc_idx][time_idx] = f

        # sort timesteps for each location
        for loc in self.loc_dict:
            self.loc_dict[loc] = dict(sorted(self.loc_dict[loc].items()))

        # build sequences with stride
        self.sequences = []
        self.seq_time_indices = []
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
        seq_files = seq_info['files']
        seq_times = self.seq_time_indices[idx]

        images_input, W_velocity, W_velocity_slices, Envelope_seq = [], [], [], []

        for f in seq_files:
            with open(f, 'rb') as pf:
                data = pickle.load(pf)

            # renamed variables
            images = data['tensors']                 # cameras
            W = np.ma.getdata(data['target'])        # target
            W_slices = data['target_slice'][8]       # target slices
            Envelope = np.ma.getdata(data['envelope']) # envelope (same shape as target)

            # select cameras 0 & 2, first timestep
            if "raw" in folder_path:
                x = images[0]
                x = x[[0, 2]]  # (2,H,W)
            else:
                x = images[0, [0, 2]]  # (2,H,W)
            images_input.append(torch.from_numpy(x).float())

            # target at first timestep
            y = W[0] if W.ndim == 3 else W
            W_velocity.append(torch.from_numpy(y).unsqueeze(0).float())

            # stack all 8 slices
            ts = torch.from_numpy(np.stack(W_slices)).float()  # [8,H,W]
            W_velocity_slices.append(ts)

            # envelope at first timestep
            e = Envelope[0] if Envelope.ndim == 3 else Envelope
            Envelope_seq.append(torch.from_numpy(e).unsqueeze(0).float())

        x_seq = torch.stack(images_input, dim=0)           # [T,2,H,W]
        W_seq = torch.stack(W_velocity, dim=0)             # [T,1,H,W]
        W_slices_seq = torch.stack(W_velocity_slices, 0)   # [T,8,H,W]
        Envelope_seq = torch.stack(Envelope_seq, 0)        # [T,1,H,W]

        return x_seq, W_seq, W_slices_seq, Envelope_seq, seq_times


# -----------------------------
# Load dataset
# -----------------------------
folder_path = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_mitsuba_transformed_iso/"
seq_len, overlap = 20, 10
dataset = PKLSequenceDataset(folder_path, seq_len=seq_len, overlap=overlap)

# Choose a sample for visualization
sample_idx = 30
images_input_seq, W_seq, W_slices_seq, Envelope_seq, time_seq = dataset[sample_idx]

sequence_cams = images_input_seq.numpy()
sequence_W = W_seq.squeeze(1).numpy()
sequence_W_slices = W_slices_seq.numpy()    # [T,8,H,W]
sequence_Envelope = Envelope_seq.squeeze(1).numpy()  # [T,H,W]

# -----------------------------
# Visualization with animation
# -----------------------------
ncols = 5
nrows = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
axes = axes.ravel()

# Cameras
frame_cam0 = axes[0].imshow(sequence_cams[0, 0], cmap='gray')
axes[0].set_title("Camera 0")
frame_cam1 = axes[1].imshow(sequence_cams[0, 1], cmap='gray')
axes[1].set_title("Camera 2")

# W_velocity
frame_W = axes[2].imshow(sequence_W[0], cmap='jet')
axes[2].set_title("W_velocity")

# W_velocity_slices
frames_slices = []
for i in range(8):
    frames_slices.append(axes[3 + i].imshow(sequence_W_slices[0, i], cmap='jet'))
    axes[3 + i].set_title(f"W_velocity_slice[{i}]")

# Envelope
frame_envelope = axes[11].imshow(sequence_Envelope[0], cmap='jet')
axes[11].set_title("Envelope")

for ax in axes:
    ax.axis('off')

def update(t):
    frame_cam0.set_data(sequence_cams[t, 0])
    frame_cam1.set_data(sequence_cams[t, 1])
    frame_W.set_data(sequence_W[t])
    for i in range(8):
        frames_slices[i].set_data(sequence_W_slices[t, i])
    frame_envelope.set_data(sequence_Envelope[t])

    fig.suptitle(
        f"Sequence {sample_idx} | timestep: {time_seq[t]} | Seq length: {seq_len} | Overlap: {overlap}"
    )
    return [frame_cam0, frame_cam1, frame_W] + frames_slices + [frame_envelope]

ani = FuncAnimation(fig, update, frames=sequence_cams.shape[0], interval=200, blit=True)

plt.tight_layout()
plt.show()
