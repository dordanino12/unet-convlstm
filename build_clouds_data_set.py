import pickle
import glob
import os
import numpy as np
import torch
from tqdm import tqdm

# -----------------------------
# Settings
# -----------------------------
folder_path = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_mitsuba_transformed_iso/"
output_path = "dataset_sequences_original.npz"
seq_len = 20
overlap = 10
stride = seq_len - overlap

# -----------------------------
# Gather all PKL files
# -----------------------------
all_files = sorted(glob.glob(os.path.join(folder_path, "*.pkl")))

# -----------------------------
# Organize files by location and time
# -----------------------------
loc_dict = {}
for f in all_files:
    base = os.path.basename(f)
    parts = base.split('_')
    time_idx = int(parts[-3])
    loc_idx = f"{parts[-2]}_{parts[-1].split('.')[0]}"
    if loc_idx not in loc_dict:
        loc_dict[loc_idx] = {}
    loc_dict[loc_idx][time_idx] = f

# Sort timesteps for each location
for loc in loc_dict:
    loc_dict[loc] = dict(sorted(loc_dict[loc].items()))

# -----------------------------
# Build sequences
# -----------------------------
sequences_X = []
sequences_Y = []

# Count total sequences for progress bar
total_sequences = sum(
    len(range(0, len(times_sorted) - seq_len + 1, stride))
    for times_sorted in [sorted(time_file_dict.keys()) for time_file_dict in loc_dict.values()]
)

# Process sequences with tqdm
with tqdm(total=total_sequences, desc="Processing all sequences") as pbar:
    for loc, time_file_dict in loc_dict.items():
        times_sorted = sorted(time_file_dict.keys())

        # Loop over all possible sequences for this location
        for i in range(0, len(times_sorted) - seq_len + 1, stride):
            seq_times = times_sorted[i:i+seq_len]
            seq_inputs = []
            seq_targets = []

            # Load each timestep in the sequence
            for t in seq_times:
                f = time_file_dict[t]
                with open(f, 'rb') as pf:
                    data = pickle.load(pf)

                tensors = data['tensors']
                target = np.ma.getdata(data['target'])

                # Select cameras 0 & 2 (like in PKLSequenceDataset)
                if "raw" in folder_path:
                    x = tensors[0]
                    x = x[[0, 2]]  # [2,H,W]
                else:
                    x = tensors[0, [0, 2]]  # [2,H,W]

                y = target[0] if target.ndim == 3 else target  # first timestep

                # Convert to numpy
                seq_inputs.append(np.array(x, dtype=np.float32))
                seq_targets.append(np.array(y, dtype=np.float32)[np.newaxis, ...])  # add channel dim

            # Stack timesteps
            sequences_X.append(np.stack(seq_inputs, axis=0))  # [seq_len,2,H,W]
            sequences_Y.append(np.stack(seq_targets, axis=0)) # [seq_len,1,H,W]

            pbar.update(1)

# Stack all sequences
X_all = np.stack(sequences_X, axis=0)  # [num_sequences, seq_len,2,H,W]
Y_all = np.stack(sequences_Y, axis=0)  # [num_sequences, seq_len,1,H,W]

# Save as compressed NPZ
np.savez_compressed(output_path, X=X_all, Y=Y_all)
print(f"Saved sequences dataset to {output_path}")

# -----------------------------
# Print shapes of saved arrays
# -----------------------------
print("Shapes of saved arrays:")
print(f"X_all.shape = {X_all.shape}")
print(f"Y_all.shape = {Y_all.shape}")
