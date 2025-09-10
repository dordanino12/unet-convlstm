import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import os

# Folder containing the PKL files
pkl_folder = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_mitsuba_transformed_iso/"

# Only files ending with "_0_0.pkl"
pkl_files = sorted(glob.glob(os.path.join(pkl_folder, "*_0_0.pkl")))

# Create plots folder if not exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------
# Collect all frames
# -------------------------------
all_tensors = []
all_targets = []
all_envelopes = []
all_labels = []  # for suptitles

for pkl_file in pkl_files:
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    tensors = data['tensors']  # shape (T, C, H, W)
    target = np.ma.getdata(data['target'])
    envelope = np.ma.getdata(data['envelope'])

    n_frames = tensors.shape[0]
    base_label = os.path.basename(pkl_file)

    for frame_idx in range(n_frames):
        all_tensors.append(tensors[frame_idx])
        all_targets.append(target[frame_idx] if target.ndim == 3 else target)
        all_envelopes.append(envelope[frame_idx] if envelope.ndim == 3 else envelope)
        all_labels.append(f"{base_label} | Frame {frame_idx}")

all_tensors = np.array(all_tensors)   # shape: [total_frames, C, H, W]
all_targets = np.array(all_targets)   # shape: [total_frames, H, W]
all_envelopes = np.array(all_envelopes)
total_frames = len(all_tensors)
n_cameras = all_tensors.shape[1]

# -------------------------------
# Create one animation
# -------------------------------
fig, axes = plt.subplots(nrows=1, ncols=n_cameras + 2, figsize=(16, 5))
if n_cameras + 2 == 1:
    axes = np.array([axes])

# Initial images
imgs = []
for c in range(n_cameras):
    ax = axes[c]
    img = ax.imshow(all_tensors[0, c], cmap='gray', animated=True)
    ax.set_title(f"Cam {c}")
    ax.axis('off')
    imgs.append(img)

ax = axes[n_cameras]
img_t = ax.imshow(all_targets[0], cmap='jet', animated=True)
ax.set_title("Target")
ax.axis('off')
imgs.append(img_t)

ax = axes[n_cameras + 1]
img_e = ax.imshow(all_envelopes[0], cmap='magma', animated=True)
ax.set_title("Envelope")
ax.axis('off')
imgs.append(img_e)

fig.suptitle(all_labels[0])
plt.tight_layout()

def update(frame):
    for c in range(n_cameras):
        imgs[c].set_array(all_tensors[frame, c])
    img_t.set_array(all_targets[frame])
    img_e.set_array(all_envelopes[frame])
    fig.suptitle(all_labels[frame])
    return imgs

ani = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)

# Save one long video
out_path = os.path.join(plots_dir, "all_sequences_combined.mp4")
ani.save(out_path, writer='ffmpeg', fps=5)
print(f"Saved combined video: {out_path}")

plt.close(fig)
