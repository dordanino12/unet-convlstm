import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Folder containing the PKL files
pkl_folder = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_mitsuba_transformed_iso/"

# Only files ending with "_0_0.pkl"
pkl_files = sorted(glob.glob(os.path.join(pkl_folder, "*_0_0.pkl")))

fig, axes = None, None

for pkl_file in pkl_files:
    # Load the PKL
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    tensors = data['tensors']  # shape (2, 3, H, W)
    target = np.ma.getdata(data['target'])
    envelope = np.ma.getdata(data['envelope'])

    n_cameras = tensors.shape[1]  # 3 cameras

    # Only time step 0
    t = 0

    # Create figure & axes if not created yet
    if fig is None:
        fig, axes = plt.subplots(nrows=1, ncols=n_cameras + 2, figsize=(16, 5))
        if n_cameras + 2 == 1:
            axes = np.array([axes])

    # Cameras
    for c in range(n_cameras):
        ax = axes[c]
        ax.clear()
        ax.imshow(tensors[t, c], cmap='gray')
        ax.set_title(f"Cam {c}")
        ax.axis('off')

    # Target
    ax = axes[n_cameras]
    img_target = target[t] if (target.ndim == 3 and target.shape[0] > t) else target
    ax.clear()
    ax.imshow(img_target, cmap='jet')
    ax.set_title("Target")
    ax.axis('off')

    # Envelope
    ax = axes[n_cameras + 1]
    img_env = envelope[t] if (envelope.ndim == 3 and envelope.shape[0] > t) else envelope
    ax.clear()
    ax.imshow(img_env, cmap='magma')
    ax.set_title("Envelope")
    ax.axis('off')

    fig.suptitle(os.path.basename(pkl_file))
    plt.tight_layout()
    plt.pause(0.25)
    plt.draw()

plt.show()
