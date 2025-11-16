import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- MODIFIED: Set the direct path to your PKL file ---
pkl_file_path = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_mitsuba_raw_iso/BOMEX_512x512x200_20m_20m_1s_512_0000005300_5_2.pkl"

# Create plots folder if not exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------
# Load one file and one frame
# -------------------------------
if not os.path.exists(pkl_file_path):
    print(f"File not found: {pkl_file_path}")
    print("Please update the 'pkl_file_path' variable with the full path to your file.")
else:
    # 1. Read the specified file
    print(f"Loading data from single file: {pkl_file_path}")

    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)

    # Extract data
    tensors = data['tensors']  # Original shape (T, C, H, W)
    target = np.ma.getdata(data['target'])
    envelope = np.ma.getdata(data['envelope'])

    # We will only show the first frame (frame_idx = 0)
    frame_idx = 0

    # Get data for the first frame
    frame_tensors = tensors[frame_idx]  # Shape (C, H, W)
    # Handle cases where target/envelope might be 2D (no time dim)
    frame_target = target[frame_idx] if target.ndim == 3 else target
    frame_envelope = envelope[frame_idx] if envelope.ndim == 3 else envelope

    n_cameras = frame_tensors.shape[0]
    base_label = os.path.basename(pkl_file_path)
    title = f"{base_label} | Frame {frame_idx}"

    print(f"Plotting frame {frame_idx} with {n_cameras} cameras.")

    # -------------------------------
    # Create one static plot
    # -------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=n_cameras + 2, figsize=(16, 5))
    # Handle case where there is only one plot total
    if not hasattr(axes, "__len__"):
        axes = np.array([axes])

    # Plot camera images
    for c in range(n_cameras):
        ax = axes[c]
        ax.imshow(frame_tensors[c], cmap='gray')
        ax.set_title(f"Cam {c}")
        ax.axis('off')

    # Plot Target
    ax = axes[n_cameras]
    ax.imshow(frame_target, cmap='jet')
    ax.set_title("Target")
    ax.axis('off')

    # Plot Envelope
    ax = axes[n_cameras + 1]
    ax.imshow(frame_envelope, cmap='magma')
    ax.set_title("Envelope")
    ax.axis('off')

    fig.suptitle(title)
    plt.tight_layout()

    # Save the static image
    out_path = os.path.join(plots_dir, "single_frame_snapshot.png")
    plt.savefig(out_path)
    print(f"Saved static image: {out_path}")

    # 2. Show the image
    print("Displaying plot...")
    plt.show()
    plt.close(fig)