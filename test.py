import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import cv2

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
# Ensure we can import from the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import model classes
from train.unet import TemporalUNetDualView, NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet
# 3D dashboard helpers
from plots.create_video_dashboard3d_from_samples import create_3d_plot_img, load_camera_csv

# -----------------------------
# Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_MASK = True  # Set to False to ignore mask in visualization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
NPZ_PATH = "data/dataset_trajectory_sequences_samples.npz"
CHECKPOINT_PATH = "models/resnet18_frozen_best.pt"  # Update this to your best model path
SEQUENCE_IDX = 1000
CSV_PATH = "data/Dor_2satellites_overpass.csv"
VIDEO_FPS = 1

# Using `NPZSequenceDataset` from `train.unet`

# -----------------------------
# 2. Load Model Logic
# -----------------------------
print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Retrieve configuration to determine model type
cfg = checkpoint.get('config', {})
model_type = cfg.get('type', 'custom') # Default to custom if key missing

print(f"[INFO] Detected Model Type: {model_type}")

if model_type == 'resnet18':
    # Initialize Pretrained ResNet Model
    model = PretrainedTemporalUNet(
        out_channels=1,
        lstm_layers=1,
        freeze_encoder=cfg.get('freeze_encoder', True)
    )
else:
    # Initialize Custom U-Net Model
    model = TemporalUNetDualView(
        in_channels_per_sat=1,
        out_channels=1,
        base_ch=cfg.get('base_ch', 64),
        lstm_layers=1,
        use_skip_lstm=cfg.get('use_skip_lstm', True),
        use_attention=cfg.get('use_attention', False)
    )

# Load weights and set to eval mode
model.load_state_dict(checkpoint['model_state'])
model.to(DEVICE)
model.eval()

# -----------------------------
# 3. Run Inference
# -----------------------------
dataset = NPZSequenceDataset(NPZ_PATH)
input_seq, gt_vel_seq, mask_seq = dataset[SEQUENCE_IDX]
T, C, H, W = input_seq.shape

# Denormalize GT for display
gt_vel_denorm = 0.5 * (gt_vel_seq + 1) * (dataset.max_vel - dataset.min_vel) + dataset.min_vel

# We run inference incrementally to visualize how predictions evolve over time
print(f"[INFO] Running inference on sequence {SEQUENCE_IDX}...")

# Load camera CSV schedule for 3D visualization (optional)
try:
    csv_times, sat_lookup = load_camera_csv(CSV_PATH)
    # compute fixed bounds similar to dashboard script
    all_x, all_y, all_z = [], [], []
    for t in csv_times:
        for pos in sat_lookup[t]:
            all_x.append(abs(pos[0] / 1000.0))
            all_y.append(abs(pos[1] / 1000.0))
            all_z.append(pos[2] / 1000.0)
    if all_y:
        global_max_x = max(all_x) * 1.2
        global_max_y = max(all_y) * 1.2
        global_max_z = max(all_z) * 1.1
    else:
        global_max_x = 100; global_max_y = 100; global_max_z = 600
    fixed_limits_3d = (global_max_x, global_max_y, global_max_z)
    have_geo = True
except Exception as e:
    print(f"[WARN] Could not load CSV for 3D visualization: {e}")
    csv_times, sat_lookup = [], {}
    fixed_limits_3d = None
    have_geo = False

video_writer = None
video_path = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(video_path, exist_ok=True)
video_file = os.path.join(video_path, f"seq{SEQUENCE_IDX}.mp4")

for t_len in range(1, T + 1):
    # Prepare input batch for current time step
    x_input = input_seq[:t_len].unsqueeze(0).to(DEVICE)  # [1, t_len, 2, H, W]
    
    with torch.no_grad():
        output, _ = model(x_input)
    
    # --- Compatibility Handling (List vs Tensor) ---
    if isinstance(output, list):
        # Custom model returns list of tensors
        pred_tensor = torch.stack(output, dim=1)
    else:
        # ResNet model returns stacked tensor
        pred_tensor = output

    # Convert to numpy [T, 1, H, W]
    pred_vel = pred_tensor.squeeze(0).cpu().numpy()

    # Denormalize Predictions
    pred_vel_denorm = 0.5 * (pred_vel + 1) * (dataset.max_vel - dataset.min_vel) + dataset.min_vel

    # --- Prepare Data for Plotting (Last Frame) ---
    last_idx = t_len - 1
    
    # Inputs (Sat 1 & Sat 2)
    sat1 = input_seq[last_idx, 0].cpu().numpy()
    sat2 = input_seq[last_idx, 1].cpu().numpy()
    
    # Ground Truth
    gt_frame = gt_vel_denorm[last_idx, 0].cpu().numpy()
    
    # Prediction
    pred_frame = pred_vel_denorm[last_idx, 0]
    
    # Masking
    mask_frame = mask_seq[last_idx, 0].cpu().numpy()
    if USE_MASK:
        pred_frame = pred_frame * mask_frame

    # Dynamic color scale based on min/max of current frame
    vmin_seq = min(gt_frame.min(), pred_frame.min())
    vmax_seq = max(gt_frame.max(), pred_frame.max())

    # --- Plotting ---
    # If 3D geo available, show a 3-column layout (sat1, sat2, GT/Pred + 3D); otherwise fallback to 2x2
    if have_geo:
        # Use GridSpec so the 3D geometry spans both rows and is larger
        fig = plt.figure(figsize=(14, 6))
        # increase vertical spacing so row titles don't overlap
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6], hspace=0.18, wspace=0.02)
        axes = np.empty((2, 3), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0])
        axes[1, 0] = fig.add_subplot(gs[1, 0])
        axes[0, 1] = fig.add_subplot(gs[0, 1])
        axes[1, 1] = fig.add_subplot(gs[1, 1])
        axes[0, 2] = fig.add_subplot(gs[:, 2])
        axes[1, 2] = None
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    fig.suptitle(f"Sequence: {SEQUENCE_IDX} | Frame: {t_len}/{T}", fontsize=16)

    # 1. Satellite View 1
    axes[0, 0].imshow(sat1, cmap='gray')
    axes[0, 0].set_title("Input (Sat 0)", pad=8)
    axes[0, 0].axis('off')

    # 2. Satellite View 2
    axes[1, 0].imshow(sat2, cmap='gray')
    axes[1, 0].set_title("Input (Sat 1)", pad=8)
    axes[1, 0].axis('off')

    # 3. Ground Truth Velocity
    if have_geo:
        im1 = axes[0, 1].imshow(gt_frame, cmap='jet', vmin=vmin_seq, vmax=vmax_seq)
        axes[0, 1].set_title("GT Velocity", pad=8)
        axes[0, 1].axis('off')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 4. Predicted Velocity
        im2 = axes[1, 1].imshow(pred_frame, cmap='jet', vmin=vmin_seq, vmax=vmax_seq)
        axes[1, 1].set_title("Predicted Velocity", pad=8)
        axes[1, 1].axis('off')
        fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 5. 3D Geometry (right column spans both rows and is larger)
        try:
            csv_ptr = (t_len - 1) % len(csv_times) if len(csv_times) else 0
            target_time_val = csv_times[csv_ptr] if len(csv_times) else None
            sat_positions = sat_lookup[target_time_val] if target_time_val is not None else []
            img_geo = create_3d_plot_img(sat_positions, look_at=[0, 0, 1500], figsize=(500, 500), fixed_bounds=fixed_limits_3d)
            # convert BGR (cv2) -> RGB for matplotlib
            img_geo_rgb = cv2.cvtColor(img_geo, cv2.COLOR_BGR2RGB)
            axes[0, 2].imshow(img_geo_rgb)
            #axes[0, 2].set_title('', pad=8)
            axes[0, 2].axis('off')
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, '3D plot error', ha='center')
            axes[0, 2].axis('off')
    else:
        im1 = axes[0, 1].imshow(gt_frame, cmap='jet', vmin=vmin_seq, vmax=vmax_seq)
        axes[0, 1].set_title("GT Velocity", pad=8)
        axes[0, 1].axis('off')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 1].imshow(pred_frame, cmap='jet', vmin=vmin_seq, vmax=vmax_seq)
        axes[1, 1].set_title("Predicted Velocity", pad=8)
        axes[1, 1].axis('off')
        fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    # If Matplotlib is using a non-GUI backend (e.g., 'agg'), save figures to disk instead of showing
    try:
        backend = matplotlib.get_backend().lower()
    except Exception:
        backend = 'agg'

    if 'agg' in backend:
        # Render figure canvas to RGB array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if video_writer is None:
            h_pad, w_pad = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_file, fourcc, VIDEO_FPS, (w_pad, h_pad))

        video_writer.write(frame_bgr)
        plt.close(fig)
        print(f"[INFO] Appended frame {t_len} to {video_file} (backend={backend})")
    else:
        plt.show()

# After loop: release video writer if created
if 'video_writer' in globals() and video_writer is not None:
    video_writer.release()
    print(f"[INFO] Saved video to {video_file}")