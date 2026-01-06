import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os
import sys
import cv2

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
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
USE_MASK = False
SHOW_MASK_IMG = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Visual Settings ---
GAMMA_VAL = 0.5  # < 1.0 brightens, > 1.0 darkens
COLORBAR_STEP = 1.0  # Sets the numerical jump (e.g., every 1 m/s)
COLORBAR_FONT_SIZE = 14  # <--- NEW: Sets the font size for the numbers
min_y = None  # 7.5987958908081055
max_y = None  # 8.784920692443848
focus_thresh = 1.0

# Paths
NPZ_PATH = "data/dataset_trajectory_sequences_samples_500m_slices_w.npz"
CHECKPOINT_PATH = "models/resnet18_frozen_2lstm_layers_500m_slice_best_skip.pt"
SEQUENCE_IDX = 1500
CSV_PATH = "data/Dor_2satellites_overpass.csv"
VIDEO_FPS = 1

# -----------------------------
# 2. Load Model Logic
# -----------------------------
print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

cfg = checkpoint.get('config', {})
model_type = cfg.get('type', 'custom')

print(f"[INFO] Detected Model Type: {model_type}")

if model_type == 'resnet18':
    model = PretrainedTemporalUNet(
        out_channels=1,
        lstm_layers=2,
        freeze_encoder=cfg.get('freeze_encoder', True)
    )
else:
    model = TemporalUNetDualView(
        in_channels_per_sat=1,
        out_channels=1,
        base_ch=cfg.get('base_ch', 64),
        lstm_layers=1,
        use_skip_lstm=cfg.get('use_skip_lstm', True),
        use_attention=cfg.get('use_attention', False)
    )

model.load_state_dict(checkpoint['model_state'])
model.to(DEVICE)
model.eval()

# -----------------------------
# 3. Run Inference
# -----------------------------
dataset = NPZSequenceDataset(NPZ_PATH, min_y=min_y, max_y=max_y)
input_seq, gt_vel_seq, mask_seq = dataset[SEQUENCE_IDX]
T, C, H, W = input_seq.shape

# Denormalize GT (use dataset.denormalize which handles non-linear transform)
gt_vel_denorm = dataset.denormalize(gt_vel_seq)

# --- Fixed plot range so color scale doesn't jump per-frame ---
# Use dataset min/max (or set symmetric range if you prefer)
vmin_fixed = dataset.min_vel
vmax_fixed = dataset.max_vel

# --- Non-linear color normalization to emphasize -3..3 while keeping full range ---
# Use SymLogNorm with a linear threshold around `focus_thresh` (e.g., 3 m/s)
norm = mcolors.SymLogNorm(linthresh=focus_thresh, linscale=1.0, vmin=vmin_fixed, vmax=vmax_fixed)

# Create a `jet`-based colormap
base_cmap = plt.get_cmap('jet', 256)
colors = base_cmap(np.linspace(0.0, 1.0, 256))

# --- UPDATED LOGIC HERE ---
# Only force zero-values to black if USE_MASK is True.
if USE_MASK:
    try:
        zero_pos = norm(0.0)
        if np.isnan(zero_pos):
            zero_pos = 0.5
    except Exception:
        zero_pos = 0.5

    idx = int(np.clip(np.round(zero_pos * (len(colors) - 1)), 0, len(colors) - 1))

    # make a small black band around the zero index
    for i in range(max(0, idx - 1), min(len(colors), idx + 2)):
        colors[i] = np.array([0.0, 0.0, 0.0, 1.0])

cmap_custom = mcolors.ListedColormap(colors, name='jet_custom')

print(f"[INFO] Running inference on sequence {SEQUENCE_IDX}...")

# Load camera CSV
try:
    csv_times, sat_lookup = load_camera_csv(CSV_PATH)
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
        global_max_x = 100
        global_max_y = 100
        global_max_z = 600
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

seq_mae = []
seq_rmse = []
seq_mean_err = []


# --- Gamma Helper Function ---
def apply_gamma(img_array, gamma=0.5):
    img_min, img_max = img_array.min(), img_array.max()
    if img_max - img_min < 1e-6:
        return img_array
    img_norm = (img_array - img_min) / (img_max - img_min)
    img_corrected = np.power(img_norm, gamma)
    return img_corrected


for t_len in range(1, T + 1):
    # Prepare input
    x_input = input_seq[:t_len].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, _ = model(x_input)

    if isinstance(output, list):
        pred_tensor = torch.stack(output, dim=1)
    else:
        pred_tensor = output

    pred_vel = pred_tensor.squeeze(0).cpu().numpy()
    pred_vel_denorm = dataset.denormalize(pred_vel)

    # Get Last Frame Data
    last_idx = t_len - 1

    # Apply Gamma
    raw_sat1 = input_seq[last_idx, 0].cpu().numpy()
    raw_sat2 = input_seq[last_idx, 1].cpu().numpy()
    sat1 = apply_gamma(raw_sat1, GAMMA_VAL)
    sat2 = apply_gamma(raw_sat2, GAMMA_VAL)

    gt_frame = gt_vel_denorm[last_idx, 0].cpu().numpy()
    pred_frame = pred_vel_denorm[last_idx, 0]
    mask_frame = mask_seq[last_idx, 0].cpu().numpy()

    # --- Metrics Calculation ---
    diff_map = pred_frame - gt_frame

    if USE_MASK:
        valid_pixels = mask_frame > 0.1
        if np.any(valid_pixels):
            valid_diff = diff_map[valid_pixels]
        else:
            valid_diff = np.array([0.0])
    else:
        valid_diff = diff_map.flatten()

    mae_score = np.mean(np.abs(valid_diff))
    rmse_score = np.sqrt(np.mean(valid_diff ** 2))
    mean_err_score = np.mean(valid_diff)

    seq_mae.append(mae_score)
    seq_rmse.append(rmse_score)
    seq_mean_err.append(mean_err_score)

    # --- Plotting Preparation ---
    if USE_MASK:
        pred_frame = pred_frame * mask_frame
        gt_frame = gt_frame * mask_frame

        # Use fixed plotting range so the color scale is stable across frames
    vmin_plot = vmin_fixed
    vmax_plot = vmax_fixed

    # --- Figure Layout Logic ---
    if have_geo:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.2, wspace=0.1)
        axes = np.empty((2, 3), dtype=object)

        # Col 0: Inputs
        axes[0, 0] = fig.add_subplot(gs[0, 0])
        axes[1, 0] = fig.add_subplot(gs[1, 0])

        # Col 1: Velocities
        axes[0, 1] = fig.add_subplot(gs[0, 1])
        axes[1, 1] = fig.add_subplot(gs[1, 1])

        # Col 2: 3D & Mask
        if SHOW_MASK_IMG:
            axes[0, 2] = fig.add_subplot(gs[0, 2])
            axes[1, 2] = fig.add_subplot(gs[1, 2])
        else:
            axes[0, 2] = fig.add_subplot(gs[:, 2])
            axes[1, 2] = None
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    fig.suptitle(f"Sequence: {SEQUENCE_IDX} | Frame: {t_len}/{T}", fontsize=16, fontweight='bold')

    # Metrics
    metrics_str = (f"MAE: {mae_score:.2f} m/s\n"
                   f"RMSE: {rmse_score:.2f} m/s\n"
                   f"Mean Error: {mean_err_score:.2f} m/s")

    fig.text(0.01, 0.93, metrics_str, fontsize=12, color='black',
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    # 1. Sat 0
    axes[0, 0].imshow(sat1, cmap='gray')
    axes[0, 0].set_title("Input Sat 0", pad=8)
    axes[0, 0].axis('off')

    # 2. Sat 1
    axes[1, 0].imshow(sat2, cmap='gray')
    axes[1, 0].set_title("Input Sat 1", pad=8)
    axes[1, 0].axis('off')

    # 3. GT (Top Middle)
    im1 = axes[0, 1].imshow(gt_frame, cmap=cmap_custom, norm=norm)
    axes[0, 1].set_title("Ground True Velocity [m/s]", pad=8)
    axes[0, 1].axis('off')

    if have_geo:
        # ### NEW: Scale Bar Config ###
        cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        # build ticks that include min, -focus_thresh, 0, focus_thresh, max (within bounds)
        ticks_parts = []
        if vmin_fixed < -focus_thresh:
            ticks_parts.append(np.linspace(vmin_fixed, -focus_thresh, 3))
        for t in (-focus_thresh, 0.0, focus_thresh):
            if (t >= vmin_fixed) and (t <= vmax_fixed):
                ticks_parts.append(np.array([t]))
        if vmax_fixed > focus_thresh:
            ticks_parts.append(np.linspace(focus_thresh, vmax_fixed, 3))
        if ticks_parts:
            ticks = np.unique(np.concatenate(ticks_parts))
            cbar1.set_ticks(ticks)
        cbar1.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)
        # #############################

    # 4. Pred (Bottom Middle)
    im2 = axes[1, 1].imshow(pred_frame, cmap=cmap_custom, norm=norm)
    axes[1, 1].set_title("Predicted Velocity [m/s]", pad=8)
    axes[1, 1].axis('off')

    if have_geo:
        # ### NEW: Scale Bar Config ###
        cbar2 = fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        # reuse tick generation from above
        if ticks_parts:
            cbar2.set_ticks(ticks)
        cbar2.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)
        # #############################

    # 5. 3D Plot
    if have_geo:
        try:
            csv_ptr = (t_len - 1) % len(csv_times) if len(csv_times) else 0
            target_time_val = csv_times[csv_ptr] if len(csv_times) else None
            sat_positions = sat_lookup[target_time_val] if target_time_val is not None else []
            img_geo = create_3d_plot_img(sat_positions, look_at=[0, 0, 1500], figsize=(500, 500),
                                         fixed_bounds=fixed_limits_3d)
            img_geo_rgb = cv2.cvtColor(img_geo, cv2.COLOR_BGR2RGB)
            axes[0, 2].imshow(img_geo_rgb)
            axes[0, 2].axis('off')
        except Exception:
            axes[0, 2].text(0.5, 0.5, '3D error', ha='center')
            axes[0, 2].axis('off')
    elif not have_geo:
        axes[0, 2].axis('off')

    # 6. Mask Plot
    if SHOW_MASK_IMG:
        if have_geo:
            axes[1, 2].imshow(mask_frame, cmap='gray', vmin=0, vmax=1)
            axes[1, 2].set_title("Cloud Mask", pad=8)
            axes[1, 2].axis('off')
        else:
            axes[1, 2].imshow(mask_frame, cmap='gray', vmin=0, vmax=1)
            axes[1, 2].set_title("Cloud Mask", pad=8)
            axes[1, 2].axis('off')

    plt.tight_layout()

    # Save logic
    try:
        backend = matplotlib.get_backend().lower()
    except Exception:
        backend = 'agg'

    if 'agg' in backend:
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        if video_writer is None:
            h_pad, w_pad = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_file, fourcc, VIDEO_FPS, (w_pad, h_pad))

        video_writer.write(frame_bgr)
        print(f"[INFO] Frame {t_len} processed. MAE={mae_score:.3f}")

    else:
        plt.show()

if 'video_writer' in globals() and video_writer is not None:
    video_writer.release()
    print(f"[INFO] Saved video to {video_file}")

    print(f"\n=== Final Sequence Stats ===")
    print(f"Average MAE:        {np.mean(seq_mae):.4f}")
    print(f"Average RMSE:       {np.mean(seq_rmse):.4f}")
    print(f"Average Mean Error: {np.mean(seq_mean_err):.4f}")