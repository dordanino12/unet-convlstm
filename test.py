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
# 3D/2D dashboard helpers
from plots.create_video_dashboard3d_from_samples import create_3d_plot_img, create_2d_plot_img, load_camera_csv

# -----------------------------
# Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_MASK = False
SHOW_MASK_IMG = True
GEO_MODE = "2d"  # "3d" or "2d" (X-Z view)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Visual Settings ---
GAMMA_VAL = 0.5  # < 1.0 brightens, > 1.0 darkens
COLORBAR_STEP = 2.0  # Sets the numerical jump (e.g., every 1 m/s)
COLORBAR_FONT_SIZE = 14  # <--- NEW: Sets the font size for the numbers
min_y = None #7.5987958908081055
max_y = None# 8.784920692443848
focus_thresh = 2.0

# Paths
# NPZ_PATH = "data/dataset_trajectory_sequences_samples_W_top.npz"
# CHECKPOINT_PATH = "models/resnet18_frozen_2lstm_layers_all_speed_skip.pt"
NPZ_PATH = "data/dataset_trajectory_sequences_samples_W_500m_w.npz"
CHECKPOINT_PATH = "models/resnet18_frozen_2lstm_layers_500m_best_skip.pt"
SEQUENCE_IDX = 1000
CSV_PATH = "data/Dor_2satellites_overpass.csv"
VIDEO_FPS = 1
SAVE_PDF_SECTIONS = True
PDF_BASE_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'frames_pdf')

# PDF Layout Settings
PDF_FIG_SIZE = (20, 20)
# Slightly smaller/taller fit: lower height and shift up a bit
PDF_AX_POS = [0.17, 0.10, 0.70, 0.75]
PDF_SUBPLOT_ADJUST = dict(left=0.17, right=0.88, top=0.95, bottom=0.08)
PDF_CBAR_PAD = 0.015
PDF_CBAR_WIDTH = 0.035
PDF_CBAR_HEIGHT = 0.80


def apply_pdf_layout(fig, ax):
    fig.set_size_inches(*PDF_FIG_SIZE)
    fig.subplots_adjust(**PDF_SUBPLOT_ADJUST)
    ax.set_position(PDF_AX_POS)


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
dataset = NPZSequenceDataset(NPZ_PATH)
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


def set_centered_meter_axis(ax, height, width, m_per_pixel=20):
    """
    Set centered meter axes like build_W_map PDF style.
    """
    half_w_m = (width * m_per_pixel) / 2.0
    half_h_m = (height * m_per_pixel) / 2.0
    ax.set_xlim(-half_w_m, half_w_m)
    ax.set_ylim(half_h_m, -half_h_m)

    tick_vals = np.array([-1100, -640, 0, 640, 1100])
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_xticklabels([f"{int(v)}" for v in tick_vals], fontsize=48, fontweight='bold')
    ax.set_yticklabels([f"{int(v)}" for v in tick_vals], fontsize=48, fontweight='bold')

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=14,
        width=4,
        color='black',
        labelsize=48
    )

    ax.set_xlabel('X [m]', fontsize=52, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=52, fontweight='bold')


def save_section_pdf(img_data, title, out_path, cmap='gray', norm_obj=None, add_colorbar=False,
                     m_per_pixel=20, extent_m=None, vmin=None, vmax=None, tick_step=None):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    if extent_m is None:
        H, W = img_data.shape[:2]
        half_w_m = (W * m_per_pixel) / 2.0
        half_h_m = (H * m_per_pixel) / 2.0
        extent_m = [-half_w_m, half_w_m, half_h_m, -half_h_m]

    if norm_obj is not None:
        im = ax.imshow(img_data, cmap=cmap, norm=norm_obj, extent=extent_m, interpolation='nearest')
    else:
        im = ax.imshow(img_data, cmap=cmap, extent=extent_m, interpolation='nearest',
                       vmin=vmin, vmax=vmax)
    ax.set_aspect('auto')
    ax.set_title(title, fontsize=56, fontweight='bold', pad=40)

    H, W = img_data.shape[:2]
    set_centered_meter_axis(ax, H, W, m_per_pixel=m_per_pixel)

    if add_colorbar:
        # Place colorbar in its own fixed axes so the main image size never changes
        # Match colorbar height to image axes height exactly
        cbar_h = PDF_AX_POS[3]
        cbar_y = PDF_AX_POS[1]
        cbar_x = PDF_AX_POS[0] + PDF_AX_POS[2] + PDF_CBAR_PAD
        cax = fig.add_axes([cbar_x, cbar_y, PDF_CBAR_WIDTH, cbar_h])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=48)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if tick_step is not None and vmin is not None and vmax is not None and tick_step > 0:
            ticks = np.arange(vmin, vmax, tick_step)
            if vmin < 0 < vmax and 0.0 not in ticks:
                ticks = np.sort(np.append(ticks, 0.0))
            cbar.set_ticks(ticks)

    # Apply fixed layout (do this after colorbar so it doesn't interfere)
    apply_pdf_layout(fig, ax)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_geo_2d_pdf(sat_positions, look_at, fixed_bounds, out_path, title):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

    def to_km(val_m):
        return val_m / 1000.0

    # Cloud Center (Y-Z)
    cy, cz = to_km(look_at[1]), to_km(look_at[2])
    ax.scatter(cy, cz, c='#555555', s=400, marker='X', label='Cloud', zorder=3)

    colors = ['#E74C3C', '#3498DB']
    for i, pos in enumerate(sat_positions):
        y_km = to_km(pos[1])
        z_km = to_km(pos[2])
        color = colors[i % len(colors)]
        ax.scatter(y_km, z_km, c=color, s=260, edgecolors='white', linewidth=2.0, zorder=4)
        ax.plot([y_km, cy], [z_km, cz], c=color, linestyle='--', alpha=0.5, linewidth=2.0, zorder=2)
        ax.annotate(
            f"S{i}",
            xy=(y_km, z_km),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center', va='bottom',
            color=color, fontsize=48, fontweight='bold'
        )

    if fixed_bounds:
        _, my, mz = fixed_bounds
        ax.set_xlim(-my, my)
        ax.set_ylim(0, mz)

        # Shift ticks inward to avoid overlap at edges (show inner values instead of edge values)
        y_ticks = np.linspace(-my * 0.85, my * 0.85, 5)
        z_ticks = np.linspace(0.15 * mz, mz * 0.9, 5)
        # Round to nearest 10 for clean display
        y_ticks = np.round(y_ticks / 10) * 10
        z_ticks = np.round(z_ticks / 10) * 10
        ax.set_xticks(y_ticks)
        ax.set_yticks(z_ticks)
        ax.set_xticklabels([f"{int(v)}" for v in y_ticks], fontsize=48, fontweight='bold')
        ax.set_yticklabels([f"{int(v)}" for v in z_ticks], fontsize=48, fontweight='bold')

    ax.set_xlabel('Y [km]', fontsize=52, fontweight='bold')
    ax.set_ylabel('Z [km]', fontsize=52, fontweight='bold')
    ax.set_title(title, fontsize=56, fontweight='bold', pad=40)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=2)
    ax.set_aspect('auto')

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=14,
        width=4,
        color='black',
        labelsize=48
    )

    apply_pdf_layout(fig, ax)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


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
        mask_invalid = mask_frame <= 0.1
        gt_display = np.ma.masked_where(mask_invalid, gt_frame)
        pred_display = np.ma.masked_where(mask_invalid, pred_frame)
        cmap_vel = cmap_custom.copy()
        cmap_vel.set_bad(color='black')
    else:
        gt_display = gt_frame
        pred_display = pred_frame
        cmap_vel = cmap_custom

    # Use fixed plotting range so the color scale is stable across frames
    vmin_plot = vmin_fixed
    vmax_plot = vmax_fixed

    # --- Figure Layout Logic ---
    if have_geo:
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.35, wspace=0.15)
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

    # --- Helper function to set meters axis ---
    def set_km_axis(ax, height, width):
        """Set axis ticks and labels in meters (each pixel = 20 m)"""
        m_per_pixel = 20

        # Set x-axis
        x_ticks_px = np.linspace(0, width - 1, 5)
        x_ticks_m = x_ticks_px * m_per_pixel
        ax.set_xticks(x_ticks_px)
        ax.set_xticklabels([f'{int(m)}' for m in x_ticks_m], fontsize=10)
        ax.set_xlabel('X [m]', fontsize=11, fontweight='bold')

        # Set y-axis
        y_ticks_px = np.linspace(0, height - 1, 5)
        y_ticks_m = y_ticks_px * m_per_pixel
        ax.set_yticks(y_ticks_px)
        ax.set_yticklabels([f'{int(m)}' for m in y_ticks_m], fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=11, fontweight='bold')

    # 1. Sat 0
    axes[0, 0].imshow(sat1, cmap='gray')
    axes[0, 0].set_title("Input Sat 0", pad=20, fontsize=12, fontweight='bold')
    set_km_axis(axes[0, 0], sat1.shape[0], sat1.shape[1])

    # 2. Sat 1
    axes[1, 0].imshow(sat2, cmap='gray')
    axes[1, 0].set_title("Input Sat 1", pad=20, fontsize=12, fontweight='bold')
    set_km_axis(axes[1, 0], sat2.shape[0], sat2.shape[1])

    # 3. GT (Top Middle)
    im1 = axes[0, 1].imshow(gt_display, cmap=cmap_vel, norm=norm)
    axes[0, 1].set_title("Ground True Velocity [m/s]", pad=15, fontsize=12, fontweight='bold')
    set_km_axis(axes[0, 1], gt_frame.shape[0], gt_frame.shape[1])

    if have_geo:
        # ### NEW: Scale Bar Config ###
        cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if COLORBAR_STEP and COLORBAR_STEP > 0:
            ticks = np.arange(vmin_fixed, vmax_fixed + COLORBAR_STEP, COLORBAR_STEP)
            if vmin_fixed < 0 < vmax_fixed and 0.0 not in ticks:
                ticks = np.sort(np.append(ticks, 0.0))
            cbar1.set_ticks(ticks)
        cbar1.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)
        # #############################

    # 4. Pred (Bottom Middle)
    im2 = axes[1, 1].imshow(pred_display, cmap=cmap_vel, norm=norm)
    axes[1, 1].set_title("Predicted Velocity [m/s]", pad=15, fontsize=12, fontweight='bold')
    set_km_axis(axes[1, 1], pred_frame.shape[0], pred_frame.shape[1])

    if have_geo:
        # ### NEW: Scale Bar Config ###
        cbar2 = fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if COLORBAR_STEP and COLORBAR_STEP > 0:
            ticks = np.arange(vmin_fixed, vmax_fixed + COLORBAR_STEP, COLORBAR_STEP)
            if vmin_fixed < 0 < vmax_fixed and 0.0 not in ticks:
                ticks = np.sort(np.append(ticks, 0.0))
            cbar2.set_ticks(ticks)
        cbar2.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)
        # #############################

    # 5. Geo Plot (3D or 2D X-Z)
    geo_image_rgb = None
    geo_positions = None
    if have_geo:
        try:
            csv_ptr = (t_len - 1) % len(csv_times) if len(csv_times) else 0
            target_time_val = csv_times[csv_ptr] if len(csv_times) else None
            sat_positions = sat_lookup[target_time_val] if target_time_val is not None else []
            geo_positions = sat_positions
            # Match geometry plot size to sat image dimensions
            geo_size = (sat1.shape[1], sat1.shape[0])  # (width, height)
            if GEO_MODE == "2d":
                img_geo = create_2d_plot_img(sat_positions, look_at=[0, 0, 1500], figsize=geo_size,
                                             fixed_bounds=fixed_limits_3d)
            else:
                img_geo = create_3d_plot_img(sat_positions, look_at=[0, 0, 1500], figsize=(500, 500),
                                             fixed_bounds=fixed_limits_3d)
            img_geo_rgb = cv2.cvtColor(img_geo, cv2.COLOR_BGR2RGB)
            geo_image_rgb = img_geo_rgb
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

    # Save per-section PDFs for this frame
    if SAVE_PDF_SECTIONS:
        seq_dir = os.path.join(PDF_BASE_DIR, f"seq{SEQUENCE_IDX}")
        frame_dir = os.path.join(seq_dir, f"frame_{t_len:03d}")
        os.makedirs(frame_dir, exist_ok=True)

        # Common extent in meters
        H_img, W_img = gt_frame.shape
        m_per_pixel = 20
        half_w_m = (W_img * m_per_pixel) / 2.0
        half_h_m = (H_img * m_per_pixel) / 2.0
        extent_m = [-half_w_m, half_w_m, half_h_m, -half_h_m]

        # Inputs
        save_section_pdf(sat1, "Input Sat 0", os.path.join(frame_dir, "sat0.pdf"),
                         cmap='gray', add_colorbar=False, m_per_pixel=m_per_pixel, extent_m=extent_m)
        save_section_pdf(sat2, "Input Sat 1", os.path.join(frame_dir, "sat1.pdf"),
                         cmap='gray', add_colorbar=False, m_per_pixel=m_per_pixel, extent_m=extent_m)

        # GT and Pred (velocity)
        save_section_pdf(gt_display, "Ground True Velocity [m/s]", os.path.join(frame_dir, "gt.pdf"),
             cmap=cmap_vel, norm_obj=norm, add_colorbar=True,
                 m_per_pixel=m_per_pixel, extent_m=extent_m, vmin=vmin_plot, vmax=vmax_plot,
                 tick_step=COLORBAR_STEP)
        save_section_pdf(pred_display, "Predicted Velocity [m/s]", os.path.join(frame_dir, "pred.pdf"),
             cmap=cmap_vel, norm_obj=norm, add_colorbar=True,
                 m_per_pixel=m_per_pixel, extent_m=extent_m, vmin=vmin_plot, vmax=vmax_plot,
                 tick_step=COLORBAR_STEP)

        # Mask
        save_section_pdf(mask_frame, "Cloud Mask", os.path.join(frame_dir, "mask.pdf"),
                         cmap='gray', add_colorbar=False, m_per_pixel=m_per_pixel, extent_m=extent_m,
                         vmin=0, vmax=1)

        # Geo plot (if exists)
        if GEO_MODE == "2d" and geo_positions is not None:
            save_geo_2d_pdf(
                geo_positions,
                look_at=[0, 0, 1500],
                fixed_bounds=fixed_limits_3d,
                out_path=os.path.join(frame_dir, "geo.pdf"),
                title="2D Geometry"
            )
        elif geo_image_rgb is not None:
            fig_geo, ax_geo = plt.subplots(figsize=(12, 12), dpi=150)
            ax_geo.imshow(geo_image_rgb)
            ax_geo.axis('off')
            plt.tight_layout()
            fig_geo.set_size_inches(20, 20)
            plt.savefig(os.path.join(frame_dir, "geo.pdf"), dpi=150)
            plt.close(fig_geo)

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