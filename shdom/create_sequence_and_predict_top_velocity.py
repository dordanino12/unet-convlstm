import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path to import train module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
from train.resnet18 import PretrainedTemporalUNetMitB1

# --- CONFIG ---
#pkl_dir = os.path.join(project_root, 'data/output/satellite_output_steps/')
#pkl_dir = "/wdata_visl/vhold/backup_from_133/CEIL/CELINE_2021/satellite_output_steps_weak_sea_low_angular_res_256pixelres/"
pkl_dir = "/wdata_visl/vhold/backup_from_133/CEIL/CELINE_2021/satellite_output_steps_no_sea_low_angular_res_256pixelres_10000-10220/"
gt_base_dir = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,reff,lwc,U,V,W)_fixed_to_shdom/10000_10220_1000m_vel/'
model_path = os.path.join(project_root, 'models/data_fix/mit_b1_1000m_data_leakag_fix_noised_with_augment_best_bin_loss.pt')
output_dir = script_dir  # Output video to script directory


def determine_movie_level_from_model_path():
    mp = model_path.lower()
    if '500m' in mp:
        return '500m'
    if '1000m' in mp:
        return '1000m'
    if '1500m' in mp:
        return '1500m'
    if 'envelop' in mp or 'envelope' in mp:
        return 'top'
    return 'top'


movie_level = determine_movie_level_from_model_path()
output_video = os.path.join(output_dir, f'vadim_predicted_{movie_level}_velocity.mp4')
temp_sequence_npy = os.path.join(output_dir, 'vadim_sequence.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- Options ---
# If True, apply dataset mask to predictions/GT when visualizing; if False, show full maps.

def determine_use_mask_from_model_path():
    mp = model_path.lower()
    return ('envelop' in mp) or ('envelope' in mp) or ('top' in mp)


USE_MASK = determine_use_mask_from_model_path()
print(f'[INFO] USE_MASK={USE_MASK} (from model_path)')

# --- Load PKL images as sequence with 2 channels per timestep ---
pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])
sequence = []
seq_files = []  # filenames corresponding to frames in `sequence`
for pkl_file in pkl_files:
    with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'sat1_image' in data and 'sat2_image' in data:
        img = np.stack([data['sat1_image'], data['sat2_image']], axis=0)  # (2, H, W)
        sequence.append(img)
        seq_files.append(pkl_file)
    else:
        print(f'PKL missing sat1_image or sat2_image in {pkl_file}')

if len(sequence) == 0:
    raise RuntimeError('No valid PKL files found with sat1_image and sat2_image.')

sequence = np.stack(sequence, axis=0)  # (T, 2, H, W)
np.save(temp_sequence_npy, sequence)

# --- Load GT top velocity maps ---
# Direct mapping from step file -> GT top velocity file.
# Examples provided by user:
#   step_0000.pkl -> .../top_vel/0000000000/sample_012_time_0_view_0_first_hit.pkl
#   step_0001.pkl -> .../top_vel/0000000001/sample_012_time_20_view_0_first_hit.pkl

def step_to_time(step_num):
    return step_num * 20


def determine_gt_suffix_from_model_path():
    """Pick GT filename suffix based on model path keywords."""
    mp = model_path.lower()
    if 'envelop' in mp or 'envelope' in mp:
        return 'first_hit'
    if '500m' in mp:
        return 'slice_500m'
    if '1000m' in mp:
        return 'slice_1000m'
    if '1500m' in mp:
        return 'slice_1500m'
    # Default keeps previous behavior when model naming is unclear.
    return 'first_hit'


gt_suffix = determine_gt_suffix_from_model_path()
print(f'[INFO] Using GT suffix={gt_suffix} (from model_path)')


def resize_gt_map(gt_img, target_shape):
    gt_img = np.asarray(gt_img)
    if gt_img.shape == target_shape:
        return gt_img
    return cv2.resize(gt_img.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def to_2d_gt_map(gt_img, target_shape):
    """Convert GT map to 2D for visualization.

    Handles 2D and common 3D layouts such as [num_heights, H, W], [H, W, num_heights],
    or volumetric-like tensors where one axis should be sliced.
    """
    arr = np.asarray(gt_img)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return resize_gt_map(arr, target_shape)

    if arr.ndim == 3:
        h_t, w_t = target_shape

        # Common case: stacked heights at axis 0 or axis -1
        if arr.shape[0] <= 16:
            arr2d = arr[0]
        elif arr.shape[-1] <= 16:
            arr2d = arr[..., 0]
        # If two axes already match target H,W, slice the remaining axis at center
        elif arr.shape[0] == h_t and arr.shape[1] == w_t:
            arr2d = arr[:, :, arr.shape[2] // 2]
        elif arr.shape[1] == h_t and arr.shape[2] == w_t:
            arr2d = arr[arr.shape[0] // 2, :, :]
        elif arr.shape[0] == h_t and arr.shape[2] == w_t:
            arr2d = arr[:, arr.shape[1] // 2, :]
        else:
            # Fallback: collapse along smallest axis
            axis = int(np.argmin(arr.shape))
            arr2d = np.mean(arr, axis=axis)

        return resize_gt_map(arr2d, target_shape)

    raise ValueError(f'Unsupported GT shape for visualization: {arr.shape}')

gt_sequence = []
for pkl_file in seq_files:
    try:
        step_num = int(pkl_file.replace('step_', '').replace('.pkl', ''))
        time_idx = step_to_time(step_num)
        gt_path = os.path.join(
            gt_base_dir,
            f'{step_num:010d}',
            f'sample_040_time_{time_idx}_view_0_{gt_suffix}.pkl'
        )

        if not os.path.exists(gt_path):
            gt_sequence.append(None)
            print(f'GT path not found: {gt_path}')
            continue

        with open(gt_path, 'rb') as f:
            gt_data = pickle.load(f)

        if isinstance(gt_data, dict):
            if 'top_vel' in gt_data:
                gt_sequence.append(gt_data['top_vel'])
                print(f'Loaded GT for {pkl_file}: {gt_path}')
            elif 'w_map' in gt_data:
                gt_sequence.append(gt_data['w_map'])
                print(f'Loaded GT w_map for {pkl_file}: {gt_path}')
            else:
                gt_sequence.append(None)
                print(f'GT missing top_vel/w_map in {gt_path}. Keys: {list(gt_data.keys())}')
        else:
            gt_sequence.append(None)
            print(f'GT file is not a dict in {gt_path}: {type(gt_data)}')

    except Exception as e:
        gt_sequence.append(None)
        print(f'Could not load GT for {pkl_file}: {e}')

# --- Normalize input sequence using norm_const (max or 1.0) ---
sequence = sequence.astype(np.float32)
sequence_shdom_to_mitusuba = sequence * 131.4 * np.cos(np.deg2rad(35))
norm_const = 31.22 # this the norm that we do in the train
sequence_norm = sequence_shdom_to_mitusuba / norm_const

# --- Prepare input for model ---
# Model expects (B, T, C, H, W), here C=2
input_seq = torch.from_numpy(sequence_norm).float().unsqueeze(0).to(device)  # (1, T, 2, H, W)

# --- Load model ---
model = PretrainedTemporalUNetMitB1(in_channels=2, out_channels=1)
state = torch.load(model_path, map_location=device)
if 'model_state' in state:
    model.load_state_dict(state['model_state'], strict=False)
else:
    model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()

# --- Run inference ---
with torch.no_grad():
    pred_seq, _ = model(input_seq)
    pred_seq = pred_seq.squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)

# --- Denormalize predicted velocity ---
# Choose scale based on model name or GT files (envelope/slice height)
scale_map = {
    'envelop': 10.55,
    '500m': 3.7231,
    '1000m': 8.4183,
    '1500m': 11.8982,
}

def determine_scale():
    # Only inspect the model filename/path for keywords
    mp = model_path.lower()
    for key in ('envelop', 'envelope'):
        if key in mp:
            return scale_map['envelop'], f'model_path contains "{key}"'
    for key in ('500m', '1000m', '1500m'):
        if key in mp:
            return scale_map[key], f'found "{key}" in model_path'

    # Fallback: default to 1000m scale
    return scale_map['1000m'], 'fallback to 1000m (no keyword in model_path)'

scale, scale_reason = determine_scale()
print(f'[INFO] Using scale={scale} ({scale_reason})')
pred_seq = pred_seq * scale

# --- Fixed colorbar axes for all frames ---
vmin_pred = -6
vmax_pred = 6

# --- Scatter Plot Configuration (same as get_metrics.py) ---
SCATTER_BIN_WIDTH = 0.02
POINTS_PER_BIN = 20
SCATTER_RANGE = (-8.5, 8.5)

# Collect all predictions and GT for scatter plot
scatter_pred_list = []
scatter_gt_list = []

# --- Gamma correction helper ---
def apply_gamma(img, gamma=0.5):
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-6:
        return img
    img_norm = (img - img_min) / (img_max - img_min)
    img_corrected = np.power(img_norm, gamma)
    return img_corrected


def sample_scatter_points(gt_vals, pred_vals, label_suffix):
    """Balanced sampling for scatter plot using bins."""
    # Safety check: ensure no NaN/Inf in input
    valid_mask = np.isfinite(gt_vals) & np.isfinite(pred_vals)
    gt_vals = gt_vals[valid_mask]
    pred_vals = pred_vals[valid_mask]
    
    if len(gt_vals) == 0:
        print(f"[WARNING] No valid points after filtering{label_suffix}.")
        return None, None, None, None, None
    
    bins = np.arange(SCATTER_RANGE[0], SCATTER_RANGE[1] + SCATTER_BIN_WIDTH, SCATTER_BIN_WIDTH)
    bin_indices = np.digitize(gt_vals, bins)
    selected_indices = []
    unique_bins = np.unique(bin_indices)

    for b_idx in unique_bins:
        points_in_bin = np.where(bin_indices == b_idx)[0]
        n_sample = min(len(points_in_bin), POINTS_PER_BIN)
        if n_sample > 0:
            chosen = np.random.choice(points_in_bin, size=n_sample, replace=False)
            selected_indices.append(chosen)

    if len(selected_indices) > 0:
        final_indices = np.concatenate(selected_indices)
        np.random.shuffle(final_indices)
        x_scatter = gt_vals[final_indices]
        y_scatter = pred_vals[final_indices]
        print(f"[INFO] Selected {len(x_scatter)} points for scatter plot{label_suffix}.")
    else:
        print(f"[WARNING] Sampling failed{label_suffix}, using all points.")
        x_scatter = gt_vals
        y_scatter = pred_vals

    scatter_min = gt_vals.min()
    scatter_max = gt_vals.max()
    scatter_range_data = max(abs(scatter_min), abs(scatter_max))
    scatter_range_padded = scatter_range_data * 1.1
    
    # Ensure scatter_range_padded is valid
    if not np.isfinite(scatter_range_padded):
        print(f"[WARNING] Computed scatter_range_padded is not finite, using default 10.0{label_suffix}.")
        scatter_range_padded = 10.0
    
    return x_scatter, y_scatter, scatter_min, scatter_max, scatter_range_padded

# --- Create video with matplotlib 2x3 grid ---
fps = 2  # Define frames per second for video
video_writer = None
for t in range(sequence.shape[0]):
    input_img1 = sequence[t, 0]
    input_img2 = sequence[t, 1]
    pred_img = pred_seq[t]
    gt_img = gt_sequence[t] if t < len(gt_sequence) and gt_sequence[t] is not None else None

    # Mask logic (as in dataset) - optional
    if USE_MASK:
        mask = (input_img1 > 0.01).astype(np.uint8)
    else:
        mask = np.ones_like(input_img1, dtype=np.uint8)
    # Gamma correction for display
    sat1_disp = apply_gamma(input_img1)
    sat2_disp = apply_gamma(input_img2)
    if USE_MASK:
        pred_disp = np.ma.masked_where(mask == 0, pred_img)
    else:
        pred_disp = pred_img
    mask_disp = mask if USE_MASK else np.ones_like(mask)

    # Build figure: Row1: Sat1, Sat2, Mask  Row2: Pred Vel, GT Vel, Diff
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Sat1, Sat2, Mask
    im0 = axs[0, 0].imshow(sat1_disp, cmap='gray', vmin=0, vmax=1)
    axs[0, 0].set_title('Sat1')
    im1 = axs[0, 1].imshow(sat2_disp, cmap='gray', vmin=0, vmax=1)
    axs[0, 1].set_title('Sat2')
    im_mask = axs[0, 2].imshow(mask_disp, cmap='gray', vmin=0, vmax=1)
    axs[0, 2].set_title('Mask')

    # Row 2: Pred Vel, GT Vel, Diff
    im2 = axs[1, 0].imshow(pred_disp, cmap='jet', vmin=vmin_pred, vmax=vmax_pred)
    axs[1, 0].set_title('Pred Top Vel')

    im3 = None
    im4 = None
    if gt_img is not None:
        gt_img = to_2d_gt_map(gt_img, pred_img.shape)
        if USE_MASK:
            gt_disp = np.ma.masked_where(mask == 0, gt_img)
        else:
            gt_disp = gt_img
        im3 = axs[1, 1].imshow(gt_disp, cmap='jet', vmin=vmin_pred, vmax=vmax_pred)
        axs[1, 1].set_title('GT Top Vel')

        # Difference
        diff_img = pred_img - gt_img
        if USE_MASK:
            diff_disp = np.ma.masked_where(mask == 0, diff_img)
        else:
            diff_disp = diff_img
        
        # Collect for scatter plot (only masked/valid regions)
        if USE_MASK:
            valid_mask = (mask == 1)
            if np.any(valid_mask):
                scatter_pred_list.append(pred_img[valid_mask])
                scatter_gt_list.append(gt_img[valid_mask])
        else:
            scatter_pred_list.append(pred_img.flatten())
            scatter_gt_list.append(gt_img.flatten())
        
        im4 = axs[1, 2].imshow(diff_disp, cmap='RdBu_r', vmin=-3.0, vmax=3.0)
        axs[1, 2].set_title('Pred - GT')
    else:
        axs[1, 1].text(0.5, 0.5, 'GT Not Available', ha='center', va='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('GT Top Vel')
        axs[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axs[1, 2].transAxes)
        axs[1, 2].set_title('Pred - GT')

    # Colorbars
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    plt.colorbar(im_mask, ax=axs[0, 2], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    if gt_img is not None:
        plt.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im4, ax=axs[1, 2], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    frame_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, 3)
    plt.close(fig)
    # Init video writer with correct size
    if video_writer is None:
        # use getattr to avoid static-analysis warning in some environments
        fourcc_fn = getattr(cv2, 'VideoWriter_fourcc', None)
        if callable(fourcc_fn):
            fourcc = fourcc_fn(*'mp4v')
        else:
            fourcc = 0
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (img_w, img_h))
    video_writer.write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))

if video_writer is not None:
    video_writer.release()
print(f'Video saved to {output_video}')

# --- Generate Scatter Plots ---
if len(scatter_pred_list) > 0 and len(scatter_gt_list) > 0:
    print("\n[INFO] Generating scatter plots...")
    all_gt = np.concatenate(scatter_gt_list)
    all_pred = np.concatenate(scatter_pred_list)
    
    # Filter out NaN and Inf values
    valid_mask = np.isfinite(all_gt) & np.isfinite(all_pred)
    all_gt_clean = all_gt[valid_mask]
    all_pred_clean = all_pred[valid_mask]
    
    print(f"[INFO] Filtered {len(all_gt) - len(all_gt_clean)} invalid (NaN/Inf) points out of {len(all_gt)}")
    
    if len(all_gt_clean) > 0:
        # Balanced sampling for scatter plot
        x_scatter, y_scatter, scatter_min, scatter_max, scatter_range_padded = sample_scatter_points(
            all_gt_clean, all_pred_clean, " (all frames)"
        )
        
        if x_scatter is not None:
            print(f"[INFO] Scatter range: [{-scatter_range_padded:.2f}, {scatter_range_padded:.2f}]")
            
            # Create and save scatter plot
            fig_scatter, ax_scatter = plt.subplots(figsize=(12, 12), dpi=150)
            ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=50, alpha=0.4, rasterized=True)
            ax_scatter.plot(
                [-scatter_range_padded, scatter_range_padded],
                [-scatter_range_padded, scatter_range_padded],
                'k--', lw=3
            )
            ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=14, fontweight='bold')
            ax_scatter.set_ylabel("Predicted [m/s]", fontsize=14, fontweight='bold')
            ax_scatter.set_title(f"Balanced Scatter Plot (All Frames)", fontsize=16, fontweight='bold', pad=20)
            ax_scatter.set_xlim(-scatter_range_padded, scatter_range_padded)
            ax_scatter.set_ylim(-scatter_range_padded, scatter_range_padded)
            ax_scatter.grid(True, alpha=0.3, linewidth=1)
            plt.tight_layout()
            
            scatter_plot_path = os.path.join(output_dir, f'scatter_plot_{movie_level}.png')
            plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
            print(f'[INFO] Scatter plot saved to {scatter_plot_path}')
            plt.close(fig_scatter)
        else:
            print("[WARNING] Sampling failed, skipping scatter plot.")
    else:
        print("[WARNING] All scatter data is invalid (NaN/Inf after filtering). Skipping scatter plot.")
