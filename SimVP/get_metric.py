"""
SimVP Evaluation Script: Metrics & Balanced Scatter Plot
Evaluates the trained SimVP model on the test set, calculates global metrics
(MAE, RMSE, Bias, Std), and generates a publication-ready balanced scatter plot.
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import SimVP modules
from model import SimVP
# Directly import the dataset class to bypass train/val loading
from API.dataloader_velocity import VelocityPredictionDataset

# ---------------------------------------------------------
# Argument Parser Setup
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='SimVP Evaluation Script')

# Paths
parser.add_argument('--fold_dir', type=str,
                    default="data/wacv_data/1000m_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2",
                    help='Path to the data fold directory')
parser.add_argument('--weights_path', type=str,
                    default="SimVP/results/velocity_simvp_binlos_1000m_fold_01_val_r0-1_c0-2/checkpoint.pth",
                    help='Path to the trained model weights')
parser.add_argument('--output_dir', type=str, default="plots/simvp_evaluation/",
                    help='Directory to save the evaluation plots')

# Mask toggle
parser.add_argument('--use_mask', action='store_true', help='Evaluate only on masked (cloud) pixels')
parser.add_argument('--no_mask', dest='use_mask', action='store_false',
                    help='Evaluate on the entire image (ignore mask)')
parser.set_defaults(use_mask=True)

args = parser.parse_args()

# ---------------------------------------------------------
# Plotting Font & Style Configuration (MUST BE SET FIRST)
# ---------------------------------------------------------
mpl.rcParams.update({
    'font.size': 60,  # base font size (bigger)
    'axes.titlesize': 64,  # axes title size
    'axes.labelsize': 56,  # X/Y label size
    'xtick.labelsize': 52,  # x tick labels
    'ytick.labelsize': 52,  # y tick labels
    'legend.fontsize': 52,  # legend if used
    'figure.titlesize': 68,  # figure suptitle
    'figure.dpi': 300,
    'savefig.dpi': 150,
    'pdf.fonttype': 42,  # embed TrueType fonts
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# -----------------------------
# Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (Assigned from arguments)
FOLD_DIR = args.fold_dir
WEIGHTS_PATH = args.weights_path
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting Configuration
SCATTER_BIN_WIDTH = 0.02  # Width of each velocity bin (e.g., 0.02 m/s)
POINTS_PER_BIN = 5  # How many points to sample from each bin
SCATTER_RANGE = (-8.5, 8.5)  # Range to define bins over
TEXT_FOR_SCATTER = "SimVP Prediction"

# -----------------------------
# 1. Load Data (TEST SET ONLY)
# -----------------------------
print(f"[INFO] Evaluating with mask: {args.use_mask}")
print(f"[INFO] Data Directory: {FOLD_DIR}")
print(f"[INFO] Initializing DataLoader for Test Set...")

# Extract test_root from fold_dir (one level up)
test_root = os.path.dirname(FOLD_DIR.rstrip('/'))
test_path = os.path.join(test_root, 'test_w.npz')

if not os.path.exists(test_path):
    raise FileNotFoundError(f"Cannot find test file at {test_path}")

# Load only the test dataset directly
test_set = VelocityPredictionDataset(
    test_path,
    is_train=False,
    use_one_satellite=False,
    augment=False
)

test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

scale = test_set.scale
print(f"[INFO] Dataset loaded successfully from {test_path}. Using velocity scale factor: {scale:.4f}")

# -----------------------------
# 2. Build & Load Model
# -----------------------------
print("[INFO] Loading SimVP Model...")
in_shape = [12, 2, 128, 128]
hid_S, hid_T, N_S, N_T = 128, 512, 4, 8

model = SimVP(tuple(in_shape), hid_S, hid_T, N_S, N_T).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"[INFO] Successfully loaded weights from: {WEIGHTS_PATH}")

# -----------------------------
# 3. Evaluation Loop
# -----------------------------
scatter_gt_list = []
scatter_pred_list = []

print("[INFO] Starting evaluation on test set...")
with torch.no_grad():
    for batch_data in tqdm(test_loader, desc="Evaluating"):
        # Unpack data
        if len(batch_data) == 3:
            batch_x, batch_y, batch_mask = batch_data
        else:
            batch_x, batch_y = batch_data
            batch_mask = None

        # Check if we should ignore the mask
        if not args.use_mask:
            batch_mask = None

        batch_x = batch_x.to(DEVICE)

        # Forward pass
        pred_y_tensor = model(batch_x)

        # Move to CPU numpy arrays
        pred_y = pred_y_tensor.cpu().numpy()
        gt_y = batch_y.numpy()

        # If mask is None (either naturally or forced by --no_mask), evaluate entire image
        if batch_mask is not None:
            mask = batch_mask.numpy()
        else:
            mask = np.ones_like(gt_y)

        # Denormalize to physical values (m/s)
        pred_y = pred_y * scale
        gt_y = gt_y * scale

        # Extract only the first channel (velocity)
        # Shapes are [B, T, C, H, W], we select C=0
        pred_vel = pred_y[:, :, 0, :, :]
        gt_vel = gt_y[:, :, 0, :, :]
        mask_vel = mask[:, :, 0, :, :]

        # Isolate pixels that belong to the cloud (mask > 0.5)
        # If no_mask was used, mask_vel is all 1s, so valid_pixels includes the whole image
        valid_pixels = (mask_vel > 0.5)

        if np.any(valid_pixels):
            scatter_gt_list.append(gt_vel[valid_pixels])
            scatter_pred_list.append(pred_vel[valid_pixels])

# -----------------------------
# 4. Global Stats & Plotting
# -----------------------------
if len(scatter_gt_list) > 0:
    # Concatenate all valid cloud pixels from the entire test set
    all_gt = np.concatenate(scatter_gt_list)
    all_pred = np.concatenate(scatter_pred_list)
    all_diff = all_pred - all_gt

    # Calculate Global Metrics
    global_mae = np.mean(np.abs(all_diff))
    global_rmse = np.sqrt(np.mean(all_diff ** 2))
    global_mean_err = np.mean(all_diff)
    global_std_err = np.std(all_diff)

    print("\n" + "=" * 40)
    print(f"Global MAE:               {global_mae:.4f} m/s")
    print(f"Global RMSE:              {global_rmse:.4f} m/s")
    print(f"Global Mean Error (Bias): {global_mean_err:.4f} m/s")
    print(f"Global Error Std:         {global_std_err:.4f} m/s")
    print("=" * 40)


    # Balanced sampling function for scatter plot
    def sample_scatter_points(gt_vals, pred_vals):
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
            print(f"[INFO] Selected {len(x_scatter)} points total for balanced scatter plot.")
        else:
            print("[WARNING] Sampling failed, using all points.")
            x_scatter = gt_vals
            y_scatter = pred_vals

        scatter_min = gt_vals.min()
        scatter_max = gt_vals.max()
        scatter_range_data = max(abs(scatter_min), abs(scatter_max))
        scatter_range_padded = scatter_range_data * 1.1

        return x_scatter, y_scatter, scatter_range_padded


    # Generate Scatter Plot
    print(f"[INFO] Performing Balanced Sampling for Scatter Plot...")
    x_scatter, y_scatter, scatter_range_padded = sample_scatter_points(all_gt, all_pred)

    fig_scatter, ax_scatter = plt.subplots(figsize=(20, 20), dpi=150)
    ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=70, alpha=0.3, rasterized=True)

    # Plot the perfect prediction line (y = x)
    ax_scatter.plot(
        [-scatter_range_padded, scatter_range_padded],
        [-scatter_range_padded, scatter_range_padded],
        'k--', lw=4
    )

    ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=80, fontweight='bold')
    ax_scatter.set_ylabel("Inferred [m/s]", fontsize=80, fontweight='bold')
    ax_scatter.set_xlim(-scatter_range_padded, scatter_range_padded)
    ax_scatter.set_ylim(-scatter_range_padded, scatter_range_padded)

    # Set tick increments manually (adjust if needed based on your velocity range)
    ax_scatter.set_xticks([-5, 0, 5])
    ax_scatter.set_yticks([-5, 0, 5])

    ax_scatter.grid(True, alpha=0.3, linewidth=2)
    ax_scatter.tick_params(axis='both', which='major', labelsize=80)

    # Add textual annotation (e.g., model name or sequence criteria)
    ax_scatter.text(
        0.05, 0.95, TEXT_FOR_SCATTER, transform=ax_scatter.transAxes,
        fontsize=80, fontweight='bold', va='top', ha='left'
    )

    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, "simvp_scatter_plot.pdf")
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)

    print(f"[INFO] Scatter plot successfully saved to: {scatter_path}")

else:
    print("[WARNING] No valid cloud pixels found in the test set. Could not calculate metrics.")