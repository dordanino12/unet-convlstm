import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
import matplotlib as mpl

# Global font settings MUST be set FIRST before creating any figures
mpl.rcParams.update({
    'font.size': 60,              # base font size (bigger)
    'axes.titlesize': 64,         # axes title size
    'axes.labelsize': 56,         # X/Y label size
    'xtick.labelsize': 52,        # x tick labels
    'ytick.labelsize': 52,        # y tick labels
    'legend.fontsize': 52,        # legend if used
    'figure.titlesize': 68,       # figure suptitle
    'figure.dpi': 300,
    'savefig.dpi': 150,
    # PDF font embedding to preserve sizes
    'pdf.fonttype': 42,           # embed TrueType fonts
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import model classes
from train.unet import TemporalUNetDualView, NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet

# -----------------------------
# Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_MASK = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_W_top.npz"
CHECKPOINT_PATH = "/home/danino/PycharmProjects/pythonProject/models/resnet18_frozen_2lstm_layers_all_speed_skip.pt"
#CHECKPOINT_PATH = "/home/danino/PycharmProjects/pythonProject/models/resnet18_frozen_2lstm_layers_500m_slice_best_skip.pt"
save_path = "/home/danino/PycharmProjects/pythonProject/plots/evaluation_comprehensive.pdf"
output_dir = "/home/danino/PycharmProjects/pythonProject/plots/"

# Plotting Configuration
# --- UPDATED CONFIG FOR BALANCED SAMPLING ---
SCATTER_BIN_WIDTH = 0.5  # Width of each velocity bin (e.g., 0.5 m/s)
POINTS_PER_BIN = 1000  # How many points to sample from each bin (The "X" you requested)
SCATTER_RANGE = (-8.0, 8.0)  # Range to define bins over

HIST_BINS = 100  # Number of bins for histograms
min_y = None  # 7.5987958908081055
max_y = None  # 8.784920692443848

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
# 3. Process Validation Dataset Only
# -----------------------------
full_dataset = NPZSequenceDataset(NPZ_PATH, min_y=min_y, max_y=max_y)


# Re-create the split exactly as in training
n_train = int(0.8 * len(full_dataset))
n_val = len(full_dataset) - n_train

# Use the same seed generator
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=generator)

print(f"[INFO] Dataset loaded. Evaluating on VALIDATION set only ({len(val_ds)} sequences)")

# Lists to store pixel values
scatter_gt_list = []
scatter_pred_list = []
scatter_time_list = []

print("[INFO] Starting evaluation...")

for i in tqdm(range(len(val_ds)), desc="Evaluating"):

    # Get item from Validation Dataset
    input_seq, gt_vel_seq, mask_seq = val_ds[i]

    x_input = input_seq.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, _ = model(x_input)

    if isinstance(output, list):
        pred_tensor = torch.stack(output, dim=1)
    else:
        pred_tensor = output

    pred_vel = pred_tensor.squeeze(0).cpu().numpy()

    # Denormalize
    gt_vel_denorm = full_dataset.denormalize(gt_vel_seq)
    pred_vel_denorm = full_dataset.denormalize(pred_vel)

    # --- Masking Logic ---
    if USE_MASK:
        mask_np = mask_seq.cpu().numpy()
        gt_np = gt_vel_denorm.cpu().numpy()

        valid_pixels = (mask_np > 0.1)

        if np.any(valid_pixels):
            # Extract valid values
            seq_gt_vals = gt_np[valid_pixels]
            seq_pred_vals = pred_vel_denorm[valid_pixels]

            # Extract time indices
            t_idx = np.nonzero(valid_pixels)[0]
            seq_time_vals = t_idx.astype(np.float32)

            scatter_gt_list.append(seq_gt_vals)
            scatter_pred_list.append(seq_pred_vals)
            scatter_time_list.append(seq_time_vals)
    else:
        # No mask logic
        seq_gt_vals = gt_vel_denorm.cpu().numpy().flatten()
        seq_pred_vals = pred_vel_denorm.flatten()
        T, H, W = gt_vel_denorm.shape
        seq_time_vals = np.repeat(np.arange(T), H * W)

        scatter_gt_list.append(seq_gt_vals)
        scatter_pred_list.append(seq_pred_vals)
        scatter_time_list.append(seq_time_vals)

# -----------------------------
# 4. Global Stats & Plotting
# -----------------------------

if len(scatter_gt_list) > 0:
    # Concatenate all pixels
    all_gt = np.concatenate(scatter_gt_list)
    all_pred = np.concatenate(scatter_pred_list)
    all_time = np.concatenate(scatter_time_list)

    all_diff = all_pred - all_gt

    # Calculate Global Metrics
    global_mae = np.mean(np.abs(all_diff))
    global_rmse = np.sqrt(np.mean(all_diff ** 2))
    global_mean_err = np.mean(all_diff)
    global_std_err = np.std(all_diff)

    print("\n" + "=" * 40)
    print(f"Global MAE:        {global_mae:.4f} m/s")
    print(f"Global RMSE:       {global_rmse:.4f} m/s")
    print(f"Global Mean Error (Bias): {global_mean_err:.4f} m/s")
    print(f"Global Error Std:  {global_std_err:.4f} m/s")
    print("=" * 40)

    # -----------------------------
    # 5. Generate Individual PDF Plots
    # -----------------------------
    print("[INFO] Generating Individual PDF Plots...")

    # --- 1. SCATTER PLOT (Updated: Balanced/Stratified Sampling) ---
    print(f"[INFO] Performing Balanced Sampling for Scatter Plot...")
    print(f"       Bins Width: {SCATTER_BIN_WIDTH}, Points per Bin: {POINTS_PER_BIN}")

    # Create bins for the Ground Truth values
    bins = np.arange(SCATTER_RANGE[0], SCATTER_RANGE[1] + SCATTER_BIN_WIDTH, SCATTER_BIN_WIDTH)

    # Assign each GT value to a bin index
    # np.digitize returns indices starting from 1
    bin_indices = np.digitize(all_gt, bins)

    selected_indices = []

    # Iterate over each bin (1 to len(bins))
    unique_bins = np.unique(bin_indices)

    for b_idx in unique_bins:
        # Find all data points falling into this bin
        points_in_bin = np.where(bin_indices == b_idx)[0]

        # Determine how many to sample (min of available points or target limit)
        n_sample = min(len(points_in_bin), POINTS_PER_BIN)

        if n_sample > 0:
            # Randomly select indices
            chosen = np.random.choice(points_in_bin, size=n_sample, replace=False)
            selected_indices.append(chosen)

    if len(selected_indices) > 0:
        final_indices = np.concatenate(selected_indices)
        # Shuffle them so they don't plot in order of bins (visual aesthetics)
        np.random.shuffle(final_indices)

        x_scatter = all_gt[final_indices]
        y_scatter = all_pred[final_indices]
        print(f"[INFO] Selected {len(x_scatter)} points total for balanced scatter plot.")
    else:
        # Fallback if something fails
        print("[WARNING] Sampling failed, using all points.")
        x_scatter = all_gt
        y_scatter = all_pred

    # Create scatter plot figure
    fig_scatter, ax_scatter = plt.subplots(figsize=(20, 20), dpi=150)
    ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=8, alpha=0.3)
    ax_scatter.plot([-7.5, 7.5], [-7.5, 7.5], 'k--', lw=4)
    ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=56, fontweight='bold')
    ax_scatter.set_ylabel("Predicted [m/s]", fontsize=56, fontweight='bold')
    ax_scatter.set_title(f"Balanced Scatter Plot", fontsize=64, fontweight='bold', pad=40)
    ax_scatter.set_xlim(-7.5, 7.5)
    ax_scatter.set_ylim(-7.5, 7.5)
    ax_scatter.grid(True, alpha=0.3, linewidth=2)
    ax_scatter.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "scatter_plot.pdf")
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)
    print(f"  Saved: scatter_plot.pdf")

    # --- 2. MAE OVER TIME (Updated: With Variance/Std Shading) ---
    unique_times = np.unique(all_time)
    time_mae = []
    time_std = []
    time_steps = []

    for t in sorted(unique_times):
        mask_t = (all_time == t)
        if np.any(mask_t):
            # Calculate Absolute Errors for this time step
            abs_errors_t = np.abs(all_diff[mask_t])

            # Mean of Absolute Errors
            mean_val = np.mean(abs_errors_t)
            # Standard Deviation of Absolute Errors
            std_val = np.std(abs_errors_t)

            time_mae.append(mean_val)
            time_std.append(std_val)
            time_steps.append(t)

    time_mae = np.array(time_mae)
    time_std = np.array(time_std)
    time_steps = np.array(time_steps)

    # Create MAE over time figure
    fig_time, ax_time = plt.subplots(figsize=(24, 12), dpi=150)
    upper_err = time_std
    lower_err = np.zeros_like(time_std)
    ax_time.errorbar(time_steps, time_mae, yerr=[lower_err, upper_err],
                     fmt='o-', color='darkblue', ecolor='red',
                     elinewidth=4, capsize=8, markersize=12, linewidth=4,
                     label='MAE with Std Dev (above) [m/s]')
    ax_time.set_xlabel("Time Step", fontsize=56, fontweight='bold')
    ax_time.set_ylabel("MAE [m/s]", fontsize=56, fontweight='bold')
    ax_time.set_ylim(-1.5, 1.5)
    ax_time.set_title("Mean Absolute Error over Sequence Time", fontsize=64, fontweight='bold', pad=40)
    ax_time.grid(True, alpha=0.3, linewidth=2)
    ax_time.legend(fontsize=48, loc='best')
    ax_time.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    time_path = os.path.join(output_dir, "mae_over_time.pdf")
    plt.savefig(time_path, dpi=150)
    plt.close(fig_time)
    print(f"  Saved: mae_over_time.pdf")

    # --- 3. HISTOGRAMS ---
    hist_range = (-7.5, 7.5)

    # A. GT Distribution
    mu_gt, std_gt = np.mean(all_gt), np.std(all_gt)
    fig_hist_gt, ax_hist_gt = plt.subplots(figsize=(20, 16), dpi=150)
    ax_hist_gt.hist(all_gt, bins=HIST_BINS, range=hist_range, color='green', alpha=0.7, density=True, linewidth=2)
    ax_hist_gt.set_title(f"Ground Truth Distribution\n$\mu={mu_gt:.2f}, \sigma={std_gt:.2f}$",
                         fontsize=64, fontweight='bold', pad=40)
    ax_hist_gt.set_xlabel("Velocity [m/s]", fontsize=56, fontweight='bold')
    ax_hist_gt.set_ylabel("Density", fontsize=56, fontweight='bold')
    ax_hist_gt.set_xlim(hist_range)
    ax_hist_gt.grid(True, alpha=0.3, linewidth=2)
    ax_hist_gt.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    gt_hist_path = os.path.join(output_dir, "histogram_gt.pdf")
    plt.savefig(gt_hist_path, dpi=150)
    plt.close(fig_hist_gt)
    print(f"  Saved: histogram_gt.pdf")

    # B. Pred Distribution
    mu_pred, std_pred = np.mean(all_pred), np.std(all_pred)
    fig_hist_pred, ax_hist_pred = plt.subplots(figsize=(20, 16), dpi=150)
    ax_hist_pred.hist(all_pred, bins=HIST_BINS, range=hist_range, color='orange', alpha=0.7, density=True, linewidth=2)
    ax_hist_pred.set_title(f"Prediction Distribution\n$\mu={mu_pred:.2f}, \sigma={std_pred:.2f}$",
                           fontsize=64, fontweight='bold', pad=40)
    ax_hist_pred.set_xlabel("Velocity [m/s]", fontsize=56, fontweight='bold')
    ax_hist_pred.set_ylabel("Density", fontsize=56, fontweight='bold')
    ax_hist_pred.set_xlim(hist_range)
    ax_hist_pred.grid(True, alpha=0.3, linewidth=2)
    ax_hist_pred.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    pred_hist_path = os.path.join(output_dir, "histogram_pred.pdf")
    plt.savefig(pred_hist_path, dpi=150)
    plt.close(fig_hist_pred)
    print(f"  Saved: histogram_pred.pdf")

    # C. Error Distribution (Pred - GT)
    err_range = (-3, 3)
    mu_err, std_err = np.mean(all_diff), np.std(all_diff)
    fig_hist_err, ax_hist_err = plt.subplots(figsize=(20, 16), dpi=150)
    ax_hist_err.hist(all_diff, bins=HIST_BINS, range=err_range, color='red', alpha=0.7, density=True, linewidth=2)
    ax_hist_err.set_title(f"Error Distribution (Pred - GT)\n$\mu={mu_err:.2f}, \sigma={std_err:.2f}$",
                          fontsize=64, fontweight='bold', pad=40)
    ax_hist_err.set_xlabel("Error [m/s]", fontsize=56, fontweight='bold')
    ax_hist_err.set_ylabel("Density", fontsize=56, fontweight='bold')
    ax_hist_err.set_xlim(err_range)
    ax_hist_err.grid(True, alpha=0.3, linewidth=2)
    ax_hist_err.axvline(0, color='k', linestyle='--', lw=4)
    ax_hist_err.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    err_hist_path = os.path.join(output_dir, "histogram_error.pdf")
    plt.savefig(err_hist_path, dpi=150)
    plt.close(fig_hist_err)
    print(f"  Saved: histogram_error.pdf")

    print(f"[INFO] All individual PDFs saved to {output_dir}")

else:
    print("[WARNING] No valid pixels found to plot.")
