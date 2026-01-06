import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

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
NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_500m_slices_w.npz"
#CHECKPOINT_PATH = "/home/danino/PycharmProjects/pythonProject/models/resnet18_frozen_2lstm_layers_all_speed_skip.pt"
CHECKPOINT_PATH = "/home/danino/PycharmProjects/pythonProject/models/resnet18_frozen_2lstm_layers_500m_slice_best_skip.pt"

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
    # 5. Generate Multi-Panel Plots
    # -----------------------------
    print("[INFO] Generating Plots...")

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(20, 11))

    fig.suptitle(f"Validation Set Evaluation Results", fontsize=20, fontweight='bold')

    gs = fig.add_gridspec(2, 3)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1:])  # Span 2 columns
    ax_hist_gt = fig.add_subplot(gs[1, 0])
    ax_hist_pred = fig.add_subplot(gs[1, 1])
    ax_hist_err = fig.add_subplot(gs[1, 2])

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

    # Plot
    ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=4, alpha=0.3)

    ax_scatter.plot([-7.5, 7.5], [-7.5, 7.5], 'k--', lw=1.5)
    ax_scatter.set_xlabel("Ground Truth [m/s]")
    ax_scatter.set_ylabel("Predicted [m/s]")
    ax_scatter.set_title(f"Balanced Scatter Plot (Fixed {POINTS_PER_BIN}/bin)")
    ax_scatter.set_xlim(-7.5, 7.5)
    ax_scatter.set_ylim(-7.5, 7.5)
    ax_scatter.grid(True, alpha=0.3)

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

    # Plot Mean line
    #ax_time.plot(time_steps, time_mae, 'o-', color='darkblue', lw=2, label='Mean Absolute Error')


    ax_time.errorbar(time_steps, time_mae, yerr=time_std,
                     fmt='o-', color='darkblue', ecolor='red',
                     elinewidth=2, capsize=4, label='MAE ± 1 Std Dev [m/s]')

    ax_time.set_xlabel("Time Step")
    ax_time.set_ylabel("MAE [m/s]")
    ax_time.set_ylim(-1.5, 1.5)
    ax_time.set_title("Mean Absolute Error & Variability over Sequence Time")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    # --- 3. HISTOGRAMS ---
    hist_range = (-7.5, 7.5)

    # A. GT Distribution
    mu_gt, std_gt = np.mean(all_gt), np.std(all_gt)
    ax_hist_gt.hist(all_gt, bins=HIST_BINS, range=hist_range, color='green', alpha=0.7, density=True)
    ax_hist_gt.set_title(f"GT Distribution\n$\mu={mu_gt:.2f}, \sigma={std_gt:.2f}$")
    ax_hist_gt.set_xlabel("Velocity [m/s]")
    ax_hist_gt.set_ylabel("Density")
    ax_hist_gt.set_xlim(hist_range)
    ax_hist_gt.grid(True, alpha=0.3)

    # B. Pred Distribution
    mu_pred, std_pred = np.mean(all_pred), np.std(all_pred)
    ax_hist_pred.hist(all_pred, bins=HIST_BINS, range=hist_range, color='orange', alpha=0.7, density=True)
    ax_hist_pred.set_title(f"Prediction Distribution\n$\mu={mu_pred:.2f}, \sigma={std_pred:.2f}$")
    ax_hist_pred.set_xlabel("Velocity [m/s]")
    ax_hist_pred.set_ylabel("Density")
    ax_hist_pred.set_xlim(hist_range)
    ax_hist_pred.grid(True, alpha=0.3)

    # UNIFY Y-AXIS FOR GT AND PRED
    _, y_max_gt = ax_hist_gt.get_ylim()
    _, y_max_pred = ax_hist_pred.get_ylim()
    common_y_max = max(y_max_gt, y_max_pred)
    ax_hist_gt.set_ylim(0, common_y_max)
    ax_hist_pred.set_ylim(0, common_y_max)

    # C. Error Distribution (Pred - GT)
    err_range = (-3, 3)
    mu_err, std_err = np.mean(all_diff), np.std(all_diff)
    ax_hist_err.hist(all_diff, bins=HIST_BINS, range=err_range, color='red', alpha=0.7, density=True)
    ax_hist_err.set_title(f"Error Distribution (Pred - GT)\n$\mu={mu_err:.2f}, \sigma={std_err:.2f}$")
    ax_hist_err.set_xlabel("Error [m/s]")
    ax_hist_err.set_ylabel("Density")
    ax_hist_err.set_xlim(err_range)
    ax_hist_err.grid(True, alpha=0.3)
    ax_hist_err.axvline(0, color='k', linestyle='--', lw=1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "/home/danino/PycharmProjects/pythonProject/plots/evaluation_comprehensive_slice_500m.png"
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Plot saved to {save_path}")
    # plt.show()

else:
    print("[WARNING] No valid pixels found to plot.")