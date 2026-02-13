import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
import matplotlib as mpl
from PIL import Image

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
from train.dataset import NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet

# -----------------------------
# Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_MASK = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_W_top_w.npz"
CHECKPOINT_PATH = "/home/danino/PycharmProjects/pythonProject/models/resnet18_trainable_2lstm_layers_envelop_best_skip.pt"
save_path = "/home/danino/PycharmProjects/pythonProject/plots/evaluation_comprehensive.pdf"
output_dir = "/home/danino/PycharmProjects/pythonProject/plots/"

# Plotting Configuration
# --- UPDATED CONFIG FOR BALANCED SAMPLING ---
SCATTER_BIN_WIDTH = 0.05  # Width of each velocity bin (e.g., 0.5 m/s)
POINTS_PER_BIN = 150  # How many points to sample from each bin (The "X" you requested)
SCATTER_RANGE = (-8.0, 8.0)  # Range to define bins over

HIST_BINS = 100  # Number of bins for histograms
min_y = None  # 7.5987958908081055
max_y = None  # 8.784920692443848

# -----------------------------
# 2. Load Model Logic
# -----------------------------
print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

cfg = checkpoint.get('config', {})

print(f"[INFO] Loading ResNet18 Model...")
model = PretrainedTemporalUNet(
    out_channels=1,
    lstm_layers=2,
    freeze_encoder=cfg.get('freeze_encoder', True)
)

model.load_state_dict(checkpoint['model_state'])
model.to(DEVICE)
model.eval()

# -----------------------------
# 3. Process Validation Dataset Only
# -----------------------------
full_dataset = NPZSequenceDataset(NPZ_PATH)


# Re-create the split exactly as in training (70% train, 15% val, 15% test)
n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

# Use the same seed generator as in training
generator = torch.Generator().manual_seed(42)
train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

# Evaluate on TEST set
eval_ds = test_ds
print(f"[INFO] Dataset loaded. Evaluating on TEST set only ({len(eval_ds)} sequences)")

# Lists to store pixel values
scatter_gt_list = []
scatter_pred_list = []
scatter_time_list = []

print("[INFO] Starting evaluation...")

for i in tqdm(range(len(eval_ds)), desc="Evaluating"):

    # Get item from Test Dataset
    input_seq, gt_vel_seq, mask_seq = eval_ds[i]

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
        # Handle both 3D (T, H, W) and 4D (T, C, H, W) shapes
        gt_shape = gt_vel_denorm.shape
        if len(gt_shape) == 4:
            T, C, H, W = gt_shape
            pixels_per_frame = C * H * W
        else:
            T, H, W = gt_shape
            pixels_per_frame = H * W
        seq_time_vals = np.repeat(np.arange(T), pixels_per_frame)

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

    def sample_scatter_points(gt_vals, pred_vals, label_suffix):
        # Balanced sampling for scatter plot
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
            print(f"[INFO] Selected {len(x_scatter)} points total for balanced scatter plot{label_suffix}.")
        else:
            print(f"[WARNING] Sampling failed{label_suffix}, using all points.")
            x_scatter = gt_vals
            y_scatter = pred_vals

        scatter_min = min(gt_vals.min(), pred_vals.min())
        scatter_max = max(gt_vals.max(), pred_vals.max())
        scatter_range_data = max(abs(scatter_min), abs(scatter_max))
        scatter_range_padded = scatter_range_data * 1.1
        return x_scatter, y_scatter, scatter_min, scatter_max, scatter_range_padded

    # -----------------------------
    # 5. Generate Individual PDF Plots
    # -----------------------------
    print("[INFO] Generating Individual PDF Plots...")

    # --- 1. SCATTER PLOT (All Time Steps) ---
    print(f"[INFO] Performing Balanced Sampling for Scatter Plot (all time steps)...")
    print(f"       Bins Width: {SCATTER_BIN_WIDTH}, Points per Bin: {POINTS_PER_BIN}")

    x_scatter, y_scatter, scatter_min, scatter_max, scatter_range_padded = sample_scatter_points(
        all_gt, all_pred, " (all time steps)"
    )

    hist_range = (scatter_min, scatter_max)
    err_min = all_diff.min()
    err_max = all_diff.max()
    err_range = (err_min * 1.1, err_max * 1.1)

    print(f"[INFO] Dynamic Ranges Calculated:")
    print(f"       Scatter Plot Range: [{-scatter_range_padded:.2f}, {scatter_range_padded:.2f}]")
    print(f"       Histogram Range: [{hist_range[0]:.2f}, {hist_range[1]:.2f}]")
    print(f"       Error Range: [{err_range[0]:.2f}, {err_range[1]:.2f}]")

    # Create scatter plot figure
    fig_scatter, ax_scatter = plt.subplots(figsize=(20, 20), dpi=150)
    ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=8, alpha=0.3)
    ax_scatter.plot([-scatter_range_padded, scatter_range_padded], [-scatter_range_padded, scatter_range_padded], 'k--', lw=4)
    ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=56, fontweight='bold')
    ax_scatter.set_ylabel("Predicted [m/s]", fontsize=56, fontweight='bold')
    ax_scatter.set_title(f"Balanced Scatter Plot", fontsize=64, fontweight='bold', pad=40)
    ax_scatter.set_xlim(-scatter_range_padded, scatter_range_padded)
    ax_scatter.set_ylim(-scatter_range_padded, scatter_range_padded)
    ax_scatter.grid(True, alpha=0.3, linewidth=2)
    ax_scatter.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "scatter_plot.pdf")
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)
    print(f"  Saved: scatter_plot.pdf")

    # --- 2. SCATTER PLOT (Time Step 5 Only) ---
    time5_mask = (all_time == 5)
    if np.any(time5_mask):
        scatter_gt_t5 = all_gt[time5_mask]
        scatter_pred_t5 = all_pred[time5_mask]
        print(f"[INFO] Performing Balanced Sampling for Scatter Plot (time step 5 only)...")

        x_scatter_t5, y_scatter_t5, scatter_min_t5, scatter_max_t5, scatter_range_padded_t5 = sample_scatter_points(
            scatter_gt_t5, scatter_pred_t5, " (time step 5)"
        )

        fig_scatter_t5, ax_scatter_t5 = plt.subplots(figsize=(20, 20), dpi=150)
        ax_scatter_t5.scatter(x_scatter_t5, y_scatter_t5, c='tab:blue', s=8, alpha=0.3)
        ax_scatter_t5.plot([-scatter_range_padded_t5, scatter_range_padded_t5], [-scatter_range_padded_t5, scatter_range_padded_t5], 'k--', lw=4)
        ax_scatter_t5.set_xlabel("Ground Truth [m/s]", fontsize=56, fontweight='bold')
        ax_scatter_t5.set_ylabel("Predicted [m/s]", fontsize=56, fontweight='bold')
        ax_scatter_t5.set_title(f"Balanced Scatter Plot (Time Step 5)", fontsize=64, fontweight='bold', pad=40)
        ax_scatter_t5.set_xlim(-scatter_range_padded_t5, scatter_range_padded_t5)
        ax_scatter_t5.set_ylim(-scatter_range_padded_t5, scatter_range_padded_t5)
        ax_scatter_t5.grid(True, alpha=0.3, linewidth=2)
        ax_scatter_t5.tick_params(axis='both', which='major', labelsize=52)
        plt.tight_layout()
        scatter_path_t5 = os.path.join(output_dir, "scatter_plot_t5.pdf")
        plt.savefig(scatter_path_t5, dpi=150)
        plt.close(fig_scatter_t5)
        print(f"  Saved: scatter_plot_t5.pdf")
    else:
        print("[WARNING] No samples found for time step 5. Skipping time-step-5 scatter plot.")

    # --- 2. MAE OVER TIME ---
    unique_times = np.unique(all_time)
    time_mae = []
    time_steps = []

    for t in sorted(unique_times):
        mask_t = (all_time == t)
        if np.any(mask_t):
            # Calculate Absolute Errors for this time step
            abs_errors_t = np.abs(all_diff[mask_t])

            # Mean of Absolute Errors
            mean_val = np.mean(abs_errors_t)
            time_mae.append(mean_val)
            time_steps.append(t)

    time_mae = np.array(time_mae)
    time_steps = np.array(time_steps)

    # Create MAE over time figure
    fig_time, ax_time = plt.subplots(figsize=(24, 12), dpi=150)
    ax_time.plot(time_steps, time_mae, 'o-', color='darkblue',
                 markersize=12, linewidth=4, label='MAE [m/s]')
    ax_time.set_xlabel("Time Step", fontsize=56, fontweight='bold')
    ax_time.set_ylabel("MAE [m/s]", fontsize=56, fontweight='bold')
    ax_time.set_ylim(0, 1)
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

    # --- 4. COMBINE ALL PDFS INTO ONE PNG GRID ---
    print("[INFO] Converting PDFs to PNG and creating combined grid...")

    pdf_files = [
        scatter_path,
        time_path,
        gt_hist_path,
        pred_hist_path,
        err_hist_path
    ]

    # Check if all PDFs exist
    existing_pdfs = [f for f in pdf_files if os.path.exists(f)]
    print(f"  Found {len(existing_pdfs)} PDF files to convert")

    # We'll recreate the plots as PNG and combine them
    images = []

    # A. Scatter Plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(12, 12), dpi=100)
    ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=8, alpha=0.3)
    ax_scatter.plot([-scatter_range_padded, scatter_range_padded], [-scatter_range_padded, scatter_range_padded], 'k--', lw=4)
    ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=56, fontweight='bold')
    ax_scatter.set_ylabel("Predicted [m/s]", fontsize=56, fontweight='bold')
    ax_scatter.set_title(f"Balanced Scatter Plot", fontsize=64, fontweight='bold', pad=40)
    ax_scatter.set_xlim(-scatter_range_padded, scatter_range_padded)
    ax_scatter.set_ylim(-scatter_range_padded, scatter_range_padded)
    ax_scatter.grid(True, alpha=0.3, linewidth=2)
    ax_scatter.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    # Save to PNG buffer
    img_scatter_path = os.path.join(output_dir, ".temp_scatter.png")
    plt.savefig(img_scatter_path, dpi=100, format='png')
    plt.close(fig_scatter)
    images.append(Image.open(img_scatter_path))
    print(f"    Converted: scatter_plot.pdf")

    # B. MAE Over Time
    fig_time, ax_time = plt.subplots(figsize=(16, 10), dpi=100)
    ax_time.plot(time_steps, time_mae, 'o-', color='darkblue',
                 markersize=12, linewidth=4, label='MAE [m/s]')
    ax_time.set_xlabel("Time Step", fontsize=56, fontweight='bold')
    ax_time.set_ylabel("MAE [m/s]", fontsize=56, fontweight='bold')
    ax_time.set_ylim(0, 1)
    ax_time.set_title("Mean Absolute Error over Sequence Time", fontsize=64, fontweight='bold', pad=40)
    ax_time.grid(True, alpha=0.3, linewidth=2)
    ax_time.legend(fontsize=48, loc='best')
    ax_time.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    img_time_path = os.path.join(output_dir, ".temp_time.png")
    plt.savefig(img_time_path, dpi=100, format='png')
    plt.close(fig_time)
    images.append(Image.open(img_time_path))
    print(f"    Converted: mae_over_time.pdf")

    # C. GT Histogram
    fig_hist_gt, ax_hist_gt = plt.subplots(figsize=(12, 12), dpi=100)
    ax_hist_gt.hist(all_gt, bins=HIST_BINS, range=hist_range, color='green', alpha=0.7, density=True, linewidth=2)
    ax_hist_gt.set_title(f"Ground Truth Distribution\n$\mu={mu_gt:.2f}, \sigma={std_gt:.2f}$",
                         fontsize=64, fontweight='bold', pad=40)
    ax_hist_gt.set_xlabel("Velocity [m/s]", fontsize=56, fontweight='bold')
    ax_hist_gt.set_ylabel("Density", fontsize=56, fontweight='bold')
    ax_hist_gt.set_xlim(hist_range)
    ax_hist_gt.grid(True, alpha=0.3, linewidth=2)
    ax_hist_gt.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    img_gt_hist_path = os.path.join(output_dir, ".temp_hist_gt.png")
    plt.savefig(img_gt_hist_path, dpi=100, format='png')
    plt.close(fig_hist_gt)
    images.append(Image.open(img_gt_hist_path))
    print(f"    Converted: histogram_gt.pdf")

    # D. Pred Histogram
    fig_hist_pred, ax_hist_pred = plt.subplots(figsize=(12, 12), dpi=100)
    ax_hist_pred.hist(all_pred, bins=HIST_BINS, range=hist_range, color='orange', alpha=0.7, density=True, linewidth=2)
    ax_hist_pred.set_title(f"Prediction Distribution\n$\mu={mu_pred:.2f}, \sigma={std_pred:.2f}$",
                           fontsize=64, fontweight='bold', pad=40)
    ax_hist_pred.set_xlabel("Velocity [m/s]", fontsize=56, fontweight='bold')
    ax_hist_pred.set_ylabel("Density", fontsize=56, fontweight='bold')
    ax_hist_pred.set_xlim(hist_range)
    ax_hist_pred.grid(True, alpha=0.3, linewidth=2)
    ax_hist_pred.tick_params(axis='both', which='major', labelsize=52)
    plt.tight_layout()
    img_pred_hist_path = os.path.join(output_dir, ".temp_hist_pred.png")
    plt.savefig(img_pred_hist_path, dpi=100, format='png')
    plt.close(fig_hist_pred)
    images.append(Image.open(img_pred_hist_path))
    print(f"    Converted: histogram_pred.pdf")

    # E. Error Histogram
    fig_hist_err, ax_hist_err = plt.subplots(figsize=(12, 12), dpi=100)
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
    img_err_hist_path = os.path.join(output_dir, ".temp_hist_err.png")
    plt.savefig(img_err_hist_path, dpi=100, format='png')
    plt.close(fig_hist_err)
    images.append(Image.open(img_err_hist_path))
    print(f"    Converted: histogram_error.pdf")

    # --- 5. CREATE COMBINED GRID PNG ---
    print("[INFO] Creating combined grid PNG...")

    if len(images) == 5:
        # Create a 3x2 grid layout (3 columns, 2 rows for 5 images)
        cols = 3
        rows = 2

        # Resize images to same size for cleaner grid
        target_width = 1200
        target_height = 900
        resized_images = []
        for img in images:
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized_images.append(img_resized)

        # Create combined grid
        grid_width = cols * target_width
        grid_height = rows * target_height
        combined_img = Image.new('RGB', (grid_width, grid_height), color='white')

        # Paste images into grid
        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            x = col * target_width
            y = row * target_height
            combined_img.paste(img, (x, y))

        # Save combined image
        combined_png_path = os.path.join(output_dir, "all_metrics_combined.png")
        combined_img.save(combined_png_path, format='PNG', quality=95)
        print(f"  [INFO] Combined PNG grid saved: {combined_png_path}")
        print(f"  Grid size: {grid_width}x{grid_height} pixels")

        # Clean up temp files
        for temp_file in [img_scatter_path, img_time_path, img_gt_hist_path, img_pred_hist_path, img_err_hist_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        print(f"  [WARNING] Expected 5 images, found {len(images)}. Skipping grid creation.")

else:
    print("[WARNING] No valid pixels found to plot.")
