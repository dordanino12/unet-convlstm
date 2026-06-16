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
from dataset import NPZSequenceDataset
from resnet18 import PretrainedTemporalUNet, PretrainedTemporalUNetMitB1, PretrainedTemporalUNetMitB2, PretrainedTemporalUNetMitB3

# -----------------------------
# Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_MASK = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GT_ENVELOPE_INPUT = False  # Set True when model expects GT envelope channel
BACKBONE = "mit_b1"  # "resnet18", "mit_b1", "mit_b2", or "mit_b3"
# Option: use only the first satellite image channel (single-sat mode)
USE_ONE_SATELLITE = False



# Paths
NPZ_TRAIN_PATH = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/1000m_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/train_w.npz"
NPZ_TEST_PATH = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/1500m_kfold_w_sensor_noise_both/test_w.npz"
CHECKPOINT_PATH = "/models/wacv/1000m/mit_b1_1000m_fold_02_val_r0-1_c4-6_best_bin_loss.pt"
KFOLD_MODELS_DIR = "/home/danino/PycharmProjects/pythonProject/models/wacv/1500m"
KFOLD_DATA_DIR = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/1500m_kfold_w_sensor_noise_both/"
#KFOLD_MODELS_DIR = None  # Set to None to disable k-fold ensemble mode
#KFOLD_DATA_DIR = None  # Set to None to disable k-fold ensemble mode
save_path = "/home/danino/PycharmProjects/pythonProject/plots/evaluation_comprehensive.pdf"
output_dir = "/home/danino/PycharmProjects/pythonProject/plots/"
# Option to disable ConvLSTM temporal processing entirely
USE_CONV_LSTM = True
ADD_SENSOR_NOISE = False  #
TEXT_FOR_SCATER = "z = 1500m"

# Plotting Configuration
# --- UPDATED CONFIG FOR BALANCED SAMPLING ---
SCATTER_BIN_WIDTH = 0.02  # Width of each velocity bin (e.g., 0.5 m/s)
POINTS_PER_BIN = 5 #How many points to sample from each bin (The "X" you requested)
SCATTER_RANGE = (-8.5, 8.5)  # Range to define bins over

HIST_BINS = 100  # Number of bins for histograms
min_y = None  # 7.5987958908081055
max_y = None  # 8.784920692443848

is_kfold_mode = (KFOLD_MODELS_DIR is not None) and (KFOLD_DATA_DIR is not None)
if is_kfold_mode:
    output_dir = os.path.join(output_dir, "ensemble_kfold")
else:
    output_dir = os.path.join(output_dir, "single_model")
os.makedirs(output_dir, exist_ok=True)


def _pick_existing(base_dir, names):
    for name in names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def _build_model_from_checkpoint(cfg, checkpoint_state, device, model_in_channels):
    has_refiner = any('refiner' in key for key in checkpoint_state.keys())
    refiner_hidden_channels = 32
    if has_refiner:
        for key in checkpoint_state.keys():
            if 'refiner.net.0.weight' in key:
                refiner_hidden_channels = checkpoint_state[key].shape[0]
                break

    if BACKBONE == "resnet18":
        print("[INFO] Loading ResNet18 Model...")
        model = PretrainedTemporalUNet(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=cfg.get('freeze_encoder', True),
            in_channels=model_in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=has_refiner,
            refiner_hidden_channels=refiner_hidden_channels
        )
    elif BACKBONE == "mit_b1":
        print("[INFO] Loading MiT-B1 Model...")
        model = PretrainedTemporalUNetMitB1(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=cfg.get('freeze_encoder', True),
            in_channels=model_in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=has_refiner,
            refiner_hidden_channels=refiner_hidden_channels
        )
    elif BACKBONE == "mit_b2":
        print("[INFO] Loading MiT-B2 Model...")
        model = PretrainedTemporalUNetMitB2(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=cfg.get('freeze_encoder', True),
            in_channels=model_in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=has_refiner,
            refiner_hidden_channels=refiner_hidden_channels
        )
    elif BACKBONE == "mit_b3":
        print("[INFO] Loading MiT-B3 Model...")
        model = PretrainedTemporalUNetMitB3(
            out_channels=1,
            lstm_layers=2 if USE_CONV_LSTM else 0,
            freeze_encoder=cfg.get('freeze_encoder', True),
            in_channels=model_in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=has_refiner,
            refiner_hidden_channels=refiner_hidden_channels
        )
    else:
        raise ValueError(f"Unsupported BACKBONE: {BACKBONE}")

    return model, has_refiner, refiner_hidden_channels

# -----------------------------
# 2. Process Validation Dataset Only
# -----------------------------
# Load train dataset for normalization/denormalization
train_dataset_for_norm = NPZSequenceDataset(
    NPZ_TRAIN_PATH,
    use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
    gt_envelope_npz_path=NPZ_TRAIN_PATH,
    use_one_satellite=USE_ONE_SATELLITE
)

# Load test dataset for evaluation
test_set = NPZSequenceDataset(
    NPZ_TEST_PATH,
    use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
    gt_envelope_npz_path=NPZ_TEST_PATH,
    use_one_satellite=USE_ONE_SATELLITE
)

# Match test normalization to train normalization for consistent metric denormalization.
test_set.scale = train_dataset_for_norm.scale
test_set.norm_const = train_dataset_for_norm.norm_const
_, C, _, _ = test_set[0][0].shape

# -----------------------------
# 3. Load Model Logic
# -----------------------------
models = []
fold_train_datasets = []

if not is_kfold_mode:
    print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint.get('config', {})
    ckpt_use_one_sat = cfg.get('use_one_satellite', None)
    if ckpt_use_one_sat is None:
        ckpt_use_one_sat = (cfg.get('in_channels', C) == 1)
    if ckpt_use_one_sat:
        model_in_channels = 1
    else:
        model_in_channels = cfg.get('in_channels', C)

    if ckpt_use_one_sat and not getattr(test_set, 'use_one_satellite', False):
        print('[INFO] Checkpoint indicates single-satellite input; reloading datasets in single-sat mode')
        train_dataset_for_norm = NPZSequenceDataset(
            NPZ_TRAIN_PATH,
            use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
            gt_envelope_npz_path=NPZ_TRAIN_PATH,
            use_one_satellite=True
        )
        test_set = NPZSequenceDataset(
            NPZ_TEST_PATH,
            use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
            gt_envelope_npz_path=NPZ_TEST_PATH,
            use_one_satellite=True
        )
        test_set.scale = train_dataset_for_norm.scale
        test_set.norm_const = train_dataset_for_norm.norm_const
        _, C, _, _ = test_set[0][0].shape

    checkpoint_state = checkpoint['model_state']
    model, has_refiner, refiner_hidden_channels = _build_model_from_checkpoint(cfg, checkpoint_state, DEVICE, model_in_channels)
    load_result = model.load_state_dict(checkpoint_state, strict=False)
    if load_result.missing_keys:
        print(f"[WARN] Missing keys when loading checkpoint: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[WARN] Unexpected keys when loading checkpoint: {load_result.unexpected_keys}")
    model.to(DEVICE)
    model.eval()
    models.append(model)
    fold_train_datasets.append(train_dataset_for_norm)

    if has_refiner:
        print(f"[INFO] ✓ Refiner ENABLED (hidden_channels={refiner_hidden_channels})")
    else:
        print(f"[INFO] ✗ Refiner DISABLED (checkpoint has no refiner weights)")
else:
    print(f"[INFO] K-FOLD MODE ENABLED")
    fold_dirs = sorted(
        d for d in os.listdir(KFOLD_DATA_DIR)
        if d.startswith("fold_") and os.path.isdir(os.path.join(KFOLD_DATA_DIR, d))
    )
    print(f"[INFO] Found {len(fold_dirs)} fold directories in {KFOLD_DATA_DIR}")
    for fold_name in fold_dirs:
        fold_data_dir = os.path.join(KFOLD_DATA_DIR, fold_name)
        fold_train_path = _pick_existing(fold_data_dir, ["train_w.npz", "train.npz", "train_uvw.npz"])
        if fold_train_path is None:
            print(f"[WARN] Skipping {fold_name}: no train npz found")
            continue

        expected_ckpt_suffix = f"_{fold_name}_best_bin_loss.pt"
        fold_ckpt_path = None
        for file_name in sorted(os.listdir(KFOLD_MODELS_DIR)):
            if file_name.endswith(expected_ckpt_suffix):
                fold_ckpt_path = os.path.join(KFOLD_MODELS_DIR, file_name)
                break
        if fold_ckpt_path is None:
            print(f"[WARN] Skipping {fold_name}: no checkpoint matching *{expected_ckpt_suffix}")
            continue

        print(f"[INFO] Loading fold model: {fold_ckpt_path}")
        checkpoint = torch.load(fold_ckpt_path, map_location=DEVICE, weights_only=False)
        cfg = checkpoint.get('config', {})
        ckpt_use_one_sat = cfg.get('use_one_satellite', None)
        if ckpt_use_one_sat is None:
            ckpt_use_one_sat = (cfg.get('in_channels', C) == 1)
        if ckpt_use_one_sat:
            model_in_channels = 1
        else:
            model_in_channels = cfg.get('in_channels', C)

        if ckpt_use_one_sat and not getattr(test_set, 'use_one_satellite', False):
            test_set = NPZSequenceDataset(
                NPZ_TEST_PATH,
                use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
                gt_envelope_npz_path=NPZ_TEST_PATH,
                use_one_satellite=True
            )

        fold_train_dataset = NPZSequenceDataset(
            fold_train_path,
            use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
            gt_envelope_npz_path=fold_train_path,
            use_one_satellite=ckpt_use_one_sat
        )
        if len(fold_train_datasets) == 0:
            test_set.scale = fold_train_dataset.scale
            test_set.norm_const = fold_train_dataset.norm_const

        checkpoint_state = checkpoint['model_state']
        model, has_refiner, refiner_hidden_channels = _build_model_from_checkpoint(cfg, checkpoint_state, DEVICE, model_in_channels)
        load_result = model.load_state_dict(checkpoint_state, strict=False)
        if load_result.missing_keys:
            print(f"[WARN] Missing keys when loading {fold_name}: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"[WARN] Unexpected keys when loading {fold_name}: {load_result.unexpected_keys}")
        model.to(DEVICE)
        model.eval()

        models.append(model)
        fold_train_datasets.append(fold_train_dataset)

    print(f"[INFO] Loaded {len(models)} fold models")
    if len(models) == 0:
        raise RuntimeError("No fold checkpoints were loaded")


# Evaluate on FULL NPZ_TEST_PATH dataset (no random splits)
# This ensures all metrics come from NPZ_TEST_PATH only
eval_ds = test_set
print(f"[INFO] Dataset loaded. Evaluating on FULL NPZ_TEST_PATH ({len(eval_ds)} sequences)")

# Lists to store pixel values
scatter_gt_list = []
scatter_pred_list = []
scatter_time_list = []
def apply_sensor_noise(img_array):
    """
    Simulates physical sensor noise: Dark Current, Read Noise, and 10-bit Quantization.
    Faithful to original parameters.
    """
    CONVERSION_FACTOR = 178.6304426659069
    EXPOSURE_TIME_US = 205
    DARK_CURRENT_RATE = 4.72 * 1e-6  # e-/sec
    FULL_WELL_CAPACITY = 10600
    BIT_DEPTH_FACTOR = 1024  # 10-bit

    # Radiance to Electrons
    electrons = img_array * CONVERSION_FACTOR

    # Dark Current Noise (Gaussian)
    dark_noise_mean = DARK_CURRENT_RATE * EXPOSURE_TIME_US
    dn_noise = np.random.normal(loc=dark_noise_mean, scale=dark_noise_mean ** 0.5, size=electrons.shape)
    electrons += dn_noise

    # Read Noise (Gaussian, mean=0)
    read_noise = np.random.normal(loc=0.0, scale=5.29 ** 0.5, size=electrons.shape)
    electrons += read_noise

    # Clipping & Quantization
    electrons = np.clip(electrons, a_min=0, a_max=FULL_WELL_CAPACITY)
    dn = electrons * (BIT_DEPTH_FACTOR / FULL_WELL_CAPACITY)
    electrons_quantized = np.round(dn) * (FULL_WELL_CAPACITY / BIT_DEPTH_FACTOR)

    # Back to Radiance (as expected by model input)
    radiance = electrons_quantized / CONVERSION_FACTOR
    return radiance.astype(np.float32)

print("[INFO] Starting evaluation...")

# for i in tqdm(range(len(eval_ds)), desc="Evaluating"):
#
#     # Get item from Test Dataset
#     input_seq, gt_vel_seq, mask_seq = eval_ds[i]
#
#     x_input = input_seq.unsqueeze(0).to(DEVICE)

for i in tqdm(range(len(eval_ds)), desc="Evaluating"):

    # 1. Get item from Test Dataset
    input_seq, gt_vel_seq, mask_seq = eval_ds[i]

    # --- NEW: Noise Support ---
    if ADD_SENSOR_NOISE:
        # Clone and move to CPU/Numpy to apply sensor effects
        input_np = input_seq.clone().cpu().numpy()  # Shape: (T, C, H, W)
        T, C, H, W = input_np.shape
        for t in range(T):
            for c in range(C):
                input_np[t, c] = apply_sensor_noise(input_np[t, c])
        # Convert back to tensor
        input_seq = torch.from_numpy(input_np)
    # ---------------------------

    x_input = input_seq.unsqueeze(0).to(DEVICE)
    all_preds = []
    with torch.no_grad():
        for current_model, current_fold_train_dataset in zip(models, fold_train_datasets):
            output, _ = current_model(x_input)

            if isinstance(output, list):
                pred_tensor = torch.stack(output, dim=1)
            else:
                pred_tensor = output

            pred_vel = pred_tensor.squeeze(0).cpu().numpy()
            pred_vel_denorm = current_fold_train_dataset.denormalize(pred_vel)
            all_preds.append(pred_vel_denorm)

    if len(all_preds) == 1:
        final_pred_vel_denorm = all_preds[0]
    else:
        final_pred_vel_denorm = np.mean(np.stack(all_preds, axis=0), axis=0)

    # Denormalize GT using test dataset stats, predictions using fold-specific train datasets
    gt_vel_denorm = test_set.denormalize(gt_vel_seq)
    pred_vel_denorm = final_pred_vel_denorm

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

        #scatter_min = min(gt_vals.min(), pred_vals.min())
        #scatter_max = max(gt_vals.max(), pred_vals.max())
        scatter_min = gt_vals.min()
        scatter_max = gt_vals.max()
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
    ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=70, alpha=0.3, rasterized=True)
    ax_scatter.plot([-scatter_range_padded, scatter_range_padded], [-scatter_range_padded, scatter_range_padded], 'k--', lw=4)
    ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=80, fontweight='bold')
    ax_scatter.set_ylabel("Inferred [m/s]", fontsize=80, fontweight='bold')
    #ax_scatter.set_title(f"Balanced Scatter Plot", fontsize=64, fontweight='bold', pad=40)
    ax_scatter.set_xlim(-scatter_range_padded, scatter_range_padded)
    ax_scatter.set_ylim(-scatter_range_padded, scatter_range_padded)
    ax_scatter.set_xticks([-5, 0, 5])
    ax_scatter.set_yticks([-5, 0, 5])
    ax_scatter.grid(True, alpha=0.3, linewidth=2)
    ax_scatter.tick_params(axis='both', which='major', labelsize=80)
    ax_scatter.text(0.05, 0.95, TEXT_FOR_SCATER, transform=ax_scatter.transAxes,
                    fontsize=80, fontweight='bold', va='top', ha='left')
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
        ax_scatter_t5.scatter(x_scatter_t5, y_scatter_t5, c='tab:blue', s=30, alpha=0.3)
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

