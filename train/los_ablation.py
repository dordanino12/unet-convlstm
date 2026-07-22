import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl

# ---------------------------------------------------------
# Global Font & Plot Settings
# ---------------------------------------------------------
mpl.rcParams.update({
    'font.size': 60,
    'axes.titlesize': 64,
    'axes.labelsize': 56,
    'xtick.labelsize': 52,
    'ytick.labelsize': 52,
    'legend.fontsize': 52,
    'figure.titlesize': 68,
    'figure.dpi': 300,
    'savefig.dpi': 150,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset import NPZSequenceDataset
from resnet18 import PretrainedTemporalUNetMitB1

# -----------------------------
# Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_MASK = True
USE_GT_ENVELOPE_INPUT = False
USE_ONE_SATELLITE = False
USE_CONV_LSTM = True
ADD_SENSOR_NOISE = False
TEXT_FOR_SCATER = ""

# -----------------------------
# Paths to Data
# -----------------------------
NPZ_TRAIN_PATH = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2/train_w.npz"
NPZ_TEST_PATH = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/envelop_kfold_w_sensor_noise_both/test_w.npz"
KFOLD_DATA_DIR = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/envelop_kfold_w_sensor_noise_both/"
output_dir = "/home/danino/PycharmProjects/pythonProject/plots/combined_scatter"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Define the 3 Models/Configs to Evaluate
# -----------------------------
MODELS_TO_EVALUATE = [
    {
        "name": "L1 loss", 
        "kfold_dir": "/home/danino/PycharmProjects/pythonProject/models/wacv/envelop_l1/", 
        "color": "tab:red"
    },
    {
        "name": "Power loss", 
        "kfold_dir": "/home/danino/PycharmProjects/pythonProject/models/wacv/envelop_exp/", 
        "color": "tab:green"
    },
    {
        "name": "Bin loss", 
        "kfold_dir": "/home/danino/PycharmProjects/pythonProject/models/wacv/envelop/", 
        "color": "tab:blue"
    }
]

# Scatter Plot Config
SCATTER_BIN_WIDTH = 0.02
POINTS_PER_BIN = 5
SCATTER_RANGE = (-8.5, 8.5)

# -----------------------------
# Helper Functions
# -----------------------------
def _pick_existing(base_dir, names):
    for name in names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None

def apply_sensor_noise(img_array):
    CONVERSION_FACTOR = 178.6304426659069
    EXPOSURE_TIME_US = 205
    DARK_CURRENT_RATE = 4.72 * 1e-6
    FULL_WELL_CAPACITY = 10600
    BIT_DEPTH_FACTOR = 1024

    electrons = img_array * CONVERSION_FACTOR
    dark_noise_mean = DARK_CURRENT_RATE * EXPOSURE_TIME_US
    dn_noise = np.random.normal(loc=dark_noise_mean, scale=dark_noise_mean ** 0.5, size=electrons.shape)
    electrons += dn_noise

    read_noise = np.random.normal(loc=0.0, scale=5.29 ** 0.5, size=electrons.shape)
    electrons += read_noise

    electrons = np.clip(electrons, a_min=0, a_max=FULL_WELL_CAPACITY)
    dn = electrons * (BIT_DEPTH_FACTOR / FULL_WELL_CAPACITY)
    electrons_quantized = np.round(dn) * (FULL_WELL_CAPACITY / BIT_DEPTH_FACTOR)

    radiance = electrons_quantized / CONVERSION_FACTOR
    return radiance.astype(np.float32)

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
    else:
        x_scatter = gt_vals
        y_scatter = pred_vals

    scatter_min = gt_vals.min()
    scatter_max = gt_vals.max()
    scatter_range_data = max(abs(scatter_min), abs(scatter_max))
    scatter_range_padded = scatter_range_data * 1.1
    return x_scatter, y_scatter, scatter_range_padded

# -----------------------------
# Initialize Base Datasets
# -----------------------------
train_dataset_for_norm = NPZSequenceDataset(
    NPZ_TRAIN_PATH, use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
    gt_envelope_npz_path=NPZ_TRAIN_PATH, use_one_satellite=USE_ONE_SATELLITE
)

test_set = NPZSequenceDataset(
    NPZ_TEST_PATH, use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
    gt_envelope_npz_path=NPZ_TEST_PATH, use_one_satellite=USE_ONE_SATELLITE
)

test_set.scale = train_dataset_for_norm.scale
test_set.norm_const = train_dataset_for_norm.norm_const
_, C, _, _ = test_set[0][0].shape

# -----------------------------
# Inference Function (For One Ensemble)
# -----------------------------
def evaluate_ensemble(model_cfg):
    config_name = model_cfg['name']
    kfold_models_dir = model_cfg['kfold_dir']
    
    print(f"\n[INFO] Loading K-Fold Ensemble for: {config_name}")
    fold_dirs = sorted(d for d in os.listdir(KFOLD_DATA_DIR) if d.startswith("fold_"))
    
    models = []
    fold_train_datasets = []

    for fold_name in fold_dirs:
        fold_data_dir = os.path.join(KFOLD_DATA_DIR, fold_name)
        fold_train_path = _pick_existing(fold_data_dir, ["train_w.npz", "train.npz", "train_uvw.npz"])
        if fold_train_path is None:
            continue

        expected_ckpt_suffix = f"_{fold_name}_best_bin_loss.pt"
        fold_ckpt_path = None
        for file_name in sorted(os.listdir(kfold_models_dir)):
            if file_name.endswith(expected_ckpt_suffix):
                fold_ckpt_path = os.path.join(kfold_models_dir, file_name)
                break
        
        if fold_ckpt_path is None:
            print(f"[WARN] Skipping {fold_name}: no checkpoint found in {kfold_models_dir}")
            continue

        checkpoint = torch.load(fold_ckpt_path, map_location=DEVICE, weights_only=False)
        cfg = checkpoint.get('config', {})
        ckpt_use_one_sat = cfg.get('use_one_satellite', (cfg.get('in_channels', C) == 1))
        model_in_channels = 1 if ckpt_use_one_sat else cfg.get('in_channels', C)

        checkpoint_state = checkpoint['model_state']
        has_refiner = any('refiner' in key for key in checkpoint_state.keys())
        refiner_hidden_channels = 32
        if has_refiner:
            for key in checkpoint_state.keys():
                if 'refiner.net.0.weight' in key:
                    refiner_hidden_channels = checkpoint_state[key].shape[0]
                    break

        model = PretrainedTemporalUNetMitB1(
            out_channels=1,
            lstm_layers=1 if USE_CONV_LSTM else 0,
            freeze_encoder=cfg.get('freeze_encoder', True),
            in_channels=model_in_channels,
            use_conv_lstm=USE_CONV_LSTM,
            use_refiner=has_refiner,
            refiner_hidden_channels=refiner_hidden_channels
        )
        
        model.load_state_dict(checkpoint_state, strict=False)
        model.to(DEVICE)
        model.eval()

        fold_train_dataset = NPZSequenceDataset(
            fold_train_path, use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
            gt_envelope_npz_path=fold_train_path, use_one_satellite=ckpt_use_one_sat
        )

        models.append(model)
        fold_train_datasets.append(fold_train_dataset)

    if len(models) == 0:
        raise RuntimeError(f"No fold checkpoints were loaded for {config_name}")
    print(f"[INFO] Successfully loaded {len(models)} folds for {config_name}.")

    # Run inference on Test Set
    scatter_gt_list = []
    scatter_pred_list = []

    for i in tqdm(range(len(test_set)), desc=f"Evaluating {config_name}"):
        input_seq, gt_vel_seq, mask_seq = test_set[i]

        if ADD_SENSOR_NOISE:
            input_np = input_seq.clone().cpu().numpy()
            T_seq, C_seq, H_seq, W_seq = input_np.shape
            for t in range(T_seq):
                for c in range(C_seq):
                    input_np[t, c] = apply_sensor_noise(input_np[t, c])
            input_seq = torch.from_numpy(input_np)

        x_input = input_seq.unsqueeze(0).to(DEVICE)
        all_preds = []
        with torch.no_grad():
            for current_model, current_fold_train_dataset in zip(models, fold_train_datasets):
                output, _ = current_model(x_input)
                pred_tensor = torch.stack(output, dim=1) if isinstance(output, list) else output
                pred_vel = pred_tensor.squeeze(0).cpu().numpy()
                pred_vel_denorm = current_fold_train_dataset.denormalize(pred_vel)
                all_preds.append(pred_vel_denorm)

        final_pred_vel_denorm = np.mean(np.stack(all_preds, axis=0), axis=0)
        gt_vel_denorm = test_set.denormalize(gt_vel_seq)

        if USE_MASK:
            mask_np = mask_seq.cpu().numpy()
            valid_pixels = (mask_np > 0.1)
            if np.any(valid_pixels):
                scatter_gt_list.append(gt_vel_denorm.cpu().numpy()[valid_pixels])
                scatter_pred_list.append(final_pred_vel_denorm[valid_pixels])
        else:
            scatter_gt_list.append(gt_vel_denorm.cpu().numpy().flatten())
            scatter_pred_list.append(final_pred_vel_denorm.flatten())

    # Collect and sample points
    all_gt = np.concatenate(scatter_gt_list)
    all_pred = np.concatenate(scatter_pred_list)
    x_scatter, y_scatter, scatter_range_padded = sample_scatter_points(all_gt, all_pred)

    return {
        "x": x_scatter,
        "y": y_scatter,
        "range": scatter_range_padded
    }

# -----------------------------
# Main Loop & Plotting
# -----------------------------
if __name__ == '__main__':
    fig_scatter, ax_scatter = plt.subplots(figsize=(20, 20), dpi=150)
    max_range = 0

    # Loop through all 3 configurations
    for model_cfg in MODELS_TO_EVALUATE:
        results = evaluate_ensemble(model_cfg)
        
        if results['range'] > max_range:
            max_range = results['range']
            
        # Plot this config's points on the shared ax_scatter
        ax_scatter.scatter(
            results['x'], 
            results['y'], 
            c=model_cfg['color'], 
            label=model_cfg['name'], 
            s=70, 
            alpha=0.2, 
            rasterized=True
        )

    # Plot the ideal diagonal line ONCE
    ax_scatter.plot([-max_range, max_range], [-max_range, max_range], 'k--', lw=4)

    # Format the shared plot
    ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=80, fontweight='bold')
    ax_scatter.set_ylabel("Inferred [m/s]", fontsize=80, fontweight='bold')
    ax_scatter.set_xlim(-max_range, max_range)
    ax_scatter.set_ylim(-max_range, max_range)
    ax_scatter.set_xticks([-5, 0, 5])
    ax_scatter.set_yticks([-5, 0, 5])
    ax_scatter.grid(True, alpha=0.3, linewidth=2)
    ax_scatter.tick_params(axis='both', which='major', labelsize=80)

    # Add the Legend mapping color to config
    leg = ax_scatter.legend(fontsize=60, loc='lower right', markerscale=3)
    
    # Override the alpha value specifically for the legend markers to make them solid
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    ax_scatter.text(0.05, 0.95, TEXT_FOR_SCATER, transform=ax_scatter.transAxes,
                    fontsize=80, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "combined_configs_scatter.pdf")
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)
    
    print(f"\n[SUCCESS] Saved combined scatter plot to: {scatter_path}")