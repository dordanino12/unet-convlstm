import os
import sys
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

from torch.amp import autocast

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train.dataset import NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNetMitB1
from plots.create_video_dashboard3d_from_samples import create_3d_plot_img, load_camera_csv


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sequence index used across all datasets
SEQUENCE_IDX = 1000

# Colorbar
FOCUS_THRESH = 3  # SymLogNorm linear threshold (matching test_multi_height_models.py)

# Part durations
TITLE_SECONDS = 5
FPS = 0.7
MAX_TIME = 200  # show time from 0..200

# Paths
CSV_PATH = os.path.join(parent_dir, "data", "Dor_2satellites_overpass.csv")
OUTPUT_VIDEO = os.path.join(parent_dir, "plots", "traj_dashboard_movie.mp4")

MODELS_CFG = {
    "500m": {
        "npz": os.path.join(parent_dir, "data", "dataset_trajectory_sequences_samples_W_500m_w.npz"),
        "ckpt": os.path.join(parent_dir, "models", "mit_b1_500m_slice_mask_no_gtenv_mix_loss_best_bin_loss.pt"),
    },
    "1000m": {
        "npz": os.path.join(parent_dir, "data", "dataset_trajectory_sequences_samples_W_1000m_w.npz"),
        "ckpt": os.path.join(parent_dir, "models", "mit_b1_1000m_slice_mask_no_gtenv_mix_loss_best_bin_loss.pt"),
    },
    "1500m": {
        "npz": os.path.join(parent_dir, "data", "dataset_trajectory_sequences_samples_W_1500m_w.npz"),
        "ckpt": os.path.join(parent_dir, "models", "mit_b1_1500m_slice_mask_no_gtenv_mix_loss_best_bin_loss.pt"),
    },
}


def apply_gamma(img_array, gamma=0.5):
    img_array = np.asarray(img_array, dtype=np.float32)
    img_min = float(np.min(img_array))
    img_max = float(np.max(img_array))
    if img_max - img_min < 1e-8:
        return np.zeros_like(img_array, dtype=np.float32)
    img_norm = (img_array - img_min) / (img_max - img_min)
    return np.power(img_norm, gamma)


def build_model_from_checkpoint(checkpoint_path, in_channels):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    checkpoint_state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_in_channels = int(cfg.get("in_channels", in_channels))

    has_refiner = any("refiner" in k for k in checkpoint_state.keys())
    refiner_hidden_channels = 32
    if has_refiner:
        for key in checkpoint_state.keys():
            if "refiner.net.0.weight" in key:
                refiner_hidden_channels = checkpoint_state[key].shape[0]
                break

    model = PretrainedTemporalUNetMitB1(
        out_channels=1,
        lstm_layers=1,
        freeze_encoder=cfg.get("freeze_encoder", True),
        in_channels=model_in_channels,
        use_refiner=has_refiner,
        refiner_hidden_channels=refiner_hidden_channels,
    )

    load_result = model.load_state_dict(checkpoint_state, strict=False)
    if hasattr(load_result, "missing_keys") and load_result.missing_keys:
        print(f"[WARN] Missing keys for {os.path.basename(checkpoint_path)}: {load_result.missing_keys}")
    if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
        print(f"[WARN] Unexpected keys for {os.path.basename(checkpoint_path)}: {load_result.unexpected_keys}")

    model.to(DEVICE)
    model.eval()
    return model


def compute_fixed_geo_limits(csv_times, sat_lookup):
    all_x, all_y, all_z = [], [], []
    for t in csv_times:
        for pos in sat_lookup[t]:
            all_x.append(abs(pos[0] / 1000.0))
            all_y.append(abs(pos[1] / 1000.0))
            all_z.append(pos[2] / 1000.0)

    if not all_x:
        return (100.0, 100.0, 600.0)

    return (max(all_x) * 1.2, max(all_y) * 1.2, max(all_z) * 1.1)


def add_title_frames(video_writer, frame_size, fps, seconds):
    n_frames = int(fps * seconds)
    
    for _ in range(n_frames):
        fig = plt.figure(figsize=(14, 8), dpi=120)
        fig.patch.set_facecolor((245/255.0, 245/255.0, 245/255.0))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Title line 1
        ax.text(0.5, 0.58, "Inferred Spaceborne Sensing of Vertical Air Flow",
                ha='center', va='center', fontsize=28, fontweight='bold', color='#1e1e1e')
        
        # Title line 2
        ax.text(0.5, 0.38, "Paper ID #10787",
                ha='center', va='center', fontsize=23, fontweight='bold', color='#5a5a5a')
        
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        frame_bgr = figure_to_bgr(fig)
        plt.close(fig)
        write_frame(video_writer, frame_bgr, frame_size)


def figure_to_bgr(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_frame(video_writer, frame_bgr, frame_size):
    target_w, target_h = frame_size
    h, w = frame_bgr.shape[:2]
    if (w, h) != (target_w, target_h):
        frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    video_writer.write(frame_bgr)


def make_part_geometry_and_render(video_writer, seq_for_render, csv_times, sat_lookup, fixed_limits, frame_size):
    # seq_for_render: (T, 2, H, W) normalized X from dataset
    t_max = min(MAX_TIME + 1, seq_for_render.shape[0]-1)

    for t in range(t_max):
        fig = plt.figure(figsize=(14, 8), dpi=120)
        gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
        time_sec = t * 20
        fig.suptitle(f"Geometry and Images | Time: {time_sec} sec", fontsize=16, fontweight="bold")

        ax_geo = fig.add_subplot(gs[:, 0])
        ax_sat0 = fig.add_subplot(gs[0, 1])
        ax_sat1 = fig.add_subplot(gs[1, 1])

        csv_ptr = t % len(csv_times)
        target_time = csv_times[csv_ptr]
        sat_positions = sat_lookup[target_time]

        geo_img = create_3d_plot_img(
            sat_positions=sat_positions,
            look_at=[0, 0, 1500],
            figsize=(700, 700),
            fixed_bounds=fixed_limits,
        )
        geo_img_rgb = cv2.cvtColor(geo_img, cv2.COLOR_BGR2RGB)

        sat_a = apply_gamma(seq_for_render[t, 0], gamma=0.5)
        sat_b = apply_gamma(seq_for_render[t, 1], gamma=0.5)

        ax_geo.imshow(geo_img_rgb)
        ax_geo.set_title("3D Satellite Geometry", fontsize=16, fontweight="bold")
        ax_geo.axis("off")

        ax_sat0.imshow(sat_a, cmap="gray", vmin=0, vmax=1)
        ax_sat0.set_title("Image at Sat A", fontsize=14, fontweight="bold")
        ax_sat0.axis("off")

        ax_sat1.imshow(sat_b, cmap="gray", vmin=0, vmax=1)
        ax_sat1.set_title("Image at Sat B", fontsize=14, fontweight="bold")
        ax_sat1.axis("off")

        frame_bgr = figure_to_bgr(fig)
        plt.close(fig)
        write_frame(video_writer, frame_bgr, frame_size)



def run_prediction(model, input_seq):
    # input_seq: (T, C, H, W)
    with torch.no_grad():
        with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            output, _ = model(input_seq.unsqueeze(0).to(DEVICE))
    if isinstance(output, list):
        output = torch.stack(output, dim=1)
    return output.squeeze(0).cpu().numpy()  # (T, 1, H, W)


def _make_five_ticks(vmin, vmax):
    """Generate 5 ticks spanning [vmin, vmax], always including 0 when range crosses it."""
    if vmin >= 0:
        return np.linspace(vmin, vmax, 5)
    elif vmax <= 0:
        return np.linspace(vmin, vmax, 5)
    else:
        return np.array([vmin, vmin / 2.0, 0.0, vmax / 2.0, vmax])


def _add_styled_colorbar(fig, im, ax, vmin, vmax):
    """Styled colorbar matching test_multi_height_models.py: 5 ticks, formatted labels."""
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = _make_five_ticks(vmin, vmax)
    cbar.set_ticks(ticks)
    cbar.ax.set_ylim(vmin, vmax)
    cbar.ax.tick_params(labelsize=18, pad=4)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    return cbar


def make_part_gt_pred_grid(video_writer, prepared_data, frame_size):
    # prepared_data[label] has keys: gt(T,1,H,W), pred(T,1,H,W), vmin, vmax, norm
    labels = ["500m", "1000m", "1500m"]
    t_max = min(MAX_TIME + 1, min(prepared_data[k]["gt"].shape[0] for k in labels))

    # Build shared custom jet colormap (matches test_multi_height_models.py)
    base_cmap = plt.get_cmap('jet', 256)
    cmap_custom = mcolors.ListedColormap(base_cmap(np.linspace(0.0, 1.0, 256)), name='jet_custom')

    # Each column gets its own SymLogNorm derived from its own vmin/vmax.
    per_label_norm = {
        label: mcolors.SymLogNorm(
            linthresh=FOCUS_THRESH, linscale=1.0,
            vmin=prepared_data[label]["vmin"],
            vmax=prepared_data[label]["vmax"],
        )
        for label in labels
    }

    for t in range(t_max):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=120)
        time_sec = t * 20
        fig.suptitle(f"GT vs Inferred Vertical Velocity | Time: {time_sec} sec", fontsize=16, fontweight="bold")

        for col, label in enumerate(labels):
            data = prepared_data[label]
            gt = data["gt"][t, 0]
            pred = data["pred"][t, 0]
            vmin = data["vmin"]
            vmax = data["vmax"]
            norm = per_label_norm[label]

            im_gt = axes[0, col].imshow(gt, cmap=cmap_custom, norm=norm, interpolation='nearest')
            axes[0, col].set_title(f"{label} GT [m/s]", fontsize=12, fontweight="bold")
            axes[0, col].axis("off")
            _add_styled_colorbar(fig, im_gt, axes[0, col], vmin, vmax)

            im_pred = axes[1, col].imshow(pred, cmap=cmap_custom, norm=norm, interpolation='nearest')
            axes[1, col].set_title(f"{label} Inferred [m/s]", fontsize=12, fontweight="bold")
            axes[1, col].axis("off")
            _add_styled_colorbar(fig, im_pred, axes[1, col], vmin, vmax)

        plt.tight_layout()
        frame_bgr = figure_to_bgr(fig)
        plt.close(fig)
        write_frame(video_writer, frame_bgr, frame_size)


def main():
    print("[INFO] Loading satellite geometry CSV...")
    csv_times, sat_lookup = load_camera_csv(CSV_PATH)
    if len(csv_times) == 0:
        raise RuntimeError("CSV has no satellite times.")

    fixed_limits = compute_fixed_geo_limits(csv_times, sat_lookup)

    print("[INFO] Loading datasets and models (500m/1000m/1500m)...")
    prepared = {}
    seq_for_render = None

    for label, cfg in MODELS_CFG.items():
        if not os.path.exists(cfg["npz"]):
            raise FileNotFoundError(f"Missing dataset for {label}: {cfg['npz']}")
        if not os.path.exists(cfg["ckpt"]):
            raise FileNotFoundError(f"Missing checkpoint for {label}: {cfg['ckpt']}")

        dataset = NPZSequenceDataset(cfg["npz"])
        if SEQUENCE_IDX >= len(dataset):
            raise IndexError(f"SEQUENCE_IDX={SEQUENCE_IDX} out of range for {label} dataset length {len(dataset)}")

        input_seq, gt_seq_norm, _ = dataset[SEQUENCE_IDX]
        model = build_model_from_checkpoint(cfg["ckpt"], in_channels=input_seq.shape[1])

        pred_norm = run_prediction(model, input_seq)

        gt_denorm = dataset.denormalize(gt_seq_norm).cpu().numpy()
        pred_denorm = dataset.denormalize(pred_norm)

        vmin = -float(dataset.max_neg_val)
        vmax = float(dataset.max_pos_val)
        norm = mcolors.SymLogNorm(linthresh=FOCUS_THRESH, linscale=1.0, vmin=vmin, vmax=vmax)

        prepared[label] = {
            "gt": gt_denorm,
            "pred": pred_denorm,
            "vmin": vmin,
            "vmax": vmax,
            "norm": norm,
        }

        if seq_for_render is None:
            seq_for_render = input_seq.cpu().numpy()

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Match figure output size from first generated frame
    warmup_fig = plt.figure(figsize=(14, 8), dpi=120)
    warmup_shape = figure_to_bgr(warmup_fig).shape
    plt.close(warmup_fig)
    h, w = warmup_shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    print("[INFO] Part 1/3: Title section...")
    add_title_frames(writer, frame_size=(w, h), fps=FPS, seconds=TITLE_SECONDS)

    print("[INFO] Part 2/3: Geometry + render section...")
    make_part_geometry_and_render(writer, seq_for_render, csv_times, sat_lookup, fixed_limits, (w, h))

    print("[INFO] Part 3/3: GT vs inferred section...")
    make_part_gt_pred_grid(writer, prepared, (w, h))

    writer.release()
    print(f"[INFO] Saved movie to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
