"""
Generate Video Comparison for SimVP Velocity Prediction
Reads trained weights, runs inference on a test sequence,
and outputs a side-by-side MP4 video of:
Input (Sat 1) | Ground Truth Velocity | Predicted Velocity
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

# Import your existing modules
from model import SimVP
from API import load_data


def generate_comparison_video(
        fold_dir,
        weights_path,
        output_video_path="cloud_velocity_comparison.mp4",
        sequence_idx=0,
        use_mask=True
):
    # 1. Setup device and configurations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Applying cloud mask: {use_mask}")

    # Set the hyperparameters based on your training configuration
    in_shape = [12, 2, 128, 128]
    hid_S = 128
    hid_T = 512
    N_S = 4
    N_T = 8

    # 2. Build model and load trained weights
    model = SimVP(tuple(in_shape), hid_S, hid_T, N_S, N_T).to(device)
    # Added weights_only=True to resolve the PyTorch security warning
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[INFO] Successfully loaded weights from: {weights_path}")

    # 3. Load the test data using the standard config dictionary
    config = {
        'dataname': 'velocity',
        'batch_size': 1,
        'val_batch_size': 1,
        'data_root': fold_dir,
        # Set test_root to the parent directory where test_w.npz is actually located
        'test_root': os.path.dirname(fold_dir.rstrip('/')),
        'num_workers': 1,
        'use_one_satellite': False,
        'augment': False
    }

    _, _, test_loader, _, _ = load_data(**config)

    # 4. Fetch the desired sequence
    test_iter = iter(test_loader)
    for _ in range(sequence_idx + 1):
        try:
            batch_data = next(test_iter)
        except StopIteration:
            print(f"[ERROR] Sequence index {sequence_idx} is out of bounds for the test set.")
            return

    if len(batch_data) == 3:
        batch_x, batch_y, batch_mask = batch_data
    else:
        batch_x, batch_y = batch_data
        batch_mask = None

    # Toggle the mask based on user input
    if not use_mask:
        batch_mask = None

    # Run inference
    with torch.no_grad():
        batch_x_dev = batch_x.to(device)
        pred_y_dev = model(batch_x_dev)
        pred_y = pred_y_dev.cpu().numpy()

    # Extract sequences (remove batch dimension)
    # Shape becomes [T, C, H, W] -> [12, C, 128, 128]
    x_seq = batch_x[0].numpy()
    y_true_seq = batch_y[0].numpy()
    y_pred_seq = pred_y[0]

    if batch_mask is not None:
        mask_seq = batch_mask[0].numpy()
    else:
        # If mask is ignored, create an array of 1s so nothing is hidden
        mask_seq = np.ones_like(y_true_seq)

    # Re-scale velocity values to physical range (m/s)
    scale = test_loader.dataset.scale
    y_true_seq = y_true_seq * scale
    y_pred_seq = y_pred_seq * scale

    # Mask the predictions and ground truth for cleaner visualization
    # We apply NaN to background so matplotlib leaves it blank (which we will color black)
    y_true_seq = np.where(mask_seq > 0.5, y_true_seq, np.nan)
    y_pred_seq = np.where(mask_seq > 0.5, y_pred_seq, np.nan)

    # 5. Create the Animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Vertical Cloud Velocity: SimVP Inference (Seq: {sequence_idx})', fontsize=16)

    # Find global min/max for stable colorbar during animation
    # Ignore NaNs when calculating max value
    vmax = np.nanmax(np.abs(y_true_seq))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0  # Fallback if the array is completely empty or zero

    vmin = -vmax

    # Diverging colormap centered at 0 using TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # --- ADDED: Custom colormap to render NaN values (background) as black ---
    jet_cmap = plt.cm.jet.copy()
    jet_cmap.set_bad(color='black')

    # Initial plot frames
    im_x = axes[0].imshow(x_seq[0, 0], cmap='gray')  # Show Satellite A
    axes[0].set_title('Satellite Input (Cam A)')
    axes[0].axis('off')

    # Render Ground Truth with the custom colormap and black facecolor
    axes[1].set_facecolor('black')
    im_true = axes[1].imshow(y_true_seq[0, 0], cmap=jet_cmap, norm=norm)
    axes[1].set_title('Ground Truth (Y)')
    axes[1].axis('off')

    # Render Prediction with the custom colormap and black facecolor
    axes[2].set_facecolor('black')
    im_pred = axes[2].imshow(y_pred_seq[0, 0], cmap=jet_cmap, norm=norm)
    axes[2].set_title('SimVP Prediction')
    axes[2].axis('off')

    # Add a single colorbar for velocities
    cbar = fig.colorbar(im_true, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Vertical Velocity (m/s)')

    def update(frame):
        """Update the plot for each timestep"""
        im_x.set_data(x_seq[frame, 0])
        im_true.set_data(y_true_seq[frame, 0])
        im_pred.set_data(y_pred_seq[frame, 0])
        fig.suptitle(f'Vertical Cloud Velocity: Timestep {frame + 1}/{in_shape[0]} (Seq: {sequence_idx})', fontsize=16)
        return [im_x, im_true, im_pred]

    print("[INFO] Generating MP4 animation... This might take a few moments.")
    ani = animation.FuncAnimation(fig, update, frames=in_shape[0], interval=300, blit=False)

    # Save the animation (requires ffmpeg installed)
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    ani.save(output_video_path, writer='ffmpeg', fps=3)
    print(f"[INFO] Video successfully saved to {output_video_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SimVP Comparison Video')

    # Path Arguments
    parser.add_argument('--fold_dir', type=str,
                        default="data/wacv_data/envelop_kfold_w_sensor_noise_both/fold_01_val_r0-1_c0-2",
                        help='Path to the data fold directory')
    parser.add_argument('--weights_path', type=str,
                        default="SimVP/results/velocity_simvp_binlos_fold_01_val_r0-1_c0-2/checkpoint.pth",
                        help='Path to the trained model weights')
    parser.add_argument('--output_video', type=str,
                        default="simvp_result_sequence.mp4",
                        help='Output path for the MP4 video')

    # Render Arguments
    parser.add_argument('--sequence_idx', type=int, default=300,
                        help='Index of the sequence in the test set to render')

    # Mask toggle
    parser.add_argument('--use_mask', action='store_true', help='Apply mask to hide background')
    parser.add_argument('--no_mask', dest='use_mask', action='store_false', help='Show entire image (no mask)')
    parser.set_defaults(use_mask=True)

    args = parser.parse_args()

    generate_comparison_video(
        fold_dir=args.fold_dir,
        weights_path=args.weights_path,
        output_video_path=args.output_video,
        sequence_idx=args.sequence_idx,
        use_mask=args.use_mask
    )