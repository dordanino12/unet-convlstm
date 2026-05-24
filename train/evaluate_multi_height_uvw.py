"""
Simple inference script for the Multi-Height UVW model.
Loads a checkpoint, runs inference on one sequence, prints per-height/channel MAE,
and saves visualizations for the last timestep of the sequence.
"""
from __future__ import annotations
import os
import sys
import copy
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train import MultiHeightTemporalModel, MultiHeightNPZDataset


def _stats_per_t(pred_t, gt_t):
    # pred_t, gt_t: (Hh, C, H, W)
    mae = np.mean(np.abs(pred_t - gt_t), axis=(2, 3))
    rmse = np.sqrt(np.mean((pred_t - gt_t) ** 2, axis=(2, 3)))
    return mae, rmse


def _to_rgb_frame(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='/home/danino/PycharmProjects/pythonProject/data/check_train_uvw.npz')
    parser.add_argument('--checkpoint', default='models/multih_best_overfit.pt')
    parser.add_argument('--sequence_idx', type=int, default=0)
    parser.add_argument('--height_idx', type=int, default=1, help='Index of the height to visualize (0-based)')
    parser.add_argument('--backbone', default='mit_b1')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--outdir', default='plots/test_multi_height')
    parser.add_argument('--out_video', default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    print('[INFO] Loading dataset:', args.npz)
    dataset = MultiHeightNPZDataset(args.npz)
    N = len(dataset)
    if args.sequence_idx < 0 or args.sequence_idx >= N:
        raise ValueError('sequence_idx out of range')

    input_seq, gt_seq = dataset[args.sequence_idx]
    # input_seq: (T, C, H, W)
    # gt_seq: (T, H_heights, C_uv_w, H, W)

    # Compute display range for input images so in_vmin/in_vmax are defined.
    # Use the full sequence & both channels to pick a sensible range for visualization.
    try:
        input_np = input_seq.numpy()  # (T, C, H, W)
    except Exception:
        # If the tensor is on GPU or otherwise, convert safely
        input_np = input_seq.cpu().numpy()
    in_vmin = float(np.min(input_np))
    in_vmax = float(np.max(input_np))

    T, C, H, W = input_seq.shape
    _, num_heights, channels_per_h, _, _ = gt_seq.shape
    if args.height_idx < 0 or args.height_idx >= num_heights:
        raise ValueError(f'height_idx out of range (0..{num_heights-1})')
    height_idx = args.height_idx
    print(f'[INFO] Sequence loaded: T={T}, inC={C}, heights={num_heights}, per-height-ch={channels_per_h}, HxW={H}x{W}')
    print(f'[INFO] Visualizing height index: {height_idx}')

    # Build model
    model = MultiHeightTemporalModel(
        in_channels=C,
        num_heights=num_heights,
        lstm_layers=1,
        pretrained=True
    )

    if args.checkpoint is not None:
        print('[INFO] Loading checkpoint:', args.checkpoint)
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        state = ck.get('model_state', ck)
        model.load_state_dict(state, strict=False)
    else:
        print('[WARN] No checkpoint provided — using random weights')

    model = model.to(device)
    model.eval()

    x = input_seq.unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        out = model(x)
        pred = out[0] if isinstance(out, (tuple, list)) else out  # (1, T, num_heights, 3, H, W)
    pred = pred.cpu().numpy()
    gt = gt_seq.numpy()

    # Denormalize both (use dataset.denormalize)
    pred_den = dataset.denormalize(pred)
    gt_den = dataset.denormalize(gt[np.newaxis, ...])[0]

    # Report MAE for last timestep
    t_idx = T - 1
    mae = np.mean(np.abs(pred_den[0, t_idx] - gt_den[t_idx]), axis=(2, 3))
    print('\nPer-height per-channel MAE (rows=height, cols=channel u,v,w):')
    print(mae)
    print('\nPer-height mean MAE:')
    print(mae.mean(axis=1))

    mae_by_vel = mae.mean(axis=0)
    print('\nPer-velocity MAE (u, v, w):')
    print(mae_by_vel)

    # Video rendering
    if args.out_video is None:
        args.out_video = os.path.join(args.outdir, f'seq{args.sequence_idx}.mp4')
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)

    # Per-channel display range for u,v,w based on GT for the chosen height across all timesteps.
    # This ensures each velocity component (u, v, w) has its own color scale and colorbar,
    # computed from GT only.
    channel_vmin = []
    channel_vmax = []
    for c in range(channels_per_h):
        gt_c = gt_den[:, height_idx, c]               # (T, H, W)
        ch_min = float(np.min(gt_c))
        ch_max = float(np.max(gt_c))
        #ch_abs = max(abs(ch_min), abs(ch_max))
        vmin, vmax = ch_min, ch_max
        channel_vmin.append(vmin)
        channel_vmax.append(vmax)

    with imageio.get_writer(args.out_video, fps=2, codec='libx264', format='ffmpeg') as writer:
        for t in range(T):
            pred_t = pred_den[0, t]          # (Nh, C, H, W)
            gt_t = gt_den[t]                # (Nh, C, H, W)
            mae_t, rmse_t = _stats_per_t(pred_t, gt_t)

            # Only one height -> two rows for that height + one row for inputs
            rows = 1 + 2
            cols = channels_per_h if channels_per_h > 0 else 3
            fig = plt.figure(figsize=(cols * 3.2, rows * 2.6))
            gs = gridspec.GridSpec(rows, cols, wspace=0.2, hspace=0.35)

            # Inputs row (first two columns)
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(input_seq[t, 0].numpy(), cmap='gray', vmin=in_vmin, vmax=in_vmax)
            ax0.set_title('input 0')
            ax0.axis('off')

            ax1 = fig.add_subplot(gs[0, 1] if cols > 1 else gs[0, 0])
            ax1.imshow(input_seq[t, 1].numpy(), cmap='gray', vmin=in_vmin, vmax=in_vmax)
            ax1.set_title('input 1')
            ax1.axis('off')

            # Put stats / summary in the remaining top-right area (if exists)
            if cols > 2:
                ax_stats = fig.add_subplot(gs[0, 2])
            else:
                ax_stats = fig.add_subplot(gs[0, -1])
            ax_stats.axis('off')
            # overall stats for the chosen height
            height_mae = mae_t[height_idx]  # (channels,)
            height_rmse = rmse_t[height_idx]
            stats_text = f'Time step: {t}/{T - 1}\n\nHeight {height_idx} stats (per-channel):\n'
            ch_names = ['u', 'v', 'w'][:channels_per_h]
            for cn, m, r in zip(ch_names, height_mae, height_rmse):
                stats_text += f'{cn}: MAE={m:.3f} RMSE={r:.3f}\n'
            stats_text += f'\nMean MAE={float(height_mae.mean()):.3f}'
            ax_stats.text(0.01, 0.99, stats_text, verticalalignment='top', fontsize=8, family='monospace')

            # GT and Pred rows for the chosen height
            row_gt = 1
            row_pr = 2
            for c_idx, ch_name in enumerate(['u', 'v', 'w'][:channels_per_h]):
                ax_gt = fig.add_subplot(gs[row_gt, c_idx])
                im = ax_gt.imshow(gt_t[height_idx, c_idx], cmap='jet',
                                  vmin=channel_vmin[c_idx], vmax=channel_vmax[c_idx])
                ax_gt.set_title(f'GT h{height_idx} {ch_name}', fontsize=8)
                ax_gt.axis('off')
                # Add a vertical colorbar next to this subplot. Adjust fraction/pad to taste.
                try:
                    fig.colorbar(im, ax=ax_gt, fraction=0.046, pad=0.04)
                except Exception:
                    # fallback: if `fig` isn't in scope or colorbar fails, ignore to avoid hard crash
                    pass

                ax_pr = fig.add_subplot(gs[row_pr, c_idx])
                im = ax_pr.imshow(pred_t[height_idx, c_idx], cmap='jet',
                                  vmin=channel_vmin[c_idx], vmax=channel_vmax[c_idx])
                ax_pr.set_title(f'Pred h{height_idx} {ch_name}', fontsize=8)
                ax_pr.axis('off')
                # Add a vertical colorbar next to this subplot. Adjust fraction/pad to taste.
                try:
                    fig.colorbar(im, ax=ax_pr, fraction=0.046, pad=0.04)
                except Exception:
                    # fallback: if `fig` isn't in scope or colorbar fails, ignore to avoid hard crash
                    pass

            frame = _to_rgb_frame(fig)
            writer.append_data(frame)
            plt.close(fig)

    print(f'[INFO] Saved video to {args.out_video}')

if __name__ == '__main__':
    main()
