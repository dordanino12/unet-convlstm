"""
SimVP Training Script for Velocity Prediction
Adapted for dual-satellite velocity estimation task
Shape: [B, T, C, H, W] = [B, 12, 2, 128, 128]
Target: [B, T, 1, H, W] = [B, 12, 1, 128, 128]
"""

import argparse
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')

from exp import Exp


def create_velocity_parser():
    """Create parser with velocity prediction defaults"""
    parser = argparse.ArgumentParser(description='SimVP for Velocity Prediction')

    # Device and I/O
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--res_dir', default='./results', type=str, help='Results directory')
    parser.add_argument('--ex_name', default='velocity_simvp', type=str, help='Experiment name')
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
    parser.add_argument('--seed', default=1, type=int)

    # Dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size for validation')
    parser.add_argument('--data_root', default='../data/', type=str,
                        help='Data root directory (contains train/val splits)')
    parser.add_argument('--test_root', default=None, type=str,
                        help='Test data root (optional, for kfold with shared test set)')
    parser.add_argument('--dataname', default='velocity', type=str, choices=['mmnist', 'taxibj', 'velocity'])
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')

    # Model parameters - optimized for 128x128 velocity fields
    parser.add_argument('--in_shape', default=[12, 2, 128, 128], type=int, nargs='*',
                        help='[T_input, C_input, H, W] = [12, 2, 128, 128]')
    parser.add_argument('--hid_S', default=128, type=int, help='Spatial hidden channels')
    parser.add_argument('--hid_T', default=512, type=int, help='Temporal hidden channels')
    parser.add_argument('--N_S', default=4, type=int, help='Spatial depth (encoder/decoder levels)')
    parser.add_argument('--N_T', default=8, type=int, help='Temporal depth')
    parser.add_argument('--groups', default=8, type=int, help='Channel groups for grouped convolutions')

    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--log_step', default=1, type=int, help='Logging frequency')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # Loss function toggle
    parser.add_argument('--use_bin_loss', action='store_true', help='Use custom physical Bin Loss')
    parser.add_argument('--no_bin_loss', dest='use_bin_loss', action='store_false', help='Use standard MSE Loss')
    parser.set_defaults(use_bin_loss=True)

    # Mask toggle
    parser.add_argument('--use_mask', action='store_true', help='Use mask to calculate loss only on cloud pixels')
    parser.add_argument('--no_mask', dest='use_mask', action='store_false',
                        help='Train on the entire image (ignore mask)')
    parser.set_defaults(use_mask=True)

    # Data-specific options
    parser.add_argument('--use_one_satellite', default=False, type=bool,
                        help='Use only first satellite channel')
    parser.add_argument('--augment', default=False, type=bool,
                        help='Apply data augmentation')

    return parser


def print_config(args):
    """Pretty print configuration"""
    print("\n" + "=" * 70)
    print("SimVP Configuration for Velocity Prediction")
    print("=" * 70)
    config = args.__dict__
    for key in sorted(config.keys()):
        if key not in ['device', 'use_gpu']:  # Skip redundant params
            print(f"  {key:.<30} {config[key]}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    parser = create_velocity_parser()
    args = parser.parse_args()
    config = args.__dict__

    # Validate input shape for velocity prediction
    assert len(args.in_shape) == 4, "in_shape must be [T, C, H, W]"
    assert args.in_shape[0] == 12, "T must be 12 (12 timesteps)"
    assert args.in_shape[3] == 128 and args.in_shape[2] == 128, "Spatial size must be 128x128"

    # Check if using kfold structure
    fold_name = os.path.basename(args.data_root.rstrip('/'))
    if 'fold_' in fold_name or args.test_root:
        print(f"\n[K-FOLD MODE] Using fold: {fold_name}")
        if args.test_root:
            print(f"[K-FOLD MODE] Shared test set: {args.test_root}/test_w.npz")

    print_config(args)

    print('>' * 35 + ' TRAINING ' + '<' * 35)
    exp = Exp(args)
    exp.train(args)

    print('>' * 35 + ' TESTING  ' + '<' * 35)
    mse = exp.test(args)

    print(f"\n{'=' * 70}")
    print(f"Final Test MSE: {mse:.6f}")
    print(f"Results saved to: {args.res_dir}/{args.ex_name}/")
    print(f"{'=' * 70}\n")