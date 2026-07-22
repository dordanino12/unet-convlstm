"""
Quick test script to verify velocity data loader works correctly
Run this before full training to debug any data loading issues
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from API.dataloader_velocity import VelocityPredictionDataset, load_velocity_data


def test_dataset_basic(npz_path):
    """Test basic dataset loading and shapes"""
    print(f"\n{'='*70}")
    print(f"Testing VelocityPredictionDataset with: {npz_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(npz_path):
        print(f"❌ File not found: {npz_path}")
        return False
    
    try:
        # Load dataset
        dataset = VelocityPredictionDataset(npz_path, is_train=True, augment=False)
        
        # Check properties
        print(f"✓ Dataset loaded successfully")
        print(f"  - Number of sequences: {len(dataset)}")
        print(f"  - Input shape: ({dataset.T}, {dataset.C}, {dataset.H}, {dataset.W})")
        print(f"  - Expected: (12, 2, 128, 128)")
        
        # Test __getitem__
        x, y, mask = dataset[0]
        print(f"\n✓ Item retrieval successful")
        print(f"  - x shape: {x.shape} (expected: [12, 2, 128, 128])")
        print(f"  - y shape: {y.shape} (expected: [12, 2, 128, 128])")
        print(f"  - mask shape: {mask.shape} (expected: [12, 1, 128, 128])")
        
        # Check normalization ranges
        print(f"\n✓ Normalization info")
        print(f"  - X normalized range: [{x.min():.4f}, {x.max():.4f}] (expected: [0, ~1])")
        print(f"  - Y normalized range: [{y.min():.4f}, {y.max():.4f}] (expected: [-0.95, 0.95])")
        
        # Check mask stats
        print(f"\n✓ Mask info")
        print(f"  - Mask values: {torch.unique(mask).tolist()}")
        print(f"  - Mask coverage: {(mask > 0.5).float().mean():.2%} (cloud pixels)")
        print(f"  - Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
        
        # Check denormalization (use first channel since y is duplicated for simvp)
        y_first_channel = y[:, 0:1, :, :]  # [T, 1, H, W]
        y_denorm = dataset.denormalize_y(y_first_channel.numpy())
        print(f"\n✓ Denormalization successful")
        print(f"  - Y denormalized range: [{y_denorm.min():.4f}, {y_denorm.max():.4f}]")
        print(f"  - Expected raw velocity range: [{-dataset.max_neg_val:.4f}, {dataset.max_pos_val:.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(data_root, batch_size=2, test_root=None):
    """Test the complete data loader pipeline"""
    print(f"\n{'='*70}")
    print(f"Testing load_velocity_data with data_root: {data_root}")
    if test_root:
        print(f"Testing with shared test_root: {test_root}")
    print(f"{'='*70}\n")
    
    try:
        kwargs = {'augment': False}
        if test_root:
            kwargs['test_root'] = test_root
        
        train_loader, val_loader, test_loader, mean, std = load_velocity_data(
            batch_size=batch_size,
            val_batch_size=batch_size,
            data_root=data_root,
            num_workers=0,  # Single worker for testing
            **kwargs
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Test batch iteration
        print(f"\n✓ Testing batch iteration...")
        for i, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                x_batch, y_batch, mask_batch = batch_data
                print(f"  Batch {i}: x={x_batch.shape}, y={y_batch.shape}, mask={mask_batch.shape}")
                print(f"    - x range: [{x_batch.min():.4f}, {x_batch.max():.4f}]")
                print(f"    - y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
                print(f"    - mask coverage: {(mask_batch > 0.5).float().mean():.2%}")
            else:
                x_batch, y_batch = batch_data
                print(f"  Batch {i}: x={x_batch.shape}, y={y_batch.shape}")
                print(f"    - x range: [{x_batch.min():.4f}, {x_batch.max():.4f}]")
                print(f"    - y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
            if i >= 2:  # Show first 3 batches
                break
        
        print(f"\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test velocity data loader')
    parser.add_argument('--data_root', default='../data/', type=str, help='Data root directory')
    parser.add_argument('--test_root', default=None, type=str, help='Shared test root (optional, for k-fold)')
    parser.add_argument('--test_file', default=None, type=str, help='Specific file to test')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for loader test')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SimVP Velocity Data Loader Test")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Find and test specific file
    if args.test_file:
        passed = test_dataset_basic(args.test_file)
        all_passed = all_passed and passed
    else:
        # Test 2: Test full pipeline
        try:
            passed = test_dataloader(args.data_root, batch_size=args.batch_size, test_root=args.test_root)
            all_passed = all_passed and passed
        except FileNotFoundError as e:
            print(f"\n⚠️  Data files not found. Try specifying with --test_file")
            print(f"   Expected structure:")
            print(f"   {args.data_root}/")
            print(f"   ├── train_w.npz")
            print(f"   ├── val_w.npz")
            print(f"   └── test_w.npz")
            all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for training!")
        print("\nNext step: Run training with:")
        print("  python train_velocity.py --dataname velocity --data_root " + args.data_root)
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
        return 1
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
