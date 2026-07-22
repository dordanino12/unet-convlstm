"""
Custom Velocity Prediction Data Loader for SimVP
Adapted from the project's NPZSequenceDataset
Format: X [N, T, C, H, W], Y [N, T, 1, H, W]
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class VelocityPredictionDataset(data.Dataset):
    """
    Dataset for dual-satellite velocity prediction from NPZ files.
    Uses symmetric MaxAbsScaler normalization matching the original project.
    """

    def __init__(self, npz_path, is_train=True, use_one_satellite=False, augment=False):
        """
        Args:
            npz_path: Path to NPZ file with 'X' and 'Y' keys
            is_train: Whether this is training set (for augmentation)
            use_one_satellite: If True, use only first satellite channel
            augment: Apply augmentation (rotations, flips) to data
        """
        super(VelocityPredictionDataset, self).__init__()
        
        # Load data
        loaded_data = np.load(npz_path, allow_pickle=True)
        self.X = loaded_data["X"].astype(np.float32)  # [N, T, C, H, W]
        self.Y = loaded_data["Y"].astype(np.float32)  # [N, T, 1, H, W]
        
        # Handle single satellite mode
        self.use_one_satellite = use_one_satellite
        if self.use_one_satellite:
            # Select first satellite channel: [N, T, 2, H, W] -> [N, T, 1, H, W]
            self.X = self.X[:, :, 0:1, :, :]
        
        self.N, self.T, self.C, self.H, self.W = self.X.shape
        self.is_train = is_train
        self.augment = augment and is_train
        
        # === Normalization Setup ===
        # Normalize X by max value
        self.x_max = np.max(np.abs(self.X))
        self.norm_const_x = max(self.x_max, 1.0)
        
        # Symmetric normalization for Y (velocity)
        self.max_pos_val = float(np.max(self.Y))
        self.max_neg_val = float(np.abs(np.min(self.Y)))
        self.max_abs_val = max(self.max_pos_val, self.max_neg_val)
        
        # Scale factor: maps ±max_abs_val to ±0.95
        self.target_norm = 0.95
        self.scale = self.max_abs_val / self.target_norm if self.max_abs_val > 0 else 1.0
        
        # Mean and std for compatibility with SimVP's expectation
        self.mean = 0.0
        self.std = 1.0
        
        mode_str = "(single-sat mode)" if self.use_one_satellite else "(dual-sat mode)"
        print(f"[VelocityDataset] Loaded {self.N} sequences from {os.path.basename(npz_path)} {mode_str}")
        print(f"  X shape: {self.X.shape}, Y shape: {self.Y.shape}")
        print(f"  X normalization: max={self.x_max:.4f}, norm_const={self.norm_const_x:.4f}")
        print(f"  Y velocity range: [{-self.max_neg_val:.4f}, {self.max_pos_val:.4f}]")
        print(f"  Y scale factor: {self.scale:.4f} (maps ±{self.max_abs_val:.4f} -> ±{self.target_norm})")

    def __len__(self):
        return self.N

    def normalize_x(self, x):
        """Normalize input satellite images"""
        return x / self.norm_const_x

    def normalize_y(self, y):
        """Symmetric normalize target velocity"""
        y_norm = y / self.scale
        y_norm = np.clip(y_norm, -1.0, 1.0).astype(np.float32)
        return y_norm

    def denormalize_y(self, y_norm):
        """Reverse symmetric normalization for velocity"""
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.detach().cpu().numpy()
        y_raw = y_norm * self.scale
        return y_raw.astype(np.float32)

    def _apply_augmentation(self, x, y, mask):
        """Apply deterministic augmentation: rotations and flips"""
        # Random rotation (0, 90, 180, 270 degrees)
        k = int(np.random.randint(0, 4))
        x = torch.tensor(np.rot90(x, k, axes=(-2, -1)))
        y = torch.tensor(np.rot90(y, k, axes=(-2, -1)))
        mask = torch.tensor(np.rot90(mask, k, axes=(-2, -1)))
        
        # Random horizontal flip
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=(-1,))
            y = torch.flip(y, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        
        # Random vertical flip
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=(-2,))
            y = torch.flip(y, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        
        return x.numpy(), y.numpy(), mask.numpy()

    def __getitem__(self, idx):
        # Get sequence
        x = self.X[idx]  # [T, C, H, W]
        y = self.Y[idx]  # [T, 1, H, W]
        
        # === CREATE MASK FROM ENVELOPE (CLOUD PIXELS) ===
        # Mask = 1.0 where x[:, 0:1] > 1.1 (cloud), 0.0 elsewhere
        # x is not yet normalized, so we use raw values
        mask = (x[:, 0:1] > 1.1).astype(np.float32)  # [T, 1, H, W]
        
        # Normalize
        x = self.normalize_x(x)
        y = self.normalize_y(y)
        
        # Apply augmentation if training
        if self.augment:
            x, y, mask = self._apply_augmentation(x, y, mask)
        
        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        mask = torch.from_numpy(mask).float()
        
        # Duplicate velocity channel to match model's 2-channel input
        # SimVP expects output to match input channels: [T, 2, H, W]
        y = y.repeat(1, 2, 1, 1)  # [T, 1, H, W] -> [T, 2, H, W]
        # Mask shape stays [T, 1, H, W]
        
        return x, y, mask


def load_velocity_data(batch_size, val_batch_size, data_root, num_workers, **kwargs):
    """
    Load velocity prediction dataset (compatible with SimVP's API).
    
    Expected files:
    - data_root/train_w.npz
    - data_root/val_w.npz
    - test_root/test_w.npz (or data_root/test_w.npz if test_root not specified)
    """
    
    test_root = kwargs.get('test_root', None)
    if test_root is None:
        test_root = data_root
    
    # Try to find data files
    def find_data_file(search_root, filename):
        # Direct path
        direct_path = os.path.join(search_root, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        # Recursive search in search_root
        for root, dirs, files in os.walk(search_root):
            if filename in files:
                return os.path.join(root, filename)
        
        raise FileNotFoundError(f"Cannot find {filename} in {search_root}")
    
    train_path = find_data_file(data_root, 'train_w.npz')
    val_path = find_data_file(data_root, 'val_w.npz')
    test_path = find_data_file(test_root, 'test_w.npz')
    
    print(f"[DataLoader] Train: {train_path}")
    print(f"[DataLoader] Val:   {val_path}")
    print(f"[DataLoader] Test:  {test_path}")
    
    # Create datasets
    train_set = VelocityPredictionDataset(
        train_path, 
        is_train=True, 
        use_one_satellite=kwargs.get('use_one_satellite', False),
        augment=kwargs.get('augment', False)
    )
    
    val_set = VelocityPredictionDataset(
        val_path, 
        is_train=False, 
        use_one_satellite=kwargs.get('use_one_satellite', False),
        augment=False
    )
    
    test_set = VelocityPredictionDataset(
        test_path, 
        is_train=False, 
        use_one_satellite=kwargs.get('use_one_satellite', False),
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Return in expected format: (train_loader, val_loader, test_loader, mean, std)
    data_mean = train_set.mean
    data_std = train_set.std
    
    return train_loader, val_loader, test_loader, data_mean, data_std
