"""
NPZ Dataset Loader
==================
Dataset class for loading velocity prediction sequences from NPZ files.
Uses Symmetric MaxAbsScaler normalization.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np


class NPZSequenceDataset(Dataset):
    """
    Dataset for dual-satellite velocity prediction.
    Implement Asymmetric Normalization to handle unbalanced positive/negative ranges.
    """

    def __init__(self, npz_path, use_gt_envelope_as_input=False, gt_envelope_npz_path=None,
                 augment=False, augment_repeats=1, deterministic_aug=False, use_one_satellite: bool = False):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
        # Optionally reduce to a single satellite channel (keep the first channel)
        self.use_one_satellite = bool(use_one_satellite)
        if self.use_one_satellite:
            # X shape: (N, T, C, H, W) -> select channel 0 -> (N, T, 1, H, W)
            self.X = self.X[:, :, 0:1, :, :]

        self.N, self.T, _, self.H, self.W = self.X.shape
        self.use_gt_envelope_as_input = use_gt_envelope_as_input
        self.gt_envelope_npz_path = gt_envelope_npz_path
        self.augment = augment
        self.augment_repeats = max(1, int(augment_repeats))
        self.deterministic_aug = deterministic_aug

        if self.use_gt_envelope_as_input:
            if self.gt_envelope_npz_path is None:
                raise ValueError("gt_envelope_npz_path is required when use_gt_envelope_as_input is True")
            env_data = np.load(self.gt_envelope_npz_path)
            self.Y_env = env_data["Y"].astype(np.float32)
            if self.Y_env.shape != self.Y.shape:
                raise ValueError(
                    f"Envelope Y shape {self.Y_env.shape} does not match target Y shape {self.Y.shape}"
                )

        # --- Statistics ---
        self.x_max = np.max(self.X)
        self.norm_const = max(self.x_max, 1.0)

        # 1. Analyze Ranges
        self.max_pos_val = np.max(self.Y)  # e.g., 3.3
        self.max_neg_val = np.abs(np.min(self.Y))  # e.g., |-1.5| = 1.5

        # 2. Define Target Norm (0.9 leaves headroom)
        self.target_norm = 0.8

        # 3. Calculate Symmetric Scale Factor
        # This ensures BOTH sides map to 0.9 magnitude
        self.max_abs_val = max(self.max_pos_val, self.max_neg_val)
        self.scale = self.max_abs_val / self.target_norm

        # Safety checks
        if self.scale == 0:
            self.scale = 1.0

        # --- Envelope Statistics (Optional) ---
        if self.use_gt_envelope_as_input:
            self.max_pos_env = np.max(self.Y_env)
            self.max_neg_env = np.abs(np.min(self.Y_env))
            self.max_abs_env = max(self.max_pos_env, self.max_neg_env)
            self.scale_env = self.max_abs_env / self.target_norm
            if self.scale_env == 0:
                self.scale_env = 1.0

        sat_info = "(single-sat mode)" if self.use_one_satellite else "(two-sat mode)"
        print(f"[INFO] Dataset Loaded. Range: [-{self.max_neg_val:.2f}, {self.max_pos_val:.2f}] {sat_info}")
        print(f"[INFO] Symmetric Norm:")
        print(f"       Scale: {self.scale:.2f} (Maps +/-{self.max_abs_val} -> +/-{self.target_norm})")

        if self.use_gt_envelope_as_input:
            print(f"[INFO] Envelope Range: [-{self.max_neg_env:.2f}, {self.max_pos_env:.2f}]")
            print(f"[INFO] Envelope Symmetric Norm:")
            print(f"       Scale: {self.scale_env:.2f} (Maps +/-{self.max_abs_env} -> +/-{self.target_norm})")

    def __len__(self):
        if self.augment and self.deterministic_aug:
            return self.N * self.augment_repeats
        return self.N

    def _apply_strict_aug(self, x, y, mask, k=0, flip_h=False, flip_v=False):
        # Apply same rotation and flips to all time steps and targets.
        if k:
            x = torch.rot90(x, k, dims=(-2, -1))
            y = torch.rot90(y, k, dims=(-2, -1))
            mask = torch.rot90(mask, k, dims=(-2, -1))

        if flip_h:
            x = torch.flip(x, dims=(-1,))
            y = torch.flip(y, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if flip_v:
            x = torch.flip(x, dims=(-2,))
            y = torch.flip(y, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))

        return x, y, mask

    def _get_aug_params(self, aug_id):
        # Deterministic dihedral-8 style transforms using one flip axis.
        ops = [
            (0, False, False),
            (1, False, False),
            (2, False, False),
            (3, False, False),
            (0, True, False),
            (1, True, False),
            (2, True, False),
            (3, True, False),
        ]

        if self.augment_repeats == 4:
            ops = [
                (0, False, False),
                (1, False, False),
                (2, True, False),
                (3, True, False),
            ]

        op = ops[aug_id % len(ops)]
        return op

    def __getitem__(self, idx):
        if self.augment and self.deterministic_aug:
            base_idx = idx % self.N
            aug_id = idx // self.N
        else:
            base_idx = idx
            aug_id = None

        x = torch.from_numpy(self.X[base_idx])
        y_raw = self.Y[base_idx]  # Keep as numpy first

        # --- STEP 1: CREATE MASK ---
        mask = (x[:, 0:1] > 1.1).float() # 1.1 for old beta, 0.9 for new beta.

        # --- STEP 2: NORMALIZE X ---
        x = x / self.norm_const

        # --- STEP 3: SYMMETRIC NORMALIZE Y ---
        y_norm = y_raw / self.scale

        # Clamp for safety
        y_norm = np.clip(y_norm, -1.0, 1.0).astype(np.float32)

        y = torch.from_numpy(y_norm)

        if self.use_gt_envelope_as_input:
            # Upper-bound experiment: feed GT envelope velocity as extra channel
            y_env_raw = self.Y_env[base_idx]
            y_env_norm = y_env_raw / self.scale_env
            y_env_norm = np.clip(y_env_norm, -1.0, 1.0).astype(np.float32)
            y_env = torch.from_numpy(y_env_norm)

            x = torch.cat([x, y_env], dim=1)

        if self.augment:
            if self.deterministic_aug:
                k, flip_h, flip_v = self._get_aug_params(aug_id)
                x, y, mask = self._apply_strict_aug(x, y, mask, k=k, flip_h=flip_h, flip_v=flip_v)
            else:
                k = int(torch.randint(0, 4, (1,)).item())
                flip_h = torch.rand(1).item() < 0.5
                flip_v = torch.rand(1).item() < 0.5
                x, y, mask = self._apply_strict_aug(x, y, mask, k=k, flip_h=flip_h, flip_v=flip_v)

        return x, y, mask

    def denormalize(self, y_norm: np.ndarray | torch.Tensor):
        """
        Invert the symmetric normalization.
        """
        is_torch = False
        device = None
        if isinstance(y_norm, torch.Tensor):
            is_torch = True
            device = y_norm.device
            y_norm = y_norm.detach().cpu().numpy()

        y_raw = y_norm * self.scale

        if is_torch:
            result = torch.from_numpy(y_raw.astype(np.float32))
            if device is not None:
                result = result.to(device)
            return result

        return y_raw

