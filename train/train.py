
"""
Train script for multi-height velocity prediction.
- Expects NPZ files where:
  X: (N, T, 2, H, W)
  Y: (N, T, Nh, 3, H, W)  # Nh heights, 3 channels (w,u,v)
- Model: shared MiT-B1 encoder + ConvLSTM bottleneck (pretrained) and one decoder+head per velocity component
- y_pred returned by the model has shape: [B, T, Nh, 3, H, W]
- Loss: bin-weighted L1 applied globally to flattened channels, plus small gradient term
- Saves best checkpoint and prints per-channel + aggregated metrics
"""
from __future__ import annotations
import argparse
import os
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from resnet18 import PretrainedTemporalUNetMitB1


# ------------------ Dataset ------------------
class MultiHeightNPZDataset(Dataset):
    def __init__(self, npz_path, target_norm=0.8):
        # We load without forcing it all into memory at once
        self.data = np.load(npz_path, mmap_mode='r')
        self.X_ref = self.data['X']  # Reference only, not loaded to RAM
        self.Y_ref = self.data['Y']  # Reference only, not loaded to RAM

        if self.X_ref.ndim != 5:
            raise ValueError(f"Unexpected X dims: {self.X_ref.shape}")
        if self.Y_ref.ndim != 6:
            raise ValueError(f"Unexpected Y dims: {self.Y_ref.shape}")

        self.N, self.T, self.C, self.H, self.W = self.X_ref.shape
        self.Ny, self.Ty, self.Nh, self.C3, self.Hy, self.Wy = self.Y_ref.shape
        assert self.N == self.Ny and self.T == self.Ty and self.H == self.Hy and self.W == self.Wy
        if self.C3 != 3:
            raise ValueError(f"Expected 3 Y channels per height, got {self.C3}")

        # Normalization constants for X
        # Calculate max directly from the memory-mapped file
        self.x_max = float(np.max(self.X_ref))
        self.norm_const = max(self.x_max, 1.0)

        # Per-height/per-channel symmetric max-absolute normalization for Y (no offsets).
        self.target_norm = float(target_norm)
        self.y_maxabs = np.zeros((self.Nh, self.C3), dtype=np.float32)
        self.y_scale = np.ones((self.Nh, self.C3), dtype=np.float32)

        for h in range(self.Nh):
            for c in range(self.C3):
                # Process channel-by-channel to save RAM during initialization
                slice_vals = self.Y_ref[:, :, h, c, :, :]
                max_pos = float(np.max(slice_vals))
                max_neg = float(abs(np.min(slice_vals)))
                max_abs = max(max_pos, max_neg)
                self.y_maxabs[h, c] = max_abs
                if np.isclose(max_abs, 0.0):
                    self.y_scale[h, c] = 1.0
                else:
                    self.y_scale[h, c] = float(max_abs / self.target_norm)

        # Print labeled per-height/per-channel symmetric scales
        channel_names = ['w', 'u', 'v']  # Adjusted to match your Y structure
        print(f"[DATA] Y symmetric max-abs scales (Nh*3={self.Nh * self.C3}):")
        for h in range(self.Nh):
            parts = []
            for c in range(self.C3):
                label = channel_names[c] if c < len(channel_names) else f"c{c}"
                parts.append(f"h{h}:{label}={self.y_scale[h, c]:.6f}")
            print("  " + ", ".join(parts))

        print(f"[DATA] Loaded {npz_path}: N={self.N}, T={self.T}, Nh={self.Nh}, C_in={self.C}, H={self.H}, W={self.W}")
        print(f"[DATA] X norm_const={self.norm_const:.3f}")
        print(
            f"[DATA] Y uses per-height/per-channel symmetric max-absolute normalization to map +/-{'{:.3f}'.format(np.max(self.y_maxabs.flat))} -> +/-{self.target_norm:.3f}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Pull only the specific index into RAM and convert to float32
        x_raw = np.array(self.X_ref[idx], dtype=np.float32)
        y_raw = np.array(self.Y_ref[idx], dtype=np.float32)

        # Normalize X
        x_raw = x_raw / self.norm_const

        # Normalize Y per height/channel
        y_norm = np.empty_like(y_raw, dtype=np.float32)
        for h in range(self.Nh):
            for c in range(self.C3):
                scale = self.y_scale[h, c]
                y_norm[:, h, c, :, :] = y_raw[:, h, c, :, :] / scale

        x = torch.from_numpy(x_raw)
        y = torch.from_numpy(y_norm)

        return x, y

    def denormalize(self, y_norm: Union[torch.Tensor, np.ndarray]):
        """Convert normalized tensor back to physical units using per-height/per-channel stats."""
        is_torch = isinstance(y_norm, torch.Tensor)
        device = None
        if is_torch:
            device = y_norm.device
            arr = y_norm.detach().cpu().numpy()
        else:
            arr = np.asarray(y_norm)

        arr = np.asarray(arr, dtype=np.float32)

        orig_was_flat5 = False
        if arr.ndim == 5:
            if arr.shape[2] != self.Nh * self.C3:
                raise ValueError(f"Unexpected flattened Y shape: {arr.shape}; expected channel dim {self.Nh * self.C3}")
            arr = arr.reshape(arr.shape[0], arr.shape[1], self.Nh, self.C3, arr.shape[3], arr.shape[4])
            orig_was_flat5 = True
        elif arr.ndim != 6:
            raise ValueError(f"Unexpected Y shape for denormalize: {arr.shape}")

        out = np.empty_like(arr, dtype=np.float32)
        for h in range(self.Nh):
            for c in range(self.C3):
                scale = self.y_scale[h, c]
                out[:, :, h, c, :, :] = arr[:, :, h, c, :, :] * scale

        if orig_was_flat5:
            out = out.reshape(out.shape[0], out.shape[1], self.Nh * self.C3, out.shape[4], out.shape[5])

        if is_torch:
            result = torch.from_numpy(out)
            if device is not None:
                result = result.to(device)
            return result
        return out


# ------------------ Model ------------------
class MultiHeightTemporalModel(nn.Module):
    """Shared encoder + ConvLSTM bottleneck (from PretrainedTemporalUNetMitB1),
    separate decoder+head per velocity type (u/v/w). Each head outputs Nh channels.
    Outputs shape: [B, T, Nh, 3, H, W]
    """

    def __init__(self, in_channels: int, num_heights: int, lstm_layers=1, pretrained=True, dropout_p=0.3):
        super().__init__()
        # Instantiate a template MiT-B1 based network to extract encoder / bottleneck machinery
        temp = PretrainedTemporalUNetMitB1(
            out_channels=3,
            lstm_layers=lstm_layers,
            freeze_encoder=False,
            in_channels=in_channels,
            dropout_p=dropout_p
        )
        # Use its encoder/bottleneck/lstm/skip projections
        self.input_adapter = getattr(temp, 'input_adapter', None)
        self.encoder = temp.encoder
        self.bottleneck_proj = getattr(temp, 'bottleneck_proj', None)
        self.bottleneck_expand = getattr(temp, 'bottleneck_expand', None)
        self.lstm = getattr(temp, 'lstm', None)
        self.skip_proj_layers = getattr(temp, 'skip_proj_layers', None)
        self.skip_expand_layers = getattr(temp, 'skip_expand_layers', None)
        self.lstm_skips = getattr(temp, 'lstm_skips', None)
        self._lstm_skip_map = getattr(temp, '_lstm_skip_map', None)
        self.dropout = getattr(temp, 'dropout', nn.Identity())

        # Separate decoder and head per velocity type (u/v/w)
        from copy import deepcopy
        self.decoders = nn.ModuleList([deepcopy(temp.decoder) for _ in range(3)])

        # The head from SMP is a simple segmentation_head (typically a Conv2d)
        original_head = deepcopy(temp.head)

        # Create a wrapper head that first applies the original head, then expands channels
        class MultiChannelHeadWrapper(nn.Module):
            def __init__(self, original_head, in_channels, out_channels):
                super().__init__()
                self.original_head = original_head
                self.expansion = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

            def forward(self, x):
                out = self.original_head(x)  # [B*T, 3, H, W]
                out = self.expansion(out)  # [B*T, Nh, H, W]
                return out

        self.heads = nn.ModuleList([
            MultiChannelHeadWrapper(deepcopy(original_head), in_channels=3, out_channels=num_heights)
            for _ in range(3)
        ])

        self.num_heights = num_heights
        del temp

    def forward(self, x_seq):
        # x_seq: [B, T, C, H, W]
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        if self.input_adapter is not None:
            x_enc = self.input_adapter(x_flat)
        else:
            x_enc = x_flat
        features = self.encoder(x_enc)

        # Bottleneck and temporal LSTM
        if self.lstm is not None and self.bottleneck_proj is not None:
            bottleneck = self.bottleneck_proj(features[-1])
            bottleneck_seq = bottleneck.view(B, T, -1, bottleneck.shape[2], bottleneck.shape[3])
            lstm_in_list = [bottleneck_seq[:, t] for t in range(T)]
            lstm_out_list, _ = self.lstm(lstm_in_list)
            lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
            lstm_out_flat = lstm_out_stacked.view(B * T, -1, bottleneck.shape[2], bottleneck.shape[3])
            bottleneck_restored = self.bottleneck_expand(lstm_out_flat)
            features[-1] = self.dropout(bottleneck_restored)

            if self.lstm_skips is not None and len(self.lstm_skips) > 0:
                lstm_idx = 0
                proj_idx = 0
                for i, use_lstm in enumerate(self._lstm_skip_map):
                    if not use_lstm:
                        proj_idx += 1
                        continue
                    feat = features[i]
                    Ck = feat.shape[1]
                    if Ck == 0:
                        proj_idx += 1
                        continue
                    hk, wk = feat.shape[2], feat.shape[3]
                    feat_proj = self.skip_proj_layers[proj_idx](feat)
                    feat_seq = feat_proj.view(B, T, -1, hk, wk)
                    lstm_in = [feat_seq[:, t] for t in range(T)]
                    lstm_out_list, _ = self.lstm_skips[lstm_idx](lstm_in)
                    lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
                    lstm_out_flat = lstm_out_stacked.view(B * T, -1, hk, wk)
                    feat_expanded = self.skip_expand_layers[proj_idx](lstm_out_flat)
                    features[i] = self.dropout(feat_expanded)
                    lstm_idx += 1
                    proj_idx += 1
        else:
            features[-1] = self.dropout(features[-1])

        # Pass through per-velocity decoder and head
        vel_outputs = []
        for dec, head in zip(self.decoders, self.heads):
            dec_out = dec(*features)
            out_flat = head(self.dropout(dec_out))  # [B*T, Nh, H, W]
            out_seq = out_flat.view(B, T, self.num_heights, H, W)
            vel_outputs.append(out_seq)

        # Stack to [B, T, 3, Nh, H, W] then permute to [B, T, Nh, 3, H, W]
        y_stack = torch.stack(vel_outputs, dim=2)
        y_pred = y_stack.permute(0, 1, 3, 2, 4, 5)
        return y_pred


# ------------------ Loss (bin-weighted L1 + small gradient) ------------------
def compute_bin_loss(y_pred, y_true, dataset_obj: MultiHeightNPZDataset, bin_min, bin_max):
    """
    y_pred, y_true: Tensors with shape [B, T, Nh, 3, H, W]
    Simple bin-weighted L1 loss. Uses global bins across all channels.
    Returns scalar loss tensor and per-channel loss list.
    """
    B, T, Nh, C3, H, W = y_pred.shape
    C = Nh * C3
    y_p = y_pred.reshape(B, T, C, H, W)
    y = y_true.reshape(B, T, C, H, W)

    y_denorm = dataset_obj.denormalize(y)

    BIN_MIN = float(bin_min)
    BIN_MAX = float(bin_max)
    BIN_WIDTH = 0.1
    NUM_BINS = int(math.ceil((BIN_MAX - BIN_MIN) / BIN_WIDTH))

    y_flat = y_denorm.flatten()
    device = y_pred.device
    bin_counts = torch.zeros(NUM_BINS, device=device)
    for i in range(NUM_BINS):
        start = BIN_MIN + i * BIN_WIDTH
        end = start + BIN_WIDTH
        bin_mask = (y_flat >= start) & (y_flat < end)
        bin_counts[i] = bin_mask.sum().float()

    total_pixels = y_flat.numel() if y_flat.numel() > 0 else 1.0
    bin_weights = torch.zeros(NUM_BINS, device=device)
    non_empty = bin_counts > 0
    if non_empty.any():
        bin_weights[non_empty] = total_pixels / (bin_counts[non_empty] + 1e-8)
        bin_weights[non_empty] = bin_weights[non_empty] / (bin_weights[non_empty].mean() + 1e-8)
    bin_weights = torch.clamp(bin_weights, max=100.0)

    pixel_bin_weights = torch.zeros_like(y_denorm, device=device)
    for i in range(NUM_BINS):
        start = BIN_MIN + i * BIN_WIDTH
        end = start + BIN_WIDTH
        bin_mask = (y_denorm >= start) & (y_denorm < end)
        pixel_bin_weights[bin_mask] = bin_weights[i]

    combined_weight = pixel_bin_weights
    combined_weight_6d = combined_weight.reshape(B, T, Nh, C3, H, W)

    abs_diff = (y_p - y).abs()
    numerator = (abs_diff * combined_weight).sum()
    denom = combined_weight.sum() + 1e-8
    weighted_l1 = numerator / denom

    per_channel_losses = []
    for c in range(C3):
        y_p_c = y_pred[:, :, :, c, :, :]
        y_c = y_true[:, :, :, c, :, :]
        w_c = combined_weight_6d[:, :, :, c, :, :]
        abs_diff_c = (y_p_c - y_c).abs()
        numerator_c = (abs_diff_c * w_c).sum()
        denom_c = w_c.sum() + 1e-8
        per_channel_losses.append(numerator_c / denom_c)

    return weighted_l1, per_channel_losses


# ------------------ Training / Evaluation ------------------

def train_one_epoch(model, loader, optimizer, device, dataset_obj, scaler, bin_min, bin_max):
    model.train()
    total_loss = 0.0
    n = 0

    sum_mae = 0.0
    total_elements = 0
    sum_per_channel_mae = [0.0 for _ in range(dataset_obj.C3)]
    total_elements_per_c = [0 for _ in range(dataset_obj.C3)]
    per_channel_loss = [0.0 for _ in range(dataset_obj.C3)]

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type):
            y_pred = model(x)
            loss, loss_per_c = compute_bin_loss(y_pred, y, dataset_obj, bin_min=bin_min, bin_max=bin_max)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.detach().item() * bs
        n += bs
        for c in range(dataset_obj.C3):
            per_channel_loss[c] += float(loss_per_c[c].detach().item()) * bs

        # Accumulate sums instead of large lists to prevent memory leak
        with torch.no_grad():
            y_den = dataset_obj.denormalize(y).cpu().numpy()
            pred_den = dataset_obj.denormalize(y_pred).cpu().numpy()
            diff = np.abs(pred_den - y_den)

            sum_mae += float(np.sum(diff))
            total_elements += diff.size

            for c in range(dataset_obj.C3):
                diff_c = diff[:, :, :, c, :, :]
                sum_per_channel_mae[c] += float(np.sum(diff_c))
                total_elements_per_c[c] += diff_c.size

    avg_loss = total_loss / n if n > 0 else 0.0
    agg_mae = sum_mae / total_elements if total_elements > 0 else 0.0
    per_c_mae = [sum_per_channel_mae[c] / total_elements_per_c[c] if total_elements_per_c[c] > 0 else 0.0 for c in
                 range(dataset_obj.C3)]
    per_c_loss = [l / n if n > 0 else 0.0 for l in per_channel_loss]

    return avg_loss, agg_mae, per_c_mae, per_c_loss


@torch.no_grad()
def evaluate(model, loader, device, dataset_obj, bin_min, bin_max):
    model.eval()
    total_loss = 0.0
    n = 0

    sum_mae = 0.0
    total_elements = 0
    sum_per_channel_mae = [0.0 for _ in range(dataset_obj.C3)]
    total_elements_per_c = [0 for _ in range(dataset_obj.C3)]
    per_channel_loss = [0.0 for _ in range(dataset_obj.C3)]

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with autocast(device_type=device.type):
            y_pred = model(x)
            loss, loss_per_c = compute_bin_loss(y_pred, y, dataset_obj, bin_min=bin_min, bin_max=bin_max)
        bs = x.size(0)
        total_loss += loss.detach().item() * bs
        n += bs
        for c in range(dataset_obj.C3):
            per_channel_loss[c] += float(loss_per_c[c].detach().item()) * bs

        y_den = dataset_obj.denormalize(y).cpu().numpy()
        pred_den = dataset_obj.denormalize(y_pred).cpu().numpy()
        diff = np.abs(pred_den - y_den)

        sum_mae += float(np.sum(diff))
        total_elements += diff.size

        for c in range(dataset_obj.C3):
            diff_c = diff[:, :, :, c, :, :]
            sum_per_channel_mae[c] += float(np.sum(diff_c))
            total_elements_per_c[c] += diff_c.size

    avg_loss = total_loss / n if n > 0 else 0.0
    agg_mae = sum_mae / total_elements if total_elements > 0 else 0.0
    per_c_mae = [sum_per_channel_mae[c] / total_elements_per_c[c] if total_elements_per_c[c] > 0 else 0.0 for c in
                 range(dataset_obj.C3)]
    per_c_loss = [l / n if n > 0 else 0.0 for l in per_channel_loss]
    return avg_loss, agg_mae, per_c_mae, per_c_loss


# ------------------ Main ------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='/home/danino/PycharmProjects/pythonProject/data/3d_688to702_train_uvw.npz',
                   help='Training NPZ path')
    p.add_argument('--val', default='/home/danino/PycharmProjects/pythonProject/data/3d_688to702_val_uvw.npz',
                   help='Validation NPZ path')
    p.add_argument('--test', default='/home/danino/PycharmProjects/pythonProject/data/3d_688to702_test_uvw.npz',
                   help='Test NPZ path')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--out', default='/home/danino/PycharmProjects/pythonProject/train/models')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = MultiHeightNPZDataset(args.train)
    val_ds = MultiHeightNPZDataset(args.val)
    test_ds = MultiHeightNPZDataset(args.test)

    val_ds.norm_const = train_ds.norm_const
    test_ds.norm_const = train_ds.norm_const

    # Quick sanity check
    sample_y = np.array(train_ds.Y_ref[:1], dtype=np.float32)
    sample_y_t = torch.from_numpy(sample_y)
    sample_y_rt = train_ds.denormalize(train_ds.__getitem__(0)[1].unsqueeze(0))
    max_rt_err = float(torch.max(torch.abs(sample_y_t - sample_y_rt)).item())
    print(f"[DATA] Y round-trip max abs error (sample): {max_rt_err:.6e}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Derive bin bounds directly from the mmap slices without fully loading to RAM
    print("Derived BIN bounds:")
    # Instead of pulling everything, find min and max directly
    bin_min = float(np.min(train_ds.Y_ref))
    bin_max = float(np.max(train_ds.Y_ref))
    print(f"  {bin_min:.4f} to {bin_max:.4f}")

    model = MultiHeightTemporalModel(
        in_channels=train_ds.C,
        num_heights=train_ds.Nh,
        dropout_p=args.dropout
    ).to(device)

    def _count_params(module):
        return int(sum(p.numel() for p in module.parameters()))

    def _count_trainable_params(module):
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

    enc_params = _count_params(getattr(model, 'encoder', nn.Identity())) if hasattr(model, 'encoder') else 0
    convlstm_params = 0
    if hasattr(model, 'lstm') and model.lstm is not None:
        convlstm_params += _count_params(model.lstm)
    if hasattr(model, 'lstm_skips') and model.lstm_skips is not None:
        try:
            for m in model.lstm_skips:
                convlstm_params += _count_params(m)
        except Exception:
            convlstm_params += _count_params(model.lstm_skips)

    decoder_params = sum(_count_params(m) for m in model.decoders) if hasattr(model, 'decoders') else 0
    head_params = sum(_count_params(m) for m in model.heads) if hasattr(model, 'heads') else 0

    total_params = _count_params(model)
    trainable_params = _count_trainable_params(model)

    print('\n[MODEL PARAMS] Breakdown:')
    print('  Encoder: parameters of the shared encoder (feature extractor)')
    print(f"    Encoder params: {enc_params:,}")
    print('  ConvLSTM: parameters of the temporal bottleneck LSTM(s) (main + any skip LSTMs)')
    print(f"    ConvLSTM params: {convlstm_params:,}")
    print(f"  Decoders (u/v/w): {decoder_params:,} params")
    print(f"  Heads (u/v/w, output Nh each): {head_params:,} params")
    print(f"  Total model params: {total_params:,} ({trainable_params:,} trainable)")
    print('[MODEL PARAMS] End of breakdown\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    os.makedirs(args.out, exist_ok=True)
    best_val = float('inf')
    best_saved = False
    best_path = os.path.join(args.out, 'multih_best.pt')

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae, tr_per_c, tr_loss_per_c = train_one_epoch(
            model, train_loader, optimizer, device, train_ds, scaler, bin_min, bin_max
        )
        val_loss, val_mae, val_per_c, val_loss_per_c = evaluate(
            model, val_loader, device, val_ds, bin_min, bin_max
        )

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {tr_loss:.4f} MAE: {tr_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f} MAE: {val_mae:.4f}")
        channel_names = ['u', 'v', 'w']
        for c, (t, v, tl, vl) in enumerate(zip(tr_per_c, val_per_c, tr_loss_per_c, val_loss_per_c)):
            label = channel_names[c] if c < len(channel_names) else f"c{c}"
            print(f"   {label}: Train MAE={t:.4f} | Val MAE={v:.4f} | Train Loss={tl:.4f} | Val Loss={vl:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, best_path)
            best_saved = True
            print(f"Saved best model (overwritten): {best_path} | Val Loss: {val_loss:.4f}")

    if best_saved:
        print(f"Loading best model from {best_path} for final test")
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck['model_state'])

    test_loss, test_mae, test_per_c, test_loss_per_c = evaluate(
        model, test_loader, device, test_ds, bin_min, bin_max
    )
    print(f"Test Loss: {test_loss:.4f} MAE: {test_mae:.4f}")
    for c, (v, vl) in enumerate(zip(test_per_c, test_loss_per_c)):
        label = channel_names[c] if c < len(channel_names) else f"c{c}"
        print(f"  {label}: Test MAE={v:.4f} | Test Loss={vl:.4f}")


if __name__ == '__main__':
    main()