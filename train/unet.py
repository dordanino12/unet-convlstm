from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
#from train.resnet18 import PretrainedTemporalUNet


# -----------------------------------------------------
# ConvLSTM building blocks
# -----------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x, state=None):
        B, C, H, W = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden_dim, H, W)
            c = x.new_zeros(B, self.hidden_dim, H, W)
        else:
            h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(ConvLSTMCell(input_dim if l == 0 else hidden_dim, hidden_dim, kernel_size))

    def forward(self, x_seq, state=None):
        T = len(x_seq)
        if state is None:
            state = [None] * len(self.layers)
        out = x_seq
        new_states = []
        for li, layer in enumerate(self.layers):
            h, c = (None, None) if state[li] is None else state[li]
            seq_out = []
            for t in range(T):
                h, (h, c) = layer(out[t], None if h is None else (h, c))
                seq_out.append(h)
            out = seq_out
            new_states.append((h, c))
        return out, new_states


# -----------------------------------------------------
# UNet blocks
# -----------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------
# Spatial Attention Module
# -----------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


# -----------------------------------------------------
# Temporal UNet Dual View with Attention (Optional)
# -----------------------------------------------------
class TemporalUNetDualView(nn.Module):
    def __init__(self, in_channels_per_sat=1, out_channels=1, base_ch=32, lstm_layers=1, use_skip_lstm=False, use_attention=False):
        super().__init__()
        in_ch_total = in_channels_per_sat * 2
        self.inc = DoubleConv(in_ch_total, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.bottleneck = Down(base_ch * 8, base_ch * 16)

        # --- OPTIONAL ATTENTION ---
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = SpatialAttention()
        # --------------------------

        # Temporal ConvLSTM
        self.temporal = ConvLSTM(base_ch * 16, base_ch * 16, num_layers=lstm_layers)

        self.use_skip_lstm = use_skip_lstm
        if use_skip_lstm:
            self.lstm_skip3 = ConvLSTM(base_ch * 8, base_ch * 8)
            self.lstm_skip2 = ConvLSTM(base_ch * 4, base_ch * 4)

        self.up3 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up0 = Up(base_ch * 2, base_ch)
        self.outc = OutConv(base_ch, out_channels)

    def encode_once(self, x_t):
        x0 = self.inc(x_t)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xb = self.bottleneck(x3)
        
        # Apply attention only if enabled
        if self.use_attention:
            xb = self.attention(xb)
            
        return xb, (x3, x2, x1, x0)

    def forward(self, x_seq, state=None):
        B, T, C, H, W = x_seq.shape
        bottlenecks = []
        skips = []

        for t in range(T):
            xb, sk = self.encode_once(x_seq[:, t])
            bottlenecks.append(xb)
            skips.append(sk)

        # Temporal ConvLSTM
        bottleneck_out, new_state = self.temporal(bottlenecks, state)

        if self.use_skip_lstm:
            x3_seq = [s[0] for s in skips]
            x2_seq = [s[1] for s in skips]
            x3_lstm, _ = self.lstm_skip3(x3_seq)
            x2_lstm, _ = self.lstm_skip2(x2_seq)
            for t in range(T):
                skips[t] = (x3_lstm[t], x2_lstm[t], skips[t][2], skips[t][3])

        out_seq = []
        for t in range(T):
            x3, x2, x1, x0 = skips[t]
            d3 = self.up3(bottleneck_out[t], x3)
            d2 = self.up2(d3, x2)
            d1 = self.up1(d2, x1)
            d0 = self.up0(d1, x0)
            out_seq.append(self.outc(d0))

        return out_seq, new_state


# -----------------------------------------------------
# NPZ Dataset loader with velocity normalization [-1,1]
# -----------------------------------------------------
class NPZSequenceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
        self.N, self.T, _, self.H, self.W = self.X.shape

        # --- Statistics ---
        self.x_max = np.max(self.X)
        self.norm_const = max(self.x_max, 1.0)

        # Symmetric MaxAbsScaler - Fixed scale factor
        self.scale_factor = 10.0

        # Store raw statistics for reference
        self.min_vel = float(np.min(self.Y))
        self.max_vel = float(np.max(self.Y))

        print(f"[INFO] Dataset Loaded. X Range: [0.0, {self.x_max:.2f}]")
        print(f"[INFO] Y Normalization (Symmetric MaxAbsScaler): scale_factor={self.scale_factor}")
        print(f"[INFO] Y Raw Range: [{self.min_vel:.3f}, {self.max_vel:.3f}] m/s")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])

        # --- STEP 1: CREATE MASK (Using raw physical values) ---
        mask = (x[:, 0:1] > 1.1).float()

        # --- STEP 2: NORMALIZE X ---
        x = x / self.norm_const

        # --- STEP 3: NORMALIZE Y (Symmetric MaxAbsScaler) ---
        y_raw = y.numpy()

        # Divide by scale_factor and clamp to [-1, 1]
        y_normalized = y_raw / self.scale_factor
        y_normalized = np.clip(y_normalized, -1.0, 1.0).astype(np.float32)

        y = torch.from_numpy(y_normalized)

        return x, y, mask

    def denormalize(self, y_norm: np.ndarray | torch.Tensor):
        """Invert the dataset normalization. Accepts numpy array or torch tensor.

        Returns raw y values in original units.
        """
        is_torch = False
        device = None
        if isinstance(y_norm, torch.Tensor):
            is_torch = True
            device = y_norm.device  # Save device
            y_norm = y_norm.cpu().numpy()

        # Simple multiplication by scale_factor
        y_raw = y_norm * self.scale_factor

        if is_torch:
            result = torch.from_numpy(y_raw.astype(np.float32))
            if device is not None:
                result = result.to(device)  # Return to original device
            return result
        return y_raw
