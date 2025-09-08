"""
Temporal UNet + ConvLSTM for dual-satellite cloud vertical velocity estimation (w-field)
--------------------------------------------------------------------------------------
- Input at each time t: two satellite images that view the same cloud field at approximately the same time t (from two satellites).
  The two frames are concatenated along the channel dimension.
- Sequence input: length T. The model outputs a velocity map per time step.
- Output: single-channel (default) vertical velocity (w) map in m/s (or normalized), per pixel.

Notes:
- If you have more than 1 channel per satellite (e.g., multi-spectral), set in_channels_per_sat accordingly.
- If you prefer a 2D horizontal velocity vector (u,v) or 3D (u,v,w), set out_channels=2 or 3 respectively.
- The model supports online/streaming inference by keeping ConvLSTM hidden state.

Author: ChatGPT — Aug 29, 2025
"""
from __future__ import annotations
import os
import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# Utility: ConvLSTM (2D) blocks
# -----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, C, H, W = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden_dim, H, W)
            c = x.new_zeros(B, self.hidden_dim, H, W)
        else:
            h, c = state
        concat = torch.cat([x, h], dim=1)
        gates = self.conv(concat)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_dim = input_dim if l == 0 else hidden_dim
            self.layers.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))

    def forward(self, x_seq: List[torch.Tensor], state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        T = len(x_seq)
        if state is None:
            state = [None] * len(self.layers)
        layer_inputs = x_seq
        new_states = []
        for li, layer in enumerate(self.layers):
            h, c = (None, None) if state[li] is None else state[li]
            out_seq = []
            for t in range(T):
                h, (h, c) = layer(layer_inputs[t], None if h is None else (h, c))
                out_seq.append(h)
            layer_inputs = out_seq
            new_states.append((h, c))
        outputs_per_t = layer_inputs  # last layer outputs per time step
        return outputs_per_t, new_states

# -----------------------------
# UNet encoder/decoder blocks
# -----------------------------
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
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad x1 to match x2
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# -------------------------------------------------
# Temporal UNet with ConvLSTM at the bottleneck
# Dual-view fusion: concatenate channels per time t
# -------------------------------------------------
class TemporalUNetDualView(nn.Module):
    def __init__(self,
                 in_channels_per_sat: int = 1,
                 out_channels: int = 1,
                 base_ch: int = 32,
                 lstm_layers: int = 1):
        super().__init__()
        in_ch_total = in_channels_per_sat * 2  # שני לוויינים בערוצים מחוברים

        # Encoder
        self.inc = DoubleConv(in_ch_total, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.bottleneck = Down(base_ch * 8, base_ch * 16)

        # ConvLSTM בבוטלנק (אותו מספר ערוצים כמו הבוטלנק)
        self.temporal = ConvLSTM(input_dim=base_ch * 16,
                                 hidden_dim=base_ch * 16,
                                 num_layers=lstm_layers)

        # Decoder
        self.up3 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up0 = Up(base_ch * 2, base_ch)
        self.outc = OutConv(base_ch, out_channels)

    def encode_once(self, x_t: torch.Tensor):
        """Encodes a single time step"""
        x0 = self.inc(x_t)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xb = self.bottleneck(x3)
        return xb, (x3, x2, x1, x0)

    def forward(self, x_seq: torch.Tensor, state=None):
        """
        x_seq: [B, T, C_total, H, W]
        מחזיר:
          y_seq: list של [B, out_channels, H, W] לכל t
          new_state: מצב חדש של ה־ConvLSTM
        """
        B, T, C, H, W = x_seq.shape
        enc_bottlenecks = []
        skips_per_t = []
        for t in range(T):
            xb, skips = self.encode_once(x_seq[:, t])
            enc_bottlenecks.append(xb)
            skips_per_t.append(skips)

        # Temporal fusion בבוטלנק
        lstm_out_seq, new_state = self.temporal(enc_bottlenecks, state)

        # Decode כל צעד בזמן עם ה־skip connections שלו
        y_seq = []
        for t in range(T):
            x3, x2, x1, x0 = skips_per_t[t]
            xb = lstm_out_seq[t]   # פלט ConvLSTM עם base_ch*16 ערוצים
            d3 = self.up3(xb, x3)
            d2 = self.up2(d3, x2)
            d1 = self.up1(d2, x1)
            d0 = self.up0(d1, x0)
            y = self.outc(d0)
            y_seq.append(y)
        return y_seq, new_state

    @torch.no_grad()
    def forward_streaming(self, x_t, state=None):
        """צעד בודד (streaming) עם שמירה על מצב ה־LSTM"""
        xb, skips = self.encode_once(x_t)
        lstm_out_seq, new_state = self.temporal([xb], state)
        x3, x2, x1, x0 = skips
        d3 = self.up3(lstm_out_seq[0], x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)
        y = self.outc(d0)
        return y, new_state

# -----------------------------
# Example Dataset (skeleton)
# -----------------------------
class SatellitePairSequenceDataset(Dataset):
    """
    Expects preprocessed tensors per sequence:
      - inputs: torch.FloatTensor [T, 2*C_in, H, W]  (two satellites concatenated by channel)
      - targets: torch.FloatTensor [T, C_out, H, W]  (velocity map per time step)
    You should adapt __getitem__ to load from your storage format (e.g., .pt/.npy files or HDF5).
    """
    def __init__(self, sequences: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.data = sequences
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

# -----------------------------
# Training / Evaluation helpers
# -----------------------------
class Averager:
    def __init__(self):
        self.reset()
    def reset(self):
        self.tot = 0.0; self.n = 0
    def add(self, val, n=1):
        self.tot += float(val) * n; self.n += n
    @property
    def avg(self):
        return self.tot / max(1, self.n)


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# Dataset
# -----------------------------
class MovingMNISTDataset(Dataset):
    def __init__(self, path, max_vel=5.0):
        npz_file = np.load(path)
        self.data = npz_file["data"]  # (N, T, 2, H, W)
        self.max_vel = max_vel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx])  # (T, 2, H, W)

        # Input: digit sequence (first channel)
        x = seq[:, 0:1]  # (T, 1, H, W)
        # duplicate channel to simulate two cameras
        x = x.repeat(1, 2, 1, 1)  # (T, 2, H, W)

        # Target: velocity map (second channel)
        y = seq[:, 1:2] / self.max_vel  # (T, 1, H, W)

        # Mask: foreground
        mask = (seq[:, 0:1] > 0).float()

        return x, y, mask


# -----------------------------
# Training function
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x, y, mask in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        y_seq_pred, _ = model(x)  # list of [B, C, H, W]
        y_pred = torch.stack(y_seq_pred, dim=1)  # [B, T, C, H, W]

        # Compute masked L1 loss
        diff = (y_pred - y) * mask
        loss = diff.abs().sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

    return total_loss / n_samples


# -----------------------------
# Evaluation function
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        y_seq_pred, _ = model(x)
        y_pred = torch.stack(y_seq_pred, dim=1)

        diff = (y_pred - y) * mask
        loss = diff.abs().sum() / (mask.sum() + 1e-8)

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

    return total_loss / n_samples


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    path = "moving_mnist_2.npz"
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MovingMNISTDataset(path)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = TemporalUNetDualView(in_channels_per_sat=1, out_channels=1, base_ch=32, lstm_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    EPOCHS = 20
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train={tr_loss:.4f} | val={val_loss:.4f}")

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'cfg': {
            'in_channels_per_sat': 1,
            'out_channels': 1
        }
    }, 'temporal_unet_convlstm_dualview.pt')
    print("Saved: temporal_unet_convlstm_dualview.pt")

#