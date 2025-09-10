"""
Temporal UNet + ConvLSTM with optional Skip-LSTM for dual-satellite velocity estimation
Improved training: weighted L1 + gradient loss to preserve fine details
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

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
# Temporal UNet Dual View
# -----------------------------------------------------
class TemporalUNetDualView(nn.Module):
    def __init__(self, in_channels_per_sat=1, out_channels=1, base_ch=32, lstm_layers=1, use_skip_lstm=False):
        super().__init__()
        in_ch_total = in_channels_per_sat * 2
        self.inc = DoubleConv(in_ch_total, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.bottleneck = Down(base_ch * 8, base_ch * 16)
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
        return xb, (x3, x2, x1, x0)

    def forward(self, x_seq, state=None):
        B, T, C, H, W = x_seq.shape
        bottlenecks = []
        skips = []
        for t in range(T):
            xb, sk = self.encode_once(x_seq[:, t])
            bottlenecks.append(xb)
            skips.append(sk)
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
    def __init__(self, npz_path, lower_percentile=0.1, upper_percentile=99.9, clip_outliers=True):
        """
        Loads NPZ dataset, applies optional clipping based on percentiles, and normalizes velocities to [-1, 1].

        Args:
            npz_path (str): Path to the NPZ file.
            lower_percentile (float): Lower percentile for clipping (default: 5).
            upper_percentile (float): Upper percentile for clipping (default: 95).
            clip_outliers (bool): Whether to clip values outside the chosen percentiles.
        """
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)  # [N, T, 2, H, W]
        self.Y = data["Y"].astype(np.float32)  # [N, T, 1, H, W]
        self.N, self.T, _, self.H, self.W = self.X.shape

        # Compute percentiles for normalization
        self.min_vel = np.percentile(self.Y, lower_percentile)
        self.max_vel = np.percentile(self.Y, upper_percentile)

        self.clip_outliers = clip_outliers

        print(f"[INFO] Velocity normalization range based on percentiles:")
        print(f"       {lower_percentile}th percentile: {self.min_vel}")
        print(f"       {upper_percentile}th percentile: {self.max_vel}")
        if not clip_outliers:
            print("[INFO] Outliers will NOT be clipped (may produce values outside [-1,1])")
        else:
            print("[INFO] Outliers will be clipped to range before normalization")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])   # [T, 2, H, W]
        y = torch.from_numpy(self.Y[idx])   # [T, 1, H, W]

        # Optional clipping of outliers
        if self.clip_outliers:
            y = torch.clamp(y, self.min_vel, self.max_vel)

        # Normalize velocity to [-1,1]
        y = 2 * (y - self.min_vel) / (self.max_vel - self.min_vel) - 1

        # Mask where either of the two input channels is greater than 0.12
        mask = ((x[:, 0:1] > 0.12) | (x[:, 1:2] > 0.12)).float()

        return x, y, mask


# -----------------------------------------------------
# Improved loss: weighted L1 + gradient loss
# -----------------------------------------------------
def compute_loss(y_pred, y, mask):
    """
    Weighted L1 loss + gradient loss.
    y_pred, y: [B, T, 1, H, W]
    mask: same shape
    """
    # Base L1
    abs_diff = (y_pred - y).abs()

    # Weighting: high-velocity pixels get higher weight
    weight = 1.0 + 2.0 * (y.abs() > 0.8).float()  # 3x weight for extreme velocities

    weighted_l1 = (abs_diff * mask * weight).sum() / (mask * weight).sum()

    # Gradient loss: encourages sharp changes to match GT
    def spatial_gradients(tensor):
        dx = tensor[..., :, 1:] - tensor[..., :, :-1]
        dy = tensor[..., 1:, :] - tensor[..., :-1, :]
        return dx, dy

    dx_pred, dy_pred = spatial_gradients(y_pred)
    dx_gt, dy_gt = spatial_gradients(y)

    grad_diff = (dx_pred - dx_gt).abs().sum() + (dy_pred - dy_gt).abs().sum()
    grad_loss = grad_diff / (mask[..., :-1, :-1].sum() + 1e-8)

    total_loss = weighted_l1 + 0.05 * grad_loss
    return total_loss


# -----------------------------------------------------
# Training & evaluation
# -----------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, mask in tqdm(loader, desc="Training", leave=False):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        y_pred_seq, _ = model(x)
        y_pred = torch.stack(y_pred_seq, dim=1)
        loss = compute_loss(y_pred, y, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        y_pred_seq, _ = model(x)
        y_pred = torch.stack(y_pred_seq, dim=1)
        loss = compute_loss(y_pred, y, mask)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    npz_path = "data/dataset_sequences_original.npz"
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NPZSequenceDataset(npz_path)
    print("Dataset length:", len(dataset))
    x, y, mask = dataset[0]
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("mask.shape:", mask.shape)

    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    USE_SKIP_LSTM = True
    model = TemporalUNetDualView(
        in_channels_per_sat=1,
        out_channels=1,
        base_ch=32,
        lstm_layers=1,
        use_skip_lstm=USE_SKIP_LSTM
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train={tr:.6f} | val={val:.6f}")

    torch.save({
        'model_state': model.state_dict(),
        'cfg': {'in_channels_per_sat': 1, 'out_channels': 1, 'use_skip_lstm': USE_SKIP_LSTM}
    }, 'models/temporal_unet_convlstm_dualview_from_npz.pt')
    print("Saved model to temporal_unet_convlstm_dualview_from_npz.pt")
