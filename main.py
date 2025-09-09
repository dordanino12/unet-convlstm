"""
Temporal UNet + ConvLSTM with optional Skip-LSTM for dual-satellite velocity estimation
Author: ChatGPT — Sept 2025
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# -----------------------------
# ConvLSTM building blocks
# -----------------------------
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

# -----------------------------
# UNet blocks
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

# -----------------------------
# Temporal UNet Dual View with optional Skip-LSTM
# -----------------------------
class TemporalUNetDualView(nn.Module):
    def __init__(self, in_channels_per_sat=1, out_channels=1, base_ch=32, lstm_layers=1, use_skip_lstm=False):
        super().__init__()
        in_ch_total = in_channels_per_sat * 2
        # Encoder
        self.inc = DoubleConv(in_ch_total, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.bottleneck = Down(base_ch*8, base_ch*16)
        # Bottleneck LSTM
        self.temporal = ConvLSTM(base_ch*16, base_ch*16, num_layers=lstm_layers)
        # Optional skip LSTMs
        self.use_skip_lstm = use_skip_lstm
        if use_skip_lstm:
            self.lstm_skip3 = ConvLSTM(base_ch*8, base_ch*8, num_layers=1)
            self.lstm_skip2 = ConvLSTM(base_ch*4, base_ch*4, num_layers=1)
        # Decoder
        self.up3 = Up(base_ch*16, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4)
        self.up1 = Up(base_ch*4, base_ch*2)
        self.up0 = Up(base_ch*2, base_ch)
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

# -----------------------------
# Dataset for Moving MNIST
# -----------------------------
class MovingMNISTDataset(Dataset):
    def __init__(self, path, max_vel=5.0):
        npz = np.load(path)
        self.data = npz["data"]  # (N, T, 2, H, W)
        self.max_vel = max_vel
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx])
        x = seq[:, 0:1].repeat(1, 2, 1, 1)    # simulate 2 satellites
        y = seq[:, 1:2] / self.max_vel
        mask = (seq[:, 0:1] > 0).float()
        return x, y, mask

# -----------------------------
# Training & Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, mask in tqdm(loader, desc="Training", leave=False):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        y_pred_seq, _ = model(x)
        y_pred = torch.stack(y_pred_seq, dim=1)
        loss = ((y_pred - y).abs() * mask).sum() / (mask.sum() + 1e-8)
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
        loss = ((y_pred - y).abs() * mask).sum() / (mask.sum() + 1e-8)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    path = "moving_mnist_2.npz"  # update path if needed
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MovingMNISTDataset(path)
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    # Choose if to use skip LSTM
    USE_SKIP_LSTM = True  # <-- change to False to disable skip-LSTM
    model = TemporalUNetDualView(in_channels_per_sat=1, out_channels=1, base_ch=32, lstm_layers=1, use_skip_lstm=USE_SKIP_LSTM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    EPOCHS = 20
    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train={tr:.4f} | val={val:.4f}")
    torch.save({
        'model_state': model.state_dict(),
        'cfg': {'in_channels_per_sat': 1, 'out_channels': 1, 'use_skip_lstm': USE_SKIP_LSTM}
    }, 'temporal_unet_convlstm_dualview.pt')
    print("Saved model to temporal_unet_convlstm_dualview.pt")
