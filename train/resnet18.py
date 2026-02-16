import torch
import torch.nn as nn
import sys
import os
import segmentation_models_pytorch as smp

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


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
# Lightweight Refiner Head (optional)
# -----------------------------------------------------
class RefinerHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

        # Zero initialization for stable training when refiner starts
        self._init_weights()

    def _init_weights(self):
        # Random init for hidden layers (default PyTorch initialization)
        # Zero init only for the final output layer for stable training
        final_conv = self.net[-1]  # Last layer
        if isinstance(final_conv, nn.Conv2d):
            nn.init.zeros_(final_conv.weight)
            if final_conv.bias is not None:
                nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------
# New Model: ResNet18 Encoder + Temporal Bottleneck
# -----------------------------------------------------
class PretrainedTemporalUNet(nn.Module):
    def __init__(self, out_channels=1, lstm_layers=1, freeze_encoder=True, in_channels=2, dropout_p=0.5,
                 use_refiner=False, refiner_hidden_channels=32):
        super().__init__()
        self.out_channels = out_channels
        self.refiner_hidden_channels = refiner_hidden_channels
        self.in_channels = in_channels

        # 1. Create base U-Net based on ResNet18
        # weights="imagenet" -> Loads pre-trained knowledge
        # in_channels is configurable; the library automatically adapts the first layer.
        self.base_model = smp.Unet(
            encoder_name="resnet18",        
            encoder_weights="imagenet",    
            in_channels=in_channels,                  
            classes=out_channels,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16) # Lightweight and fast decoder
        )
        
        # 2. Decompose the model into components
        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.head = self.base_model.segmentation_head

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.use_refiner = use_refiner
        self.refiner = (
            RefinerHead(out_channels + in_channels, out_channels, refiner_hidden_channels)
            if use_refiner else None
        )

        # 3. Freeze the Encoder (saves memory and prevents Overfitting)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] ResNet Encoder is FROZEN.")

        # 4. Add ConvLSTM at the bottleneck
        # In ResNet18, the final output (stage 5) has 512 channels
        self.lstm_input_dim = 512
        self.proj_channels = 128  # Reduce to 128 channels for efficiency

        # Bottleneck projection and expansion layers
        self.bottleneck_proj = nn.Conv2d(self.lstm_input_dim, self.proj_channels, kernel_size=1, bias=False)
        self.bottleneck_expand = nn.Conv2d(self.proj_channels, self.lstm_input_dim, kernel_size=1, bias=False)

        self.lstm = ConvLSTM(
            input_dim=self.proj_channels,  # Now operates on reduced dimension
            hidden_dim=self.proj_channels, # Keep the same width
            num_layers=lstm_layers,
            kernel_size=3
        )
        # Skip connections will be used as-is without ConvLSTM processing
        # Only the bottleneck will have temporal processing via ConvLSTM

    def enable_refiner(self, hidden_channels=None):
        if hidden_channels is None:
            hidden_channels = self.refiner_hidden_channels
        if self.refiner is None:
            self.refiner = RefinerHead(self.out_channels + self.in_channels, self.out_channels, hidden_channels)
            # Move refiner to the same device as the model
            if next(self.parameters(), None) is not None:
                device = next(self.parameters()).device
                self.refiner = self.refiner.to(device)
        self.use_refiner = True

    def forward(self, x_seq):
        # x_seq shape: [B, T, C, H, W]
        B, T, C, H, W = x_seq.shape
        
        # --- A. ENCODER (Frame by Frame) ---
        # We merge B and T dimensions to pass everything through the Encoder at once
        # Reshape: [B*T, 1, H, W]
        x_flat = x_seq.view(B * T, C, H, W)
        
        # The Encoder returns a list of features (Skip Connections)
        # features[0] -> High resolution ... features[-1] -> The Bottleneck
        features = self.encoder(x_flat)

        # Note: do not enforce exact skip counts here; let mismatches raise
        # natural errors during execution so failures are explicit in stack traces.
        
        # Extract the Bottleneck (the deepest feature)
        bottleneck = features[-1] # Shape: [B*T, 512, H/32, W/32]
        
        # --- B. TEMPORAL PROCESSING (LSTM) ---
        # Project bottleneck to lower dimension
        # Shape: [B*T, 128, H/32, W/32]
        bottleneck_proj = self.bottleneck_proj(bottleneck)

        # Reshape back to time dimension so the LSTM understands the sequence
        # Shape: [B, T, 128, H/32, W/32]
        bottleneck_seq = bottleneck_proj.view(B, T, -1, bottleneck_proj.shape[2], bottleneck_proj.shape[3])

        # Your ConvLSTM expects a list of tensors
        lstm_in_list = [bottleneck_seq[:, t] for t in range(T)]
        
        # Run the LSTM
        lstm_out_list, _ = self.lstm(lstm_in_list)
        
        # Stack back into a single tensor
        lstm_out_stacked = torch.stack(lstm_out_list, dim=1) # [B, T, 128, H/32, W/32]

        # --- C. DECODER (Frame by Frame) ---
        # Flatten again to [B*T, ...] to enter the Decoder
        lstm_out_flat = lstm_out_stacked.view(B * T, -1, bottleneck_proj.shape[2], bottleneck_proj.shape[3])

        # Expand back to original bottleneck dimension
        # Shape: [B*T, 512, H/32, W/32]
        bottleneck_expanded = self.bottleneck_expand(lstm_out_flat)

        # The Trick: Replace the last feature in the list (which was static)
        # with the LSTM output (which is dynamic)
        features[-1] = self.dropout(bottleneck_expanded)

        # Decode (skip connections are used as-is without temporal processing)
        decoder_out = self.decoder(*features)
        
        # Final output layer
        output_flat = self.head(self.dropout(decoder_out))
        if self.use_refiner:
            ref_input = torch.cat([output_flat, x_flat], dim=1)
            output_flat = output_flat + self.refiner(ref_input)
        # Reshape back to original shape: [B, T, 1, H, W]
        output_seq = output_flat.view(B, T, -1, H, W)
        
        return output_seq, None # (None because there is no external state currently)


# -----------------------------------------------------
# New Model: MiT-B3 Encoder + Temporal Bottleneck
# -----------------------------------------------------
class PretrainedTemporalUNetMitB3(nn.Module):
    def __init__(self, out_channels=1, lstm_layers=1, freeze_encoder=True, in_channels=2, dropout_p=0.2,
                 use_refiner=False, refiner_hidden_channels=32):
        super().__init__()
        self.out_channels = out_channels
        self.refiner_hidden_channels = refiner_hidden_channels
        self.in_channels = in_channels
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = None

        self.base_model = smp.MAnet(
            encoder_name="mit_b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=out_channels,
            encoder_depth=5,
            decoder_channels=(512, 256, 128, 64, 32)
        )

        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.head = self.base_model.segmentation_head

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.use_refiner = use_refiner
        self.refiner = (
            RefinerHead(out_channels + in_channels, out_channels, refiner_hidden_channels)
            if use_refiner else None
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] MiT-B3 Encoder is FROZEN.")

        encoder_out_channels = None
        if hasattr(self.encoder, 'out_channels'):
            encoder_out_channels = getattr(self.encoder, 'out_channels')
        elif hasattr(self.base_model, 'encoder_out_channels'):
            encoder_out_channels = getattr(self.base_model, 'encoder_out_channels')

        if encoder_out_channels is None:
            raise RuntimeError("MiT-B3 encoder channels are unavailable.")

        self.lstm_input_dim = encoder_out_channels[-1]
        self.lstm = ConvLSTM(
            input_dim=self.lstm_input_dim,
            hidden_dim=self.lstm_input_dim,
            num_layers=lstm_layers,
            kernel_size=3
        )

        skip_channels = encoder_out_channels[:-1]
        self.skip_channels = list(skip_channels)
        lstm_modules = []
        for ch in skip_channels:
            if ch <= 0:
                lstm_modules.append(None)
            else:
                lstm_modules.append(ConvLSTM(input_dim=ch, hidden_dim=ch, num_layers=lstm_layers, kernel_size=3))
        self.lstm_skips = nn.ModuleList([m for m in lstm_modules if m is not None])
        self._lstm_skip_map = [m is not None for m in lstm_modules]

    def enable_refiner(self, hidden_channels=None):
        if hidden_channels is None:
            hidden_channels = self.refiner_hidden_channels
        if self.refiner is None:
            self.refiner = RefinerHead(self.out_channels + self.in_channels, self.out_channels, hidden_channels)
            # Move refiner to the same device as the model
            if next(self.parameters(), None) is not None:
                device = next(self.parameters()).device
                self.refiner = self.refiner.to(device)
        self.use_refiner = True

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        x_flat_orig = x_flat
        if self.input_adapter is not None:
            x_flat = self.input_adapter(x_flat)
        features = self.encoder(x_flat)

        bottleneck = features[-1]
        bottleneck_seq = bottleneck.view(B, T, -1, bottleneck.shape[2], bottleneck.shape[3])
        lstm_in_list = [bottleneck_seq[:, t] for t in range(T)]
        lstm_out_list, _ = self.lstm(lstm_in_list)
        lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
        lstm_out_flat = lstm_out_stacked.view(B * T, -1, bottleneck.shape[2], bottleneck.shape[3])
        features[-1] = self.dropout(lstm_out_flat)

        lstm_idx = 0
        for i, use_lstm in enumerate(self._lstm_skip_map):
            if not use_lstm:
                continue
            feat = features[i]
            Ck = feat.shape[1]
            if Ck == 0:
                continue
            hk, wk = feat.shape[2], feat.shape[3]
            feat_seq = feat.view(B, T, Ck, hk, wk)
            lstm_in = [feat_seq[:, t] for t in range(T)]
            lstm_out_list, _ = self.lstm_skips[lstm_idx](lstm_in)
            lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
            features[i] = self.dropout(lstm_out_stacked.view(B * T, Ck, hk, wk))
            lstm_idx += 1

        decoder_out = self.decoder(*features)
        output_flat = self.head(self.dropout(decoder_out))
        if self.use_refiner:
            ref_input = torch.cat([output_flat, x_flat_orig], dim=1)
            output_flat = output_flat + self.refiner(ref_input)
        output_seq = output_flat.view(B, T, -1, H, W)
        return output_seq, None


# -----------------------------------------------------
# New Model: MiT-B2 Encoder + Temporal Bottleneck
# -----------------------------------------------------
class PretrainedTemporalUNetMitB2(nn.Module):
    def __init__(self, out_channels=1, lstm_layers=1, freeze_encoder=True, in_channels=2, dropout_p=0.2,
                 use_refiner=False, refiner_hidden_channels=32):
        super().__init__()
        self.out_channels = out_channels
        self.refiner_hidden_channels = refiner_hidden_channels
        self.in_channels = in_channels
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = None

        self.base_model = smp.MAnet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=out_channels,
            encoder_depth=5,
            decoder_channels=(512, 256, 128, 64, 32)
        )

        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.head = self.base_model.segmentation_head

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.use_refiner = use_refiner
        self.refiner = (
            RefinerHead(out_channels + in_channels, out_channels, refiner_hidden_channels)
            if use_refiner else None
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] MiT-B2 Encoder is FROZEN.")

        encoder_out_channels = None
        if hasattr(self.encoder, 'out_channels'):
            encoder_out_channels = getattr(self.encoder, 'out_channels')
        elif hasattr(self.base_model, 'encoder_out_channels'):
            encoder_out_channels = getattr(self.base_model, 'encoder_out_channels')

        if encoder_out_channels is None:
            raise RuntimeError("MiT-B2 encoder channels are unavailable.")

        bottleneck_in_channels = encoder_out_channels[-1]
        self.lstm_input_dim = bottleneck_in_channels
        self.lstm = ConvLSTM(
            input_dim=self.lstm_input_dim,
            hidden_dim=self.lstm_input_dim,
            num_layers=lstm_layers,
            kernel_size=3
        )

        skip_channels = encoder_out_channels[:-1]
        self.skip_channels = list(skip_channels)
        lstm_modules = []
        for ch in skip_channels:
            if ch <= 0:
                lstm_modules.append(None)
            else:
                lstm_modules.append(ConvLSTM(input_dim=ch, hidden_dim=ch, num_layers=lstm_layers, kernel_size=3))
        self.lstm_skips = nn.ModuleList([m for m in lstm_modules if m is not None])
        self._lstm_skip_map = [m is not None for m in lstm_modules]

    def enable_refiner(self, hidden_channels=None):
        if hidden_channels is None:
            hidden_channels = self.refiner_hidden_channels
        if self.refiner is None:
            self.refiner = RefinerHead(self.out_channels + self.in_channels, self.out_channels, hidden_channels)
            # Move refiner to the same device as the model
            if next(self.parameters(), None) is not None:
                device = next(self.parameters()).device
                self.refiner = self.refiner.to(device)
        self.use_refiner = True

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        x_flat_orig = x_flat
        if self.input_adapter is not None:
            x_flat = self.input_adapter(x_flat)
        features = self.encoder(x_flat)

        bottleneck = features[-1]
        bottleneck_seq = bottleneck.view(B, T, -1, bottleneck.shape[2], bottleneck.shape[3])
        lstm_in_list = [bottleneck_seq[:, t] for t in range(T)]
        lstm_out_list, _ = self.lstm(lstm_in_list)
        lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
        lstm_out_flat = lstm_out_stacked.view(B * T, -1, bottleneck.shape[2], bottleneck.shape[3])
        features[-1] = self.dropout(lstm_out_flat)

        lstm_idx = 0
        for i, use_lstm in enumerate(self._lstm_skip_map):
            if not use_lstm:
                continue
            feat = features[i]
            Ck = feat.shape[1]
            if Ck == 0:
                continue
            hk, wk = feat.shape[2], feat.shape[3]
            feat_seq = feat.view(B, T, Ck, hk, wk)
            lstm_in = [feat_seq[:, t] for t in range(T)]
            lstm_out_list, _ = self.lstm_skips[lstm_idx](lstm_in)
            lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
            features[i] = self.dropout(lstm_out_stacked.view(B * T, Ck, hk, wk))
            lstm_idx += 1

        decoder_out = self.decoder(*features)
        output_flat = self.head(self.dropout(decoder_out))
        if self.use_refiner:
            ref_input = torch.cat([output_flat, x_flat_orig], dim=1)
            output_flat = output_flat + self.refiner(ref_input)
        output_seq = output_flat.view(B, T, -1, H, W)
        return output_seq, None


# -----------------------------------------------------
# New Model: MiT-B1 Encoder + Temporal Bottleneck
# -----------------------------------------------------
class PretrainedTemporalUNetMitB1(nn.Module):
    def __init__(self, out_channels=1, lstm_layers=1, freeze_encoder=True, in_channels=2, dropout_p=0.2,
                 proj_channels=64, use_refiner=False, refiner_hidden_channels=32):
        super().__init__()
        self.out_channels = out_channels
        self.refiner_hidden_channels = refiner_hidden_channels
        self.in_channels = in_channels
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = None

        self.base_model = smp.MAnet(
            encoder_name="mit_b1",
            encoder_weights="imagenet",
            in_channels=3,
            classes=out_channels,
            encoder_depth=5,
            decoder_channels=(512, 256, 128, 64, 32)
        )

        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.head = self.base_model.segmentation_head

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.use_refiner = use_refiner
        self.refiner = (
            RefinerHead(out_channels + in_channels, out_channels, refiner_hidden_channels)
            if use_refiner else None
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] MiT-B1 Encoder is FROZEN.")

        encoder_out_channels = None
        if hasattr(self.encoder, 'out_channels'):
            encoder_out_channels = getattr(self.encoder, 'out_channels')
        elif hasattr(self.base_model, 'encoder_out_channels'):
            encoder_out_channels = getattr(self.base_model, 'encoder_out_channels')

        if encoder_out_channels is None:
            raise RuntimeError("MiT-B1 encoder channels are unavailable.")

        bottleneck_in_channels = encoder_out_channels[-1]
        self.bottleneck_proj = nn.Conv2d(bottleneck_in_channels, proj_channels, kernel_size=1, bias=False)
        self.bottleneck_expand = nn.Conv2d(proj_channels, bottleneck_in_channels, kernel_size=1, bias=False)

        self.lstm_input_dim = proj_channels
        self.lstm = ConvLSTM(
            input_dim=self.lstm_input_dim,
            hidden_dim=self.lstm_input_dim,
            num_layers=lstm_layers,
            kernel_size=3
        )

        skip_channels = encoder_out_channels[:-1]
        self.skip_channels = list(skip_channels)

        # Create projection layers for each skip connection
        self.skip_proj_layers = nn.ModuleList()
        self.skip_expand_layers = nn.ModuleList()

        lstm_modules = []
        for ch in skip_channels:
            if ch <= 0:
                lstm_modules.append(None)
                self.skip_proj_layers.append(nn.Identity())
                self.skip_expand_layers.append(nn.Identity())
            else:
                # Add projection layer for this skip connection
                self.skip_proj_layers.append(nn.Conv2d(ch, proj_channels, kernel_size=1, bias=False))
                self.skip_expand_layers.append(nn.Conv2d(proj_channels, ch, kernel_size=1, bias=False))
                lstm_modules.append(ConvLSTM(input_dim=proj_channels, hidden_dim=proj_channels, num_layers=lstm_layers, kernel_size=3))

        self.lstm_skips = nn.ModuleList([m for m in lstm_modules if m is not None])
        self._lstm_skip_map = [m is not None for m in lstm_modules]

    def enable_refiner(self, hidden_channels=None):
        if hidden_channels is None:
            hidden_channels = self.refiner_hidden_channels
        if self.refiner is None:
            self.refiner = RefinerHead(self.out_channels + self.in_channels, self.out_channels, hidden_channels)
            # Move refiner to the same device as the model
            if next(self.parameters(), None) is not None:
                device = next(self.parameters()).device
                self.refiner = self.refiner.to(device)
        self.use_refiner = True

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        x_flat_orig = x_flat
        if self.input_adapter is not None:
            x_flat = self.input_adapter(x_flat)
        features = self.encoder(x_flat)

        bottleneck = self.bottleneck_proj(features[-1])
        bottleneck_seq = bottleneck.view(B, T, -1, bottleneck.shape[2], bottleneck.shape[3])
        lstm_in_list = [bottleneck_seq[:, t] for t in range(T)]
        lstm_out_list, _ = self.lstm(lstm_in_list)
        lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
        lstm_out_flat = lstm_out_stacked.view(B * T, -1, bottleneck.shape[2], bottleneck.shape[3])
        bottleneck_restored = self.bottleneck_expand(lstm_out_flat)
        features[-1] = self.dropout(bottleneck_restored)

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

            # Project skip connection
            feat_proj = self.skip_proj_layers[proj_idx](feat)

            feat_seq = feat_proj.view(B, T, -1, hk, wk)
            lstm_in = [feat_seq[:, t] for t in range(T)]
            lstm_out_list, _ = self.lstm_skips[lstm_idx](lstm_in)
            lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
            lstm_out_flat = lstm_out_stacked.view(B * T, -1, hk, wk)

            # Expand back to original channels
            feat_expanded = self.skip_expand_layers[proj_idx](lstm_out_flat)
            features[i] = self.dropout(feat_expanded)

            lstm_idx += 1
            proj_idx += 1

        decoder_out = self.decoder(*features)
        output_flat = self.head(self.dropout(decoder_out))
        if self.use_refiner:
            ref_input = torch.cat([output_flat, x_flat_orig], dim=1)
            output_flat = output_flat + self.refiner(ref_input)
        output_seq = output_flat.view(B, T, -1, H, W)
        return output_seq, None

