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
# New Model: ResNet18 Encoder + Temporal Bottleneck
# -----------------------------------------------------
class PretrainedTemporalUNet(nn.Module):
    def __init__(self, out_channels=1, lstm_layers=1, freeze_encoder=True):
        super().__init__()
        
        # 1. Create base U-Net based on ResNet18
        # weights="imagenet" -> Loads pre-trained knowledge
        # in_channels=2 -> The library automatically adapts the first layer!
        self.base_model = smp.Unet(
            encoder_name="resnet18",        
            encoder_weights="imagenet",    
            in_channels=2,                  
            classes=out_channels,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16) # Lightweight and fast decoder
        )
        
        # 2. Decompose the model into components
        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.head = self.base_model.segmentation_head
        
        # 3. Freeze the Encoder (saves memory and prevents Overfitting)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] ResNet Encoder is FROZEN.")

        # 4. Add ConvLSTM at the bottleneck
        # In ResNet18, the final output (stage 5) has 512 channels
        self.lstm_input_dim = 512
        self.lstm = ConvLSTM(
            input_dim=self.lstm_input_dim, 
            hidden_dim=self.lstm_input_dim, # Keep the same width
            num_layers=lstm_layers,
            kernel_size=3
        )
        # Add ConvLSTM modules for all encoder skip features (excluding bottleneck)
        # Try to obtain encoder channel list from the encoder object; fall back
        # to ResNet18 defaults if unavailable.
        encoder_out_channels = None
        if hasattr(self.encoder, 'out_channels'):
            encoder_out_channels = getattr(self.encoder, 'out_channels')
        elif hasattr(self.base_model, 'encoder_out_channels'):
            encoder_out_channels = getattr(self.base_model, 'encoder_out_channels')

        if encoder_out_channels is None:
            # Typical ResNet18 encoder channels: [64, 64, 128, 256, 512]
            encoder_out_channels = [64, 64, 128, 256, 512]

        # Create a ConvLSTM module for each skip feature (all except last bottleneck)
        skip_channels = encoder_out_channels[:-1]
        self.skip_channels = list(skip_channels)
        self.lstm_skips = nn.ModuleList([
            ConvLSTM(input_dim=ch, hidden_dim=ch, num_layers=lstm_layers, kernel_size=3)
            for ch in skip_channels
        ])

    def forward(self, x_seq):
        # x_seq shape: [B, T, 1, H, W]
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
        # Reshape back to time dimension so the LSTM understands the sequence
        # Shape: [B, T, 512, H/32, W/32]
        bottleneck_seq = bottleneck.view(B, T, -1, bottleneck.shape[2], bottleneck.shape[3])
        
        # Your ConvLSTM expects a list of tensors
        lstm_in_list = [bottleneck_seq[:, t] for t in range(T)]
        
        # Run the LSTM
        lstm_out_list, _ = self.lstm(lstm_in_list)
        
        # Stack back into a single tensor
        lstm_out_stacked = torch.stack(lstm_out_list, dim=1) # [B, T, 512, H/32, W/32]
        
        # --- C. DECODER (Frame by Frame) ---
        # Flatten again to [B*T, ...] to enter the Decoder
        lstm_out_flat = lstm_out_stacked.view(B * T, -1, bottleneck.shape[2], bottleneck.shape[3])
        
        # The Trick: Replace the last feature in the list (which was static) 
        # with the LSTM output (which is dynamic)
        features[-1] = lstm_out_flat

        # --- OPTIONAL: temporal processing for selected skip connections ---
        # Process all skip connections (features[0..-2]) with their ConvLSTMs
        # Assumes `self.lstm_skips` was constructed with matching order and channels.
        for i in range(len(self.lstm_skips)):
            feat = features[i]
            Ck = feat.shape[1]
            hk, wk = feat.shape[2], feat.shape[3]
            feat_seq = feat.view(B, T, Ck, hk, wk)
            lstm_in = [feat_seq[:, t] for t in range(T)]
            lstm_out_list, _ = self.lstm_skips[i](lstm_in)
            lstm_out_stacked = torch.stack(lstm_out_list, dim=1)
            features[i] = lstm_out_stacked.view(B * T, Ck, hk, wk)
        
        # Decode
        decoder_out = self.decoder(*features)
        
        # Final output layer
        output_flat = self.head(decoder_out)
        
        # Reshape back to original shape: [B, T, 1, H, W]
        output_seq = output_flat.view(B, T, -1, H, W)
        
        return output_seq, None # (None because there is no external state currently)