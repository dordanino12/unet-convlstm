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

from train.unet import TemporalUNetDualView, NPZSequenceDataset, ConvLSTM

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
        
        # Decode
        decoder_out = self.decoder(*features)
        
        # Final output layer
        output_flat = self.head(decoder_out)
        
        # Reshape back to original shape: [B, T, 1, H, W]
        output_seq = output_flat.view(B, T, -1, H, W)
        
        return output_seq, None # (None because there is no external state currently)