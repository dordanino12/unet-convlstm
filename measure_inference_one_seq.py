#!/usr/bin/env python3
"""Measure inference time for one sequence (frame-by-frame loop used in test.py).
"""
import time
import os
import sys
import torch
import numpy as np

# Ensure project root is on path
proj_root = os.path.dirname(os.path.abspath(__file__))
if proj_root not in sys.path:
    sys.path.append(proj_root)

from train.dataset import NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNetMitB1

# Config (match main/test settings)
SPLIT_PATH = os.path.join(proj_root, 'data', 'dataset_envelop_w_fix_leak_test_w.npz')
CHECKPOINT = os.path.join(proj_root, 'models', 'mit_b1_envelop_data_leakag_fix_best_bin_loss.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_INDEX_IN_SPLIT = 0  # which sequence in the split to time
WARMUP_ITERS = 2

print(f"Device: {DEVICE}")
print(f"Loading dataset from: {SPLIT_PATH}")

ds = NPZSequenceDataset(SPLIT_PATH, use_gt_envelope_as_input=False)
print(f"Dataset length: {len(ds)}")

# Get one sequence
x_seq, y_seq, mask = ds[SEQ_INDEX_IN_SPLIT]
# x_seq shape: (T, C, H, W)
T = x_seq.shape[0]
C = x_seq.shape[1]
print(f"Using sequence index {SEQ_INDEX_IN_SPLIT} with T={T}, C={C}")

# Build model
model = PretrainedTemporalUNetMitB1(
    out_channels=1,
    lstm_layers=1,
    freeze_encoder=True,
    in_channels=C,
    use_refiner=False
)

# Load checkpoint if available
if os.path.exists(CHECKPOINT):
    print(f"Loading checkpoint: {CHECKPOINT}")
    ck = torch.load(CHECKPOINT, map_location=DEVICE)
    # checkpoint may be dict
    if isinstance(ck, dict):
        state = ck.get('model_state', ck.get('model_state_dict', ck))
    else:
        state = ck
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"Warning: failed to load full state dict: {e}")
else:
    print("Warning: checkpoint not found, using freshly initialized model")

model.to(DEVICE)
model.eval()

# Warmup
print(f"Warming up {WARMUP_ITERS} iterations...")
with torch.no_grad():
    for _ in range(WARMUP_ITERS):
        for t in range(1, min(3, T) + 1):
            x_input = x_seq[:t].unsqueeze(0).to(DEVICE)
            out, _ = model(x_input)
            # small sync
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()

# Timed run: measure end-to-end time for processing full sequence (t=1..T)
print("Starting timed inference for one full sequence (frame-by-frame)...")
start = time.time()
with torch.no_grad():
    for t in range(1, T + 1):
        x_input = x_seq[:t].unsqueeze(0).to(DEVICE)
        out, _ = model(x_input)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
end = time.time()

total = end - start
per_frame = total / T if T > 0 else float('nan')
print(f"Total inference time (frames 1..{T}): {total:.4f} s")
print(f"Average per-frame time: {per_frame:.6f} s/frame")
print(f"Throughput: {1.0/total:.6f} sequences/s")

# Update todo: run done
