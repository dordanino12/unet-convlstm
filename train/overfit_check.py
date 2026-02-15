import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2

# ---------------------------------------------------------
# FIX IMPORT PATH
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import both model classes
from train.dataset import NPZSequenceDataset
from train.resnet18 import PretrainedTemporalUNet

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- CONFIGURATION ---
FREEZE_ENCODER = True  # True: Freeze encoder (faster), False: Train encoder (slower, more capacity)
USE_GT_ENVELOPE_INPUT = True  # Set True when model expects GT envelope channel
SAVE_OVERFIT_VIDEO = True
OVERFIT_VIDEO_FPS = 2
USE_MASK = "slice_mask"  # True, False, or "slice_mask"
UNMASKED_WEIGHT_FACTOR = 0.2
# ---------------------

npz_path = os.path.join(parent_dir, "data/dataset_trajectory_sequences_samples_W_500m_w.npz")
gt_envelope_npz_path = os.path.join(parent_dir, "data/dataset_trajectory_sequences_samples_W_top_w.npz")
if not os.path.exists(npz_path):
    npz_path = "/data/dataset_trajectory_sequences_samples_W_500m_w.npz"
if not os.path.exists(gt_envelope_npz_path):
    gt_envelope_npz_path = "/data/dataset_trajectory_sequences_samples_W_top_w.npz"

# ---------------------------------------------------------
# LOGIC
# ---------------------------------------------------------
def _to_color_frame(frame, vmin, vmax):
    if vmax <= vmin:
        vmax = vmin + 1e-6
    frame_norm = (frame - vmin) / (vmax - vmin)
    frame_norm = np.clip(frame_norm, 0.0, 1.0)
    frame_u8 = (frame_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(frame_u8, cv2.COLORMAP_JET)


def compute_loss(y_pred, y, mask=None, use_mask=True, unmasked_weight_factor=0.1):
    abs_diff = (y_pred - y).abs()
    weight = torch.exp(8.0 * y.abs())

    mask_broadcasted = None
    spatial_mask = None

    if use_mask == "slice_mask" and mask is not None:
        mask_slice_5 = mask[:, 5:6, :, :]
        mask_broadcasted = mask_slice_5.expand_as(mask)

        spatial_mask = torch.ones_like(mask_broadcasted)
        spatial_mask[mask_broadcasted > 0.5] = 1.0
        spatial_mask[mask_broadcasted <= 0.5] = unmasked_weight_factor

        combined_weight = weight * spatial_mask
        numerator = (abs_diff * combined_weight).sum()
        denominator = combined_weight.sum() + 1e-8
        weighted_l1 = numerator / denominator
    elif use_mask is True and mask is not None:
        numerator = (abs_diff * mask * weight).sum()
        denominator = (mask * weight).sum() + 1e-8
        weighted_l1 = numerator / denominator
    else:
        weighted_l1 = (abs_diff * weight).sum() / (weight.sum() + 1e-8)

    def spatial_gradients(tensor):
        dx = tensor[..., :, 1:] - tensor[..., :, :-1]
        dy = tensor[..., 1:, :] - tensor[..., :-1, :]
        return dx, dy

    dx_pred, dy_pred = spatial_gradients(y_pred)
    dx_gt, dy_gt = spatial_gradients(y)

    H_min = min(dx_pred.shape[3], dy_pred.shape[3])
    W_min = min(dx_pred.shape[4], dy_pred.shape[4])

    grad_diff = (dx_pred[..., :H_min, :W_min] - dx_gt[..., :H_min, :W_min]).abs() + \
                (dy_pred[..., :H_min, :W_min] - dy_gt[..., :H_min, :W_min]).abs()

    if use_mask == "slice_mask" and mask is not None:
        spatial_mask_c = spatial_mask[..., :H_min, :W_min]
        grad_loss = (grad_diff * spatial_mask_c).sum() / (spatial_mask_c.sum() + 1e-8)
    elif use_mask is True and mask is not None:
        mask_c = mask[..., :H_min, :W_min]
        grad_loss = (grad_diff * mask_c).sum() / (mask_c.sum() + 1e-8)
    else:
        grad_loss = grad_diff.mean()

    total_loss = weighted_l1 + 0.005 * grad_loss
    return total_loss


def save_overfit_video(model, x_batch, y_batch, dataset, sample_index, out_path, fps=2):
    model.eval()
    with torch.no_grad():
        output, _ = model(x_batch)
        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output

    y_denorm = dataset.denormalize(y_batch).cpu().numpy()
    pred_denorm = dataset.denormalize(y_pred).cpu().numpy()

    gt_seq = y_denorm[sample_index, :, 0]
    pred_seq = pred_denorm[sample_index, :, 0]

    vmin = -dataset.max_neg_val
    vmax = dataset.max_pos_val

    t_len, h, w = gt_seq.shape
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

    for t in range(t_len):
        gt_color = _to_color_frame(gt_seq[t], vmin, vmax)
        pred_color = _to_color_frame(pred_seq[t], vmin, vmax)
        frame = np.concatenate([gt_color, pred_color], axis=1)

        cv2.putText(frame, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Pred", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        video_writer.write(frame)

    video_writer.release()
    print(f"[INFO] Saved overfit video to: {out_path}")

def run_overfit_test_and_save():
    # 1. Load Full Dataset
    dataset = NPZSequenceDataset(
        npz_path,
        use_gt_envelope_as_input=USE_GT_ENVELOPE_INPUT,
        gt_envelope_npz_path=gt_envelope_npz_path
    )
    total_samples = len(dataset)
    
    # 2. Select specific indices manually
    num_samples = 1
    selected_indices = np.random.choice(total_samples, num_samples, replace=False)
    print(f"\n[INFO] Selected Sequence Indices for Training: {selected_indices}")
    
    # Create a subset with only these indices
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
    
    # Get the single batch of data
    batch = next(iter(loader)) 
    x, y, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)

    print(f"Batch Shapes -> X: {x.shape}, Y: {y.shape}")
    in_channels = x.shape[2]

    # 3. Initialize Model
    print("[INFO] Initializing Pre-trained ResNet18 Model...")
    encoder_state = "frozen" if FREEZE_ENCODER else "trainable"
    print(f"[INFO] Encoder is {'FROZEN' if FREEZE_ENCODER else 'TRAINABLE'}")

    model = PretrainedTemporalUNet(
        out_channels=1,
        lstm_layers=1,
        freeze_encoder=FREEZE_ENCODER,
        in_channels=in_channels
    ).to(device)

    # Optimize only trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    model_type_str = f'resnet18_{encoder_state}'
    save_cfg = {'type': 'resnet18', 'freeze_encoder': FREEZE_ENCODER, 'lstm_layers': 1}

    print(f"\n--- Starting Overfit Test ({model_type_str}) ---")
    
    # Run for many iterations to force memorization
    for i in range(3001): 
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(x)
        
        # Compatibility handling:
        # Custom model returns a LIST of tensors -> needs stack
        # ResNet model returns a TENSOR -> no stack needed
        if isinstance(output, list):
            y_pred = torch.stack(output, dim=1)
        else:
            y_pred = output

        # Regular loss (weighted L1 + gradient loss) with mask
        loss = compute_loss(
            y_pred,
            y,
            mask=mask,
            use_mask=USE_MASK,
            unmasked_weight_factor=UNMASKED_WEIGHT_FACTOR
        )
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Iter {i:04d} | Loss: {loss.item():.6f}")

            # If converged, save and exit
            if loss.item() < 0.035:
                print("\n[SUCCESS] Loss is near zero!")
                
                # --- SAVE LOGIC ---
                filename = f"overfitted_{model_type_str}_100_model.pt"
                save_path = os.path.join(current_dir, filename)
                
                print(f"[INFO] Saving model to: {save_path}")
                print(f"[INFO] This model is overfitted on indices: {selected_indices}")
                
                torch.save({
                    'model_state': model.state_dict(),
                    'config': save_cfg,
                    'indices': selected_indices
                }, save_path)

                if SAVE_OVERFIT_VIDEO:
                    video_dir = os.path.join(parent_dir, "plots", "overfit_videos")
                    video_name = f"overfit_sample_{selected_indices[0]}.mp4"
                    video_path = os.path.join(video_dir, video_name)
                    save_overfit_video(model, x, y, dataset, 0, video_path, fps=OVERFIT_VIDEO_FPS)
                
                return

    print("\n[WARNING] Did not reach perfect convergence, but saving anyway.")
    torch.save({
        'model_state': model.state_dict(),
        'config': save_cfg,
        'train_indices': selected_indices
    }, os.path.join(current_dir, f"overfitted_{model_type_str}_failed.pt"))

    if SAVE_OVERFIT_VIDEO:
        video_dir = os.path.join(parent_dir, "plots", "overfit_videos")
        video_name = f"overfit_sample_{selected_indices[0]}_failed.mp4"
        video_path = os.path.join(video_dir, video_name)
        save_overfit_video(model, x, y, dataset, 0, video_path, fps=OVERFIT_VIDEO_FPS)

if __name__ == "__main__":
    run_overfit_test_and_save()