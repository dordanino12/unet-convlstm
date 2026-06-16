import os
import sys
import pickle
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2

# Add parent directory to path to import train module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
from train.resnet18 import (
    PretrainedTemporalUNetMitB1, 
    PretrainedTemporalUNetMitB2,
    PretrainedTemporalUNetMitB3,
    PretrainedTemporalUNet
)
from train.dataset import NPZSequenceDataset  

# --- CONFIG ---
pkl_dir = "/wdata_visl/vhold/backup_from_133/CEIL/CELINE_2021/satellite_output_steps_no_sea_low_angular_res_256pixelres_10000-10220/"
gt_base_dir = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,reff,lwc,U,V,W)_fixed_to_shdom/10000_10220_top_vel/'

USE_KFOLD = True
KFOLD_MODELS_DIR = os.path.join(project_root, 'models/wacv/envelop')
KFOLD_DATA_DIR = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/envelop_kfold_w_sensor_noise_both/"

# pkl_dir = '/wdata_visl/vhold/backup_from_133/CEIL/CELINE_2021/satellite_output_steps_no_sea_low_angular_res_256pixelres_5920_6140/'
# gt_base_dir = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,reff,lwc,U,V,W)_fixed_to_shdom/top_vel/'

# pkl_dir = "/wdata_visl/vhold/backup_from_133/CEIL/CELINE_2021/satellite_output_steps_18000-18220_low_angular_res/"
# gt_base_dir = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,reff,lwc,U,V,W)_fixed_to_shdom/18000_18220_top_vel/'

model_path = os.path.join(project_root, 'models/data_fix/mit_b1_1000m_leakag_fix_noised_best_bin_loss.pt')
output_dir = script_dir  # Output video to script directory

# --- PDF Configuration ---
SAVE_PDF_SECTIONS = True
PDF_BASE_DIR = os.path.join(output_dir, 'frames_pdf')
PDF_FIG_SIZE = (12, 12)
PDF_DPI = 150
PDF_AX_POS = [0.17, 0.10, 0.70, 0.75]
PDF_SUBPLOT_ADJUST = dict(left=0.17, right=0.88, top=0.95, bottom=0.08)
PDF_CBAR_PAD = 0.015
PDF_CBAR_WIDTH = 0.035
PDF_CBAR_HEIGHT = 0.80

def _pick_existing(base_dir, names):
    for name in names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    return None

def detect_model_type_from_path(path):
    """Auto-detect model type from file path."""
    path_lower = path.lower()
    if 'resnet' in path_lower:
        return 'resnet18'
    if 'mit_b3' in path_lower or 'b3' in path_lower:
        return 'mit_b3'
    if 'mit_b2' in path_lower or 'b2' in path_lower:
        return 'mit_b2'
    if 'mit_b1' in path_lower or 'b1' in path_lower:
        return 'mit_b1'
    return 'mit_b1'  # Fallback to mit_b1


def create_model(model_type, in_channels=2, out_channels=1):
    """Factory function to create the appropriate model."""
    model_type = model_type.lower()
    
    if model_type == 'resnet18':
        print(f'[INFO] Creating ResNet18-based model...')
        model = PretrainedTemporalUNet(
            out_channels=out_channels,
            lstm_layers=1,
            freeze_encoder=True,
            in_channels=in_channels,
            dropout_p=0.3,
            use_conv_lstm=True
        )
    elif model_type == 'mit_b3':
        print(f'[INFO] Creating MiT-B3-based model...')
        model = PretrainedTemporalUNetMitB3(
            in_channels=in_channels,
            out_channels=out_channels,
            lstm_layers=1,
            freeze_encoder=True,
            dropout_p=0.3
        )
    elif model_type == 'mit_b2':
        print(f'[INFO] Creating MiT-B2-based model...')
        model = PretrainedTemporalUNetMitB2(
            in_channels=in_channels,
            out_channels=out_channels,
            lstm_layers=1,
            freeze_encoder=True,
            dropout_p=0.2
        )
    elif model_type == 'mit_b1':
        print(f'[INFO] Creating MiT-B1-based model...')
        model = PretrainedTemporalUNetMitB1(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f'Unknown model type: {model_type}. Choose from: resnet18, mit_b1, mit_b2, mit_b3')
    
    return model

def get_reference_model_path():
    """Returns the relevant path to check for keywords (either single model or first fold)"""
    if USE_KFOLD:
        # Get the first model in the kfold dir to determine type/scale
        fold_files = [f for f in os.listdir(KFOLD_MODELS_DIR) if f.endswith('.pt')]
        if fold_files:
            return os.path.join(KFOLD_MODELS_DIR, fold_files[0])
    return model_path

def determine_movie_level_from_model_path():
    mp = get_reference_model_path().lower()
    if '500m' in mp:
        return '500m'
    if '1000m' in mp:
        return '1000m'
    if '1500m' in mp:
        return '1500m'
    if 'envelop' in mp or 'envelope' in mp:
        return 'top'
    return 'top'


movie_level = determine_movie_level_from_model_path()
output_video = os.path.join(output_dir, f'vadim_predicted_{movie_level}_velocity.mp4')
print (f'[INFO] Output video will be saved to: {output_video}')
temp_sequence_npy = os.path.join(output_dir, 'vadim_sequence.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- Options ---
# If True, apply dataset mask to predictions/GT when visualizing; if False, show full maps.
def determine_use_mask_from_model_path():
    mp = get_reference_model_path().lower()
    return ('envelop' in mp) or ('envelope' in mp) or ('top' in mp)

USE_MASK = determine_use_mask_from_model_path()
print(f'[INFO] USE_MASK={USE_MASK} (from model_path)')

MASK_THRESHOLD = 1.1 if 'mitsuba' in pkl_dir.lower() else 0.01
if 'mitsuba' in pkl_dir.lower():
    print(f'[INFO] Mitsuba input detected, using mask threshold={MASK_THRESHOLD}')

# --- Load PKL images as sequence with 2 channels per timestep ---
pkl_files = []
for dirpath, _, filenames in os.walk(pkl_dir):
    for fname in filenames:
        if fname.endswith('.pkl'):
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, pkl_dir)
            pkl_files.append(rel_path)
pkl_files = sorted(pkl_files)

if not pkl_files:
    raise RuntimeError(f'No PKL files found in {pkl_dir} (including subfolders)')

print(f'[INFO] Found {len(pkl_files)} pkl files. Checking format...')

# Debug: print first few files and their structure
for i, pkl_file in enumerate(pkl_files[:3]):
    with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
        data = pickle.load(f)
    print(f'[DEBUG] {pkl_file}: type={type(data)}, keys={list(data.keys()) if isinstance(data, dict) else "N/A"}')

sequence = []
seq_files = []  # filenames corresponding to frames in `sequence`

# Detect file format (old format: sat1_image/sat2_image, new format: render)
is_new_format = None

# For Mitsuba format, accumulate one frame per (step/time/sample) with per-view channels.
mitsuba_groups = {}


def render_to_single_channel(img):
    """Convert Mitsuba render output to a single 2D channel."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Common case: RGB/RGBA image -> grayscale channel for this satellite view.
        if arr.shape[-1] >= 3:
            return np.mean(arr[..., :3], axis=-1)
        return arr[..., 0]
    return np.squeeze(arr)

for pkl_file in pkl_files:
    with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        # Try new format (Mitsuba render output): single 'render' key
        if 'render' in data:
            if is_new_format is None:
                is_new_format = True
                print(f'[INFO] Detected new format (Mitsuba render output)')

            base_name = os.path.basename(pkl_file)
            sample_match = re.search(r'sample_(\d+)_', base_name)
            time_match = re.search(r'_time_(\d+)_', base_name)
            view_match = re.search(r'_view_(\d+)', base_name)

            if view_match is None:
                print(f'[WARNING] Mitsuba file missing view index, skipping: {pkl_file}')
                continue

            sample_num = int(sample_match.group(1)) if sample_match else 40
            time_idx = int(time_match.group(1)) if time_match else -1
            view_idx = int(view_match.group(1))

            parent_dir = os.path.basename(os.path.dirname(pkl_file))
            step_num = int(parent_dir) if parent_dir.isdigit() else -1

            frame_key = (step_num, time_idx, sample_num)
            if frame_key not in mitsuba_groups:
                mitsuba_groups[frame_key] = {
                    'views': {},
                    'view_files': {}
                }

            mitsuba_groups[frame_key]['views'][view_idx] = render_to_single_channel(data['render'])
            mitsuba_groups[frame_key]['view_files'][view_idx] = pkl_file
        # Try old format: sat1_image and sat2_image keys
        elif 'sat1_image' in data and 'sat2_image' in data:
            if is_new_format is None:
                is_new_format = False
                print(f'[INFO] Detected old format (satellite output)')
            img = np.stack([data['sat1_image'], data['sat2_image']], axis=0)  # (2, H, W)
            sequence.append(img)
            seq_files.append(pkl_file)
        else:
            print(f'PKL file {pkl_file} has unrecognized format. Keys: {list(data.keys())}')
    else:
        print(f'PKL file {pkl_file} is not a dict: {type(data)}')

# Finalize Mitsuba grouped frames: require at least view_0 and view_1.
if is_new_format and len(mitsuba_groups) > 0:
    sequence = []
    seq_files = []
    missing_pairs = 0

    for frame_key in sorted(mitsuba_groups.keys()):
        group = mitsuba_groups[frame_key]
        views = group['views']
        if 0 in views and 1 in views:
            img = np.stack([views[0], views[1]], axis=0)  # (2, H, W)
            sequence.append(img)
            seq_files.append(group['view_files'][0])  # representative file for GT parsing
        else:
            missing_pairs += 1

    if missing_pairs > 0:
        print(f'[WARNING] Skipped {missing_pairs} Mitsuba frames missing view_0/view_1 pairs.')

if len(sequence) == 0:
    raise RuntimeError('No valid PKL files found. Expected either "render" key (new format) or "sat1_image"/"sat2_image" keys (old format).')

format_name = 'new (Mitsuba render)' if is_new_format else 'old (satellite output)'
print(f'[INFO] Loaded {len(sequence)} valid PKL files ({format_name})')

sequence = np.stack(sequence, axis=0)  # (T, 2, H, W)
np.save(temp_sequence_npy, sequence)

# --- Load GT top velocity maps ---
# Direct mapping from step file -> GT top velocity file.
# Examples provided by user:
#   step_0000.pkl -> .../top_vel/0000000000/sample_012_time_0_view_0_first_hit.pkl
#   step_0001.pkl -> .../top_vel/0000000001/sample_012_time_20_view_0_first_hit.pkl

def step_to_time(step_num):
    return step_num * 20


def parse_step_time_sample(seq_file):
    """Parse step/time/sample from either old or Mitsuba-style filenames."""
    base_name = os.path.basename(seq_file)
    parent_dir = os.path.basename(os.path.dirname(seq_file))

    step_num = None
    time_idx = None
    sample_num = 40

    # Old format: step_0001.pkl
    if base_name.startswith('step_') and base_name.endswith('.pkl'):
        raw = base_name.replace('step_', '').replace('.pkl', '')
        if raw.isdigit():
            step_num = int(raw)

    # New format: step from parent directory (e.g. .../0000002000/sample_...)
    if step_num is None and parent_dir.isdigit():
        step_num = int(parent_dir)

    # Sample from filename if present (e.g. sample_040_...)
    sample_match = re.search(r'sample_(\d+)_', base_name)
    if sample_match:
        sample_num = int(sample_match.group(1))

    # Time from filename if present (e.g. _time_7028_)
    time_match = re.search(r'_time_(\d+)_', base_name)
    if time_match:
        time_idx = int(time_match.group(1))
    elif step_num is not None:
        time_idx = step_to_time(step_num)

    return step_num, time_idx, sample_num


def determine_gt_suffix_from_model_path():
    """Pick GT filename suffix based on model path keywords."""
    mp = get_reference_model_path().lower()
    if 'envelop' in mp or 'envelope' in mp:
        return 'first_hit'
    if '500m' in mp:
        return 'slice_500m'
    if '1000m' in mp:
        return 'slice_1000m'
    if '1500m' in mp:
        return 'slice_1500m'
    return 'first_hit'


gt_suffix = determine_gt_suffix_from_model_path()
print(f'[INFO] Using GT suffix={gt_suffix} (from model_path)')


def resize_gt_map(gt_img, target_shape):
    gt_img = np.asarray(gt_img)
    if gt_img.shape == target_shape:
        return gt_img
    return cv2.resize(gt_img.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def to_2d_gt_map(gt_img, target_shape):
    """Convert GT map to 2D for visualization.

    Handles 2D and common 3D layouts such as [num_heights, H, W], [H, W, num_heights],
    or volumetric-like tensors where one axis should be sliced.
    """
    arr = np.asarray(gt_img)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return resize_gt_map(arr, target_shape)

    if arr.ndim == 3:
        h_t, w_t = target_shape

        # Common case: stacked heights at axis 0 or axis -1
        if arr.shape[0] <= 16:
            arr2d = arr[0]
        elif arr.shape[-1] <= 16:
            arr2d = arr[..., 0]
        # If two axes already match target H,W, slice the remaining axis at center
        elif arr.shape[0] == h_t and arr.shape[1] == w_t:
            arr2d = arr[:, :, arr.shape[2] // 2]
        elif arr.shape[1] == h_t and arr.shape[2] == w_t:
            arr2d = arr[arr.shape[0] // 2, :, :]
        elif arr.shape[0] == h_t and arr.shape[2] == w_t:
            arr2d = arr[:, arr.shape[1] // 2, :]
        else:
            # Fallback: collapse along smallest axis
            axis = int(np.argmin(arr.shape))
            arr2d = np.mean(arr, axis=axis)

        return resize_gt_map(arr2d, target_shape)

    raise ValueError(f'Unsupported GT shape for visualization: {arr.shape}')

gt_sequence = []
for pkl_file in seq_files:
    try:
        step_num, time_idx, sample_num = parse_step_time_sample(pkl_file)

        if step_num is None or time_idx is None:
            gt_sequence.append(None)
            print(f'Could not infer step/time for GT from: {pkl_file}')
            continue

        gt_path = os.path.join(
            gt_base_dir,
            f'{step_num:010d}',
            f'sample_{sample_num:03d}_time_{time_idx}_view_0_{gt_suffix}.pkl'
        )

        if not os.path.exists(gt_path):
            gt_sequence.append(None)
            print(f'GT path not found: {gt_path}')
            continue

        with open(gt_path, 'rb') as f:
            gt_data = pickle.load(f)

        if isinstance(gt_data, dict):
            if 'top_vel' in gt_data:
                gt_sequence.append(gt_data['top_vel'])
                print(f'Loaded GT for {pkl_file}: {gt_path}')
            elif 'w_map' in gt_data:
                gt_sequence.append(gt_data['w_map'])
                print(f'Loaded GT w_map for {pkl_file}: {gt_path}')
            else:
                gt_sequence.append(None)
                print(f'GT missing top_vel/w_map in {gt_path}. Keys: {list(gt_data.keys())}')
        else:
            gt_sequence.append(None)
            print(f'GT file is not a dict in {gt_path}: {type(gt_data)}')

    except Exception as e:
        gt_sequence.append(None)
        print(f'Could not load GT for {pkl_file}: {e}')

# --- Normalize input sequence using norm_const (max or 1.0) ---
sequence = sequence.astype(np.float32)
is_mitsuba_input = 'mitsuba' in pkl_dir.lower()
if is_mitsuba_input:
    print('[INFO] Mitsuba input detected from pkl_dir, skipping SHDOM->Mitsuba scaling.')
    sequence_shdom_to_mitsuba = sequence
else:
    sequence_shdom_to_mitsuba = sequence * 131.4 * np.cos(np.deg2rad(35))
norm_const = 31.22 # this the norm that we do in the train
sequence_norm = sequence_shdom_to_mitsuba / norm_const

# --- Prepare input for model ---
# Model expects (B, T, C, H, W), here C=2
input_seq = torch.from_numpy(sequence_norm).float().unsqueeze(0).to(device)  # (1, T, 2, H, W)

# --- Load model ---
# Auto-detect model type from model path
detected_type = detect_model_type_from_path(model_path)
print(f'[INFO] Model path: {model_path}')
print(f'[INFO] Detected model type: {detected_type}')

model = create_model(detected_type, in_channels=2, out_channels=1)
state = torch.load(model_path, map_location=device)
if 'model_state' in state:
    model.load_state_dict(state['model_state'], strict=False)
else:
    model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()

# --- Load models & Datasets ---
models = []
fold_train_datasets = []  # רשימה לשמירת ה-Dataset של כל Fold (עבור De-normalization)
is_kfold_mode = USE_KFOLD and (KFOLD_MODELS_DIR is not None) and (KFOLD_DATA_DIR is not None)

print("\n" + "="*50)
if is_kfold_mode:
    print(f'[INFO] K-FOLD MODE ENABLED: searching {KFOLD_DATA_DIR} and {KFOLD_MODELS_DIR}')
    fold_dirs = sorted(d for d in os.listdir(KFOLD_DATA_DIR) if d.startswith("fold_") and os.path.isdir(os.path.join(KFOLD_DATA_DIR, d)))
    
    for fold_name in fold_dirs:
        fold_data_dir = os.path.join(KFOLD_DATA_DIR, fold_name)
        fold_train_path = _pick_existing(fold_data_dir, ["train_w.npz", "train.npz", "train_uvw.npz"])
        
        if fold_train_path is None:
            print(f"[WARN] {fold_name}: no train npz found, skipping")
            continue
            
        expected_suffix = f"_{fold_name}_best_bin_loss.pt"
        fold_ckpt = None
        for f in sorted(os.listdir(KFOLD_MODELS_DIR)):
            if f.endswith(expected_suffix):
                fold_ckpt = os.path.join(KFOLD_MODELS_DIR, f)
                break
                
        if fold_ckpt is None:
            print(f"[WARN] {fold_name}: no matching checkpoint in {KFOLD_MODELS_DIR}, skipping")
            continue

        print(f"[INFO] Loading fold {fold_name} model {fold_ckpt}")
        
        # 1. טעינת ה-Dataset של ה-Fold לצורך חילוץ הסטטיסטיקות (Scale / Norm)
        fold_train_dataset = NPZSequenceDataset(
            fold_train_path,
            use_gt_envelope_as_input=False,
            gt_envelope_npz_path=fold_train_path,
            use_one_satellite=False
        )
        fold_train_datasets.append(fold_train_dataset)

        # 2. טעינת המודל
        detected_type = detect_model_type_from_path(fold_ckpt)
        mdl = create_model(detected_type, in_channels=2, out_channels=1)
        state = torch.load(fold_ckpt, map_location=device)
        
        if 'model_state' in state:
            mdl.load_state_dict(state['model_state'], strict=False)
        else:
            mdl.load_state_dict(state, strict=False)
            
        mdl = mdl.to(device)
        mdl.eval()
        models.append(mdl)
        
    print(f"[INFO] Loaded {len(models)} models for ensemble")
else:
    print(f'[INFO] SINGLE MODEL MODE: {model_path}')
    detected_type = detect_model_type_from_path(model_path)
    print(f'[INFO] Detected model type: {detected_type}')
    
    mdl = create_model(detected_type, in_channels=2, out_channels=1)
    state = torch.load(model_path, map_location=device)
    if 'model_state' in state:
        mdl.load_state_dict(state['model_state'], strict=False)
    else:
        mdl.load_state_dict(state, strict=False)
    mdl = mdl.to(device)
    mdl.eval()
    models.append(mdl)

print("="*50 + "\n")

# --- Run inference & Denormalize per fold ---
print(f'[INFO] Running inference with {len(models)} model(s)...')
all_preds = []

with torch.no_grad():
    if is_kfold_mode and len(fold_train_datasets) == len(models):
        # במצב K-Fold: הרצה של כל מודל וביצוע De-normalization עם הסטטיסטיקות הספציפיות שלו
        for i, (mdl, fold_train) in enumerate(zip(models, fold_train_datasets)):
            pred_tensor, _ = mdl(input_seq)
            pred_np = pred_tensor.squeeze(0).squeeze(1).cpu().numpy()  
            
            # דה-נורמליזציה ייחודית ל-Fold!
            pred_np_denorm = fold_train.denormalize(pred_np)
            all_preds.append(pred_np_denorm)
    else:
        # מצב מודל בודד: משתמשים במילון הישן כברירת מחדל אם אין Datasets זמינים
        for i, mdl in enumerate(models):
            pred_tensor, _ = mdl(input_seq)
            pred_np = pred_tensor.squeeze(0).squeeze(1).cpu().numpy()  
            
            scale_map = {'envelop': 10.55, '500m': 3.7231, '1000m': 7.0, '1500m': 11.8982}
            mp = get_reference_model_path().lower()
            scale = scale_map['1000m'] # default
            for key in scale_map.keys():
                if key in mp:
                    scale = scale_map[key]
                    break
                    
            pred_np_denorm = pred_np * scale
            all_preds.append(pred_np_denorm)

if len(all_preds) == 0:
    raise RuntimeError('No models were available for inference.')

# --- Ensemble (Average) ---
if len(all_preds) == 1:
    pred_seq = all_preds[0]
else:
    print(f'[INFO] Ensembling (Mean) predictions from {len(all_preds)} models AFTER denormalization')
    pred_seq = np.mean(np.stack(all_preds, axis=0), axis=0)

# # --- Denormalize predicted velocity ---
# scale_map = {
#     'envelop': 10.55,
#     '500m': 3.7231,
#     '1000m': 8.5,
#     '1500m': 11.8982,
# }

# def determine_scale():
#     mp = get_reference_model_path().lower()
#     for key in ('envelop', 'envelope'):
#         if key in mp:
#             return scale_map['envelop'], f'reference path contains "{key}"'
#     for key in ('500m', '1000m', '1500m'):
#         if key in mp:
#             return scale_map[key], f'found "{key}" in reference path'
#     return scale_map['1000m'], 'fallback to 1000m (no keyword in reference path)'

# scale, scale_reason = determine_scale()
# print(f'[INFO] Using scale={scale} ({scale_reason})')
# pred_seq = pred_seq * scale
# --- Fixed colorbar axes for all frames ---
vmin_pred = -6
vmax_pred = 6

# --- Scatter Plot Configuration (same as get_metrics.py) ---
SCATTER_BIN_WIDTH = 0.02
POINTS_PER_BIN = 7
SCATTER_RANGE = (-8.5, 8.5)

# Collect all predictions and GT for scatter plot
scatter_pred_list = []
scatter_gt_list = []

# --- Gamma correction helper ---
def apply_gamma(img, gamma=0.5):
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-6:
        return img
    img_norm = (img - img_min) / (img_max - img_min)
    img_corrected = np.power(img_norm, gamma)
    return img_corrected


def apply_pdf_layout(fig, ax):
    """Apply consistent PDF layout settings."""
    fig.set_size_inches(*PDF_FIG_SIZE)
    fig.subplots_adjust(**PDF_SUBPLOT_ADJUST)
    ax.set_position(PDF_AX_POS)


def set_centered_meter_axis(ax, H, W, m_per_pixel=20):
    """Set up coordinate axes in meters with center origin."""
    half_w_m = (W * m_per_pixel) / 2.0
    half_h_m = (H * m_per_pixel) / 2.0

    tick_vals = np.linspace(-half_w_m, half_w_m, 5)
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_xticklabels([f"{int(v)}" for v in tick_vals], fontsize=48, fontweight='bold')
    ax.set_yticklabels([f"{int(v)}" for v in tick_vals], fontsize=48, fontweight='bold')

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=14,
        width=4,
        color='black',
        labelsize=48
    )

    ax.set_xlabel('X [m]', fontsize=52, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=52, fontweight='bold')


def save_section_pdf(img_data, title, out_path, cmap='gray', norm_obj=None, add_colorbar=False,
                     m_per_pixel=20, extent_m=None, vmin=None, vmax=None, tick_step=None):
    """Save a single image section as PDF with consistent formatting."""
    fig, ax = plt.subplots(figsize=PDF_FIG_SIZE, dpi=PDF_DPI)

    if extent_m is None:
        H, W = img_data.shape[:2]
        half_w_m = (W * m_per_pixel) / 2.0
        half_h_m = (H * m_per_pixel) / 2.0
        extent_m = [-half_w_m, half_w_m, half_h_m, -half_h_m]

    if norm_obj is not None:
        im = ax.imshow(img_data, cmap=cmap, norm=norm_obj, extent=extent_m, interpolation='nearest')
    else:
        im = ax.imshow(img_data, cmap=cmap, extent=extent_m, interpolation='nearest',
                       vmin=vmin, vmax=vmax)
    ax.set_aspect('auto')
    ax.set_title(title, fontsize=56, fontweight='bold', pad=40)

    H, W = img_data.shape[:2]
    set_centered_meter_axis(ax, H, W, m_per_pixel=m_per_pixel)

    if add_colorbar:
        # Place colorbar in fixed axes so main image size never changes
        cbar_h = PDF_AX_POS[3]
        cbar_y = PDF_AX_POS[1]
        cbar_x = PDF_AX_POS[0] + PDF_AX_POS[2] + PDF_CBAR_PAD
        cax = fig.add_axes([cbar_x, cbar_y, PDF_CBAR_WIDTH, cbar_h])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=48)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if tick_step is not None and vmin is not None and vmax is not None and tick_step > 0:
            ticks = np.arange(vmin, vmax + tick_step, tick_step)
            if vmin < 0 < vmax and 0.0 not in ticks:
                ticks = np.sort(np.append(ticks, 0.0))
            cbar.set_ticks(ticks)

    # Apply fixed layout (after colorbar so it doesn't interfere)
    apply_pdf_layout(fig, ax)
    plt.savefig(out_path, dpi=PDF_DPI)
    plt.close(fig)


def sample_scatter_points(gt_vals, pred_vals, label_suffix):
    """Balanced sampling for scatter plot using bins."""
    # Safety check: ensure no NaN/Inf in input
    valid_mask = np.isfinite(gt_vals) & np.isfinite(pred_vals)
    gt_vals = gt_vals[valid_mask]
    pred_vals = pred_vals[valid_mask]
    
    if len(gt_vals) == 0:
        print(f"[WARNING] No valid points after filtering{label_suffix}.")
        return None, None, None, None, None
    
    bins = np.arange(SCATTER_RANGE[0], SCATTER_RANGE[1] + SCATTER_BIN_WIDTH, SCATTER_BIN_WIDTH)
    bin_indices = np.digitize(gt_vals, bins)
    selected_indices = []
    unique_bins = np.unique(bin_indices)

    for b_idx in unique_bins:
        points_in_bin = np.where(bin_indices == b_idx)[0]
        n_sample = min(len(points_in_bin), POINTS_PER_BIN)
        if n_sample > 0:
            chosen = np.random.choice(points_in_bin, size=n_sample, replace=False)
            selected_indices.append(chosen)

    if len(selected_indices) > 0:
        final_indices = np.concatenate(selected_indices)
        np.random.shuffle(final_indices)
        x_scatter = gt_vals[final_indices]
        y_scatter = pred_vals[final_indices]
        print(f"[INFO] Selected {len(x_scatter)} points for scatter plot{label_suffix}.")
    else:
        print(f"[WARNING] Sampling failed{label_suffix}, using all points.")
        x_scatter = gt_vals
        y_scatter = pred_vals

    scatter_min = gt_vals.min()
    scatter_max = gt_vals.max()
    scatter_range_data = max(abs(scatter_min), abs(scatter_max))
    scatter_range_padded = scatter_range_data * 1.1
    
    # Ensure scatter_range_padded is valid
    if not np.isfinite(scatter_range_padded):
        print(f"[WARNING] Computed scatter_range_padded is not finite, using default 10.0{label_suffix}.")
        scatter_range_padded = 10.0
    
    return x_scatter, y_scatter, scatter_min, scatter_max, scatter_range_padded

# --- Create video with matplotlib 2x3 grid ---
fps = 2  # Define frames per second for video
video_writer = None
for t in range(sequence.shape[0]):
    input_img1 = sequence[t, 0]
    input_img2 = sequence[t, 1]
    pred_img = pred_seq[t]
    gt_img = gt_sequence[t] if t < len(gt_sequence) and gt_sequence[t] is not None else None

    # Mask logic (as in dataset) - optional
    if USE_MASK:
        mask = (input_img1 > MASK_THRESHOLD).astype(np.uint8)
    else:
        mask = np.ones_like(input_img1, dtype=np.uint8)
    # Gamma correction for display
    sat1_disp = apply_gamma(input_img1)
    sat2_disp = apply_gamma(input_img2)
    if USE_MASK:
        pred_disp = np.ma.masked_where(mask == 0, pred_img)
    else:
        pred_disp = pred_img
    mask_disp = mask if USE_MASK else np.ones_like(mask)

    # Build figure: Row1: Sat1, Sat2, Mask  Row2: Pred Vel, GT Vel, Diff
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Sat1, Sat2, Mask
    im0 = axs[0, 0].imshow(sat1_disp, cmap='gray', vmin=0, vmax=1)
    axs[0, 0].set_title('Sat1')
    im1 = axs[0, 1].imshow(sat2_disp, cmap='gray', vmin=0, vmax=1)
    axs[0, 1].set_title('Sat2')
    im_mask = axs[0, 2].imshow(mask_disp, cmap='gray', vmin=0, vmax=1)
    axs[0, 2].set_title('Mask')

    # Row 2: Pred Vel, GT Vel, Diff
    im2 = axs[1, 0].imshow(pred_disp, cmap='jet', vmin=vmin_pred, vmax=vmax_pred)
    axs[1, 0].set_title('Pred Top Vel')

    im3 = None
    im4 = None
    gt_disp = None
    diff_disp = None
    if gt_img is not None:
        gt_img = to_2d_gt_map(gt_img, pred_img.shape)
        if USE_MASK:
            gt_disp = np.ma.masked_where(mask == 0, gt_img)
        else:
            gt_disp = gt_img
        im3 = axs[1, 1].imshow(gt_disp, cmap='jet', vmin=vmin_pred, vmax=vmax_pred)
        axs[1, 1].set_title('GT Top Vel')

        # Difference
        diff_img = pred_img - gt_img
        if USE_MASK:
            diff_disp = np.ma.masked_where(mask == 0, diff_img)
        else:
            diff_disp = diff_img
        
        # Collect for scatter plot (only masked/valid regions)
        if USE_MASK:
            valid_mask = (mask == 1)
            if np.any(valid_mask):
                scatter_pred_list.append(pred_img[valid_mask])
                scatter_gt_list.append(gt_img[valid_mask])
        else:
            scatter_pred_list.append(pred_img.flatten())
            scatter_gt_list.append(gt_img.flatten())
        
        im4 = axs[1, 2].imshow(diff_disp, cmap='RdBu_r', vmin=-3.0, vmax=3.0)
        axs[1, 2].set_title('Pred - GT')
    else:
        axs[1, 1].text(0.5, 0.5, 'GT Not Available', ha='center', va='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('GT Top Vel')
        axs[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axs[1, 2].transAxes)
        axs[1, 2].set_title('Pred - GT')

    # Colorbars
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    plt.colorbar(im_mask, ax=axs[0, 2], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    if gt_img is not None:
        plt.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im4, ax=axs[1, 2], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if video_writer is None:
        h_pad, w_pad = frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w_pad, h_pad))

    video_writer.write(frame_bgr)
    # ---------------------------

    plt.close(fig)

    # Save per-section PDFs for this frame
    if SAVE_PDF_SECTIONS:
        frame_dir = os.path.join(PDF_BASE_DIR, f"frame_{t:04d}")
        os.makedirs(frame_dir, exist_ok=True)

        # Common extent in meters
        H_img, W_img = pred_img.shape
        m_per_pixel = 20
        half_w_m = (W_img * m_per_pixel) / 2.0
        half_h_m = (H_img * m_per_pixel) / 2.0
        extent_m = [-half_w_m, half_w_m, half_h_m, -half_h_m]

        # Inputs (satellite images)
        save_section_pdf(sat1_disp, "Input Satellite A", os.path.join(frame_dir, "sat0.pdf"),
                         cmap='gray', add_colorbar=False, m_per_pixel=m_per_pixel, extent_m=extent_m)
        save_section_pdf(sat2_disp, "Input Satellite B", os.path.join(frame_dir, "sat1.pdf"),
                         cmap='gray', add_colorbar=False, m_per_pixel=m_per_pixel, extent_m=extent_m)

        # Mask
        save_section_pdf(mask_disp, "Cloud Mask", os.path.join(frame_dir, "mask.pdf"),
                         cmap='gray_r', add_colorbar=False, m_per_pixel=m_per_pixel, extent_m=extent_m,
                         vmin=0, vmax=1)

        # Prediction (velocity)
        save_section_pdf(pred_disp, "Predicted Top Velocity [m/s]", os.path.join(frame_dir, "pred.pdf"),
                         cmap='jet', add_colorbar=True,
                         m_per_pixel=m_per_pixel, extent_m=extent_m, vmin=vmin_pred, vmax=vmax_pred,
                         tick_step=1.0)

        # GT and Difference (if available)
        if gt_img is not None:
            save_section_pdf(gt_disp, "Ground Truth Top Velocity [m/s]", os.path.join(frame_dir, "gt.pdf"),
                             cmap='jet', add_colorbar=True,
                             m_per_pixel=m_per_pixel, extent_m=extent_m, vmin=vmin_pred, vmax=vmax_pred,
                             tick_step=1.0)

            save_section_pdf(diff_disp, "Predicted - GT [m/s]", os.path.join(frame_dir, "diff.pdf"),
                             cmap='RdBu_r', add_colorbar=True,
                             m_per_pixel=m_per_pixel, extent_m=extent_m, vmin=-3.0, vmax=3.0,
                             tick_step=1.0)

print(f'[INFO] PDF frames saved to {PDF_BASE_DIR}')

if video_writer is not None:
    video_writer.release()
    print(f"[INFO] Successfully saved output video to: {output_video}")

# # --- Generate Scatter Plots ---
# if len(scatter_pred_list) > 0 and len(scatter_gt_list) > 0:
#     print("\n[INFO] Generating scatter plots...")
#     all_gt = np.concatenate(scatter_gt_list)
#     all_pred = np.concatenate(scatter_pred_list)
#
#     # Filter out NaN and Inf values
#     valid_mask = np.isfinite(all_gt) & np.isfinite(all_pred)
#     all_gt_clean = all_gt[valid_mask]
#     all_pred_clean = all_pred[valid_mask]
#
#     print(f"[INFO] Filtered {len(all_gt) - len(all_gt_clean)} invalid (NaN/Inf) points out of {len(all_gt)}")
#
#     if len(all_gt_clean) > 0:
#         # Balanced sampling for scatter plot
#         x_scatter, y_scatter, scatter_min, scatter_max, scatter_range_padded = sample_scatter_points(
#             all_gt_clean, all_pred_clean, " (all frames)"
#         )
#
#         if x_scatter is not None:
#             print(f"[INFO] Scatter range: [{-scatter_range_padded:.2f}, {scatter_range_padded:.2f}]")
#
#             # Create and save scatter plot
#             fig_scatter, ax_scatter = plt.subplots(figsize=(12, 12), dpi=150)
#             ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=50, alpha=0.4, rasterized=True)
#             ax_scatter.plot(
#                 [-scatter_range_padded, scatter_range_padded],
#                 [-scatter_range_padded, scatter_range_padded],
#                 'k--', lw=3
#             )
#             ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=14, fontweight='bold')
#             ax_scatter.set_ylabel("Predicted [m/s]", fontsize=14, fontweight='bold')
#             ax_scatter.set_title(f"Balanced Scatter Plot (All Frames)", fontsize=16, fontweight='bold', pad=20)
#             ax_scatter.set_xlim(-scatter_range_padded, scatter_range_padded)
#             ax_scatter.set_ylim(-scatter_range_padded, scatter_range_padded)
#             ax_scatter.grid(True, alpha=0.3, linewidth=1)
#             plt.tight_layout()
#
#             scatter_plot_path = os.path.join(output_dir, f'scatter_plot_{movie_level}.pdf')
#             plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
#             print(f'[INFO] Scatter plot saved to {scatter_plot_path}')
#             plt.close(fig_scatter)
#         else:
#             print("[WARNING] Sampling failed, skipping scatter plot.")
#     else:
#         print("[WARNING] All scatter data is invalid (NaN/Inf after filtering). Skipping scatter plot.")

# --- Calculate Metrics & Generate Scatter Plots ---
if len(scatter_pred_list) > 0 and len(scatter_gt_list) > 0:
    print("\n[INFO] Calculating Global Metrics and generating scatter plots...")
    all_gt = np.concatenate(scatter_gt_list)
    all_pred = np.concatenate(scatter_pred_list)

    # Filter out NaN and Inf values
    valid_mask = np.isfinite(all_gt) & np.isfinite(all_pred)
    all_gt_clean = all_gt[valid_mask]
    all_pred_clean = all_pred[valid_mask]

    print(f"[INFO] Filtered {len(all_gt) - len(all_gt_clean)} invalid (NaN/Inf) points out of {len(all_gt)}")

    if len(all_gt_clean) > 0:
        # ---> NEW: Calculate Metrics <---
        all_diff = all_pred_clean - all_gt_clean
        global_mae = np.mean(np.abs(all_diff))
        global_rmse = np.sqrt(np.mean(all_diff ** 2))
        global_mean_err = np.mean(all_diff)

        print("\n" + "=" * 40)
        print(f"=== Final Sequence Stats ({movie_level}) ===")
        print(f"Average MAE:        {global_mae:.4f} m/s")
        print(f"Average RMSE:       {global_rmse:.4f} m/s")
        print(f"Average Mean Error: {global_mean_err:.4f} m/s")
        print("=" * 40 + "\n")
        # --------------------------------

        # Balanced sampling for scatter plot
        x_scatter, y_scatter, scatter_min, scatter_max, scatter_range_padded = sample_scatter_points(
            all_gt_clean, all_pred_clean, " (all frames)"
        )

        if x_scatter is not None:
            print(f"[INFO] Scatter range: [{-scatter_range_padded:.2f}, {scatter_range_padded:.2f}]")

            # Create and save scatter plot
            fig_scatter, ax_scatter = plt.subplots(figsize=(12, 12), dpi=150)
            ax_scatter.scatter(x_scatter, y_scatter, c='tab:blue', s=50, alpha=0.4, rasterized=True)
            ax_scatter.plot(
                [-scatter_range_padded, scatter_range_padded],
                [-scatter_range_padded, scatter_range_padded],
                'k--', lw=3
            )
            ax_scatter.set_xlabel("Ground Truth [m/s]", fontsize=14, fontweight='bold')
            ax_scatter.set_ylabel("Predicted [m/s]", fontsize=14, fontweight='bold')
            ax_scatter.set_title(f"Balanced Scatter Plot (All Frames)", fontsize=16, fontweight='bold', pad=20)
            ax_scatter.set_xlim(-scatter_range_padded, scatter_range_padded)
            ax_scatter.set_ylim(-scatter_range_padded, scatter_range_padded)
            ax_scatter.grid(True, alpha=0.3, linewidth=1)

            # ---> NEW: Add text box with metrics on the scatter plot <---
            metrics_text = f"MAE: {global_mae:.2f}\nRMSE: {global_rmse:.2f}\nBias: {global_mean_err:.2f}"
            ax_scatter.text(0.05, 0.95, metrics_text, transform=ax_scatter.transAxes,
                            fontsize=16, fontweight='bold', va='top', ha='left',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            # -----------------------------------------------------------

            plt.tight_layout()

            scatter_plot_path = os.path.join(output_dir, f'scatter_plot_{movie_level}.pdf')
            plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
            print(f'[INFO] Scatter plot saved to {scatter_plot_path}')
            plt.close(fig_scatter)
        else:
            print("[WARNING] Sampling failed, skipping scatter plot.")
    else:
        print("[WARNING] All scatter data is invalid (NaN/Inf after filtering). Skipping scatter plot.")