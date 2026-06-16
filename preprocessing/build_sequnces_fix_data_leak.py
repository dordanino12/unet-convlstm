import os
import pickle
import time
import gc
from concurrent.futures import ThreadPoolExecutor

import cv2  # <--- Added for resizing
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

root_images = "/wdata_visl/danino/dataset_rendered_data_spp8192_g85/render_images/"
root_maps = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split_vel_maps_slice_1500m_nadir/"
output_path = "/home/danino/PycharmProjects/pythonProject/data/wacv_data/1500m.npz"

SEQ_LEN = 12  # Time 0 to 220 (12 frames)
NUM_SAMPLES = 49  # Samples 000 to 048 (7x7 spatial grid)

# --- NEW PARAMETERS ---

MAX_CHUNKS = None  # Set to None to run ALL. Set to 5, 10, etc. for partial runs.
USE_K_FOLD = True  # If True, creates K_FOLDS with spatial block splits. If False, uses a single fixed validation block.

MAP_TYPE = 'w'  # <--- Select map type here: 'w', 'u', or 'v'
# If True, ignore MAP_TYPE and save all three components (u,v,w).
# When SAVE_UVW=True the saved Y per-timestep will have shape (N_heights, 3, H, W)
SAVE_UVW = False

# --- NEW: Optional deterministic pixel adjustment settings for input views ---
# Adjustment is applied per pixel as: pixel * (1 + p), where p = NOISE_PERCENT / 100.
NOISE_PERCENT = 0.0  # Example: 5.0 means add 5% of each pixel value
NOISE_TARGET = 'both'  # One of: 'none', 'view0', 'view1', 'both'

# --- NEW: Optional physical sensor noise settings for input views ---
ADD_SENSOR_NOISE = True  # If True, applies a physical sensor noise model instead of simple relative noise.

# --- NEW: Specify valid folder ranges ---

VALID_RANGES = [
     (2000, 19740)
    #(5600, 6000)
]

# --- SPATIAL BLOCK K-FOLD SETTINGS ---
# The grid is split into 4 rectangular validation blocks.
# For each fold, one block becomes validation and its 8-connected halo is ignored.
if USE_K_FOLD:
    VAL_BLOCKS = [
        ((0, 1), (0, 2)),  # Top left band
        ((0, 1), (4, 6)),  # top right band
        ((2, 3), (0, 6)),  # Middle band
        ((5, 6), (0, 2)),  # Bottom-left band
    ]
else:
    VAL_BLOCKS = [
        ((5, 6), (0, 2)),  # Fixed validation block when not using k-fold
    ]

K_FOLDS = len(VAL_BLOCKS)

STATIC_TEST_CELLS = {
    (5, 4), (5, 5), (5, 6),
    (6, 4), (6, 5), (6, 6),
}

STATIC_IGNORE_CELLS = {
                          (4, c) for c in range(3, 7)
                      } | {
                          (5, 3), (6, 3)
                      }

GRID_SIZE = 7

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS - OPTIMIZED
# ---------------------------------------------------------

_dir_cache = {}


def apply_relative_noise(image, noise_percent):
    """Apply deterministic per-pixel relative adjustment by noise_percent."""
    if noise_percent <= 0:
        return image

    p = noise_percent / 100.0
    return (image * (1.0 + p)).astype(np.float32)


def apply_sensor_noise(img_array):
    """
    Simulates physical sensor noise: Dark Current, Read Noise, and 10-bit Quantization.
    Uses the same parameters as train/get_metrics.py.
    """
    CONVERSION_FACTOR = 178.6304426659069
    EXPOSURE_TIME_US = 205
    DARK_CURRENT_RATE = 4.72 * 1e-6  # e-/sec
    FULL_WELL_CAPACITY = 10600
    BIT_DEPTH_FACTOR = 1024  # 10-bit

    electrons = img_array * CONVERSION_FACTOR

    dark_noise_mean = DARK_CURRENT_RATE * EXPOSURE_TIME_US
    dn_noise = np.random.normal(loc=dark_noise_mean, scale=dark_noise_mean ** 0.5, size=electrons.shape)
    electrons += dn_noise

    read_noise = np.random.normal(loc=0.0, scale=5.29 ** 0.5, size=electrons.shape)
    electrons += read_noise

    electrons = np.clip(electrons, a_min=0, a_max=FULL_WELL_CAPACITY)
    dn = electrons * (BIT_DEPTH_FACTOR / FULL_WELL_CAPACITY)
    electrons_quantized = np.round(dn) * (FULL_WELL_CAPACITY / BIT_DEPTH_FACTOR)

    radiance = electrons_quantized / CONVERSION_FACTOR
    return radiance.astype(np.float32)


def get_files_in_dir(folder):
    """Cache directory listings to avoid repeated glob calls."""
    if folder not in _dir_cache:
        _dir_cache[folder] = {}

        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith('.pkl'):
                    _dir_cache[folder][f] = os.path.join(folder, f)

    return _dir_cache[folder]


def get_file_path(folder, sample_idx, view_idx=None, is_map=False):
    """
    Finds file path for a specific sample ID. Uses cached directory listings.
    """
    s_id_str = f"sample_{sample_idx:03d}"
    files_dict = get_files_in_dir(folder)

    for filename, filepath in files_dict.items():
        if is_map:
            if f"{s_id_str}_" in filename and "_view_0" in filename:
                return filepath
        else:
            if f"{s_id_str}_" in filename and f"_view_{view_idx}" in filename:
                return filepath

    return None


def load_triplet(folder_name, sample_idx):
    """Load view0, view1, and map pickles for a single timestep."""
    path_img_dir = os.path.join(root_images, folder_name)
    path_map_dir = os.path.join(root_maps, folder_name)

    f_v0 = get_file_path(path_img_dir, sample_idx, view_idx=0)
    f_v1 = get_file_path(path_img_dir, sample_idx, view_idx=1)
    f_map = get_file_path(path_map_dir, sample_idx, is_map=True)

    if not f_v0 or not f_v1 or not f_map:
        return None

    try:
        with open(f_v0, 'rb') as f:
            d0 = pickle.load(f)
        with open(f_v1, 'rb') as f:
            d1 = pickle.load(f)
        with open(f_map, 'rb') as f:
            dm = pickle.load(f)
        return d0, d1, dm
    except Exception:
        return None


def _cell_in_rect(cell, rect):
    (row_start, row_end), (col_start, col_end) = rect
    row, col = cell
    return row_start <= row <= row_end and col_start <= col <= col_end


def _expand_halo(cells):
    halo = set()
    for row, col in cells:
        for neighbor_row in range(row - 1, row + 2):
            for neighbor_col in range(col - 1, col + 2):
                if 0 <= neighbor_row < GRID_SIZE and 0 <= neighbor_col < GRID_SIZE:
                    neighbor = (neighbor_row, neighbor_col)
                    if neighbor not in cells:
                        halo.add(neighbor)
    return halo


def build_spatial_geometry():
    """Build validation blocks and their dynamic halo buffers."""
    block_cells = []
    block_halos = []

    for block_rect in VAL_BLOCKS:
        cells = {
            (row, col)
            for row in range(block_rect[0][0], block_rect[0][1] + 1)
            for col in range(block_rect[1][0], block_rect[1][1] + 1)
        }
        block_cells.append(cells)
        block_halos.append(_expand_halo(cells))

    return block_cells, block_halos


FOLD_BLOCK_CELLS, FOLD_HALO_CELLS = build_spatial_geometry()


def get_spatial_split(s_idx, fold_idx):
    """Return the split for a sample index in the selected fold."""
    r = s_idx // 7
    c = s_idx % 7
    cell = (r, c)

    if cell in STATIC_TEST_CELLS:
        return 'test'

    if cell in STATIC_IGNORE_CELLS:
        return None

    if cell in FOLD_BLOCK_CELLS[fold_idx]:
        return 'val'

    if cell in FOLD_HALO_CELLS[fold_idx]:
        return None

    return 'train'


# ---------------------------------------------------------
# 3. MAIN BUILDER
# ---------------------------------------------------------


def main():
    start_time = time.time()

    if not os.path.exists(root_images) or not os.path.exists(root_maps):
        print("Error: Root paths not found.")
        return

    if NOISE_TARGET not in {'none', 'view0', 'view1', 'both'}:
        print("Error: NOISE_TARGET must be one of 'none', 'view0', 'view1', 'both'.")
        return

    if NOISE_PERCENT < 0:
        print("Error: NOISE_PERCENT must be >= 0.")
        return

    map_label = 'uvw' if SAVE_UVW else f"{MAP_TYPE}_map"
    print(f"[INFO] Selected Map Type: {map_label}")
    if ADD_SENSOR_NOISE:
        print(f"[INFO] Noise: sensor model enabled, target={NOISE_TARGET}")
    else:
        print(f"[INFO] Noise: target={NOISE_TARGET}, max={NOISE_PERCENT}%")

    all_folders = sorted([f for f in os.listdir(root_images) if f.isdigit()], key=int)
    valid_folders = []
    for f in all_folders:
        num = int(f)
        for start, end in VALID_RANGES:
            if start <= num <= end:
                valid_folders.append(f)
                break

    print(f"Found {len(valid_folders)} valid time folders in specified ranges.")

    if K_FOLDS != len(VAL_BLOCKS):
        print(f"Error: K_FOLDS={K_FOLDS} must match the number of validation blocks ({len(VAL_BLOCKS)}).")
        return

    fold_data = []
    for block_idx, block_rect in enumerate(VAL_BLOCKS):
        fold_data.append({
            'val_block': block_rect,
            'train_X': [],
            'train_Y': [],
            'val_X': [],
            'val_Y': [],
            'test_X': [],
            'test_Y': [],
        })

    chunk_indices = list(range(0, len(valid_folders), SEQ_LEN))
    if MAX_CHUNKS is not None:
        chunk_indices = chunk_indices[:MAX_CHUNKS]
        print(f"Limiting execution to first {MAX_CHUNKS} chunks.")

    for i in tqdm(chunk_indices, desc="Time Chunks"):
        batch_folders = valid_folders[i: i + SEQ_LEN]
        if len(batch_folders) < SEQ_LEN:
            continue

        for s_idx in range(NUM_SAMPLES):
            cell = (s_idx // GRID_SIZE, s_idx % GRID_SIZE)

            if cell in STATIC_TEST_CELLS:
                target_splits = ['test'] * K_FOLDS
            elif cell in STATIC_IGNORE_CELLS:
                continue
            else:
                target_splits = [get_spatial_split(s_idx, fold_idx) for fold_idx in range(K_FOLDS)]

            seq_inputs = []
            seq_targets = []
            valid_sequence = True

            # --- 2. Build the Sequence ---
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_triplet, batch_folders[0], s_idx)

                for idx, folder_name in enumerate(batch_folders):
                    data = future.result()

                    if idx + 1 < len(batch_folders):
                        future = executor.submit(load_triplet, batch_folders[idx + 1], s_idx)
                    else:
                        future = None

                    if data is None:
                        valid_sequence = False
                        break

                    d0, d1, dm = data

                    img0 = np.nan_to_num(d0['render'], nan=0.0)
                    img1 = np.nan_to_num(d1['render'], nan=0.0)

                    img0 = cv2.resize(img0, (128, 128), interpolation=cv2.INTER_AREA)
                    img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
                    if ADD_SENSOR_NOISE:
                        if NOISE_TARGET in {'view0', 'both'}:
                            img0 = apply_sensor_noise(img0)
                        if NOISE_TARGET in {'view1', 'both'}:
                            img1 = apply_sensor_noise(img1)
                    else:
                        if NOISE_TARGET in {'view0', 'both'}:
                            img0 = apply_relative_noise(img0, NOISE_PERCENT)
                        if NOISE_TARGET in {'view1', 'both'}:
                            img1 = apply_relative_noise(img1, NOISE_PERCENT)

                    img_stack = np.stack([img0, img1], axis=0).astype(np.float32)

                    # Build the target map. Support multi-height maps and optionally u/v/w components.
                    if SAVE_UVW:
                        # require all three components
                        if not all(k in dm for k in ('u_map', 'v_map', 'w_map')):
                            valid_sequence = False
                            break

                        comp_arrays = []
                        # For each component, ensure shape (N_heights, H, W)
                        for comp_key in ('u_map', 'v_map', 'w_map'):
                            arr = np.nan_to_num(dm[comp_key], nan=0.0)
                            arr = np.asarray(arr)
                            if arr.ndim == 2:
                                arr = arr[np.newaxis, ...]
                            # resize each height slice to target 128x128
                            resized_slices = []
                            for h_idx in range(arr.shape[0]):
                                slice_h = arr[h_idx]
                                slice_h = cv2.resize(slice_h, (128, 128), interpolation=cv2.INTER_AREA)
                                resized_slices.append(slice_h)
                            comp_arrays.append(np.stack(resized_slices, axis=0))

                        # comp_arrays: list of 3 arrays each (N_heights, 128,128)
                        # We want final_map shape (N_heights, 3, 128,128) as requested (heights first)
                        stacked_comps = np.stack(comp_arrays, axis=1)  # shape (N_heights, 3, H, W)
                        final_map = stacked_comps.astype(np.float32)
                    else:
                        map_key = f"{MAP_TYPE}_map"
                        if map_key not in dm:
                            valid_sequence = False
                            break

                        target_map_raw = np.nan_to_num(dm[map_key], nan=0.0)
                        target_map_raw = np.asarray(target_map_raw)
                        # If single 2D map, make it (1, H, W). If already (N, H, W), keep N.
                        if target_map_raw.ndim == 2:
                            target_map_raw = target_map_raw[np.newaxis, ...]

                        # Resize each height slice to target 128x128
                        resized = []
                        for h_idx in range(target_map_raw.shape[0]):
                            slice_h = target_map_raw[h_idx]
                            slice_h = cv2.resize(slice_h, (128, 128), interpolation=cv2.INTER_AREA)
                            resized.append(slice_h)

                        # final_map shape (N_heights, 128, 128)
                        final_map = np.stack(resized, axis=0).astype(np.float32)

                    seq_inputs.append(img_stack)
                    seq_targets.append(final_map)

            # --- 3. Append to correct spatial list ---
            if valid_sequence:
                # STACK ONCE TO SAVE MEMORY - Reduces memory usage by ~75%
                # By stacking here, all folds share the same array object in memory
                seq_inputs_stacked = np.stack(seq_inputs, axis=0)
                seq_targets_stacked = np.stack(seq_targets, axis=0)

                for fold_idx, store in enumerate(fold_data):
                    target_split = target_splits[fold_idx]
                    if target_split == 'train':
                        store['train_X'].append(seq_inputs_stacked)
                        store['train_Y'].append(seq_targets_stacked)
                    elif target_split == 'val':
                        store['val_X'].append(seq_inputs_stacked)
                        store['val_Y'].append(seq_targets_stacked)
                    elif target_split == 'test':
                        store['test_X'].append(seq_inputs_stacked)
                        store['test_Y'].append(seq_targets_stacked)

    # 4. Final Save helper function
    def save_split_data(X_list, Y_list, split_name, output_file):
        if X_list:
            print(f"Stacking {split_name} in memory...")
            X_all = np.stack(X_list, axis=0)
            Y_all = np.stack(Y_list, axis=0)

            # For scalar-map targets, remove the height axis only when every sample
            # in this split is single-height (H==1).
            # if not SAVE_UVW:
            #     all_single_height = all((y.ndim == 4 and y.shape[1] == 1) for y in Y_list)
            #     if all_single_height and Y_all.ndim == 5 and Y_all.shape[2] == 1:
            #         Y_all = np.squeeze(Y_all, axis=2)
            #         print(f"[INFO] {split_name}: squeezed single-height axis -> Y shape {Y_all.shape}")

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            print(f"Writing {split_name} to disk...")
            np.savez_compressed(output_file, X=X_all, Y=Y_all)
            print(f"Saved {split_name} -> X: {X_all.shape}, Y: {Y_all.shape} | Path: {output_file}")

            # Explicitly delete the large arrays to free memory instantly
            del X_all
            del Y_all
            gc.collect()
        else:
            print(f"No valid sequences found for {split_name}.")

    print("\n--- Saving Datasets ---")

    noise_suffix = ""
    if ADD_SENSOR_NOISE:
        noise_suffix = f"_sensor_noise_{NOISE_TARGET}"
    elif NOISE_TARGET != 'none' and NOISE_PERCENT > 0:
        noise_str = f"{NOISE_PERCENT:g}".replace('.', 'p')
        noise_suffix = f"_noise_{NOISE_TARGET}_{noise_str}pct"

    map_label = 'uvw' if SAVE_UVW else MAP_TYPE
    output_root = output_path.replace(".npz", f"_kfold_{map_label}{noise_suffix}")
    os.makedirs(output_root, exist_ok=True)

    # Save test data once, then clear it entirely from all folds
    if fold_data and fold_data[0]['test_X']:
        save_split_data(
            fold_data[0]['test_X'],
            fold_data[0]['test_Y'],
            'test',
            os.path.join(output_root, f"test_{map_label}.npz"),
        )
    # Clear test lists
    for store in fold_data:
        store['test_X'].clear()
        store['test_Y'].clear()
    gc.collect()

    # Iterate over folds, process, and clear memory sequentially
    for fold_idx, store in enumerate(fold_data, start=1):
        row_start, row_end = store['val_block'][0]
        col_start, col_end = store['val_block'][1]
        fold_dir = os.path.join(output_root, f"fold_{fold_idx:02d}_val_r{row_start}-{row_end}_c{col_start}-{col_end}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n[INFO] Saving fold {fold_idx}/{K_FOLDS} with validation block {store['val_block']} -> {fold_dir}")

        # Save and clear train
        save_split_data(store['train_X'], store['train_Y'], 'train', os.path.join(fold_dir, f"train_{map_label}.npz"))
        store['train_X'].clear()
        store['train_Y'].clear()
        gc.collect()

        # Save and clear val
        save_split_data(store['val_X'], store['val_Y'], 'val', os.path.join(fold_dir, f"val_{map_label}.npz"))
        store['val_X'].clear()
        store['val_Y'].clear()
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n[TIMING] Total execution time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()