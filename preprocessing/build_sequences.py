import os
import pickle
import numpy as np
import cv2  # <--- Added for resizing
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
root_images = '/wdata_visl/danino/dataset_rendered_data_spp512_g0/'
root_maps = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(vel_maps_slice_1000m_nadir)/'
output_path = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_W_1000m.npz"

SEQ_LEN = 12  # Time 0 to 220 (12 frames)
NUM_SAMPLES = 49  # Samples 000 to 048

# --- NEW PARAMETERS ---
MAX_CHUNKS = None # Set to None to run ALL. Set to 5, 10, etc. for partial runs.
MAP_TYPE = 'w'  # <--- Select map type here: 'w', 'u', or 'v'


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS - OPTIMIZED
# ---------------------------------------------------------
# Cache directory listings to avoid repeated glob calls
_dir_cache = {}

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
            if f"{s_id_str}_" in filename and "_view_0_slice_1000m" in filename:
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

# ---------------------------------------------------------
# 3. MAIN BUILDER
# ---------------------------------------------------------
def main():
    start_time = time.time()

    if not os.path.exists(root_images) or not os.path.exists(root_maps):
        print("Error: Root paths not found.")
        return

    print(f"[INFO] Selected Map Type: {MAP_TYPE}_map")

    # Get all numbered folders (2000, 2020, etc.)
    all_folders = sorted([f for f in os.listdir(root_images) if f.isdigit()], key=int)
    print(f"Found {len(all_folders)} time folders.")

    sequences_X = []
    sequences_Y = []

    # Calculate indices for the chunks
    chunk_indices = list(range(0, len(all_folders), SEQ_LEN))

    # Apply Limit if requested
    if MAX_CHUNKS is not None:
        chunk_indices = chunk_indices[:MAX_CHUNKS]
        print(f"Limiting execution to first {MAX_CHUNKS} chunks.")

    # --- LOOP 1: Process Time Chunks (Trajectories) ---
    for i in tqdm(chunk_indices, desc="Time Chunks"):

        batch_folders = all_folders[i: i + SEQ_LEN]

        # Ensure we have a full sequence of 12 folders
        if len(batch_folders) < SEQ_LEN:
            continue

        # --- LOOP 2: Process Known Samples (000 to 048) ---
        for s_idx in range(NUM_SAMPLES):

            seq_inputs = []  # List for X
            seq_targets = []  # List for Y
            valid_sequence = True

            # --- LOOP 3: Build the Sequence (Time 0 -> 220) ---
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_triplet, batch_folders[0], s_idx)

                for idx, folder_name in enumerate(batch_folders):
                    data = future.result()

                    # Prefetch next timestep while processing current
                    if idx + 1 < len(batch_folders):
                        future = executor.submit(load_triplet, batch_folders[idx + 1], s_idx)
                    else:
                        future = None

                    # Skip this sample if ANY file is missing
                    if data is None:
                        valid_sequence = False
                        break

                    d0, d1, dm = data

                    # Extract and clean image data
                    img0 = np.nan_to_num(d0['render'], nan=0.0)
                    img1 = np.nan_to_num(d1['render'], nan=0.0)

                    # Batch resize for efficiency
                    img0 = cv2.resize(img0, (128, 128), interpolation=cv2.INTER_AREA)
                    img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
                    img_stack = np.stack([img0, img1], axis=0).astype(np.float32)

                    # Extract and clean map data
                    map_key = f"{MAP_TYPE}_map"
                    if map_key not in dm:
                        valid_sequence = False
                        break

                    target_map_raw = np.nan_to_num(dm[map_key], nan=0.0)
                    target_map_raw = cv2.resize(target_map_raw, (128, 128), interpolation=cv2.INTER_AREA)
                    final_map = target_map_raw[np.newaxis, ...].astype(np.float32)

                    seq_inputs.append(img_stack)
                    seq_targets.append(final_map)


            if valid_sequence:
                sequences_X.append(np.stack(seq_inputs, axis=0))
                sequences_Y.append(np.stack(seq_targets, axis=0))

    # 4. Final Save
    if sequences_X:
        X_all = np.stack(sequences_X, axis=0)
        Y_all = np.stack(sequences_Y, axis=0)

        print(f"\nSaved {X_all.shape[0]} sequences.")
        print(f"X: {X_all.shape}")
        print(f"Y: {Y_all.shape} (Map Type: {MAP_TYPE})")

        # You might want to append the map type to filename to avoid overwriting
        # e.g., dataset_trajectory_w.npz
        final_output_path = output_path.replace(".npz", f"_{MAP_TYPE}.npz")

        np.savez_compressed(final_output_path, X=X_all, Y=Y_all)
        print(f"File saved to: {final_output_path}")
    else:
        print("No valid sequences found.")

    elapsed = time.time() - start_time
    print(f"\n[TIMING] Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")


if __name__ == "__main__":
    main()