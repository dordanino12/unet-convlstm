import os
import glob
import pickle
import numpy as np
import cv2  # <--- Added for resizing
from tqdm import tqdm

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
root_images = '/wdata_visl/danino/dataset_rendered_data/'
root_maps = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(vel_maps_slice_500m_nadir)/'
output_path = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_500m_slices.npz"

SEQ_LEN = 12  # Time 0 to 220 (12 frames)
NUM_SAMPLES = 49  # Samples 000 to 048

# --- NEW PARAMETERS ---
MAX_CHUNKS = None # Set to None to run ALL. Set to 5, 10, etc. for partial runs.
MAP_TYPE = 'w'  # <--- Select map type here: 'w', 'u', or 'v'


# ---------------------------------------------------------
# 2. HELPER TO FIND FILE
# ---------------------------------------------------------
def get_file_path(folder, sample_idx, view_idx=None, is_map=False):
    """
    Constructs the search pattern for a specific sample ID inside a time folder.
    """
    s_id_str = f"sample_{sample_idx:03d}"

    if is_map:
        pattern = os.path.join(folder, f"{s_id_str}_*_view_0_slice_500m.pkl")
    else:
        pattern = os.path.join(folder, f"{s_id_str}_*_view_{view_idx}.pkl")

    files = glob.glob(pattern)
    return files[0] if files else None


# ---------------------------------------------------------
# 3. MAIN BUILDER
# ---------------------------------------------------------
def main():
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
            for folder_name in batch_folders:
                path_img_dir = os.path.join(root_images, folder_name)
                path_map_dir = os.path.join(root_maps, folder_name)

                f_v0 = get_file_path(path_img_dir, s_idx, view_idx=0)
                f_v1 = get_file_path(path_img_dir, s_idx, view_idx=1)
                f_map = get_file_path(path_map_dir, s_idx, is_map=True)

                if not f_v0 or not f_v1 or not f_map:
                    valid_sequence = False
                    break

                try:
                    # 1. Load Images
                    with open(f_v0, 'rb') as f:
                        d0 = pickle.load(f)
                    with open(f_v1, 'rb') as f:
                        d1 = pickle.load(f)

                    # A. Fix NaNs (Set to 0.0)
                    img0 = np.nan_to_num(d0['render'], nan=0.0)
                    img1 = np.nan_to_num(d1['render'], nan=0.0)

                    # B. RESIZE (256 -> 128) using INTER_AREA (best for shrinking)
                    # ------------------------------------------------------
                    img0 = cv2.resize(img0, (128, 128), interpolation=cv2.INTER_AREA)
                    img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
                    # ------------------------------------------------------

                    img_stack = np.stack([img0, img1], axis=0).astype(np.float32)

                    # 2. Load Map
                    with open(f_map, 'rb') as f:
                        dm = pickle.load(f)

                    # --- NEW LOGIC: Select specific map key ---
                    map_key = f"{MAP_TYPE}_map"  # e.g. 'w_map', 'u_map', 'v_map'

                    if map_key not in dm:
                        print(f"Error: Key '{map_key}' not found in pickle.")
                        valid_sequence = False
                        break

                    # A. Fix NaNs
                    target_map_raw = np.nan_to_num(dm[map_key], nan=0.0)

                    # B. RESIZE (256 -> 128)
                    # ------------------------------------------------------
                    target_map_raw = cv2.resize(target_map_raw, (128, 128), interpolation=cv2.INTER_AREA)
                    # ------------------------------------------------------

                    final_map = target_map_raw[np.newaxis, ...].astype(np.float32)

                    seq_inputs.append(img_stack)
                    seq_targets.append(final_map)

                except Exception as e:
                    print(f"Read Error: Sample {s_idx} in {folder_name} - {e}")
                    valid_sequence = False
                    break

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


if __name__ == "__main__":
    main()