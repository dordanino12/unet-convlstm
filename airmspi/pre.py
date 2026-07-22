import pickle
import numpy as np
import os
import cv2  # Added for image resizing

def convert_airmspi_pkl_to_npz_2channels(pkl_path: str, out_npz_path: str, target_size: tuple = (128, 128)) -> None:
    """
    Converts an AirMSPI .pkl file into a .npz format compatible with NPZSequenceDataset.
    Resizes images to target_size (default 128x128) and duplicates to simulate 2 channels.
    """
    if not os.path.exists(pkl_path):
        print(f"[ERROR] File not found: {pkl_path}")
        return

    # 1. Load the original .pkl file
    with open(pkl_path, "rb") as f:
        image_data = pickle.load(f)

    # Extract the array
    img_array = None
    if isinstance(image_data, dict):
        for key, val in image_data.items():
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                img_array = val
                break
    elif isinstance(image_data, np.ndarray):
        img_array = image_data

    if img_array is None or img_array.ndim != 3:
        print(f"[ERROR] Expected a 3D array (e.g., 9x350x350), but got unsupported format.")
        return

    # img_array is currently (9, H, W)
    num_cameras = img_array.shape[0]

    # --- NEW: Resize to 128x128 ---
    resized_images = []
    for i in range(num_cameras):
        # cv2.resize expects dimensions as (width, height)
        # INTER_AREA is recommended for downscaling images
        resized_img = cv2.resize(img_array[i], target_size, interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)
        
    img_array = np.array(resized_images)

    # Update dimensions after resize
    num_cameras, height, width = img_array.shape

    # 2. Duplicate the images to create 2 channels
    # np.stack creates a new axis for the channels: shape becomes (9, 2, H, W)
    img_2ch = np.stack([img_array, img_array], axis=1)
    
    # Add the batch dimension (N=1): shape becomes (1, 9, 2, H, W)
    X = np.expand_dims(img_2ch, axis=0).astype(np.float32)
    
    # 3. Create dummy Ground Truth (Y) and masks
    # The GT and mask should remain 1-channel since the model predicts a single velocity map
    # Shape for Y and mask: (1, 9, 1, 128, 128)
    Y = np.zeros((1, num_cameras, 1, height, width), dtype=np.float32)
    mask = np.ones((1, num_cameras, 1, height, width), dtype=np.float32)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

    # 4. Save as .npz
    np.savez(out_npz_path, X=X, Y=Y, mask=mask)
    
    print(f"[INFO] Successfully converted '{os.path.basename(pkl_path)}'")
    print(f"       Saved to: {out_npz_path}")
    print(f"       Input (X) Shape (N, T, C, H, W): {X.shape}")
    print(f"       GT (Y) Shape (N, T, C, H, W): {Y.shape}")

if __name__ == "__main__":
    # Point this to one of your extracted PKL files
    input_pkl = r"/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/AirMSPI_pushbroom_camera/SIMULATED_AIRMSPI_TRAIN_20160826_104727Z_SouthAtlanticOcean-14S19W/satellites_images_1000.pkl"
    
    # The destination for the new 2-channel NPZ file
    output_npz = r"/home/danino/PycharmProjects/pythonProject/airmspi/airmspi_test_seq_2sat.npz"
    
    convert_airmspi_pkl_to_npz_2channels(input_pkl, output_npz)