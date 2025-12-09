import os
import pickle
import numpy as np
from netCDF4 import Dataset
from mitsuba3.calc_beta import process_cloud_vars


def generate_velocity_beta_dataset_overlapping(nc_path, output_dir):
    """
    Splits a 512x512x200 NetCDF file into patches of size 128x128x200 with overlap.
    Stride is set to 64, meaning 50% overlap.
    """
    # 1. Create output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Open NetCDF
    print(f"Opening file: {nc_path}")
    nc = Dataset(nc_path, 'r')

    # 3. Get Dimensions
    x_len = nc.variables['x'].shape[0]  # 512
    y_len = nc.variables['y'].shape[0]  # 512

    # --- Configuration for 128x128 patches with overlap ---
    patch_size = 128
    stride = 64  # This gives 50% overlap

    # Calculate how many steps fit in each dimension
    # For 512 size, 128 patch, 64 stride:
    # Starts at: 0, 64, 128, 192, 256, 320, 384.
    # Last start index must ensure we don't go over bounds (384 + 128 = 512)
    x_steps = (x_len - patch_size) // stride + 1
    y_steps = (y_len - patch_size) // stride + 1

    print(f"--- Splitting Configuration ---")
    print(f"Input Grid: {x_len}x{y_len}")
    print(f"Patch Size: {patch_size}x{patch_size}")
    print(f"Stride: {stride} (Overlap: {patch_size - stride} pixels)")
    print(f"Total Patches: {x_steps}x{y_steps} = {x_steps * y_steps}")
    print("-------------------------------")

    # Load Global Pressure (p) - needed for the calculation
    global_p = nc.variables['p'][:]

    count = 0

    # 4. Loop with Overlap
    for i in range(y_steps):  # Y-axis loop
        for j in range(x_steps):  # X-axis loop

            # Calculate start/end indices based on stride
            y_start = i * stride
            y_end = y_start + patch_size

            x_start = j * stride
            x_end = x_start + patch_size

            # Extract Variables (taking time index 0)
            patch_QN = nc.variables['QN'][0, :, y_start:y_end, x_start:x_end]
            patch_NC = nc.variables['NC'][0, :, y_start:y_end, x_start:x_end]
            patch_TABS = nc.variables['TABS'][0, :, y_start:y_end, x_start:x_end]

            # --- Execute Calculation ---
            _, _, patch_beta = process_cloud_vars(patch_QN, patch_NC, patch_TABS, global_p)

            # --- Extract Targets ---
            patch_U = nc.variables['U'][0, :, y_start:y_end, x_start:x_end]
            patch_V = nc.variables['V'][0, :, y_start:y_end, x_start:x_end]
            patch_W = nc.variables['W'][0, :, y_start:y_end, x_start:x_end]

            # --- Save Data ---
            data = {
                'metadata': {
                    'id': count,
                    'grid_idx': (i, j),
                    'coords_x': (x_start, x_end),
                    'coords_y': (y_start, y_end)
                },
                'U': np.ma.filled(patch_U, 0.0).astype(np.float32),
                'V': np.ma.filled(patch_V, 0.0).astype(np.float32),
                'W': np.ma.filled(patch_W, 0.0).astype(np.float32),
                'beta_ext': np.ma.filled(patch_beta, 0.0).astype(np.float32)
            }

            filename = f"sample_{count:03d}.pkl"
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(data, f)

            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{x_steps * y_steps} patches...")

    nc.close()
    print(f"Done. Generated {count} patches in total.")


# --- Run ---
nc_file_path = '/wdata_visl/udigal/netCDF_20X20/BOMEX_512x512x200_20m_20m_1s_512_0000006040.nc'
output_folder_path = '/wdata_visl/danino/dataset_128_overlap'

generate_velocity_beta_dataset_overlapping(nc_file_path, output_folder_path)