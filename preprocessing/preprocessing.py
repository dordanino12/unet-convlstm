import os
import pickle
import numpy as np
import glob
from netCDF4 import Dataset
# Assuming this module exists in your environment
from mitsuba3.calc_beta import process_cloud_vars


def generate_patches_from_nc(nc_path, output_dir):
    """
    Process a single NetCDF file: splits it into 128x128 patches with overlap
    and saves them as .pkl files in the specific output_dir.
    """
    # 1. Create output folder for this specific file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Open NetCDF
    print(f"Processing file: {os.path.basename(nc_path)}")
    try:
        nc = Dataset(nc_path, 'r')
    except Exception as e:
        print(f"Error opening {nc_path}: {e}")
        return

    # 3. Get Dimensions
    x_len = nc.variables['x'].shape[0]  # 512
    y_len = nc.variables['y'].shape[0]  # 512

    # --- Configuration for 128x128 patches with overlap ---
    patch_size = 512
    stride = 64  # This gives 50% overlap

    x_steps = (x_len - patch_size) // stride + 1
    y_steps = (y_len - patch_size) // stride + 1

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
            # Using try/except in case of corrupt data segments
            try:
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
                        'source_file': os.path.basename(nc_path),
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

            except Exception as e:
                print(f"Error processing patch {i},{j} in {os.path.basename(nc_path)}: {e}")

    nc.close()
    print(f"Finished {os.path.basename(nc_path)}: Generated {count} patches.")


def process_all_nc_files(input_folder, base_output_folder):
    """
    Finds all .nc files in input_folder, SORTS THEM NUMERICALLY, and processes them.
    """
    # Find all .nc files
    nc_files = glob.glob(os.path.join(input_folder, "*.nc"))

    if not nc_files:
        print(f"No .nc files found in {input_folder}")
        return

    # --- KEY UPDATE: SORT BY NUMBER ---
    # This lambda function extracts the last part of the filename (the number)
    # splits by '_' and converts it to an integer for correct numerical sorting.
    try:
        nc_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
        print("Files sorted numerically.")
    except Exception as e:
        print(f"Warning: Could not sort numerically (filenames might not match format). Running default order. Error: {e}")

    print(f"Found {len(nc_files)} NetCDF files. Starting process...")
    print("-" * 50)

    for nc_file in nc_files:
        # 1. Get the filename without extension
        # e.g., "BOMEX_512x512x200_20m_20m_1s_512_0000002144"
        base_name = os.path.splitext(os.path.basename(nc_file))[0]

        # 2. Split by '_' and take the last element
        # e.g., "0000002144"
        folder_name = base_name.split('_')[-1]

        # Determine specific output path: base_output_folder / folder_name
        specific_output_dir = os.path.join(base_output_folder, folder_name)

        # Run the generator
        generate_patches_from_nc(nc_file, specific_output_dir)
        print("-" * 50)


# --- Run Configuration ---
if __name__ == "__main__":
    # Update these paths to your directories
    input_directory = '/wdata_visl/udigal/netCDF_20X20/'
    output_directory = '/wdata_visl/danino/dataset_512x512x200_overlap_64_stride_7x7_split(beta,U,V,W)'

    process_all_nc_files(input_directory, output_directory)