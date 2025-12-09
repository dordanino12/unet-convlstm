import numpy as np
import pickle
import os
import pandas as pd
import ast
from build_W_map import CloudRayCaster
import matplotlib.pyplot as plt


def load_camera_csv(csv_path):
    """
    Reads the CSV and returns a dictionary:
    { utc_time_int: [ (pos_array, lookat_array), (pos_array, lookat_array) ] }

    UPDATES:
    1. Forces look_at to [0, 0, 1500].
    2. Parses 'sat ENU' assuming input format is [-y, x, z] and converts to [x, y, z].
    """
    df = pd.read_csv(csv_path)

    camera_schedule = {}

    # Iterate through unique times in the CSV
    unique_times = sorted(df['utc time'].unique())

    for t in unique_times:
        rows = df[df['utc time'] == t]
        configs = []
        for _, row in rows.iterrows():
            # 1. Parse string "[v0, v1, v2]" to a list/array
            # This handles the text reading of brackets and commas
            raw_coords = ast.literal_eval(row['sat ENU coordinates [km]'])

            # 2. Apply Coordinate Transformation
            # User Input Format: [-y, x, z] -> [raw[0], raw[1], raw[2]]
            # Target Format:     [x, y, z]

            x_km = raw_coords[1]  # x is the second element
            y_km = -raw_coords[0]  # y is the negative of the first element (-y)
            z_km = raw_coords[2]  # z is the third element

            sat_enu_km = np.array([x_km, y_km, z_km])

            # 3. Convert KM to Meters
            sat_enu_m = sat_enu_km * 1000.0

            # --- FORCE FIXED LOOK AT ---
            lookat_enu_m = np.array([0, 0, 1500])

            configs.append((sat_enu_m, lookat_enu_m))

        camera_schedule[t] = configs

    return unique_times, camera_schedule
# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":

    # PATHS
    input_root = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split[beta,U,V,W]//'
    output_root = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(U,V,W_vel_maps)/'
    # Updated CSV path as requested
    csv_file_path = '/home/danino/PycharmProjects/pythonProject/data/Dor_2satellites_overpass.csv'

    # 1. Load Camera Schedule
    print("Loading CSV...")
    csv_times_list, camera_lookup = load_camera_csv(csv_file_path)
    num_csv_states = len(csv_times_list)
    print(f"Loaded {num_csv_states} unique timestamps from CSV.")

    # 2. Get and Sort Input Folders
    # Filter only numeric folders
    all_items = os.listdir(input_root)
    folders = sorted([d for d in all_items if os.path.isdir(os.path.join(input_root, d)) and d.isdigit()])

    print(f"Found {len(folders)} data folders. Starting batch processing...")

    # 3. Iterate Folders with Stride/Cycle Logic
    for folder_idx, folder_name in enumerate(folders):

        # A. Determine which CSV time to use (Cyclic/Modulo)
        # If folder_idx is 0 -> index 0
        # If folder_idx is larger than available CSV times -> wrap around
        csv_ptr = folder_idx % num_csv_states
        target_time = csv_times_list[csv_ptr]

        # Get the list of camera views for this time (usually 2 views)
        current_cameras = camera_lookup[target_time]

        print(
            f"[{folder_idx + 1}/{len(folders)}] Folder: {folder_name} | Mapped to CSV Time: {target_time} ({len(current_cameras)} views)")

        # B. Setup paths
        current_input_dir = os.path.join(input_root, folder_name)
        current_output_dir = os.path.join(output_root, folder_name)
        os.makedirs(current_output_dir, exist_ok=True)

        # C. Find all PKL files in this folder
        pkl_files = sorted([f for f in os.listdir(current_input_dir) if f.endswith('.pkl')])

        # D. Process each sample
        for pkl_file in pkl_files:
            full_pkl_path = os.path.join(current_input_dir, pkl_file)

            try:
                caster = CloudRayCaster(full_pkl_path)

                # E. Render for each View in the CSV row
                for view_idx, (cam_pos, look_at) in enumerate(current_cameras):
                    # Double check we are passing the fixed look_at here
                    # Generate Maps
                    u_map, v_map, w_map = caster.render_velocity_maps_first_hit(
                        cam_pos=cam_pos,
                        look_at=look_at,
                        resolution=(128, 128)
                    )

                    # Save RAW data
                    data_packet = {'u_map': u_map, 'u_map': v_map, 'u_map': w_map}

                    # Construct filename: sample_000_view_0.pkl
                    base_name = os.path.splitext(pkl_file)[0]
                    save_name = f"{base_name}_view_{view_idx}.pkl"
                    save_path = os.path.join(current_output_dir, save_name)

                    with open(save_path, 'wb') as f_out:
                        pickle.dump(data_packet, f_out)

            except Exception as e:
                print(f"Failed {pkl_file}: {e}")

    print("Processing Complete.")