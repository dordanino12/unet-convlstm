import numpy as np
import pickle
import os
import pandas as pd
import ast
from build_W_map import CloudRayCaster
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_camera_csv(csv_path):
    """
    Reads the CSV and returns a dictionary:
    { utc_time_int: [ (pos_array, lookat_array), (pos_array, lookat_array) ] }
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
            raw_coords = ast.literal_eval(row['sat ENU coordinates [km]'])

            # 2. Apply Coordinate Transformation
            x_km = -raw_coords[1]
            y_km = raw_coords[0]
            z_km = raw_coords[2]

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

    # ================= CONFIGURATION =================
    # Paths
    input_root = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)//'
    output_root = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(vel_maps_slice_1000m_nadir)/'
    csv_file_path = '/home/danino/PycharmProjects/pythonProject/data/Dor_2satellites_overpass.csv'

    # Rendering Mode
    RENDER_MODE = 'slice'  # Options: 'slice' OR 'first_hit'

    # Parameters for Slice Mode
    SLICE_HEIGHT_M = 1000.0
    REFERENCE_PLANE_Z = 750.0

    # --- CAMERA OVERRIDE SETTINGS ---
    # Set this to True to ignore the CSV camera position and use the fixed one below
    USE_FIXED_CAMERA = True

    # Fixed Camera Position (Meters) - e.g. [0, 0, 600km]
    FIXED_CAMERA_POS = np.array([0.0, 0.0, 600.0 * 1000.0])

    # Resolution
    RES = (256, 256)
    # =================================================

    print(f"--- Starting Batch Processing ---")
    print(f"Mode: {RENDER_MODE}")
    if USE_FIXED_CAMERA:
        print(f"(!) Using FIXED Camera Position: {FIXED_CAMERA_POS}")

    if RENDER_MODE == 'slice':
        print(f"Slice Height: {SLICE_HEIGHT_M}m (Ref Plane: {REFERENCE_PLANE_Z}m)")

    # 1. Load Camera Schedule
    print("Loading CSV...")
    csv_times_list, camera_lookup = load_camera_csv(csv_file_path)
    num_csv_states = len(csv_times_list)
    print(f"Loaded {num_csv_states} unique timestamps from CSV.")

    # 2. Get and Sort Input Folders
    folders = []
    if os.path.exists(input_root):
        with os.scandir(input_root) as entries:
            for entry in tqdm(entries, desc="Finding Folders"):
                if entry.is_dir() and entry.name.isdigit():
                    folders.append(entry.name)
        folders = sorted(folders)
    else:
        print(f"Error: Input root {input_root} does not exist.")
        exit()

    print(f"Found {len(folders)} data folders. Starting batch processing...")

    # 3. Iterate Folders
    for folder_idx, folder_name in enumerate(tqdm(folders, desc="Processing")):

        # A. Determine which CSV time to use
        csv_ptr = folder_idx % num_csv_states
        target_time = csv_times_list[csv_ptr]

        # Get the list of camera views for this time
        current_cameras = camera_lookup[target_time]

        # B. Setup paths
        current_input_dir = os.path.join(input_root, folder_name)
        current_output_dir = os.path.join(output_root, folder_name)
        os.makedirs(current_output_dir, exist_ok=True)

        # C. Find all PKL files
        pkl_files = sorted([f for f in os.listdir(current_input_dir) if f.endswith('.pkl')])

        # D. Process each sample
        for pkl_file in pkl_files:
            full_pkl_path = os.path.join(current_input_dir, pkl_file)

            try:
                caster = CloudRayCaster(full_pkl_path)

                # E. Render for each View in the CSV row
                for view_idx, (csv_cam_pos, look_at) in enumerate(current_cameras):

                    # --- APPLY OVERRIDE IF NEEDED ---
                    if USE_FIXED_CAMERA:
                        render_cam_pos = FIXED_CAMERA_POS
                    else:
                        render_cam_pos = csv_cam_pos

                    # --- RENDERING LOGIC ---
                    if RENDER_MODE == 'first_hit':
                        u_map, v_map, w_map = caster.render_velocity_maps_first_hit(
                            cam_pos=render_cam_pos,
                            look_at=look_at,
                            resolution=RES
                        )
                        mode_suffix = "first_hit"

                    elif RENDER_MODE == 'slice':
                        u_map, v_map, w_map = caster.render_z_slice(
                            cam_pos=render_cam_pos,
                            look_at=look_at,
                            target_z_height=SLICE_HEIGHT_M,
                            resolution=RES,
                            reference_plane_z=REFERENCE_PLANE_Z
                        )
                        mode_suffix = f"slice_{int(SLICE_HEIGHT_M)}m"

                    else:
                        raise ValueError(f"Unknown RENDER_MODE: {RENDER_MODE}")

                    # Save RAW data
                    data_packet = {'u_map': u_map, 'v_map': v_map, 'w_map': w_map}

                    # Naming: add "_fixedcam" to filename if override is on, to distinguish files?
                    # Or keep same format. I'll append a flag if needed, but for now keeping format as requested.
                    base_name = os.path.splitext(pkl_file)[0]

                    # Optional: Add indicator in filename if fixed cam was used?
                    # For now keeping it standard to your previous code.
                    save_name = f"{base_name}_time_{target_time}_view_{view_idx}_{mode_suffix}.pkl"
                    save_path = os.path.join(current_output_dir, save_name)

                    with open(save_path, 'wb') as f_out:
                        pickle.dump(data_packet, f_out)

            except Exception as e:
                print(f"Failed {pkl_file}: {e}")

    print("Processing Complete.")