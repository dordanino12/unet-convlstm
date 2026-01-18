import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from render import MitsubaRenderer
from concurrent.futures import ThreadPoolExecutor 

def get_csv_indices_grouped_by_time(csv_path):
    df = pd.read_csv(csv_path)
    unique_times = sorted(df['utc time'].unique())
    
    time_to_indices = {}
    for t in unique_times:
        indices = df.index[df['utc time'] == t].tolist()
        time_to_indices[t] = indices
        
    return unique_times, time_to_indices

def main(start_folder_name=None, end_folder_name=None):
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    input_root = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/'
    output_root = '/wdata_visl/danino/dataset_rendered_data_spp8192_g085/' 
    csv_path = '/home/danino/PycharmProjects/pythonProject/data/Dor_2satellites_overpass.csv'
    
    SPP = 8192
    RES = 256
    G_VALUE = 0.85
    start_folder_name = '0000005520'
    end_folder_name = '0000009520'
    # ==========================================
    # 2. SETUP DATA & FOLDERS
    # ==========================================
    print("Parsing CSV...")
    unique_times, time_lookup = get_csv_indices_grouped_by_time(csv_path)
    num_csv_states = len(unique_times)
    
    # Optimized folder search
    folders = []
    if os.path.exists(input_root):
        with os.scandir(input_root) as entries:
            for entry in tqdm(entries, desc="Finding Folders"):
                if entry.is_dir() and entry.name.isdigit():
                    folders.append(entry.name)
        folders = sorted(folders)
    else:
        print(f"Error: Input root {input_root} does not exist.")
        return

    print(f"Found {len(folders)} folders to process.")

    # ==========================================
    # 3. CONVERT FOLDER NAMES TO INDICES
    # ==========================================
    start_idx = 0
    end_idx = len(folders)

    if start_folder_name is not None:
        if start_folder_name in folders:
            start_idx = folders.index(start_folder_name)
        else:
            print(f"Error: start_folder_name '{start_folder_name}' not found in folders")
            print(f"Available range: {folders[0]} to {folders[-1]}")
            return

    if end_folder_name is not None:
        if end_folder_name in folders:
            end_idx = folders.index(end_folder_name) + 1  # +1 because end is exclusive
        else:
            print(f"Error: end_folder_name '{end_folder_name}' not found in folders")
            print(f"Available range: {folders[0]} to {folders[-1]}")
            return

    if end_idx <= start_idx:
        print(f"Error: end_folder_name must be after start_folder_name")
        return

    original_start_idx = start_idx
    folders = folders[start_idx:end_idx]
    print(f"Processing folders {folders[0]} to {folders[-1]} ({len(folders)} folders)")

    # ==========================================
    # 4. MAIN BATCH LOOP
    # ==========================================
    for folder_idx, folder_name in enumerate(tqdm(folders, desc="Folders")):
        
        # A. Cyclic Logic - use original index position for consistency
        csv_ptr = (original_start_idx + folder_idx) % num_csv_states
        target_time = unique_times[csv_ptr]
        target_indices = time_lookup[target_time]
        
        # B. Setup Paths
        current_input_dir = os.path.join(input_root, folder_name)
        current_output_dir = os.path.join(output_root, folder_name)
        os.makedirs(current_output_dir, exist_ok=True)
        # Save temp in script directory (same location as render_all.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_vol_path = os.path.join(script_dir, f"temp_cloud_{folder_name}.vol")

        # C. Find Files FIRST (We need one to initialize the volume)
        pkl_files = sorted([f for f in os.listdir(current_input_dir) if f.endswith('.pkl')])
        
        if not pkl_files:
            continue # Skip empty folders

        # D. Initialize Renderer
        renderer = MitsubaRenderer(
            overpass_csv=csv_path,
            overpass_indices=target_indices, 
            spp=SPP,
            image_res=RES,
            satellites=len(target_indices),
            timestamps=1,
            vol_path=temp_vol_path,
            dynamic_emitter=True,
            bitmaps_required=False,
            pad_image=False,
            centralize_cloud=True,
            voxel_res=0.02,
            scene_scale=1000.0,
            cloud_zrange=[0.0, 4.0],
            g_value=G_VALUE
        )
        
        renderer.read_overpass_csv()
        renderer.camera_params()
        renderer.create_sensors()

        # --- FIX START: Create Initial Volume File ---
        # We must generate a .vol file BEFORE asking Mitsuba to load the scene.
        first_file_path = os.path.join(current_input_dir, pkl_files[0])
        renderer.write_vol_file(sample_path=first_file_path, vol_path=temp_vol_path)
        
        # NOW we can safely initialize the scenes
        renderer.set_scenes()
        # --- FIX END ---
        
        # E. Process All Files - Load next while rendering current
        # This overlaps disk I/O with GPU rendering for maximum efficiency
        
        # Pre-load first file
        first_pkl_path = os.path.join(current_input_dir, pkl_files[0])
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_next = None
            current_path = first_pkl_path
            
            for i, pkl_file in enumerate(pkl_files):
                # Start loading NEXT file in background (if exists)
                if i + 1 < len(pkl_files):
                    next_pkl_file = pkl_files[i + 1]
                    next_pkl_path = os.path.join(current_input_dir, next_pkl_file)
                    # Background thread loads next file
                    future_next = executor.submit(
                        lambda path: (path, os.path.exists(path)),  # Load it (warm cache)
                        next_pkl_path
                    )
                else:
                    future_next = None
                
                try:
                    # 1. Convert & Update current file
                    renderer.write_vol_file(sample_path=current_path, vol_path=temp_vol_path)
                    renderer.update_scenes() 
                    
                    # 2. Render (GPU is busy here, next file loads in background!)
                    tensor_stacks, _ = renderer.render_scenes()
                    
                    # 3. Wait for next file to be loaded/cached
                    if future_next is not None:
                        current_path, _ = future_next.result()
                    
                    # 4. Save current results
                    base_name = os.path.splitext(pkl_file)[0]
                    current_tensors = tensor_stacks[0] 
                    
                    for sat_idx in range(len(target_indices)):
                        img_array = current_tensors[sat_idx]
                        
                        data_packet = {
                            'render': img_array,
                            'timestamp': target_time,
                            'satellite_idx': sat_idx
                        }

                        save_name = f"{base_name}_time_{target_time}_view_{sat_idx}.pkl"
                        save_path = os.path.join(current_output_dir, save_name)
                        
                        with open(save_path, 'wb') as f_out:
                            pickle.dump(data_packet, f_out)

                except Exception as e:
                    tqdm.write(f"Error rendering {pkl_file}: {e}")
        
        # Cleanup temp vol file
        if os.path.exists(temp_vol_path):
            os.remove(temp_vol_path)

if __name__ == "__main__":
    main()