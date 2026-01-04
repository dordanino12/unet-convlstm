import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from render import MitsubaRenderer 

def get_csv_indices_grouped_by_time(csv_path):
    df = pd.read_csv(csv_path)
    unique_times = sorted(df['utc time'].unique())
    
    time_to_indices = {}
    for t in unique_times:
        indices = df.index[df['utc time'] == t].tolist()
        time_to_indices[t] = indices
        
    return unique_times, time_to_indices

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    input_root = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/'
    output_root = '/wdata_visl/danino/dataset_rendered_data/' 
    csv_path = '/home/danino/PycharmProjects/pythonProject/data/Dor_2satellites_overpass.csv'
    
    SPP = 512
    RES = 256
    
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
    # 3. MAIN BATCH LOOP
    # ==========================================
    for folder_idx, folder_name in enumerate(tqdm(folders, desc="Folders")):
        
        # A. Cyclic Logic
        csv_ptr = folder_idx % num_csv_states
        target_time = unique_times[csv_ptr]
        target_indices = time_lookup[target_time]
        
        # B. Setup Paths
        current_input_dir = os.path.join(input_root, folder_name)
        current_output_dir = os.path.join(output_root, folder_name)
        os.makedirs(current_output_dir, exist_ok=True)
        temp_vol_path = os.path.join(current_output_dir, "temp_cloud.vol")

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
            g_value=0.0
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
        
        # E. Process All Files (Loop)
        for pkl_file in pkl_files:
            full_pkl_path = os.path.join(current_input_dir, pkl_file)
            
            try:
                # 1. Convert & Update 
                # (Overwrite the temp vol file with the current sample)
                renderer.write_vol_file(sample_path=full_pkl_path, vol_path=temp_vol_path)
                
                renderer.update_scenes() 
                
                # 2. Render
                tensor_stacks, _ = renderer.render_scenes()
                
                # 3. Save as Pickle
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