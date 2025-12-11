import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# --- Helper Function for Dynamic Limits ---
def get_dynamic_limit(data_map, fallback=10.0):
    valid_data = data_map[~np.isnan(data_map)]
    if len(valid_data) == 0: 
        return fallback
    return np.percentile(np.abs(valid_data), 99)

def visualize_combined_data(vel_pkl_path, render_pkl_path, save_path):
    """
    Loads Velocity Data (U,V,W) AND Render Data (Brightness/Radiance)
    and saves raw PNGs for all of them using default orientation (Top-Left origin).
    """
    
    # --- 1. Load Velocity Data ---
    if not os.path.exists(vel_pkl_path):
        print(f"Error: Velocity file not found: {vel_pkl_path}")
        return

    print(f"Reading Velocity: {vel_pkl_path}")
    try:
        with open(vel_pkl_path, 'rb') as f:
            vel_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading velocity pickle: {e}")
        return

    u_map = vel_data.get('u_map')
    v_map = vel_data.get('v_map')
    w_map = vel_data.get('w_map')

    if u_map is None:
        print("Error: Missing velocity keys.")
        return

    # --- 2. Load Render Data ---
    render_map = None
    if os.path.exists(render_pkl_path):
        print(f"Reading Render:   {render_pkl_path}")
        try:
            with open(render_pkl_path, 'rb') as f:
                r_data = pickle.load(f)
            render_map = r_data.get('render') 
        except Exception as e:
            print(f"Error loading render pickle: {e}")
    else:
        print(f"Warning: Render file not found at {render_pkl_path}")

    # --- 3. Stats & Setup ---
    # Velocity Limits
    lim_u = get_dynamic_limit(u_map, fallback=10.0)
    lim_v = get_dynamic_limit(v_map, fallback=10.0)
    lim_w = 2.0 

    print("-" * 40)
    print(f" -> U Stats: min={np.nanmin(u_map):.2f}, max={np.nanmax(u_map):.2f} | Lim: +/-{lim_u:.1f}")
    print(f" -> V Stats: min={np.nanmin(v_map):.2f}, max={np.nanmax(v_map):.2f} | Lim: +/-{lim_v:.1f}")
    print(f" -> W Stats: min={np.nanmin(w_map):.2f}, max={np.nanmax(w_map):.2f} | Lim: +/-{lim_w:.1f}")
    
    if render_map is not None:
        print(f" -> Render : min={np.min(render_map):.4f}, max={np.max(render_map):.4f}")
    print("-" * 40)

    # Ensure output directory exists
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Setup Colormaps
    vel_cmap = plt.cm.jet.copy()
    vel_cmap.set_bad(color='black')
    
    # --- 4. Save Raw Images (Default Origin = Upper/Top-Left) ---
    base_name_no_ext = os.path.splitext(os.path.basename(save_path))[0]
    print("Saving raw images...")

    # Save U
    plt.imsave(os.path.join(output_dir, f"{base_name_no_ext}_U_raw.png"), 
               u_map, cmap=vel_cmap, vmin=-lim_u, vmax=lim_u)
    
    # Save V
    plt.imsave(os.path.join(output_dir, f"{base_name_no_ext}_V_raw.png"), 
               v_map, cmap=vel_cmap, vmin=-lim_v, vmax=lim_v)
    
    # Save W
    plt.imsave(os.path.join(output_dir, f"{base_name_no_ext}_W_raw.png"), 
               w_map, cmap=vel_cmap, vmin=-lim_w, vmax=lim_w)

    # Save Render (If available)
    if render_map is not None:
        plt.imsave(os.path.join(output_dir, f"{base_name_no_ext}_Render_raw.png"), 
                   render_map, cmap='gray') 

    print("Done.")

if __name__ == "__main__":
    # --- PATHS ---
    
    # 1. The Velocity File
    vel_file = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(U,V,W_vel_maps)/0000002000/sample_004_time_0_view_0.pkl"
    
    # 2. The Render File 
    # (Update this path to point to a real file you created in the previous step)
    render_file = "/wdata_visl/danino/dataset_rendered_data/0000002000/sample_004_time_0_view_0.pkl"
    
    # 3. Output Prefix
    output_png_prefix = '/home/danino/PycharmProjects/pythonProject/data/output/check_combined.png'

    visualize_combined_data(vel_file, render_file, output_png_prefix)