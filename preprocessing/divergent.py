import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def check_physics_full_breakdown(pkl_path, voxel_res=20.0, save_dir=None):
    """
    Plots Raw Fields (U, V, W) and their derivatives.
    Layout: 2 Rows x 4 Columns.
    """
    
    # 1. Load Data
    if not os.path.exists(pkl_path):
        print(f"Error: File not found {pkl_path}")
        return

    print(f"Loading: {os.path.basename(pkl_path)}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Handle Key Variations
    keys = data.keys()
    try:
        k_u = 'U' if 'U' in keys else 'u'
        k_v = 'V' if 'V' in keys else 'v'
        k_w = 'W' if 'W' in keys else 'w'
        k_beta = 'beta_ext' if 'beta_ext' in keys else 'beta'

        vol_u = data[k_u]
        vol_v = data[k_v]
        vol_w = data[k_w]
        vol_beta = data[k_beta]
    except KeyError:
        print(f"Error: Missing keys. Found: {list(keys)}")
        return

    # 2. Calculate Gradients
    print("Calculating 3D gradients...")
    
    grads_u = np.gradient(vol_u, voxel_res)
    du_dx = grads_u[2]
    
    grads_v = np.gradient(vol_v, voxel_res)
    dv_dy = grads_v[1]
    
    grads_w = np.gradient(vol_w, voxel_res)
    dw_dz = grads_w[0]

    # 3. Calculate Divergence
    divergence_3d = du_dx + dv_dy + dw_dz

    # 4. Stats
    full_div_flat = divergence_3d.flatten()
    mean_div = np.mean(np.abs(full_div_flat))
    print(f" -> Mean Abs Divergence: {mean_div:.6f}")

    # 5. Prepare Saving
    if save_dir is None:
        save_dir = os.path.dirname(pkl_path)
    
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]

    # ==========================================
    # IMAGE 1: THE 8-PANEL MAP (2x4 Grid)
    # ==========================================
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))

    # Find best Z slice
    cloud_mask_for_slice = vol_beta > 0.001
    if np.sum(cloud_mask_for_slice) == 0:
        best_z = vol_beta.shape[0] // 2 
    else:
        z_counts = np.sum(cloud_mask_for_slice, axis=(1, 2))
        best_z = np.argmax(z_counts)
    
    print(f"Visualizing Slice Z={best_z}")

    # --- ROW 1: RAW FIELDS ---
    
    # 1. Cloud Density
    axes[0, 0].imshow(vol_beta[best_z], cmap='gray')
    axes[0, 0].set_title(f'Cloud Density (Z={best_z})')
    
    # 2. U Velocity
    lim_u = np.percentile(np.abs(vol_u), 99)
    im_u = axes[0, 1].imshow(vol_u[best_z], cmap='seismic', vmin=-lim_u, vmax=lim_u)
    axes[0, 1].set_title(f'U Velocity (East-West)')
    plt.colorbar(im_u, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. V Velocity
    lim_v = np.percentile(np.abs(vol_v), 99)
    im_v = axes[0, 2].imshow(vol_v[best_z], cmap='seismic', vmin=-lim_v, vmax=lim_v)
    axes[0, 2].set_title(f'V Velocity (North-South)')
    plt.colorbar(im_v, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. W Velocity
    lim_w = np.percentile(np.abs(vol_w), 99)
    im_w = axes[0, 3].imshow(vol_w[best_z], cmap='seismic', vmin=-lim_w, vmax=lim_w)
    axes[0, 3].set_title(f'W Velocity (Vertical)')
    plt.colorbar(im_w, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # --- ROW 2: DERIVATIVES ---
    
    diff_cmap = 'PRGn' 

    # 5. dU / dx
    lim_du = np.percentile(np.abs(du_dx), 99)
    im_du = axes[1, 1].imshow(du_dx[best_z], cmap=diff_cmap, vmin=-lim_du, vmax=lim_du)
    axes[1, 1].set_title(f'dU / dx (Stretch X)')
    plt.colorbar(im_du, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. dV / dy
    lim_dv = np.percentile(np.abs(dv_dy), 99)
    im_dv = axes[1, 2].imshow(dv_dy[best_z], cmap=diff_cmap, vmin=-lim_dv, vmax=lim_dv)
    axes[1, 2].set_title(f'dV / dy (Stretch Y)')
    plt.colorbar(im_dv, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 7. dW / dz
    lim_dw = np.percentile(np.abs(dw_dz), 99)
    im_dw = axes[1, 3].imshow(dw_dz[best_z], cmap=diff_cmap, vmin=-lim_dw, vmax=lim_dw)
    axes[1, 3].set_title(f'dW / dz (Stretch Z)')
    plt.colorbar(im_dw, ax=axes[1, 3], fraction=0.046, pad=0.04)

    # 8. Total Divergence
    lim_div = np.percentile(np.abs(divergence_3d), 99)
    im_div = axes[1, 0].imshow(divergence_3d[best_z], cmap='coolwarm', vmin=-lim_div, vmax=lim_div)
    axes[1, 0].set_title(f'Total Divergence')
    plt.colorbar(im_div, ax=axes[1, 0], fraction=0.046, pad=0.04)

    plt.suptitle(f"3D Physics Full Breakdown: {base_name}", fontsize=16)
    plt.tight_layout()
    
    # SAVE MAPS
    save_path_map = os.path.join(save_dir, f"{base_name}_physics_full.png")
    plt.savefig(save_path_map)
    plt.close(fig)
    print(f"Saved Maps: {save_path_map}")

    # ==========================================
    # IMAGE 2: THE HISTOGRAM
    # ==========================================
    plt.figure(figsize=(8, 4))
    
    plt.hist(full_div_flat, bins=100, range=(-lim_div*2, lim_div*2), color='blue', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Histogram of Total Divergence")
    plt.xlabel("Divergence")
    plt.ylabel("Count")
    
    # SAVE HISTOGRAM
    save_path_hist = os.path.join(save_dir, f"{base_name}_div_hist.png")
    plt.savefig(save_path_hist)
    plt.close() 
    print(f"Saved Hist: {save_path_hist}")

if __name__ == "__main__":
    # --- INPUT ---
    file_path = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/0000015040/sample_022.pkl"
    
    # --- OUTPUT ---
    output_folder = "/home/danino/PycharmProjects/pythonProject/data/output/"
    
    check_physics_full_breakdown(file_path, voxel_res=20.0, save_dir=output_folder)