import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use Agg backend for non-interactive plotting (faster/safer for loops)
matplotlib.use('Agg')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
NPZ_PATH = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_500m_slices_w.npz"
OUTPUT_DIR = "/home/danino/PycharmProjects/pythonProject/plots/mask_tuning/"
NUM_VIDEOS_TO_GENERATE = 1
FPS = 1

# --- TUNING PARAMETER ---
MASK_THRESHOLD = 1.1  # The Red line on the graph

# ---------------------------------------------------------
# HELPER 1: Satellite Image Normalization
# ---------------------------------------------------------
def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0)
    img = img.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val - min_val > 1e-5:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)

# ---------------------------------------------------------
# HELPER 2: Jet Colormap
# ---------------------------------------------------------
def apply_jet_colormap(data):
    mask = np.isnan(data)
    clean_data = np.nan_to_num(data, nan=0.0)
    limit = np.percentile(np.abs(clean_data), 99)
    if limit == 0: limit = 1
    norm_data = np.clip(clean_data, -limit, limit)
    norm_data = (norm_data + limit) / (2 * limit) 
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='black')
    colored = cmap(norm_data) 
    colored[mask] = [0, 0, 0, 1] 
    colored_bgr = (colored[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored_bgr, cv2.COLOR_RGB2BGR)
    return colored_bgr, limit

# ---------------------------------------------------------
# HELPER 3: Create Color Bar
# ---------------------------------------------------------
def create_colorbar(height, width, limit):
    gradient = np.linspace(1, 0, height)
    gradient = np.tile(gradient[:, np.newaxis], (1, width))
    cmap = plt.cm.jet
    colored_bar = cmap(gradient)
    colored_bar = (colored_bar[:, :, :3] * 255).astype(np.uint8)
    colored_bar = cv2.cvtColor(colored_bar, cv2.COLOR_RGB2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    
    cv2.putText(colored_bar, f"{limit:.1f}", (5, 15), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(colored_bar, "0.0", (5, height // 2), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(colored_bar, f"-{limit:.1f}", (5, height - 5), font, font_scale, color, 1, cv2.LINE_AA)
    
    return colored_bar

# ---------------------------------------------------------
# HELPER 4: Create Histogram Plot (Left Image Only)
# ---------------------------------------------------------
def create_histogram_plot(img0, threshold, target_height, target_width):
    """
    Creates a histogram of pixel values from ONLY the left satellite image
    and draws a red line at the threshold.
    """
    # Use only img0 (Left image) data
    data = img0.flatten()
    data = np.nan_to_num(data, nan=0.0)
    
    # Filter out absolute 0s to make the plot readable (ignore background)
    data_nonzero = data[data > 1e-5]

    fig, ax = plt.subplots(figsize=(target_width/100, target_height/100), dpi=100)
    
    if len(data_nonzero) > 0:
        ax.hist(data_nonzero, bins=50, color='gray', alpha=0.7, log=True)
    
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Thresh {threshold}')
    
    # Updated title to be specific
    ax.set_title("Sat 0 Dist (Log Scale)", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig.canvas.draw()
    img_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) 
    
    img_plot = cv2.resize(img_plot, (target_width, target_height))
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
    
    return img_plot

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    if not os.path.exists(NPZ_PATH):
        print(f"Error: Dataset not found at {NPZ_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    data = np.load(NPZ_PATH)
    X_all = data['X'] 
    Y_all = data['Y'] 
    
    total_seqs = X_all.shape[0]
    seq_len = X_all.shape[1]
    height, width = X_all.shape[3], X_all.shape[4]
    
    cbar_width = 40
    hist_width = width * 2 
    
    print(f"Loaded {total_seqs} sequences.")

    indices = np.random.choice(total_seqs, min(NUM_VIDEOS_TO_GENERATE, total_seqs), replace=False)

    print(f"Generating {len(indices)} videos...")

    for idx in tqdm(indices, desc="Rendering"):
        
        # Width = 3 images + 1 mask + histogram + color bar
        total_width = (width * 4) + hist_width + cbar_width
        
        video_filename = os.path.join(OUTPUT_DIR, f"seq_{idx:04d}_thresh_{MASK_THRESHOLD}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(video_filename, fourcc, FPS, (total_width, height))

        for t in range(seq_len):
            # 1. Get Data
            img_v0 = X_all[idx, t, 0]
            img_v1 = X_all[idx, t, 1]
            map_w  = Y_all[idx, t, 0]

            # 2. Process Visuals
            view0_vis = normalize_image(img_v0)
            view0_vis = cv2.cvtColor(view0_vis, cv2.COLOR_GRAY2BGR)
            
            view1_vis = normalize_image(img_v1)
            view1_vis = cv2.cvtColor(view1_vis, cv2.COLOR_GRAY2BGR)

            target_vis, limit_val = apply_jet_colormap(map_w)
            
            # 3. Create Binary Mask Visualization
            raw_0 = np.nan_to_num(img_v0, nan=0.0)
            raw_1 = np.nan_to_num(img_v1, nan=0.0)
            
            # --- CALCULATE MIN VALUES ---
            min_val_0 = np.min(raw_0)
            min_val_1 = np.min(raw_1)
            # ----------------------------
            
            binary_mask = ((raw_0 > MASK_THRESHOLD) | (raw_1 > MASK_THRESHOLD)).astype(np.uint8) * 255
            mask_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

            # 4. Generate Histogram (Passing ONLY raw_0 now)
            hist_vis = create_histogram_plot(raw_0, MASK_THRESHOLD, height, hist_width)

            # 5. Generate Color Bar
            cbar_vis = create_colorbar(height, cbar_width, limit_val)

            # 6. Labels
            # Sat View 0 + Min Value
            cv2.putText(view0_vis, f"S0 Min:{min_val_0:.4f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Sat View 1 + Min Value
            cv2.putText(view1_vis, f"S1 Min:{min_val_1:.4f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.putText(target_vis, "Target W", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(mask_vis, f"Mask > {MASK_THRESHOLD}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 7. Stack
            frame = np.hstack((view0_vis, view1_vis, target_vis, mask_vis, hist_vis, cbar_vis))
            out.write(frame)

        out.release()

    print("Done!")

if __name__ == "__main__":
    main()