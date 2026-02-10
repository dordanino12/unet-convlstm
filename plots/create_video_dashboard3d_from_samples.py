import os
import pickle
import numpy as np
import cv2
import glob
import pandas as pd
import ast
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. CSV & GEOMETRY HELPERS
# ==========================================
def load_camera_csv(csv_path):
    df = pd.read_csv(csv_path)
    camera_schedule = {}
    unique_times = sorted(df['utc time'].unique())

    for t in unique_times:
        current_time_rows = df[df['utc time'] == t]
        positions = []
        for _, row in current_time_rows.iterrows():
            raw_coords = ast.literal_eval(row['sat ENU coordinates [km]'])
            # Transformation: [-y, x, z] -> [x, y, z]
            x_km = -raw_coords[1]
            y_km = raw_coords[0]
            z_km = raw_coords[2]
            sat_enu_m = np.array([x_km, y_km, z_km]) * 1000.0
            positions.append(sat_enu_m)
        camera_schedule[t] = positions 

    return unique_times, camera_schedule

def create_3d_plot_img(sat_positions, look_at=[0, 0, 1500], figsize=(400, 400), fixed_bounds=None):
    """
    Generates a Modern 3D Scatter Plot.
    """
    w_in, h_in = figsize[0] / 100.0, figsize[1] / 100.0
    fig = plt.figure(figsize=(w_in, h_in), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    def to_km(val_m): return val_m / 1000.0
    
    # Cloud Center
    cx, cy, cz = to_km(look_at[0]), to_km(look_at[1]), to_km(look_at[2])
    ax.scatter(cx, cy, cz, c='#555555', s=250, marker='X', label='Cloud', depthshade=False)

    colors = ['#E74C3C', '#3498DB'] 
    
    for i, pos in enumerate(sat_positions):
        x_km = to_km(pos[0])
        y_km = to_km(pos[1])
        z_km = to_km(pos[2])
        color = colors[i % len(colors)]
        
        # Scatter
        ax.scatter(x_km, y_km, z_km, c=color, s=120, depthshade=False, edgecolors='white', linewidth=1.5)
        
        # Line to Cloud
        ax.plot([x_km, cx], [y_km, cy], [z_km, cz], c=color, linestyle='--', alpha=0.4, linewidth=1.5)
        
        # Label (offset so it doesn't overlap the marker)
        offset_z = (fixed_bounds[2] * 0.08) if fixed_bounds else 40
        ax.text(x_km, y_km, z_km + offset_z, f"S{i}", color=color, fontsize=10,
            fontweight='bold', ha='center', va='bottom')

    # Minimalist Axis Styling
    ax.set_xlabel('X (km)', fontsize=9, fontweight='bold', labelpad=5, color='#333333')
    ax.set_ylabel('Y (km)', fontsize=9, fontweight='bold', labelpad=5, color='#333333')
    ax.set_zlabel('Z (km)', fontsize=9, fontweight='bold', labelpad=5, color='#333333')
    ax.set_title('3D Geometry', fontsize=12, fontweight='bold', color='#333333')
    
    # Transparent panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Fixed Camera Angle
    ax.view_init(elev=25, azim=-45)

    if fixed_bounds:
        mx, my, mz = fixed_bounds
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-my, my)
        ax.set_zlim(0, mz) 
    
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_2d_plot_img(sat_positions, look_at=[0, 0, 1500], figsize=(400, 400), fixed_bounds=None, fill_axes=False):
    """
    Generates a 2D Y-Z side view scatter plot (Y horizontal, Z vertical).
    """
    w_in, h_in = figsize[0] / 100.0, figsize[1] / 100.0
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=100)

    def to_km(val_m):
        return val_m / 1000.0

    # Cloud Center (Y-Z)
    cy, cz = to_km(look_at[1]), to_km(look_at[2])
    ax.scatter(cy, cz, c='#555555', s=180, marker='X', label='Cloud', zorder=3)

    colors = ['#E74C3C', '#3498DB']
    for i, pos in enumerate(sat_positions):
        y_km = to_km(pos[1])
        z_km = to_km(pos[2])
        color = colors[i % len(colors)]
        ax.scatter(y_km, z_km, c=color, s=100, edgecolors='white', linewidth=1.2, zorder=4)
        ax.plot([y_km, cy], [z_km, cz], c=color, linestyle='--', alpha=0.4, linewidth=1.5, zorder=2)
        ax.annotate(
            f"S{i}",
            xy=(y_km, z_km),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center', va='bottom',
            color=color, fontsize=9, fontweight='bold'
        )

    ax.set_xlabel('Y (km)', fontsize=9, fontweight='bold', labelpad=8, color='#333333')
    ax.set_ylabel('Z (km)', fontsize=9, fontweight='bold', labelpad=8, color='#333333')
    ax.set_title('2D Geometry (Y-Z)', fontsize=12, fontweight='bold', color='#333333')
    ax.grid(True, linestyle='--', alpha=0.3)

    if fixed_bounds:
        _, my, mz = fixed_bounds
        ax.set_xlim(-my, my)
        ax.set_ylim(0, mz)

    if fill_axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    else:
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.9)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ==========================================
# 2. IMAGE PROCESSING
# ==========================================
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
    colored = (colored[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

# ==========================================
# 3. TEXT RENDERER
# ==========================================
def draw_modern_text(image_bgr, text_list):
    img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')

    def get_font(size, bold=False):
        try:
            if bold:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
            else:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()

    for text, (x, y), color, size, bold in text_list:
        font = get_font(size, bold)
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        padding = 4
        # Background Box
        draw.rectangle(
            (left - padding, top - padding, right + padding, bottom + padding), 
            fill=(30, 30, 30, 160) 
        )
        draw.text((x, y), text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 4. MAIN VIDEO GENERATOR
# ==========================================
def create_dashboard_3d_padded(
    render_root,
    velocity_root,
    csv_path,
    output_video_path, 
    sample_id="sample_000", 
    start_folder=2000,
    end_folder=2220,
    fps=5,
    geo_mode="3d"
):
    # --- A. Load Metadata ---
    print("Loading Satellite Data...")
    csv_times, sat_lookup = load_camera_csv(csv_path)
    num_csv_states = len(csv_times)

    # --- UPDATED: Calculate Global 3D Bounds ---
    print("Calculating Global 3D Bounds...")
    all_x, all_y, all_z = [], [], []
    for t in csv_times:
        for pos in sat_lookup[t]:
            all_x.append(abs(pos[0] / 1000.0))
            all_y.append(abs(pos[1] / 1000.0))
            all_z.append(pos[2] / 1000.0)
            
    if all_y:
        global_max_x = max(all_x) * 1.2
        global_max_y = max(all_y) * 1.2
        global_max_z = max(all_z) * 1.1
    else:
        global_max_x = 100; global_max_y = 100; global_max_z = 600
    
    fixed_limits_3d = (global_max_x, global_max_y, global_max_z)
    print(f"Fixed Limits -> X: +/-{global_max_x:.0f}km, Y: +/-{global_max_y:.0f}km, Z: {global_max_z:.0f}km")

    # --- B. Find Folders ---
    if not os.path.exists(render_root):
        print(f"Error: Render root not found {render_root}")
        return

    all_items = os.listdir(render_root)
    folder_map = {}
    for d in all_items:
        if d.isdigit() and os.path.isdir(os.path.join(render_root, d)):
            val = int(d)
            if start_folder <= val <= end_folder:
                folder_map[val] = d

    sorted_keys = sorted(folder_map.keys())
    print(f"Found {len(sorted_keys)} folders. Processing '{sample_id}'...")

    video = None

    for folder_idx, key in enumerate(tqdm(sorted_keys, desc="Generating Video")):
        folder_real_name = folder_map[key]
        csv_ptr = folder_idx % num_csv_states
        target_time_val = csv_times[csv_ptr]
        
        # Paths
        render_dir = os.path.join(render_root, folder_real_name)
        vel_dir = os.path.join(velocity_root, folder_real_name)
        
        files_v0 = glob.glob(os.path.join(render_dir, f"{sample_id}_*_view_0.pkl"))
        files_v1 = glob.glob(os.path.join(render_dir, f"{sample_id}_*_view_1.pkl"))
        vel_files_v0 = glob.glob(os.path.join(vel_dir, f"{sample_id}_*_view_0.pkl"))
        vel_files_v1 = glob.glob(os.path.join(vel_dir, f"{sample_id}_*_view_1.pkl"))
        
        if not files_v0 or not files_v1:
            continue
            
        try:
            # --- Load Data ---
            with open(files_v0[0], 'rb') as f:
                img0 = pickle.load(f)['render']
            with open(files_v1[0], 'rb') as f:
                img1 = pickle.load(f)['render']

            w0 = np.zeros_like(img0)
            if vel_files_v0:
                with open(vel_files_v0[0], 'rb') as f:
                    w0 = pickle.load(f)['w_map']
            w1 = np.zeros_like(img1)
            if vel_files_v1:
                with open(vel_files_v1[0], 'rb') as f:
                    w1 = pickle.load(f)['w_map']

            # --- Process Images ---
            def norm_gray_with_gamma(img, gamma=0.5):
                # 1. Handle NaNs
                img = np.nan_to_num(img, nan=0.0)

                # 2. Linear Normalization (0.0 to 1.0 float)
                mi, ma = np.min(img), np.max(img)
                if ma > mi:
                    norm_float = (img - mi) / (ma - mi)
                else:
                    norm_float = np.zeros_like(img)

                # 3. Apply Gamma Correction
                # Gamma < 1.0 will make the image lighter (expands shadows)
                # Gamma > 1.0 will make the image darker
                gamma_corrected = np.power(norm_float, gamma)

                # 4. Convert to 0-255 uint8
                norm_uint8 = (gamma_corrected * 255).astype(np.uint8)

                return cv2.cvtColor(norm_uint8, cv2.COLOR_GRAY2BGR)

            render0 = norm_gray_with_gamma(img0)
            render1 = norm_gray_with_gamma(img1)
            w0_col = apply_jet_colormap(w0)
            w1_col = apply_jet_colormap(w1)

            # --- Layout ---
            col0 = np.vstack([render0, w0_col])
            col1 = np.vstack([render1, w1_col])
            h_col, w_col, _ = col0.shape
            separator = np.ones((h_col, 20, 3), dtype=np.uint8) * 230 

            # Geometry (Now 3D)
            sat_positions = sat_lookup[target_time_val]
            plot_w = int(h_col * 0.8)
            
            # --- CALLING GEO PLOT ---
            if geo_mode == "2d":
                img_geo = create_2d_plot_img(
                    sat_positions,
                    figsize=(plot_w, h_col),
                    fixed_bounds=fixed_limits_3d
                )
            else:
                img_geo = create_3d_plot_img(
                    sat_positions,
                    figsize=(plot_w, h_col),
                    fixed_bounds=fixed_limits_3d
                )
            
            if img_geo.shape[0] != h_col:
                img_geo = cv2.resize(img_geo, (img_geo.shape[1], h_col))

            combined_frame = np.hstack([col0, separator, col1, separator, img_geo])

            # --- Draw Labels ---
            labels_to_draw = []
            w_r = render0.shape[1]
            h_r = render0.shape[0]
            
            # 1. Global Info
            info_text = f"Folder: {key} | Time: {target_time_val}"
            labels_to_draw.append((info_text, (10, 10), (255, 255, 255), 16, True))
            
            # 2. Row Labels
            labels_to_draw.append(("Render Image", (10, 40), (200, 200, 200), 14, True))
            labels_to_draw.append(("W Map", (10, h_r + 10), (200, 200, 200), 14, True))

            # 3. Satellite View Labels (Bottom Left)
            labels_to_draw.append(("S0", (10, h_r - 30), (52, 152, 219), 16, True)) 
            s1_x_start = w_r + 20 + 10
            labels_to_draw.append(("S1", (s1_x_start, h_r - 30), (52, 152, 219), 16, True))
            
            # 4. Geometry Header (Optional/Empty)
            geo_x_start = w_r * 2 + 40 + 20
            # labels_to_draw.append(("", (geo_x_start, 20), (231, 76, 60), 16, True))

            content_frame = draw_modern_text(combined_frame, labels_to_draw)

            # --- Global Padding ---
            PADDING = 40
            h_cont, w_cont, c_cont = content_frame.shape
            padded_frame = np.full((h_cont + 2*PADDING, w_cont + 2*PADDING, c_cont), (50, 50, 50), dtype=np.uint8)
            padded_frame[PADDING:PADDING+h_cont, PADDING:PADDING+w_cont] = content_frame

            # --- Write Video ---
            if video is None:
                h_pad, w_pad, _ = padded_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(output_video_path, fourcc, fps, (w_pad, h_pad))
            
            video.write(padded_frame)

        except Exception as e:
            tqdm.write(f"Error {key}: {e}")

    if video is not None:
        video.release()
        print(f"Saved: {output_video_path}")
    else:
        print("No video created.")

if __name__ == "__main__":
    # --- CONFIG ---
    render_root = "/wdata_visl/danino/dataset_rendered_data_spp8192_g085/"
    velocity_root = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(vel_maps)/"
    csv_file = '/home/danino/PycharmProjects/pythonProject/data/Dor_2satellites_overpass.csv'
    
    output_video = "/home/danino/PycharmProjects/pythonProject/data/output/dashboard_3d.mp4"
    target_sample = "sample_010"
    
    create_dashboard_3d_padded(
        render_root, 
        velocity_root, 
        csv_file,
        output_video,
        sample_id=target_sample,
        start_folder=2000,
        end_folder=2600,
        fps=2
    )