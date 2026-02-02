# Original imports
from render import MitsubaRenderer
import numpy as np
from PIL import Image
import os
import mitsuba as mi
import pickle

# --- Imports for 3D Plotting ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
import matplotlib.ticker as ticker

# Global font settings MUST be set FIRST before creating any figures
mpl.rcParams.update({
    'font.size': 60,              # base font size (bigger)
    'axes.titlesize': 64,         # axes title size
    'axes.labelsize': 56,         # X/Y label size
    'xtick.labelsize': 52,        # x tick labels
    'ytick.labelsize': 52,        # y tick labels
    'legend.fontsize': 52,        # legend if used
    'figure.titlesize': 68,       # figure suptitle
    'figure.dpi': 300,
    'savefig.dpi': 150,
    # PDF font embedding to preserve sizes
    'pdf.fonttype': 42,           # embed TrueType fonts
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# --- Helper function for axes ---
def set_centered_meter_axis(ax, height, width, m_per_pixel):
    """
    Set X and Y axes in meters with (0,0) at the center.
    Fix axis limits to [-1280, 1280] meters.
    Show intermediate ticks to avoid overlap.
    Add tick lines connecting labels to the image.
    """
    # Fixed meter range
    ax.set_xlim(-1280, 1280)
    ax.set_ylim(1280, -1280)  # invert Y so top is positive in image coordinates

    # Build ticks WITHOUT first and last values to prevent overlap
    tick_vals = np.array([-1100, -640, 0, 640, 1100])
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_xticklabels([f"{int(v)}" for v in tick_vals], fontsize=48, fontweight='bold')
    ax.set_yticklabels([f"{int(v)}" for v in tick_vals], fontsize=48, fontweight='bold')

    # Enable tick marks with custom styling
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',  # ticks point outward from plot
        length=14,        # length of tick lines
        width=4,          # width of tick lines
        color='black',    # color of tick lines
        labelsize=48
    )

    ax.set_xlabel('X [m]', fontsize=52, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=52, fontweight='bold')


# --- 1. Define Your Input Data Paths ---
# You must have these files.
csv_file = '/home/danino/PycharmProjects/pythonProject/data/Dor_2satellites_overpass.csv'
#csv_file = '/home/danino/PycharmProjects/pythonProject/data/debug.csv'

#cloud_data_file = '/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_3D/BOMEX_512x512x200_20m_20m_1s_512_0000005200_1_5'  # This is the pkl file you have
#cloud_data_file= "/wdata_visl/danino/dataset_256x256x200_overlap_64_stride_7x7_split(beta,U,V,W)/0000002000/sample_001.pkl"
cloud_data_file= '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/0000005920/sample_012.pkl'
#cloud_data_file= "/wdata_visl/danino/dataset_512x512x200_overlap_64_stride_7x7_split(beta,U,V,W)/0000002000/sample_000.pkl"

output_vol_file = 'temp/my_cloud.vol'  # A temporary file this script will create
output_image_dir = '/home/danino/PycharmProjects/pythonProject/data/output'
# Create temp directory if it doesn't exist
os.makedirs('temp', exist_ok=True)

# Define which rows from your CSV to use
overpass_indices = [0, 1, 3, 5, 7, 9]
overpass_indices = [11, 13, 15, 17, 19, 21]
overpass_indices = [0, 1, 9, 17, 19, 21]
overpass_indices = [9,10,11,12]


# overpass_indices = [0, 3, 6, 9, 12, 15]
# overpass_indices = [6, 7, 8, 9, 10, 11]
# overpass_indices = [9, 10, 11, 15, 16, 17]
# overpass_indices = [0, 1, 2, 30, 31, 32]
# overpass_indices = [12,13,14, 15, 16, 17]
# overpass_indices = [7, 8,9]
# overpass_indices = [0, 1, 2]


# --- 2. Set Up the Renderer Parameters ---
renderer_params = {
    'overpass_csv': csv_file,
    'overpass_indices': overpass_indices,
    'spp': 512,
    'g_value': 0,
    'cloud_width': 128,
    'image_res': 256,
    'fov' : 0.25,
    'voxel_res': 0.02,
    'scene_scale': 1000.0,
    'cloud_zrange': [0.0, 4.0],
    'satellites': 2,
    'timestamps': 2,
    'pad_image': False,
    'dynamic_emitter': True,
    'centralize_cloud': True,
    'bitmaps_required': False,
    'vol_path': output_vol_file
}

print("Initializing MitsubaRenderer...")
# --- 3. Create an instance of the renderer ---
renderer = MitsubaRenderer(**renderer_params)

print("Reading CSV and setting up cameras...")
# --- 4. Run the setup methods ---
renderer.read_overpass_csv()
renderer.camera_params()
renderer.create_sensors()

print(f"Loading cloud data from '{cloud_data_file}'...")
# --- 5. Load your cloud data from the .pkl file ---
renderer.write_vol_file(sample_path=cloud_data_file,
                        vol_path=output_vol_file,
                        param_type='beta_ext',  # This must match a key in your pkl file!
                        sample_ext='pkl')

print("Setting up the 3D scene(s)...")
# --- 6. Build the 3D scene(s) in memory ---
renderer.set_scenes()

print("Rendering... This may take a while.")
# --- 7. Run the render! ---
tensor_stacks, bitmap_stacks = renderer.render_scenes()

print("Rendering complete!")

# --- 8. Do something with the output ---
# (Your original 2D plotting code is here)
if not tensor_stacks or tensor_stacks[0] is None:
    print("No images were rendered.")
else:
    print("Processing images for plotting...")
    n_timestamps = renderer.timestamps
    n_satellites = renderer.satellites

    GAMMA_VAL = 0.5

    # Determine resolution for axis scaling
    image_res =renderer_params['image_res']
    if image_res == 128:
        m_per_pixel = 20
    elif image_res == 256:
        m_per_pixel = 10
    else:
        m_per_pixel = 20  # default fallback

    # Loop through all results and save individual PDFs
    print("Saving individual render PDFs...")
    for t in range(n_timestamps):
        for s in range(n_satellites):
            # Get the raw image data
            float_data = tensor_stacks[t][s]

            # 1. Normalize to 0-1
            if float_data.max() > 0:
                normalized_data = float_data / float_data.max()
            else:
                normalized_data = float_data

            # 2. Apply Gamma Correction
            corrected_data = np.power(normalized_data, GAMMA_VAL)

            # Get dimensions
            H, W = corrected_data.shape

            # Compute extent in meters so that (0,0) is centered
            half_w_m = (W * m_per_pixel) / 2.0
            half_h_m = (H * m_per_pixel) / 2.0
            extent_m = [-half_w_m, half_w_m, half_h_m, -half_h_m]

            # Create figure for individual PDF
            fig_single, ax_single = plt.subplots(figsize=(12, 12), dpi=150)
            im_single = ax_single.imshow(corrected_data, cmap='gray', extent=extent_m, interpolation='nearest')
            ax_single.set_title('Render Cloud Image', fontsize=56, fontweight='bold', pad=40)
            set_centered_meter_axis(ax_single, H, W, m_per_pixel)

            # Add colorbar
            cbar_single = plt.colorbar(im_single, ax=ax_single, fraction=0.046, pad=0.04)
            cbar_single.ax.tick_params(labelsize=48)

            # Save PDF
            fig_single.set_size_inches(20, 20)
            plt.subplots_adjust(left=0.17, right=0.92, top=0.95, bottom=0.08)
            filename = f"render_t{t:02d}_s{s:02d}.pdf"
            filepath = os.path.join(output_image_dir, filename)
            plt.savefig(filepath, dpi=150)
            plt.close(fig_single)
            print(f"Saved: {filepath}")

    # Create combined overview plot
    print("Creating combined overview plot...")
    if n_timestamps == 1 and n_satellites == 1:
        fig, axes = plt.subplots(1, 1, figsize=(20, 20))
        axes = np.array([[axes]])  # Make it 2D for consistent indexing
    elif n_timestamps == 1:
        fig, axes = plt.subplots(n_timestamps, n_satellites, figsize=(n_satellites * 20, n_timestamps * 20))
        axes = np.array([axes])  # Make it 2D
    elif n_satellites == 1:
        fig, axes = plt.subplots(n_timestamps, n_satellites, figsize=(n_satellites * 20, n_timestamps * 20))
        axes = axes.reshape(-1, 1)  # Make it 2D
    else:
        fig, axes = plt.subplots(n_timestamps, n_satellites,
                                 figsize=(n_satellites * 20, n_timestamps * 20),
                                 squeeze=False)

    fig.suptitle('All Rendered Satellite Images', fontsize=68, fontweight='bold')

    for t in range(n_timestamps):
        for s in range(n_satellites):
            float_data = tensor_stacks[t][s]
            if float_data.max() > 0:
                normalized_data = float_data / float_data.max()
            else:
                normalized_data = float_data

            # Apply gamma correction
            corrected_data = np.power(normalized_data, GAMMA_VAL)

            # Get dimensions
            H, W = corrected_data.shape
            half_w_m = (W * m_per_pixel) / 2.0
            half_h_m = (H * m_per_pixel) / 2.0
            extent_m = [-half_w_m, half_w_m, half_h_m, -half_h_m]

            ax = axes[t, s]
            im = ax.imshow(corrected_data, cmap='gray', extent=extent_m, interpolation='nearest')
            ax.set_title(f'Timestamp {t}, Satellite {s}', fontsize=56, fontweight='bold', pad=20)
            set_centered_meter_axis(ax, H, W, m_per_pixel)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=48)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)

    # Save combined overview as PDF
    overview_path = os.path.join(output_image_dir, "render_overview.pdf")
    plt.savefig(overview_path, dpi=150)
    print(f"Saved combined overview: {overview_path}")
    plt.close(fig)

# ==============================================================================
# 8.5 NEW SECTION: LOAD AND SHOW TARGET IMAGE
# ==============================================================================
print(f"Loading 'target' image directly from {cloud_data_file}...")
try:
    cloud_data_file = cloud_data_file + '.pkl'
    with open(cloud_data_file, 'rb') as f:
        data_dict = pickle.load(f)

    if 'target' in data_dict:
        target_img = np.ma.getdata(data_dict['target'])

        # Ensure dimensions are correct (remove singleton dims like 1x128x128 -> 128x128)
        #target_img = np.squeeze(target_img)

        plt.figure(figsize=(6, 6))
        plt.imshow(target_img, cmap='gray')
        plt.title(f"Target Image (Ground Truth)\nShape: {target_img.shape}")
        plt.axis('off')
        print("Target image loaded and ready to display.")
    else:
        print(f"Warning: 'target' key not found in the .pkl file. Available keys: {list(data_dict.keys())}")

except Exception as e:
    print(f"Error loading target image: {e}")

# ==============================================================================
first_tensor_data = tensor_stacks[0][0]
print(f"Shape of the first tensor: {first_tensor_data.shape}")

# --- 9. 3D Debug Plot ---
print("Generating 3D debug plot...")


def plot_scene_geometry(renderer, zoom_on_cloud=True):
    """
    Plots the 3D positions of the cloud, cameras, and sun direction.

    Args:
        renderer: The MitsubaRenderer instance.
        zoom_on_cloud (bool): If True, zooms the plot to the cloud and origin.
                              If False, shows the full scene including satellites.
    """

    # --- Helper function to draw a cube ---
    def plot_cube(center, size, ax, color='cyan', alpha=0.1, edge_color='r'):
        """Plots a 3D cube."""
        half_size = size / 2.0
        x, y, z = center

        # 8 vertices of the cube
        vertices = np.array([
            [x - half_size, y - half_size, z - half_size],
            [x + half_size, y - half_size, z - half_size],
            [x + half_size, y + half_size, z - half_size],
            [x - half_size, y + half_size, z - half_size],
            [x - half_size, y - half_size, z + half_size],
            [x + half_size, y - half_size, z + half_size],
            [x + half_size, y + half_size, z + half_size],
            [x - half_size, y + half_size, z + half_size]
        ])

        # Define faces by vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[3], vertices[7], vertices[4]]  # Left
        ]

        # Add faces to the plot
        ax.add_collection3d(Poly3DCollection(faces,
                                             facecolors=color, linewidths=1, edgecolors=edge_color, alpha=alpha))

    # --- Start 3D Plot ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Override global font settings for 3D plot (they're too large from PDF settings)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.zaxis.label.set_size(12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # 1. Plot the Cloud Volume
    cloud_size = renderer.W  # e.g., 2.56 km
    cloud_center = [0, 0, renderer.cloud_zcenter * 2]  # e.g., [0, 0, 4.0]
    plot_cube(cloud_center, cloud_size, ax, color='cyan', alpha=0.1, edge_color='green')

    # 2. Plot the Cameras (Satellites)
    # Coordinate swap from your create_sensors() function!
    # In create_sensors: origin=[self.sat_Wy[i], self.sat_Wx[i], self.sat_H[i]]
    cam_x = np.array(renderer.sat_Wy)  # World X corresponds to CSV Wy
    cam_y = np.array(renderer.sat_Wx)  # World Y corresponds to CSV Wx
    cam_z = np.array(renderer.sat_H)  # World Z corresponds to CSV H

    ax.scatter(cam_x, cam_y, cam_z, c='blue', marker='^', s=100, label='Satellites (Cameras)')

    # 3. Plot Camera View Lines
    # --- THIS IS THE FIX ---
    # The target point is now the cloud's center, matching the fix in create_sensors
    target_point = cloud_center
    # --- END OF FIX ---

    for i in range(len(cam_x)):
        ax.plot([cam_x[i], target_point[0]],
                [cam_y[i], target_point[1]],
                [cam_z[i], target_point[2]],
                c='blue', linestyle='--', alpha=0.5)

    # 4. Plot the Sun Direction
    try:
        # Get the direction vector from the first timestamp's scene dict
        sun_dir_raw = np.array(renderer.scenes_dict[0]['emitter']['direction'])
        sun_direction = sun_dir_raw / np.linalg.norm(sun_dir_raw)  # Normalize

        # The sun is a 'directional' emitter, so it's infinitely far away.
        # We draw an arrow starting far away and pointing TO the cloud center.
        sun_arrow_origin_offset = 100  # How far from the cloud center to start the arrow for visualization

        # Calculate a point far away along the sun's inverse direction
        sun_start_point = np.array(cloud_center) - sun_direction * sun_arrow_origin_offset

        # Draw the arrow from the start point to the cloud center
        ax.quiver(sun_start_point[0], sun_start_point[1], sun_start_point[2],
                  sun_direction[0], sun_direction[1], sun_direction[2],
                  length=sun_arrow_origin_offset, color='orange', linewidth=3, arrow_length_ratio=0.1,
                  label='Sun Direction')

        # Also plot a point for the sun's "effective position" if we want to visualize it
        # as a source (even though it's directional)
        ax.scatter([sun_start_point[0]], [sun_start_point[1]], [sun_start_point[2]],
                   c='yellow', marker='*', s=300, label='Sun Source (Illustrative)')

    except Exception as e:
        print(f"Could not plot sun direction: {e}")

    # 5. Plot the Origin
    # --- THIS IS THE FIX ---
    # The label is now correct, it's just the origin, not the camera target.
    ax.scatter([0], [0], [0], c='red', marker='o', s=100, label='World Origin (0,0,0)')
    # --- END OF FIX ---

    # 6. Set Labels and Title
    ax.set_xlabel('X [km] (from CSV: sat_Wy)', fontsize=12)
    ax.set_ylabel('Y [km] (from CSV: sat_Wx)', fontsize=12)
    ax.set_zlabel('Z [km] (Altitude)', fontsize=12)

    if zoom_on_cloud:
        ax.set_title('3D Scene Geometry (Zoomed on Cloud)', fontsize=14, fontweight='bold')
        # Set plot limits to zoom in on the origin/cloud
        # A slightly larger range than just the cloud to see context
        max_coord = max(cloud_size / 2, np.abs(cloud_center[2])) + 5  # Adjust 5 for padding
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        ax.set_zlim([-max_coord, max_coord * 2])  # Z-axis might need more range for cloud altitude
    else:
        ax.set_title('3D Scene Geometry (Global View)', fontsize=14, fontweight='bold')
        # Automatically determine limits to fit everything
        all_x = np.concatenate(
            ([0, cloud_center[0]], cam_x, [sun_start_point[0]] if 'sun_start_point' in locals() else []))
        all_y = np.concatenate(
            ([0, cloud_center[1]], cam_y, [sun_start_point[1]] if 'sun_start_point' in locals() else []))
        all_z = np.concatenate(
            ([0, cloud_center[2]], cam_z, [sun_start_point[2]] if 'sun_start_point' in locals() else []))

        # Add buffer for better visualization
        x_min, x_max = all_x.min() - 50, all_x.max() + 50
        y_min, y_max = all_y.min() - 50, all_y.max() + 50
        z_min, z_max = all_z.min() - 50, all_z.max() + 50

        # Ensure cloud is visible if satellites are very far
        z_min = min(z_min, -50)  # Ensure a floor is seen
        z_max = max(z_max, 600)  # Ensure satellites are seen

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

    ax.legend(fontsize=10, loc='best')

    # Set equal aspect ratio if possible (can make plots look weird if ranges are very different)
    # ax.set_box_aspect([1,1,1]) # Newer matplotlib version

    print(
        f"Displaying 3D debug plot ({'Zoomed on Cloud' if zoom_on_cloud else 'Global View'})... (You can click and drag this window)")
    #plt.show()


# --- Call the new 3D plot function ---
try:
    # First, a zoomed-in view to check cloud and target
    plot_scene_geometry(renderer, zoom_on_cloud=True)
    # Then, a global view to see satellites and sun in relation to the center
    plot_scene_geometry(renderer, zoom_on_cloud=False)
except Exception as e:
    print(f"Could not generate 3D plot: {e}")
    print("Check your 'renderer' object. Did 'read_overpass_csv' run correctly?")

# <-- MODIFIED: Added this section -->
# --- 10. Show all plots ---
print("Displaying all plot windows at the same time...")
plt.show()