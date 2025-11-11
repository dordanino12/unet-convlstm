# Original imports
from rednder_from_udi_class import MitsubaRenderer
import numpy as np
from PIL import Image
import os  # <-- Added import
import mitsuba as mi  # <-- Added import

# --- Imports for 3D Plotting ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. Define Your Input Data Paths ---
# You must have these files.
csv_file = '/home/danino/PycharmProjects/pythonProject/data/Udi_3satellites_overpass.csv'
# This file is no longer used in this script
# cloud_data_file = '/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_3D/BOMEX_512x512x200_20m_20m_1s_512_0000005300_5_2'
output_vol_file = 'temp/my_cloud.vol'  # A temporary file this script will create

# Create temp directory if it doesn't exist
os.makedirs('temp', exist_ok=True)

# Define which rows from your CSV to use
overpass_indices = [0, 1, 2, 15, 16, 17]
overpass_indices = [0, 3, 6, 9, 12, 15]
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
    'spp': 8,  # <-- Lowered SPP for a fast test
    'g_value': 0,
    'cloud_width': 128,
    'voxel_res': 0.02,
    'scene_scale': 1000.0,
    'cloud_zrange': [0.0, 4.0],
    'satellites': 2,
    'timestamps': 3,
    'pad_image': True,
    'dynamic_emitter': True,
    'centralize_cloud': True,
    'bitmaps_required': False,
    'vol_path': output_vol_file  # <-- This path is now used for our synthetic file
}

print("Initializing MitsubaRenderer...")
# --- 3. Create an instance of the renderer ---
renderer = MitsubaRenderer(**renderer_params)

print("Reading CSV and setting up cameras...")
# --- 4. Run the setup methods ---
# These are crucial! They calculate renderer.W, .cloud_zrange, etc.
renderer.read_overpass_csv()
renderer.camera_params()
renderer.create_sensors()

# --- 5. [MODIFIED] Create a SYNTHETIC cloud volume ---
print("--- Creating Synthetic Cloud Data ---")
try:
    # 1. Define voxel dimensions
    W_voxels = renderer_params['cloud_width']  # e.g., 128
    H_voxels = renderer_params['cloud_width']  # e.g., 128
    # Infer Z dimension from your pkl file name '...x200...'
    # If your class's .vol writer uses a different dimension, adjust this
    D_voxels = 200
    print(f"Creating synthetic grid of size: (X={W_voxels}, Y={H_voxels}, Z={D_voxels})")

    # 2. Create an empty 3D array (in X, Y, Z order for clarity)
    synthetic_data_xyz = np.zeros((W_voxels, H_voxels, D_voxels), dtype=np.float32)

    # 3. Create three "axes" (X, Y, Z) with different densities
    center_x, center_y, center_z = W_voxels // 2, H_voxels // 2, D_voxels // 2
    thickness = 4  # Make the lines voxels thick

    print("Drawing synthetic axes (X=1.0, Y=2.0, Z=3.0)...")

    # X-axis (density 1.0) - Shortest
    x_len = 60
    synthetic_data_xyz[
    center_x: center_x + x_len,
    center_y - thickness: center_y + thickness,
    center_z - thickness: center_z + thickness
    ] = 0.1

    # Y-axis (density 2.0) - Medium
    y_len = 10
    synthetic_data_xyz[
    center_x - thickness: center_x + thickness,
    center_y: center_y + y_len,
    center_z - thickness: center_z + thickness
    ] = 0.1

    # Z-axis (density 3.0) - Longest
    z_len = 30
    synthetic_data_xyz[
    center_x - thickness: center_x + thickness,
    center_y - thickness: center_y + thickness,
    center_z: center_z + z_len
    ] = 0.1

    print(f"Added synthetic axes with densities 1.0, 2.0, 3.0.")

    # 4. Transpose data to the (Z, Y, X) format that the .pkl file would have
    # This is the format 'renderer.write_vol_file' expects
    # (X, Y, Z) -> (Z, Y, X)
    print("Transposing synthetic data to (Z, Y, X) for the class's writer...")
    synthetic_data_for_class = synthetic_data_xyz.transpose(2, 1, 0)
    print(f"Data shape is now: {synthetic_data_for_class.shape}")

    # 5. Call the renderer's write_vol_file method with our synthetic data
    print(f"Calling renderer.write_vol_file, saving to {output_vol_file}")
    renderer.write_vol_file(data=synthetic_data_for_class, vol_path=output_vol_file)
    print("Synthetic .vol file written successfully.")

except Exception as e:
    print(f"Error creating synthetic cloud: {e}")
    print("Please ensure 'numpy as np' is imported.")

# --- 5. [ORIGINAL - NOW COMMENTED OUT] ---
# print(f"Loading cloud data from '{cloud_data_file}'...")
# # We are skipping this and using our synthetic file instead.
# renderer.write_vol_file(sample_path=cloud_data_file,
#                         vol_path=output_vol_file,
#                         param_type='beta_ext',  # This must match a key in your pkl file!
#                         sample_ext='pkl')
# --- [END MODIFIED SECTION] ---


print("Setting up the 3D scene(s)...")
# --- 6. Build the 3D scene(s) in memory ---
# This will now load the synthetic 'output_vol_file' we just created
renderer.set_scenes()

print("Rendering... This may take a while.")
# --- 7. Run the render! ---
tensor_stacks, bitmap_stacks = renderer.render_scenes()

print("Rendering complete!")

# --- 8. Do something with the output ---
if not tensor_stacks or tensor_stacks[0] is None:
    print("No images were rendered.")
else:
    print("Processing images for plotting...")
    n_timestamps = renderer.timestamps
    n_satellites = renderer.satellites

    # Handle subplot indexing for 1D cases
    if n_timestamps == 1 and n_satellites == 1:
        fig, axes = plt.subplots(1, 1, figsize=(n_satellites * 4, n_timestamps * 4))
        axes = np.array([[axes]])
    elif n_timestamps == 1:
        fig, axes = plt.subplots(n_timestamps, n_satellites, figsize=(n_satellites * 4, n_timestamps * 4))
        axes = np.array([axes])
    elif n_satellites == 1:
        fig, axes = plt.subplots(n_timestamps, n_satellites, figsize=(n_satellites * 4, n_timestamps * 4))
        axes = axes.reshape(-1, 1)
    else:
        fig, axes = plt.subplots(n_timestamps, n_satellites,
                                 figsize=(n_satellites * 4, n_timestamps * 4),
                                 squeeze=False)

    fig.suptitle('All Rendered Satellite Images (SYNTHETIC CUBE)', fontsize=16)

    for t in range(n_timestamps):
        for s in range(n_satellites):
            float_data = tensor_stacks[t][s]
            if float_data.max() > 0:
                normalized_data = float_data / float_data.max()
            else:
                normalized_data = float_data

            image_data_uint8 = (normalized_data * 255).astype(np.uint8)

            ax = axes[t, s]
            ax.imshow(image_data_uint8, cmap='gray')
            ax.set_title(f'Timestamp {t}, Satellite {s}')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Displaying 2D render plot...")


first_tensor_data = tensor_stacks[0][0]
print(f"Shape of the first tensor: {first_tensor_data.shape}")

# --- 9. 3D Debug Plot ---
print("Generating 3D debug plot...")


def plot_scene_geometry(renderer, zoom_on_cloud=True):
    """
    Plots the 3D positions of the cloud, cameras, and sun direction.
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

        ax.add_collection3d(Poly3DCollection(faces,
                                               facecolors=color, linewidths=1, edgecolors=edge_color, alpha=alpha))

    # --- Start 3D Plot ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot the Cloud Volume Bounding Box
    cloud_size = renderer.W
    cloud_center = [0, 0, renderer.cloud_zcenter * 2]
    plot_cube(cloud_center, cloud_size, ax, color='cyan', alpha=0.1, edge_color='green')

    # 2. Plot the Cameras (Satellites)
    cam_x = np.array(renderer.sat_Wy)
    cam_y = np.array(renderer.sat_Wx)
    cam_z = np.array(renderer.sat_H)
    ax.scatter(cam_x, cam_y, cam_z, c='blue', marker='^', s=100, label='Satellites (Cameras)')

    # 3. Plot Camera View Lines
    target_point = cloud_center
    for i in range(len(cam_x)):
        ax.plot([cam_x[i], target_point[0]],
                [cam_y[i], target_point[1]],
                [cam_z[i], target_point[2]],
                c='blue', linestyle='--', alpha=0.5)

    # 4. Plot the Sun Direction
    try:
        sun_dir_raw = np.array(renderer.scenes_dict[0]['emitter']['direction'])
        sun_direction = sun_dir_raw / np.linalg.norm(sun_dir_raw)  # Normalize
        sun_arrow_origin_offset = 100
        sun_start_point = np.array(cloud_center) - sun_direction * sun_arrow_origin_offset

        ax.quiver(sun_start_point[0], sun_start_point[1], sun_start_point[2],
                  sun_direction[0], sun_direction[1], sun_direction[2],
                  length=sun_arrow_origin_offset, color='orange', linewidth=3, arrow_length_ratio=0.1,
                  label='Sun Direction')
        ax.scatter([sun_start_point[0]], [sun_start_point[1]], [sun_start_point[2]],
                   c='yellow', marker='*', s=300, label='Sun Source (Illustrative)')

    except Exception as e:
        print(f"Could not plot sun direction: {e}")

    # 5. Plot the Origin
    ax.scatter([0], [0], [0], c='red', marker='o', s=100, label='World Origin (0,0,0)')

    # 6. Set Labels and Title
    ax.set_xlabel('X [km] (from CSV: sat_Wy)')
    ax.set_ylabel('Y [km] (from CSV: sat_Wx)')
    ax.set_zlabel('Z [km] (Altitude)')

    if zoom_on_cloud:
        ax.set_title('3D Scene Geometry (Zoomed on Cloud)')
        max_coord = max(cloud_size / 2, np.abs(cloud_center[2])) + 5
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        ax.set_zlim([-max_coord, max_coord * 2])
    else:
        ax.set_title('3D Scene Geometry (Global View)')
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

        z_min = min(z_min, -50)
        z_max = max(z_max, 600)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

    ax.legend()
    print(
        f"Displaying 3D debug plot ({'Zoomed on Cloud' if zoom_on_cloud else 'Global View'})... (You can click and drag this window)")


# --- Call the new 3D plot function ---
try:
    plot_scene_geometry(renderer, zoom_on_cloud=True)
    plot_scene_geometry(renderer, zoom_on_cloud=False)
except Exception as e:
    print(f"Could not generate 3D plot: {e}")
    print("Check your 'renderer' object. Did 'read_overpass_csv' run correctly?")


# --- 10. Show all plots ---
print("Displaying all plot windows at the same time...")
plt.show()