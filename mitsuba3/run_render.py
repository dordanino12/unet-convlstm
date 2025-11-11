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
cloud_data_file = '/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_3D/BOMEX_512x512x200_20m_20m_1s_512_0000005300_5_2'  # This is the pkl file you have
output_vol_file = 'temp/my_cloud.vol'  # A temporary file this script will create

# Create temp directory if it doesn't exist
os.makedirs('temp', exist_ok=True)

# Define which rows from your CSV to use
overpass_indices = [0, 1, 2, 9, 12, 15]
#overpass_indices = [0, 3, 6, 9, 12, 15]
#overpass_indices = [6, 7, 8, 9, 10, 11]
#overpass_indices = [9, 10, 11, 15, 16, 17]
#overpass_indices = [0, 1, 2, 30, 31, 32]
#overpass_indices = [12,13,14, 15, 16, 17]
#overpass_indices = [7, 8,9]
#overpass_indices = [0, 1, 2]

# --- 2. Set Up the Renderer Parameters ---
renderer_params = {
    'overpass_csv': csv_file,
    'overpass_indices': overpass_indices,
    'spp': 1024,
    'g_value': 0.7,
    'cloud_width': 128,
    'voxel_res': 0.02,
    'scene_scale': 1000.0,
    'cloud_zrange': [0.0, 4.0],
    'satellites': 3,
    'timestamps': 2,
    'pad_image': True,
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

    # Handle the case where n_satellites is 1, so subplots isn't 2D
    if n_timestamps == 1 and n_satellites == 1:
        fig, axes = plt.subplots(1, 1, figsize=(n_satellites * 4, n_timestamps * 4))
        axes = np.array([[axes]])  # Make it 2D for consistent indexing
    elif n_timestamps == 1:
        fig, axes = plt.subplots(n_timestamps, n_satellites, figsize=(n_satellites * 4, n_timestamps * 4))
        axes = np.array([axes])  # Make it 2D
    elif n_satellites == 1:
        fig, axes = plt.subplots(n_timestamps, n_satellites, figsize=(n_satellites * 4, n_timestamps * 4))
        axes = axes.reshape(-1, 1)  # Make it 2D
    else:
        fig, axes = plt.subplots(n_timestamps, n_satellites,
                                 figsize=(n_satellites * 4, n_timestamps * 4),
                                 squeeze=False)

    fig.suptitle('All Rendered Satellite Images', fontsize=16)

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
    #plt.show()  # Show the 2D plot window

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
    ax.set_xlabel('X [km] (from CSV: sat_Wy)')
    ax.set_ylabel('Y [km] (from CSV: sat_Wx)')
    ax.set_zlabel('Z [km] (Altitude)')

    if zoom_on_cloud:
        ax.set_title('3D Scene Geometry (Zoomed on Cloud)')
        # Set plot limits to zoom in on the origin/cloud
        # A slightly larger range than just the cloud to see context
        max_coord = max(cloud_size / 2, np.abs(cloud_center[2])) + 5  # Adjust 5 for padding
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        ax.set_zlim([-max_coord, max_coord * 2])  # Z-axis might need more range for cloud altitude
    else:
        ax.set_title('3D Scene Geometry (Global View)')
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

    ax.legend()

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