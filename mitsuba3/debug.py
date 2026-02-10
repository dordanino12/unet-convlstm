from render import MitsubaRenderer
import numpy as np
import matplotlib.pyplot as plt
import os

# --- New Imports for 3D Plotting ---
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Import alignment functions from the external module ---
from image_alignment_utils import warp_to_virtual_camera

# ==============================================================================
# 1. SETUP & INITIALIZATION
# ==============================================================================

# --- Setup Paths ---
csv_file = '/home/danino/PycharmProjects/pythonProject/data/Udi_3satellites_overpass.csv'
output_vol_file = 'temp/my_cloud.vol'
os.makedirs('temp', exist_ok=True)
overpass_indices = [0, 1, 2, 15, 16, 17]
overpass_indices = [15,16, 17, 33, 34, 35]
overpass_indices = [15,1, 2, 33, 34, 35]



# --- Init Renderer ---
renderer_params = {
    'overpass_csv': csv_file,
    'overpass_indices': overpass_indices,
    'spp': 512,
    'g_value': 0.85,
    'cloud_width': 128,
    'voxel_res': 0.02,
    'scene_scale': 1000.0,
    'cloud_zrange': [2.0, 6.0],
    'satellites': 3,
    'timestamps': 2,
    'pad_image': True,
    'dynamic_emitter': True,
    'centralize_cloud': True,
    'bitmaps_required': False,
    'vol_path': output_vol_file
}

print("Initializing Renderer...")
renderer = MitsubaRenderer(**renderer_params)
renderer.read_overpass_csv()
renderer.camera_params()
renderer.create_sensors()

# ==============================================================================
# 2. CREATE SYNTHETIC DATA & RENDER
# ==============================================================================

print("Generating Synthetic Volume...")
try:
    W_voxels = renderer_params['cloud_width']
    H_voxels = renderer_params['cloud_width']
    D_voxels = 200
    synthetic_data_xyz = np.zeros((W_voxels, H_voxels, D_voxels), dtype=np.float32)
    center_x, center_y, center_z = W_voxels // 2, H_voxels // 2, D_voxels // 2
    x, y, z = np.indices(synthetic_data_xyz.shape)

    # Ball
    mask_sphere = ((x - (center_x - 40)) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2) <= 20 ** 2
    synthetic_data_xyz[mask_sphere] = 0.02
    # Cube
    synthetic_data_xyz[center_x - 15:center_x + 15, center_y - 15:center_y + 15, center_z - 15:center_z + 15] = 0.2
    # Pyramid
    pyr_center_x = center_x + 40
    pyr_h = 40
    pyr_base_z = center_z - 15
    h = z - pyr_base_z
    L_h = 40 * (1.0 - h / pyr_h)
    mask_pyr = (z >= pyr_base_z) & (z < pyr_base_z + pyr_h) & (np.abs(x - pyr_center_x) <= L_h / 2) & (
            np.abs(y - center_y) <= L_h / 2)
    synthetic_data_xyz[mask_pyr] = 0.02

    # --- ADD NEW BORDER FRAME ---
    border_density = 0.005
    border_thickness = 2
    mask_x_border = (x < border_thickness) | (x >= W_voxels - border_thickness)
    mask_y_border = (y < border_thickness) | (y >= H_voxels - border_thickness)
    mask_z_border = (z < border_thickness) | (z >= D_voxels - border_thickness)
    mask_frame = mask_x_border | mask_y_border | mask_z_border

    synthetic_data_xyz[mask_frame] = border_density

    # Save volume
    synthetic_data_zxy = np.transpose(synthetic_data_xyz, (2, 0, 1))
    renderer.write_vol_file(data=synthetic_data_zxy, vol_path=output_vol_file)

except Exception as e:
    print(f"Error creating data: {e}")

# --- Render ---
print("Rendering Scenes...")
renderer.set_scenes()
tensor_stacks, bitmap_stacks = renderer.render_scenes()

# ==============================================================================
# 3. PROCESS: WARP TO VIRTUAL CAMERA (HOMOGRAPHY)
# ==============================================================================

if not tensor_stacks or tensor_stacks[0] is None:
    print("No images were rendered.")
else:
    print("Processing Image Alignment (Virtual Camera @ 600km)...")

    n_ts = renderer.timestamps
    n_sat = renderer.satellites

    # Setup Plot
    fig, axes = plt.subplots(n_ts * 2, n_sat, figsize=(n_sat * 4, n_ts * 8), squeeze=False)
    fig.suptitle('Original (Top) vs. Aligned to Virtual Cam (Bottom)', fontsize=16)

    # 1. Camera Intrinsics (K)
    fov_rad = np.deg2rad(renderer.fov)
    f_px = (renderer.film_dim / 2) / np.tan(fov_rad / 2)

    K_matrix = np.array([
        [f_px, 0, renderer.film_dim / 2],
        [0, f_px, renderer.film_dim / 2],
        [0, 0, 1]
    ])

    # 2. Define "Ideal" Virtual Camera and Alignment Plane
    virtual_cam_pos = np.array([0.0, 0.0, 600.0])
    cloud_center_z = renderer.cloud_zcenter * 2

    print(f"Reference View: Camera at 600km, Aligning to Z={cloud_center_z:.2f}km")

    for t in range(n_ts):
        for s in range(n_sat):
            # Prepare Image
            float_data = tensor_stacks[t][s]
            norm_data = float_data / (float_data.max() + 1e-6)
            image_data_uint8 = (norm_data * 255).astype(np.uint8)

            # Get Real Camera Position & Index
            global_idx = t * n_sat + s
            ov_idx = renderer.overpass_indices[global_idx]  # <--- RETRIEVE INDEX

            real_cam_pos = np.array([
                renderer.sat_Wy[global_idx],
                renderer.sat_Wx[global_idx],
                renderer.sat_H[global_idx]
            ])

            # --- Warp! ---
            warped_image = warp_to_virtual_camera(
                image_data_uint8,
                real_cam_pos,
                virtual_cam_pos,
                K_matrix,
                renderer.film_dim,
                plane_height=cloud_center_z
            )

            # --- Plotting ---
            # Plot Original
            ax_orig = axes[t * 2, s]
            ax_orig.imshow(image_data_uint8, cmap='gray')

            # --- UPDATED TITLE: ONLY IDX ---
            ax_orig.set_title(f'Idx: {ov_idx}')
            ax_orig.axis('off')

            # Plot Warped
            ax_rect = axes[t * 2 + 1, s]
            ax_rect.imshow(warped_image, cmap='gray')
            ax_rect.set_title(f'Aligned (Idx: {ov_idx})')
            ax_rect.axis('off')

    plt.tight_layout()


# ==============================================================================
# 4. 3D SCENE VISUALIZATION
# ==============================================================================

def plot_cube(center, size, ax, color='cyan', alpha=0.1, edge_color='r'):
    """Plots a 3D cube representing the cloud volume boundaries."""
    half_size = size / 2.0
    x, y, z = center
    vertices = np.array([
        [x - half_size, y - half_size, z - half_size], [x + half_size, y - half_size, z - half_size],
        [x + half_size, y + half_size, z - half_size], [x - half_size, y + half_size, z - half_size],
        [x - half_size, y - half_size, z + half_size], [x + half_size, y - half_size, z + half_size],
        [x + half_size, y + half_size, z + half_size], [x - half_size, y + half_size, z + half_size]
    ])
    faces_idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    faces = [vertices[i] for i in faces_idx]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors=edge_color, alpha=alpha))


def _plot_common_elements(renderer, ax):
    """Plots the cloud box, cameras (satellites), sun vector, and origin."""
    cloud_size = renderer.W
    cloud_center = [0, 0, renderer.cloud_zcenter * 2]

    # Plot Cloud Box
    plot_cube(cloud_center, cloud_size, ax, color='cyan', alpha=0.1, edge_color='green')

    # Plot Cameras (Satellites)
    cam_x = np.array(renderer.sat_Wy)
    cam_y = np.array(renderer.sat_Wx)
    cam_z = np.array(renderer.sat_H)
    ax.scatter(cam_x, cam_y, cam_z, c='blue', marker='^', s=100, label='Satellites')

    # <--- UPDATED LABELS: ONLY IDX ---
    for i in range(len(cam_x)):
        ov_idx = renderer.overpass_indices[i]
        label_text = f"Idx: {ov_idx}"
        ax.text(cam_x[i], cam_y[i], cam_z[i] + 10, label_text, color='black', fontsize=9, fontweight='bold')

    # Plot View Lines
    target_point = cloud_center
    for i in range(len(cam_x)):
        ax.plot([cam_x[i], target_point[0]], [cam_y[i], target_point[1]],
                [cam_z[i], target_point[2]], c='blue', linestyle='--', alpha=0.5)

    # Plot Sun Direction
    sun_start_point = None
    try:
        sun_dir = np.array(renderer.scenes_dict[0]['emitter']['direction'])
        sun_dir_norm = sun_dir / np.linalg.norm(sun_dir)
        sun_arrow_origin_offset = 100
        sun_start_point = np.array(cloud_center) - sun_dir_norm * sun_arrow_origin_offset

        ax.quiver(sun_start_point[0], sun_start_point[1], sun_start_point[2],
                  sun_dir_norm[0], sun_dir_norm[1], sun_dir_norm[2],
                  length=sun_arrow_origin_offset, color='orange', linewidth=3, arrow_length_ratio=0.1,
                  label='Sun Direction')
        ax.scatter([sun_start_point[0]], [sun_start_point[1]], [sun_start_point[2]],
                   c='yellow', marker='*', s=300, label='Sun Source')
    except Exception as e:
        pass

    # Plot Origin and Labels
    ax.scatter([0], [0], [0], c='red', marker='o', s=100, label='Origin (0,0,0)')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()

    return cam_x, cam_y, cam_z, cloud_center, sun_start_point


def plot_scene_geometry_global(renderer):
    """Initializes the 3D figure and sets axis limits."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cam_x, cam_y, cam_z, cloud_center, sun_start_point = _plot_common_elements(renderer, ax)

    ax.set_title('3D Scene Geometry & Satellite Positions')

    sun_points = [sun_start_point[0], sun_start_point[1], sun_start_point[2]] if sun_start_point is not None else [0, 0,
                                                                                                                   0]
    all_x = np.concatenate(([0, cloud_center[0]], cam_x, [sun_points[0]]))
    all_y = np.concatenate(([0, cloud_center[1]], cam_y, [sun_points[1]]))
    all_z = np.concatenate(([0, cloud_center[2]], cam_z, [sun_points[2]]))

    x_min, x_max = all_x.min() - 50, all_x.max() + 50
    y_min, y_max = all_y.min() - 50, all_y.max() + 50
    z_min, z_max = all_z.min() - 50, all_z.max() + 50
    z_min = min(z_min, -50)
    z_max = max(z_max, 650)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    print("Displaying 3D debug plot...")


# --- Run the 3D Plotting ---
print("Generating 3D Visualization...")
try:
    plot_scene_geometry_global(renderer)
except Exception as e:
    print(f"Error generating 3D plot: {e}")

# --- Final Show ---
print("Displaying all plots. Close windows to exit.")
plt.show()