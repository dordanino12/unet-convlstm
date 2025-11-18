from render import MitsubaRenderer
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # חובה להתקין: pip install opencv-python


# ==============================================================================
# 1. HELPER FUNCTIONS (IMAGE-TO-IMAGE HOMOGRAPHY) - UPDATED FOR ROLL CONTROL
# ==============================================================================

def get_camera_matrices(camera_pos, target_pos, world_up=np.array([0, 1, 0])):
    """
    Computes R and t, allowing explicit control over the 'Up' vector
    (world_up) to fix the horizontal rotation (Roll). Default is World Y-Up [0, 1, 0].
    """
    # 1. Forward Vector (Camera Z-axis)
    forward = target_pos - camera_pos
    dist = np.linalg.norm(forward)
    if dist < 1e-6:
        forward = np.array([0, 0, -1])
    else:
        forward = forward / np.linalg.norm(forward)

    # 2. Right Vector (Camera X-axis) - must be perpendicular to Forward and World Up
    right = np.cross(forward, world_up)

    # Handle singularity (looking straight down/up)
    if np.linalg.norm(right) < 1e-6:
        # If looking straight down/up (Forward || Up), define Right arbitrarily as World X
        if np.abs(world_up[1]) > 0.99:
            right = np.array([1, 0, 0])
        else:
            right = np.array([0, 0, 1])
    else:
        right = right / np.linalg.norm(right)

    # 3. Down Vector (Camera Y-axis) - must be perpendicular to Forward and Right
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Build Rotation Matrix (R)
    R = np.vstack([right, down, forward])

    # Translation (t)
    t = -R @ camera_pos.reshape(3, 1)

    return R, t


def warp_to_virtual_camera(src_image, src_pose, virtual_pose, K, img_dim, plane_height):
    """
    Warps 'src_image' to match the view of a 'virtual_camera', ensuring fixed Roll alignment.
    """

    # [NEW] Define Alignment Vector: World Y (0, 1, 0)
    world_up_vector = np.array([0, 1, 0])

    w, h = img_dim, img_dim
    target_center = np.array([0.0, 0.0, plane_height])

    # 1. Get Matrices for the Virtual Camera (Destination) - FORCED ROLL
    # We pass world_up_vector=Y to fix the Roll of the virtual camera.
    R_v, t_v = get_camera_matrices(virtual_pose, target_center, world_up=world_up_vector)

    # 2. Get Matrices for the Real/Source Camera - FORCED ROLL
    # We MUST pass the same world_up_vector to ensure a correct relative rotation.
    R_s, t_s = get_camera_matrices(src_pose, target_center, world_up=world_up_vector)

    # 3. Define the 4 corners of the Virtual Image (Pixels)
    dst_corners_px = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)

    # 4. Back-project Virtual Pixels -> World Points (on the plane)
    K_inv = np.linalg.inv(K)
    world_points = []

    for px in dst_corners_px:
        p_uv = np.array([px[0], px[1], 1.0])
        p_cam_norm = K_inv @ p_uv
        ray_dir_world = R_v.T @ p_cam_norm
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)

        if abs(ray_dir_world[2]) < 1e-6:
            scale_t = 1e6
        else:
            scale_t = (plane_height - virtual_pose[2]) / ray_dir_world[2]

        p_world = virtual_pose + scale_t * ray_dir_world
        world_points.append(p_world)

    world_points = np.array(world_points)

    # 5. Project World Points -> Source Camera Pixels
    points_cam = (R_s @ world_points.T + t_s).T

    src_corners_px = []
    for p in points_cam:
        x, y, z = p[0], p[1], p[2]
        if z < 0.1: z = 0.1

        u = K[0, 0] * (x / z) + K[0, 2]
        v = K[1, 1] * (y / z) + K[1, 2]
        src_corners_px.append([u, v])

    src_corners_px = np.array(src_corners_px, dtype=np.float32)

    # 6. Compute Homography and Warp
    M_inv = cv2.getPerspectiveTransform(dst_corners_px, src_corners_px)

    if src_image.dtype != np.uint8 and src_image.dtype != np.float32:
        src_image = src_image.astype(np.float32)

    warped = cv2.warpPerspective(src_image, M_inv, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
    return warped

# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================

# --- Setup Paths ---
csv_file = '/home/danino/PycharmProjects/pythonProject/data/debug.csv'
output_vol_file = 'temp/my_cloud.vol'
os.makedirs('temp', exist_ok=True)
overpass_indices = [0, 1, 2, 3, 4, 5]

# --- Init Renderer ---
renderer_params = {
    'overpass_csv': csv_file,
    'overpass_indices': overpass_indices,
    'spp': 64,
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

print("Initializing Renderer...")
renderer = MitsubaRenderer(**renderer_params)
renderer.read_overpass_csv()
renderer.camera_params()
renderer.create_sensors()

# --- Create Synthetic Data (Ball + Cube + Pyramid) ---
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
    synthetic_data_xyz[mask_sphere] = 0.1
    # Cube
    synthetic_data_xyz[center_x - 15:center_x + 15, center_y - 15:center_y + 15, center_z - 15:center_z + 15] = 0.1
    # Pyramid
    pyr_center_x = center_x + 40
    pyr_h = 40
    pyr_base_z = center_z - 15
    h = z - pyr_base_z
    L_h = 40 * (1.0 - h / pyr_h)
    mask_pyr = (z >= pyr_base_z) & (z < pyr_base_z + pyr_h) & (np.abs(x - pyr_center_x) <= L_h / 2) & (
            np.abs(y - center_y) <= L_h / 2)
    synthetic_data_xyz[mask_pyr] = 0.1

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
    fig, axes = plt.subplots(n_ts * 2, n_sat, figsize=(n_sat * 4, n_ts * 8))
    if n_ts * 2 == 1 and n_sat == 1:
        axes = np.array([[axes]])
    elif n_ts * 2 == 1:
        axes = np.array([axes])
    elif n_sat == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Original (Top) vs. Aligned to Virtual Cam (Bottom)', fontsize=16)

    # 1. Camera Intrinsics (K)
    fov_rad = np.deg2rad(renderer.fov)
    f_px = (renderer.film_dim / 2) / np.tan(fov_rad / 2)

    K_matrix = np.array([
        [f_px, 0, renderer.film_dim / 2],
        [0, f_px, renderer.film_dim / 2],
        [0, 0, 1]
    ])

    # 2. Define "Ideal" Virtual Camera
    # This is the reference view we want everyone to match.
    # Position: 0,0, 600km height. Looking straight down.
    virtual_cam_pos = np.array([0.0, 0.0, 600.0])

    # 3. Define Alignment Plane (The "Sweet Spot")
    # We align everything based on the cloud center height to minimize parallax shifts.
    cloud_center_z = renderer.cloud_zcenter

    print(f"Reference View: Camera at 600km, Aligning to Z={cloud_center_z:.2f}km")

    for t in range(n_ts):
        for s in range(n_sat):
            # Prepare Image
            float_data = tensor_stacks[t][s]
            norm_data = float_data / (float_data.max() + 1e-6)
            image_data_uint8 = (norm_data * 255).astype(np.uint8)

            # Get Real Camera Position
            # Calculate the correct index in the flattened list
            global_idx = t * n_sat + s

            # Use global_idx instead of s
            real_cam_pos = np.array([
                renderer.sat_Wy[global_idx],
                renderer.sat_Wx[global_idx],
                renderer.sat_H[global_idx]
            ])

            # Warp!
            # Instead of inventing map coordinates, we just ask:
            # "Make this image look like it was taken by the virtual camera"
            warped_image = warp_to_virtual_camera(
                image_data_uint8,
                real_cam_pos,
                virtual_cam_pos,
                K_matrix,
                renderer.film_dim,
                plane_height=cloud_center_z
            )

            # Plot Original
            ax_orig = axes[t * 2, s]
            ax_orig.imshow(image_data_uint8, cmap='gray')
            ax_orig.set_title(f'Sat {s}\nH={real_cam_pos[2]:.0f}km')
            ax_orig.axis('off')

            # Plot Warped
            ax_rect = axes[t * 2 + 1, s]
            ax_rect.imshow(warped_image, cmap='gray')
            ax_rect.set_title(f'Aligned to\n600km View')
            ax_rect.axis('off')

    plt.tight_layout()
    plt.show()