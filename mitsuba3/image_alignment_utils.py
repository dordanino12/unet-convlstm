# image_alignment_utils.py
# Contains helper functions for performing image-to-image homography (warping)

import numpy as np
import cv2  # Requires: pip install opencv-python


def get_camera_matrices(camera_pos, target_pos, world_up=np.array([0, 1, 0])):
    """
    Computes Rotation (R) and Translation (t) matrices for a camera.
    Uses world_up vector to explicitly control image Roll/rotation.
    """
    # 1. Forward Vector (Camera Z-axis)
    forward = target_pos - camera_pos
    dist = np.linalg.norm(forward)
    if dist < 1e-6:
        forward = np.array([0, 0, -1])
    else:
        forward = forward / dist

    # 2. Right Vector (Camera X-axis)
    right = np.cross(forward, world_up)

    if np.linalg.norm(right) < 1e-6:
        # Handle singularity (looking straight down/up)
        if np.abs(world_up[1]) > 0.99:
            right = np.array([1, 0, 0])
        else:
            right = np.array([0, 0, 1])
    else:
        right = right / np.linalg.norm(right)

    # 3. Down Vector (Camera Y-axis)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Build Rotation Matrix (R)
    R = np.vstack([right, down, forward])

    # Translation (t)
    t = -R @ camera_pos.reshape(3, 1)

    return R, t


def warp_to_virtual_camera(src_image, src_pose, virtual_pose, K, img_dim, plane_height):
    """
    Warps 'src_image' to match the view of a 'virtual_camera' using Homography.
    """
    # Define Alignment Vector: World Y (0, 1, 0) for fixed roll
    world_up_vector = np.array([0, 1, 0])

    w, h = img_dim, img_dim
    target_center = np.array([0.0, 0.0, plane_height])

    # 1. Get Matrices for Virtual (Destination) and Source Cameras
    R_v, t_v = get_camera_matrices(virtual_pose, target_center, world_up=world_up_vector)
    R_s, t_s = get_camera_matrices(src_pose, target_center, world_up=world_up_vector)

    # 2. Define 4 corners of the Virtual Image (Pixels)
    dst_corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # 3. Back-project Virtual Pixels -> World Points (on the plane)
    K_inv = np.linalg.inv(K)
    world_points = []

    for px in dst_corners_px:
        p_uv = np.array([px[0], px[1], 1.0])
        p_cam_norm = K_inv @ p_uv
        ray_dir_world = R_v.T @ p_cam_norm
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)

        # Intersect ray with the alignment plane (Z=plane_height)
        if abs(ray_dir_world[2]) < 1e-6:
            scale_t = 1e6
        else:
            scale_t = (plane_height - virtual_pose[2]) / ray_dir_world[2]

        p_world = virtual_pose + scale_t * ray_dir_world
        world_points.append(p_world)

    world_points = np.array(world_points)

    # 4. Project World Points -> Source Camera Pixels
    points_cam = (R_s @ world_points.T + t_s).T
    src_corners_px = []

    for p in points_cam:
        x, y, z = p[0], p[1], p[2]
        if z < 0.1: z = 0.1  # Avoid division by zero/near-zero
        u = K[0, 0] * (x / z) + K[0, 2]
        v = K[1, 1] * (y / z) + K[1, 2]
        src_corners_px.append([u, v])

    src_corners_px = np.array(src_corners_px, dtype=np.float32)

    # 5. Compute Homography and Warp
    M_inv = cv2.getPerspectiveTransform(dst_corners_px, src_corners_px)

    if src_image.dtype != np.uint8 and src_image.dtype != np.float32:
        src_image = src_image.astype(np.float32)

    warped = cv2.warpPerspective(src_image, M_inv, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
    return warped