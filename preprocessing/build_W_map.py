import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


class CloudRayCaster:
    def __init__(self, pkl_path, voxel_size=20.0):
        """
        Loads the cloud volume.
        """
        # 1. Load Data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.vol_beta = data['beta_ext']
        self.vol_u = data['U']
        self.vol_v = data['V']
        self.vol_w = data['W']

        # 2. Define Grid Dimensions
        self.nz, self.ny, self.nx = self.vol_beta.shape
        self.voxel_size = voxel_size

        # 3. Calculate Physical Sizes
        self.size_x = self.nx * voxel_size
        self.size_y = self.ny * voxel_size
        self.size_z = self.nz * voxel_size

        # 4. Define Bounds
        self.min_bound = np.array([-self.size_x / 2, -self.size_y / 2, 0.0])
        self.max_bound = np.array([self.size_x / 2, self.size_y / 2, self.size_z])

        # print(f"--- Volume Loaded ---")
        # print(f"Grid Shape: {self.vol_beta.shape}")
        # print(f"World Bounds:")
        # print(f"  X: [{self.min_bound[0]}, {self.max_bound[0]}]")
        # print(f"  Y: [{self.min_bound[1]}, {self.max_bound[1]}]")
        # print(f"  Z: [{self.min_bound[2]}, {self.max_bound[2]}]")

    def get_rays(self, cam_pos, look_at, resolution=(128, 128), fov=0.25): # fov for 128 on 128 need set to 0.25 , for 256 on 256 0.115
        H, W = resolution

        # Camera Coordinate System
        cam_dir = (look_at - cam_pos)
        cam_dir = cam_dir / np.linalg.norm(cam_dir)

        world_up = np.array([-1, 0, 0])

        cam_right = np.cross(cam_dir, world_up)
        cam_right = cam_right / np.linalg.norm(cam_right)

        cam_up = np.cross(cam_right, cam_dir)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # Image Plane
        aspect_ratio = W / H
        scale = np.tan(np.deg2rad(fov * 0.5))

        i, j = np.meshgrid(np.arange(W), np.arange(H))

        x = -((2 * (i + 0.5) / W - 1) * aspect_ratio * scale)
        y = (1 - 2 * (j + 0.5) / H) * scale

        rays_d = x[..., np.newaxis] * cam_right + \
                 y[..., np.newaxis] * cam_up + \
                 cam_dir

        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_o = np.broadcast_to(cam_pos, rays_d.shape)

        return rays_o, rays_d

    def render_velocity_maps_first_hit(self, cam_pos, look_at, resolution=(128, 128), step_size=20.0):
        """
        Ray Marching: Finds the surface of the cloud (First Hit)
        """
        cam_pos = np.array(cam_pos)
        look_at = np.array(look_at)

        # 1. Get Rays
        rays_o, rays_d = self.get_rays(cam_pos, look_at, resolution)
        H, W_res = resolution

        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        num_rays = rays_o_flat.shape[0]

        # 2. AABB Intersection
        inv_d = 1.0 / (rays_d_flat + 1e-6)
        t0 = (self.min_bound - rays_o_flat) * inv_d
        t1 = (self.max_bound - rays_o_flat) * inv_d
        tmin = np.maximum(np.minimum(t0, t1).max(axis=1), 0)
        tmax = np.minimum(np.maximum(t0, t1).min(axis=1), 100000000000)
        mask_box_hit = tmax > tmin

        u_map = np.full(num_rays, np.nan)
        v_map = np.full(num_rays, np.nan)
        w_map = np.full(num_rays, np.nan)

        # 3. Ray Marching
        valid_indices = np.where(mask_box_hit)[0]
        if len(valid_indices) == 0:
            return u_map.reshape(H, W_res), v_map.reshape(H, W_res), w_map.reshape(H, W_res)

        r_o = rays_o_flat[valid_indices]
        r_d = rays_d_flat[valid_indices]
        t_start = tmin[valid_indices]
        t_end = tmax[valid_indices]

        max_dist = np.max(t_end - t_start)
        num_steps = int(max_dist / step_size) + 2
        steps = np.arange(num_steps) * step_size
        t_vals = t_start[:, np.newaxis] + steps[np.newaxis, :]
        points = r_o[:, np.newaxis, :] + r_d[:, np.newaxis, :] * t_vals[:, :, np.newaxis]

        # Convert to Grid
        grid_indices = (points - self.min_bound) / self.voxel_size
        grid_indices = grid_indices.astype(int)
        gx = np.clip(grid_indices[:, :, 0], 0, self.nx - 1)
        gy = np.clip(grid_indices[:, :, 1], 0, self.ny - 1)
        gz = np.clip(grid_indices[:, :, 2], 0, self.nz - 1)
        in_bounds_mask = t_vals <= t_end[:, np.newaxis]

        sampled_beta = self.vol_beta[gz, gy, gx]
        sampled_beta[~in_bounds_mask] = 0

        # Find First Hit
        hit_cloud = sampled_beta > 0
        first_hit_idx = np.argmax(hit_cloud, axis=1)
        has_hit = np.any(hit_cloud, axis=1)

        # Sample U, V, W
        ray_idx_range = np.arange(len(valid_indices))
        hit_z = gz[ray_idx_range, first_hit_idx]
        hit_y = gy[ray_idx_range, first_hit_idx]
        hit_x = gx[ray_idx_range, first_hit_idx]

        final_u = np.full(len(valid_indices), np.nan)
        final_v = np.full(len(valid_indices), np.nan)
        final_w = np.full(len(valid_indices), np.nan)

        final_u[has_hit] = self.vol_u[hit_z, hit_y, hit_x][has_hit]
        final_v[has_hit] = self.vol_v[hit_z, hit_y, hit_x][has_hit]
        final_w[has_hit] = self.vol_w[hit_z, hit_y, hit_x][has_hit]

        u_map[valid_indices] = final_u
        v_map[valid_indices] = final_v
        w_map[valid_indices] = final_w

        return u_map.reshape(H, W_res), v_map.reshape(H, W_res), w_map.reshape(H, W_res)

    def render_z_slice(self, cam_pos, look_at, target_z_height, resolution=(128, 128), reference_plane_z=750.0):
        """
        Modified Ray-Plane Intersection:
        1. Casts rays to hit a fixed Reference Plane (default 750m).
        2. Keeps the X, Y from that intersection.
        3. Forces the Z to be the 'target_z_height'.

        This prevents the image from 'shifting' sideways when changing slice height due to parallax.
        """
        cam_pos = np.array(cam_pos)
        look_at = np.array(look_at)
        H, W_res = resolution

        # 1. Get Rays
        rays_o, rays_d = self.get_rays(cam_pos, look_at, resolution)

        # Flatten for vectorized calc
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        num_rays = rays_o_flat.shape[0]

        # 2. Calculate Intersection with REFERENCE Plane (Z = 750)
        # We solve t for the reference plane, not the target slice
        d_z = rays_d_flat[:, 2]
        # Avoid division by zero
        d_z[np.abs(d_z) < 1e-6] = 1e-6

        # t_hit is now the distance to the 750m plane
        t_hit = (reference_plane_z - rays_o_flat[:, 2]) / d_z

        # 3. Calculate 3D Hit Points on Reference Plane
        ref_hit_points = rays_o_flat + t_hit[:, np.newaxis] * rays_d_flat

        # 4. Create Sampling Points
        # Take X and Y from the reference hit, but set Z to the requested target_z_height
        points_to_sample = np.copy(ref_hit_points)
        points_to_sample[:, 2] = target_z_height

        # 5. Filter Invalid Points
        # Check if the generated points are within the volume bounds
        valid_mask = (t_hit > 0) & \
                     (points_to_sample[:, 0] >= self.min_bound[0]) & (points_to_sample[:, 0] <= self.max_bound[0]) & \
                     (points_to_sample[:, 1] >= self.min_bound[1]) & (points_to_sample[:, 1] <= self.max_bound[1]) & \
                     (points_to_sample[:, 2] >= self.min_bound[2]) & (points_to_sample[:, 2] <= self.max_bound[2])

        # Initialize Output Maps
        u_map = np.full(num_rays, np.nan)
        v_map = np.full(num_rays, np.nan)
        w_map = np.full(num_rays, np.nan)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # print("No rays hit the volume bounds at the requested projection.")
            return u_map.reshape(H, W_res), v_map.reshape(H, W_res), w_map.reshape(H, W_res)

        # 6. Convert World Coords to Grid Indices
        final_points = points_to_sample[valid_indices]

        grid_indices = (final_points - self.min_bound) / self.voxel_size
        grid_indices = grid_indices.astype(int)

        gx = np.clip(grid_indices[:, 0], 0, self.nx - 1)
        gy = np.clip(grid_indices[:, 1], 0, self.ny - 1)
        gz = np.clip(grid_indices[:, 2], 0, self.nz - 1)

        # 7. Sample Data
        u_map[valid_indices] = self.vol_u[gz, gy, gx]
        v_map[valid_indices] = self.vol_v[gz, gy, gx]
        w_map[valid_indices] = self.vol_w[gz, gy, gx]

        return u_map.reshape(H, W_res), v_map.reshape(H, W_res), w_map.reshape(H, W_res)


# --- Usage Example ---
if __name__ == "__main__":
    pkl_file = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/0000007000/sample_019.pkl'
    output_dir = "/home/danino/PycharmProjects/pythonProject/data/output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caster = CloudRayCaster(pkl_file)

    # Define Camera
    camera_pos = [214*1000, 1044*1000, 515*1000]
    camera_pos = [0, 0, 600*1000]
    #camera_pos = [-154*1000, -747*1000, 558*1000]
    look_at = [0, 0, 1500]

    # --- CONFIGURATION ---
    # Choose mode: 'first_hit' (Cloud Surface) or 'slice' (Specific Height)
    render_mode = 'slice'
    slice_height_m = 750.0  # Height in meters to slice (e.g., middle of cloud)

    print(f"Cam: {camera_pos}, Mode: {render_mode}")

    if render_mode == 'first_hit':
        u_map, v_map, w_map = caster.render_velocity_maps_first_hit(
            cam_pos=camera_pos, look_at=look_at, resolution=(128, 128))
        title_prefix = "First Hit"
        file_prefix = "first_hit"

    elif render_mode == 'slice':
        print(f"Slicing at Height: {slice_height_m}m")
        u_map, v_map, w_map = caster.render_z_slice(
            cam_pos=camera_pos, look_at=look_at, target_z_height=slice_height_m, resolution=(256, 256))
        title_prefix = f"Z-Slice {int(slice_height_m)}m"
        file_prefix = f"slice_{int(slice_height_m)}m"


    # --- Limit Calculation ---
    def get_dynamic_limit(data_map, fallback=10.0):
        valid_data = data_map[~np.isnan(data_map)]
        if len(valid_data) == 0: return fallback
        return np.percentile(np.abs(valid_data), 99)


    lim_u = get_dynamic_limit(u_map, fallback=10.0)
    lim_v = get_dynamic_limit(v_map, fallback=10.0)
    lim_w = 2.0  # Fixed for W

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    current_cmap = plt.cm.jet.copy()
    current_cmap.set_bad(color='black')

    # Plot U
    im0 = axes[0].imshow(u_map, cmap=current_cmap, vmin=-lim_u, vmax=lim_u)
    axes[0].set_title(f"{title_prefix} - Velocity U\nLimit: +/-{lim_u:.1f}")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot V
    im1 = axes[1].imshow(v_map, cmap=current_cmap, vmin=-lim_v, vmax=lim_v)
    axes[1].set_title(f"{title_prefix} - Velocity V\nLimit: +/-{lim_v:.1f}")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot W
    im2 = axes[2].imshow(w_map, cmap=current_cmap, vmin=-lim_w, vmax=lim_w)
    axes[2].set_title(f"{title_prefix} - Velocity W\nLimit: +/-{lim_w:.1f}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # --- Saving ---
    print("Saving raw images...")
    plt.imsave(os.path.join(output_dir, f"{file_prefix}_U_raw.png"), u_map, cmap=current_cmap, vmin=-lim_u, vmax=lim_u)
    plt.imsave(os.path.join(output_dir, f"{file_prefix}_V_raw.png"), v_map, cmap=current_cmap, vmin=-lim_v, vmax=lim_v)
    plt.imsave(os.path.join(output_dir, f"{file_prefix}_W_raw.png"), w_map, cmap=current_cmap, vmin=-lim_w, vmax=lim_w)
    print("Done.")