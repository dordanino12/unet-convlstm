import numpy as np
import pickle
import matplotlib.pyplot as plt


class CloudRayCaster:
    def __init__(self, pkl_path, voxel_size=20.0):
        """
        Loads the cloud volume.
        Coordinate System:
          X, Y: Centered at 0 (Range: -Size/2 to +Size/2)
          Z:    Starts at 0   (Range: 0 to +Size)
        """
        # 1. Load Data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        #
        self.vol_beta = data['beta_ext']
        self.vol_w = data['W']

        # 2. Define Grid Dimensions
        self.nz, self.ny, self.nx = self.vol_beta.shape
        self.voxel_size = voxel_size

        # 3. Calculate Physical Sizes
        self.size_x = self.nx * voxel_size  # 1280m
        self.size_y = self.ny * voxel_size  # 1280m
        self.size_z = self.nz * voxel_size  # 4000m

        # 4. Define Bounds (The logic you requested)
        # X and Y are centered around 0 (-640 to +640)
        # Z starts at 0 (0 to 4000)
        self.min_bound = np.array([-self.size_x / 2, -self.size_y / 2, 0.0])
        self.max_bound = np.array([self.size_x / 2, self.size_y / 2, self.size_z])

        print(f"--- Volume Loaded ---")
        print(f"Grid Shape: {self.vol_beta.shape}")
        print(f"World Bounds:")
        print(f"  X: [{self.min_bound[0]}, {self.max_bound[0]}]")
        print(f"  Y: [{self.min_bound[1]}, {self.max_bound[1]}]")
        print(f"  Z: [{self.min_bound[2]}, {self.max_bound[2]}]")

    def get_rays(self, cam_pos, look_at, resolution=(128, 128), fov=0.25):
        H, W = resolution

        # Camera Coordinate System
        cam_dir = (look_at - cam_pos)
        cam_dir = cam_dir / np.linalg.norm(cam_dir)

        world_up = np.array([1, 0, 0])
        # if np.abs(np.dot(cam_dir, world_up)) > 0.99:
        #     world_up = np.array([0, 1, 0])

        cam_right = np.cross(cam_dir, world_up)
        cam_right = cam_right / np.linalg.norm(cam_right)

        cam_up = np.cross(cam_right, cam_dir)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # Image Plane
        aspect_ratio = W / H
        scale = np.tan(np.deg2rad(fov * 0.5))

        i, j = np.meshgrid(np.arange(W), np.arange(H))

        x = ((2 * (i + 0.5) / W - 1) * aspect_ratio * scale)
        y = (1 - 2 * (j + 0.5) / H) * scale

        rays_d = x[..., np.newaxis] * cam_right + \
                 y[..., np.newaxis] * cam_up + \
                 cam_dir

        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_o = np.broadcast_to(cam_pos, rays_d.shape)

        return rays_o, rays_d

    def render_velocity_map(self, cam_pos, look_at, resolution=(128, 128), step_size=20.0):
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

        velocity_map = np.full(num_rays, np.nan)

        # 3. Ray Marching
        valid_indices = np.where(mask_box_hit)[0]

        if len(valid_indices) == 0:
            return velocity_map.reshape(H, W_res)

        print(f"Marching {len(valid_indices)} rays...")

        r_o = rays_o_flat[valid_indices]
        r_d = rays_d_flat[valid_indices]
        t_start = tmin[valid_indices]
        t_end = tmax[valid_indices]

        max_dist = np.max(t_end - t_start)
        num_steps = int(max_dist / step_size) + 2

        steps = np.arange(num_steps) * step_size
        t_vals = t_start[:, np.newaxis] + steps[np.newaxis, :]

        points = r_o[:, np.newaxis, :] + r_d[:, np.newaxis, :] * t_vals[:, :, np.newaxis]

        # 4. Convert World Positions to Grid Indices
        # Formula: Index = (Position - MinBound) / VoxelSize
        # Example for X: (Pos - (-640)) / 20  => (Pos + 640) / 20
        grid_indices = (points - self.min_bound) / self.voxel_size
        grid_indices = grid_indices.astype(int)

        # Mapping: World(X,Y,Z) -> Grid(Z,Y,X) standard numpy indexing
        # World X (index 0) -> Grid X (index 2)
        # World Y (index 1) -> Grid Y (index 1)
        # World Z (index 2) -> Grid Z (index 0)

        gx = np.clip(grid_indices[:, :, 0], 0, self.nx - 1)
        gy = np.clip(grid_indices[:, :, 1], 0, self.ny - 1)
        gz = np.clip(grid_indices[:, :, 2], 0, self.nz - 1)

        in_bounds_mask = t_vals <= t_end[:, np.newaxis]

        sampled_beta = self.vol_beta[gz, gy, gx]
        sampled_beta[~in_bounds_mask] = 0

        # 5. Find First Hit
        hit_cloud = sampled_beta > 1e-5
        first_hit_idx = np.argmax(hit_cloud, axis=1)
        has_hit = np.any(hit_cloud, axis=1)

        # 6. Sample W
        ray_idx_range = np.arange(len(valid_indices))

        hit_z = gz[ray_idx_range, first_hit_idx]
        hit_y = gy[ray_idx_range, first_hit_idx]
        hit_x = gx[ray_idx_range, first_hit_idx]

        w_values = self.vol_w[hit_z, hit_y, hit_x]

        final_values = np.full(len(valid_indices), np.nan)
        final_values[has_hit] = w_values[has_hit]

        velocity_map[valid_indices] = final_values

        return velocity_map.reshape(H, W_res)


# --- Usage Example ---
if __name__ == "__main__":
    pkl_file = '/wdata_visl/danino/dataset_128_overlap/sample_006.pkl'
    caster = CloudRayCaster(pkl_file)

    # Define Camera
    # Camera is at (0,0,600) -> Center X, Center Y, 600m Height
    camera_pos = [0, 0, 600000] # 600km

    # Look slightly up to catch more clouds if needed, or straight ahead
    # If we look at (0, 0, 600) we look straight forward relative to camera height
    # Let's look at (0, 100, 600) to look "Forward" along Y axis
    look_at = [0, 0, 1500]

    print(f"Cam: {camera_pos}, LookAt: {look_at}")

    w_map = caster.render_velocity_map(
        cam_pos=camera_pos,
        look_at=look_at,
        resolution=(256, 256),
        step_size=20.0
    )

    plt.figure(figsize=(8, 6))
    current_cmap = plt.cm.jet.copy()
    current_cmap.set_bad(color='black')

    plt.imshow(w_map, cmap=current_cmap, vmin=-2, vmax=2)
    plt.colorbar(label='Vertical Velocity (W) [m/s]')
    plt.title(f"Velocity Map (W)\nCam: {camera_pos}")
    plt.show()