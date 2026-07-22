import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
import matplotlib.ticker as ticker

# Global font settings MUST be set FIRST before creating any figures
mpl.rcParams.update({
    'font.size': 60,              # base font size (bigger)
    'axes.titlesize': 64,         # axes title size
    'axes.labelsize': 56,         # X/Y label size
    'xtick.labelsize': 70,        # x tick labels
    'ytick.labelsize': 70,        # y tick labels
    'legend.fontsize': 52,        # legend if used
    'figure.titlesize': 68,       # figure suptitle
    'figure.dpi': 300,
    'savefig.dpi': 150,
    # PDF font embedding to preserve sizes
    'pdf.fonttype': 42,           # embed TrueType fonts
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


class CloudRayCaster:
    def __init__(self, pkl_path, voxel_size=20.0):
        """
        Loads the cloud volume.
        If U or V do not exist, uses W as fallback.
        """
        # 1. Load Data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.vol_beta = data['beta_ext']
        self.vol_w = data['W']
        
        # Check if U and V exist, otherwise use W as fallback
        self.vol_u = data.get('U', self.vol_w)
        self.vol_v = data.get('V', self.vol_w)
        self.has_u = 'U' in data
        self.has_v = 'V' in data

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

        # Return None for U/V if they don't exist
        u_ret = u_map.reshape(H, W_res) if self.has_u else None
        v_ret = v_map.reshape(H, W_res) if self.has_v else None
        w_ret = w_map.reshape(H, W_res)
        return u_ret, v_ret, w_ret

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

        # Return None for U/V if they don't exist
        u_ret = u_map.reshape(H, W_res) if self.has_u else None
        v_ret = v_map.reshape(H, W_res) if self.has_v else None
        w_ret = w_map.reshape(H, W_res)
        return u_ret, v_ret, w_ret


# --- Usage Example ---
if __name__ == "__main__":
    pkl_file = '/wdata_visl/danino/BOMEX_1CLD_256x200_20m_processed/0000003420/sample_004.pkl'
    output_dir = "/home/danino/PycharmProjects/pythonProject/data/output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caster = CloudRayCaster(pkl_file)

    # Define Camera
    camera_pos = [214*1000, 1044*1000, 515*1000]
    camera_pos = [-30*1000, 149*1000, 600*1000]
    camera_pos = [0*1000, 0*1000, 600*1000]

    #camera_pos = [-154*1000, -747*1000, 558*1000]
    look_at = [0, 0, 1500]

    # --- CONFIGURATION ---
    # Choose mode: 'first_hit' (Cloud Surface) or 'slice' (Specific Height)
    render_mode = 'slice'
    slice_height_m = 1000.0  # Height in meters to slice (e.g., middle of cloud)
    
    # Output mode: True = clean image (only velocity map), False = full (with axes, titles, colorbar)
    clean_image_mode = True

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
        if data_map is None:
            return fallback
        valid_data = data_map[~np.isnan(data_map)]
        if len(valid_data) == 0: return fallback
        return np.percentile(np.abs(valid_data), 99)

    lim_u = get_dynamic_limit(u_map, fallback=10.0) if u_map is not None else None
    lim_v = get_dynamic_limit(v_map, fallback=10.0) if v_map is not None else None
    lim_w = 2.0  # Fixed for W

    # --- Resolution-dependent pixel size ---
    H, W = w_map.shape if w_map is not None else (u_map.shape if u_map is not None else v_map.shape)
    if W == 128:
        m_per_pixel = 20
    elif W == 256:
        m_per_pixel = 10
    else:
        m_per_pixel = 20  # default fallback

    # --- Helper to set centered axes ---
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
        # Show: -1100, -500, 0, 500, 1100 instead of -1280, -640, 0, 640, 1280
        tick_vals = np.array([-1100, -500, 0, 500, 1100])
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.set_xticklabels([f"{int(v)}" for v in tick_vals], fontsize=70, fontweight='bold')
        ax.set_yticklabels([f"{int(v)}" for v in tick_vals], fontsize=70, fontweight='bold')

        # Enable tick marks with custom styling
        ax.tick_params(
            axis='both',
            which='major',
            direction='out',  # ticks point outward from plot
            length=14,        # length of tick lines
            width=4,          # width of tick lines
            color='black',    # color of tick lines
            labelsize=70    # make axis numbers large (match set_xticklabels)
        )

        ax.set_xlabel('X [m]', fontsize=70, fontweight='bold')
        ax.set_ylabel('Y [m]', fontsize=70, fontweight='bold')

    # --- Plotting with centered axes and jet colormap ---
    # Determine number of subplots based on available data
    num_plots = sum([u_map is not None, v_map is not None, w_map is not None])
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  # Make axes iterable for single plot case
    
    current_cmap = plt.cm.jet.copy()
    current_cmap.set_bad(color='black')

    # Mask zeros to show them as black (only if they exist)
    u_plot = np.ma.masked_where(u_map == 0, u_map) if u_map is not None else None
    v_plot = np.ma.masked_where(v_map == 0, v_map) if v_map is not None else None
    w_plot = np.ma.masked_where(w_map == 0, w_map) if w_map is not None else None

    # Create normalization for each component (using vmin/vmax with jet)
    from matplotlib.colors import Normalize
    norm_u = Normalize(vmin=-lim_u, vmax=lim_u) if lim_u is not None else None
    norm_v = Normalize(vmin=-lim_v, vmax=lim_v) if lim_v is not None else None
    norm_w = Normalize(vmin=-lim_w, vmax=lim_w)

    # Compute extent in meters so that (0,0) is centered and axes are in meters
    half_w_m = (W * m_per_pixel) / 2.0
    half_h_m = (H * m_per_pixel) / 2.0
    extent_m = [-half_w_m, half_w_m, half_h_m, -half_h_m]

    plot_idx = 0
    
    # Plot U if it exists
    if u_map is not None:
        im0 = axes[plot_idx].imshow(u_plot, cmap=current_cmap, norm=norm_u, extent=extent_m, interpolation='nearest')
        axes[plot_idx].set_title(f"{title_prefix} - Velocity U [m/s]\nLimit: +/-{lim_u:.1f}", pad=24, fontsize=20, fontweight='bold')
        set_centered_meter_axis(axes[plot_idx], H, W, m_per_pixel)
        cbar0 = plt.colorbar(im0, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        cbar0.ax.tick_params(labelsize=70)
        cbar0.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cbar0.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        for label in cbar0.ax.get_yticklabels():
            label.set_fontsize(70)
            label.set_fontweight('bold')
        plot_idx += 1

    # Plot V if it exists
    if v_map is not None:
        im1 = axes[plot_idx].imshow(v_plot, cmap=current_cmap, norm=norm_v, extent=extent_m, interpolation='nearest')
        axes[plot_idx].set_title(f"{title_prefix} - Velocity V [m/s]\nLimit: +/-{lim_v:.1f}", pad=24, fontsize=20, fontweight='bold')
        set_centered_meter_axis(axes[plot_idx], H, W, m_per_pixel)
        cbar1 = plt.colorbar(im1, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=70)
        cbar1.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        for label in cbar1.ax.get_yticklabels():
            label.set_fontsize(70)
            label.set_fontweight('bold')
        plot_idx += 1

    # Plot W (always plot if it exists)
    if w_map is not None:
        im2 = axes[plot_idx].imshow(w_plot, cmap=current_cmap, norm=norm_w, extent=extent_m, interpolation='nearest')
        axes[plot_idx].set_title(f"{title_prefix} - $V_z$ [m/s]\nLimit: +/-{lim_w:.1f}", pad=24, fontsize=20, fontweight='bold')
        set_centered_meter_axis(axes[plot_idx], H, W, m_per_pixel)
        cbar2 = plt.colorbar(im2, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=70)
        cbar2.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cbar2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        for label in cbar2.ax.get_yticklabels():
            label.set_fontsize(70)
            label.set_fontweight('bold')

    # --- Saving with proper colormaps ---
    print("Saving PDFs...")
    # Save U (only if it exists)
    if u_plot is not None:
        fig_u, ax_u = plt.subplots(figsize=(12, 12), dpi=150)
        im_u = ax_u.imshow(u_plot, cmap=current_cmap, norm=norm_u, extent=extent_m, interpolation='nearest')
        
        if clean_image_mode:
            # Clean mode: no axes or title, but with colorbar
            ax_u.axis('off')
            cbar_u = plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
            cbar_u.ax.tick_params(labelsize=70)
            cbar_u.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            cbar_u.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            # Set colorbar tick labels to be large and bold
            for label in cbar_u.ax.get_yticklabels():
                label.set_fontsize(70)
                label.set_fontweight('bold')

            plt.tight_layout(pad=0)
            fig_u.set_size_inches(20, 18)
            plt.subplots_adjust(left=0, right=0.88, top=1, bottom=0)
        else:
            # Full mode: with axes, title, and colorbar
            ax_u.set_title(f"{title_prefix} - Velocity U [m/s]", fontsize=70, fontweight='bold', pad=40)
            set_centered_meter_axis(ax_u, H, W, m_per_pixel)
            cbar_u = plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
            cbar_u.ax.tick_params(labelsize=70)
            cbar_u.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            cbar_u.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            # Set colorbar tick labels to be large and bold
            for label in cbar_u.ax.get_yticklabels():
                label.set_fontsize(70)
                label.set_fontweight('bold')

            plt.tight_layout()
            fig_u.set_size_inches(20, 18)
            plt.subplots_adjust(left=0.19, right=0.87, top=0.99, bottom=0.01)
        
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_U.pdf"), dpi=150)
        plt.close(fig_u)
        print(f"  Saved: {file_prefix}_U.pdf")
    else:
        print("  Skipped: U data not available")

    # Save V (only if it exists)
    if v_plot is not None:
        fig_v, ax_v = plt.subplots(figsize=(12, 12), dpi=150)
        im_v = ax_v.imshow(v_plot, cmap=current_cmap, norm=norm_v, extent=extent_m, interpolation='nearest')
        
        if clean_image_mode:
            # Clean mode: no axes or title, but with colorbar
            ax_v.axis('off')
            cbar_v = plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
            cbar_v.ax.tick_params(labelsize=70)
            cbar_v.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            cbar_v.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            # Set colorbar tick labels to be large and bold
            for label in cbar_v.ax.get_yticklabels():
                label.set_fontsize(70)
                label.set_fontweight('bold')

            plt.tight_layout(pad=0)
            fig_v.set_size_inches(20, 18)
            plt.subplots_adjust(left=0, right=0.88, top=1, bottom=0)
        else:
            # Full mode: with axes, title, and colorbar
            ax_v.set_title(f"{title_prefix} - Velocity V [m/s]", fontsize=70, fontweight='bold', pad=40)
            set_centered_meter_axis(ax_v, H, W, m_per_pixel)
            cbar_v = plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
            cbar_v.ax.tick_params(labelsize=70)
            cbar_v.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            cbar_v.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            # Set colorbar tick labels to be large and bold
            for label in cbar_v.ax.get_yticklabels():
                label.set_fontsize(70)
                label.set_fontweight('bold')

            plt.tight_layout()
            fig_v.set_size_inches(20, 18)
            plt.subplots_adjust(left=0.19, right=0.87, top=0.99, bottom=0.01)
        
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_V.pdf"), dpi=150)
        plt.close(fig_v)
        print(f"  Saved: {file_prefix}_V.pdf")
    else:
        print("  Skipped: V data not available")

    # Save W
    fig_w, ax_w = plt.subplots(figsize=(12, 12), dpi=150)
    im_w = ax_w.imshow(w_plot, cmap=current_cmap, norm=norm_w, extent=extent_m, interpolation='nearest')
    
    if clean_image_mode:
        # Clean mode: no axes or title, but with colorbar
        ax_w.axis('off')
        cbar_w = plt.colorbar(im_w, ax=ax_w, fraction=0.046, pad=0.04)
        cbar_w.ax.tick_params(labelsize=70)
        cbar_w.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cbar_w.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # Set colorbar tick labels to be large and bold
        for label in cbar_w.ax.get_yticklabels():
            label.set_fontsize(70)
            label.set_fontweight('bold')

        plt.tight_layout(pad=0)
        fig_w.set_size_inches(20, 18)
        plt.subplots_adjust(left=0, right=0.88, top=1, bottom=0)
    else:
        # Full mode: with axes, title, and colorbar
        ax_w.set_title(f"{title_prefix} - $V_z$ [m/s]", fontsize=70, fontweight='bold', pad=40)
        set_centered_meter_axis(ax_w, H, W, m_per_pixel)
        cbar_w = plt.colorbar(im_w, ax=ax_w, fraction=0.046, pad=0.04)
        cbar_w.ax.tick_params(labelsize=70)
        cbar_w.ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        cbar_w.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # Set colorbar tick labels to be large and bold
        for label in cbar_w.ax.get_yticklabels():
            label.set_fontsize(70)
            label.set_fontweight('bold')

        plt.tight_layout()
        fig_w.set_size_inches(20, 18)
        plt.subplots_adjust(left=0.23, right=0.88, top=0.99, bottom=0.01)
    
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_W.pdf"), dpi=150)
    plt.close(fig_w)
    print(f"  Saved: {file_prefix}_W.pdf")

    print("Done.")
