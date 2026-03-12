import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------
# CONFIG (edit these values directly)
# ---------------------------------------------------------
PKL_N_MINUS_1 = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split_beta,U,V,W_fixed/0000002160/sample_013.pkl"
PKL_N = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split_beta,U,V,W_fixed/0000002160/sample_013.pkl"
PKL_N_PLUS_1 = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split_beta,U,V,W_fixed/0000002160/sample_013.pkl"
DT_SECONDS = 20.0
VOXEL_SIZE_M = 20.0
BETA_THRESHOLD = 0.0
TEMPORAL_XY_WINDOW = 9
Z_MIN_FOCUS_M = 300.0
Z_MAX_FOCUS_M = 1700.0
OUTPUT_FIG = "/home/danino/PycharmProjects/pythonProject/plots/envelope_vs_flow_side_xz.pdf"
OUTPUT_SCATTER_FIG = "/home/danino/PycharmProjects/pythonProject/plots/flow_vs_envelope_change_scatter.pdf"

# Use many cloud files for the flow-vs-structure scatter.
# Expected example: .../0000003000/sample_010.pkl, .../0000003020/sample_010.pkl, ...
SCATTER_PKL_GLOB = "/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split_beta,U,V,W_fixed/*/sample_010.pkl"
SCATTER_TRIPLET_STRIDE = 1
SCATTER_MAX_TRIPLETS = 300
SCATTER_TYPE_RANGES = [
    ("Type 1", 2000, 3000),
    ("Type 2", 5000, 6000),
    ("Type 3", 12000, 13000),
]


def load_beta_w(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "beta_ext" in data:
        beta = np.asarray(data["beta_ext"], dtype=np.float32)
    elif "beta" in data:
        beta = np.asarray(data["beta"], dtype=np.float32)
    else:
        raise KeyError(f"No beta field found in {pkl_path}. Expected 'beta_ext' or 'beta'.")

    if "W" in data:
        w = np.asarray(data["W"], dtype=np.float32)
    elif "w" in data:
        w = np.asarray(data["w"], dtype=np.float32)
    else:
        raise KeyError(f"No W field found in {pkl_path}. Expected 'W' or 'w'.")

    if beta.shape != w.shape:
        raise ValueError(f"beta shape {beta.shape} != W shape {w.shape} in {pkl_path}")

    return beta, w


def first_hit_from_top(beta, threshold=0.0):
    """
    Returns top-most z index per (y, x), i.e. first beta > threshold when looking from above.
    beta shape is (z, y, x).
    """
    cloud = beta > threshold
    valid = np.any(cloud, axis=0)  # (y, x)

    rev = cloud[::-1, :, :]
    first_rev_idx = np.argmax(rev, axis=0)

    nz = beta.shape[0]
    z_top = (nz - 1 - first_rev_idx).astype(np.int32)
    z_top[~valid] = -1
    return z_top, valid


def sample_w_at_envelope(w, z_top, valid):
    """Sample W at envelope z index for each (y, x)."""
    ny, nx = valid.shape
    w_env = np.full((ny, nx), np.nan, dtype=np.float32)
    yy, xx = np.where(valid)
    zz = z_top[yy, xx]
    w_env[yy, xx] = w[zz, yy, xx]
    return w_env


def top_line_profile_per_x(z_top, valid):
        """
        Build one side-view envelope line by taking, for each x,
        the highest envelope height across y.

        Returns:
            z_line: (x,) top z index per x, -1 when invalid
            y_line: (x,) y index where z_line was taken, -1 when invalid
            valid_x: (x,) bool, whether line exists at this x
        """
        ny, nx = z_top.shape
        z_masked = np.where(valid, z_top, -1)
        y_line = np.argmax(z_masked, axis=0).astype(np.int32)
        z_line = z_masked[y_line, np.arange(nx)].astype(np.int32)
        valid_x = z_line >= 0
        y_line[~valid_x] = -1
        return z_line, y_line, valid_x


def bottom_line_profile_per_x(z_top, valid):
        """
        Build one side-view bottom-envelope line by taking, for each x,
        the lowest envelope height across y.

        Returns:
            z_bottom: (x,) bottom z index per x, -1 when invalid
            y_bottom: (x,) y index where z_bottom was taken, -1 when invalid
            valid_x: (x,) bool, whether line exists at this x
        """
        ny, nx = z_top.shape
        z_masked = np.where(valid, z_top, np.iinfo(np.int32).max)
        y_bottom = np.argmin(z_masked, axis=0).astype(np.int32)
        z_bottom = z_masked[y_bottom, np.arange(nx)].astype(np.int32)
        valid_x = z_bottom != np.iinfo(np.int32).max
        z_bottom[~valid_x] = -1
        y_bottom[~valid_x] = -1
        return z_bottom, y_bottom, valid_x


def robust_sym_limit(arr, fallback=1.0):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return fallback
    lim = np.percentile(np.abs(vals), 99)
    if lim <= 0:
        lim = fallback
    return float(lim)


def sample_highest_z_in_xy_window(beta, y_idx, x_idx, threshold=0.0, window_size=8):
    """
    For each reference (y, x), search an XY window in the given 3D beta volume and
    return the highest z index where beta > threshold.

    Returns:
        z_out: int32 array, -1 when no voxel passes threshold in the search window.
        valid_out: bool array
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")

    nz, ny, nx = beta.shape
    y_idx = np.asarray(y_idx, dtype=np.int32)
    x_idx = np.asarray(x_idx, dtype=np.int32)

    if y_idx.shape != x_idx.shape:
        raise ValueError("y_idx and x_idx must have the same shape")

    n = y_idx.size
    z_out = np.full(n, -1, dtype=np.int32)
    valid_out = np.zeros(n, dtype=bool)

    half = int(window_size) // 2
    # Keep exactly window_size samples away from borders when possible.
    extra = int(window_size) - half

    for i in range(n):
        y = int(y_idx[i])
        x = int(x_idx[i])

        y0 = max(0, y - half)
        y1 = min(ny, y + extra)
        x0 = max(0, x - half)
        x1 = min(nx, x + extra)

        local_cloud = beta[:, y0:y1, x0:x1] > threshold
        if not np.any(local_cloud):
            continue

        z_any = np.any(local_cloud, axis=(1, 2))
        z_out[i] = int(np.max(np.where(z_any)[0]))
        valid_out[i] = True

    return z_out, valid_out


def extract_flow_vs_env_pairs(pkl_prev, pkl_curr, pkl_next):
    """Return paired samples of flow, envelope speed, and envelope height for one triplet."""
    beta_prev, _ = load_beta_w(pkl_prev)
    beta_curr, w_curr = load_beta_w(pkl_curr)
    beta_next, _ = load_beta_w(pkl_next)

    if not (beta_prev.shape == beta_curr.shape == beta_next.shape):
        raise ValueError(
            f"beta shapes mismatch: n-1 {beta_prev.shape}, n {beta_curr.shape}, n+1 {beta_next.shape}"
        )

    z_prev, valid_prev = first_hit_from_top(beta_prev, threshold=BETA_THRESHOLD)
    z_curr, valid_curr = first_hit_from_top(beta_curr, threshold=BETA_THRESHOLD)
    z_next, valid_next = first_hit_from_top(beta_next, threshold=BETA_THRESHOLD)

    z_line_curr, y_line_curr, valid_x_curr = top_line_profile_per_x(z_curr, valid_curr)

    nx = z_curr.shape[1]
    w_line = np.full(nx, np.nan, dtype=np.float32)
    idx_w = np.where(valid_x_curr)[0]
    if idx_w.size > 0:
        z_idx = z_line_curr[idx_w]
        y_idx = y_line_curr[idx_w]
        w_line[idx_w] = w_curr[z_idx, y_idx, idx_w]

    env_vz_line = np.full(nx, np.nan, dtype=np.float32)
    valid_x_all = np.zeros(nx, dtype=bool)
    idx_curr = np.where(valid_x_curr)[0]
    if idx_curr.size > 0:
        y_ref = y_line_curr[idx_curr]

        z_prev_ref, prev_ok = sample_highest_z_in_xy_window(
            beta_prev,
            y_ref,
            idx_curr,
            threshold=BETA_THRESHOLD,
            window_size=TEMPORAL_XY_WINDOW,
        )
        z_next_ref, next_ok = sample_highest_z_in_xy_window(
            beta_next,
            y_ref,
            idx_curr,
            threshold=BETA_THRESHOLD,
            window_size=TEMPORAL_XY_WINDOW,
        )

        keep = prev_ok & next_ok
        idx_v = idx_curr[keep]
        if idx_v.size > 0:
            z_prev_keep = z_prev_ref[keep].astype(np.float32)
            z_next_keep = z_next_ref[keep].astype(np.float32)
            dz_m = (z_next_keep - z_prev_keep) * float(VOXEL_SIZE_M)
            env_vz_line[idx_v] = dz_m / float(2.0 * DT_SECONDS)
            valid_x_all[idx_v] = True

    pair_mask = np.isfinite(w_line) & np.isfinite(env_vz_line) & valid_x_all
    z_line_m = z_line_curr.astype(np.float32) * float(VOXEL_SIZE_M)
    return w_line[pair_mask], env_vz_line[pair_mask], z_line_m[pair_mask]


def collect_multi_file_pairs(range_start, range_end):
    """Collect flow/envelope-speed pairs from many triplets in a given folder-index range."""
    all_paths = sorted(glob(SCATTER_PKL_GLOB))

    # Keep only files whose parent folder index is inside [range_start, range_end].
    filtered_paths = []
    for p in all_paths:
        folder_name = os.path.basename(os.path.dirname(p))
        try:
            idx = int(folder_name)
        except ValueError:
            continue
        if range_start <= idx <= range_end:
            filtered_paths.append(p)

    all_paths = filtered_paths
    if len(all_paths) < 3:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0

    triplet_indices = list(range(1, len(all_paths) - 1, max(1, int(SCATTER_TRIPLET_STRIDE))))
    if SCATTER_MAX_TRIPLETS is not None:
        triplet_indices = triplet_indices[: int(SCATTER_MAX_TRIPLETS)]

    flow_chunks = []
    env_chunks = []
    z_chunks = []
    used_triplets = 0

    for i in triplet_indices:
        p_prev = all_paths[i - 1]
        p_curr = all_paths[i]
        p_next = all_paths[i + 1]
        try:
            flow_vals, env_vals, z_vals = extract_flow_vs_env_pairs(p_prev, p_curr, p_next)
        except Exception:
            continue

        if flow_vals.size == 0:
            continue

        flow_chunks.append(flow_vals)
        env_chunks.append(env_vals)
        z_chunks.append(z_vals)
        used_triplets += 1

    if not flow_chunks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0

    return np.concatenate(flow_chunks), np.concatenate(env_chunks), np.concatenate(z_chunks), used_triplets


def main():
    title_fs = 40
    label_fs = 34
    tick_fs = 32
    cbar_label_fs = 30
    cbar_tick_fs = 32
    suptitle_fs = 30

    beta_prev, _ = load_beta_w(PKL_N_MINUS_1)
    beta_curr, w_curr = load_beta_w(PKL_N)
    beta_next, _ = load_beta_w(PKL_N_PLUS_1)

    if not (beta_prev.shape == beta_curr.shape == beta_next.shape):
        raise ValueError(
            f"beta shapes mismatch: n-1 {beta_prev.shape}, n {beta_curr.shape}, n+1 {beta_next.shape}"
        )

    nz, ny, nx = beta_curr.shape

    z_prev, valid_prev = first_hit_from_top(beta_prev, threshold=BETA_THRESHOLD)
    z_curr, valid_curr = first_hit_from_top(beta_curr, threshold=BETA_THRESHOLD)
    z_next, valid_next = first_hit_from_top(beta_next, threshold=BETA_THRESHOLD)

    # Build one top-envelope line per x at time n.
    # IMPORTANT: This (x,y) selection is the reference location for temporal comparison.
    z_line_curr, y_line_curr, valid_x_curr = top_line_profile_per_x(z_curr, valid_curr)
    z_bottom_curr, y_bottom_curr, valid_x_bottom = bottom_line_profile_per_x(z_curr, valid_curr)

    # Left panel values: W at time n, sampled exactly on the current top-envelope line.
    w_line = np.full(nx, np.nan, dtype=np.float32)
    idx_w = np.where(valid_x_curr)[0]
    if idx_w.size > 0:
        z_idx = z_line_curr[idx_w]
        y_idx = y_line_curr[idx_w]
        w_line[idx_w] = w_curr[z_idx, y_idx, idx_w]

    # Right panel values: envelope vertical speed at n using centered finite difference.
    # For n-1 and n+1, search a TEMPORAL_XY_WINDOW x TEMPORAL_XY_WINDOW area around
    # the (x,y) selected at time n and take the highest z where beta > threshold.
    # vz_env(n) = (z_{n+1}(window around x,y_n) - z_{n-1}(window around x,y_n)) * voxel / (2*dt)
    env_vz_line = np.full(nx, np.nan, dtype=np.float32)
    valid_x_all = np.zeros(nx, dtype=bool)
    idx_curr = np.where(valid_x_curr)[0]
    if idx_curr.size > 0:
        y_ref = y_line_curr[idx_curr]

        z_prev_ref, prev_ok = sample_highest_z_in_xy_window(
            beta_prev,
            y_ref,
            idx_curr,
            threshold=BETA_THRESHOLD,
            window_size=TEMPORAL_XY_WINDOW,
        )
        z_next_ref, next_ok = sample_highest_z_in_xy_window(
            beta_next,
            y_ref,
            idx_curr,
            threshold=BETA_THRESHOLD,
            window_size=TEMPORAL_XY_WINDOW,
        )

        keep = prev_ok & next_ok
        idx_v = idx_curr[keep]
        if idx_v.size > 0:
            z_prev_keep = z_prev_ref[keep].astype(np.float32)
            z_next_keep = z_next_ref[keep].astype(np.float32)
            dz_m = (z_next_keep - z_prev_keep) * float(VOXEL_SIZE_M)
            env_vz_line[idx_v] = dz_m / float(2.0 * DT_SECONDS)
            valid_x_all[idx_v] = True

    lim_w = robust_sym_limit(w_line, fallback=1.0)
    lim_vz = robust_sym_limit(env_vz_line, fallback=1.0)
    diff_line = w_line - env_vz_line
    lim_diff = robust_sym_limit(diff_line, fallback=1.0)
    shared_lim = max(lim_w, lim_vz)

    x_coords_m = (np.arange(nx, dtype=np.float32) - (nx / 2.0)) * float(VOXEL_SIZE_M)
    z_line_m = z_line_curr.astype(np.float32) * float(VOXEL_SIZE_M)
    z_bottom_m = z_bottom_curr.astype(np.float32) * float(VOXEL_SIZE_M)
    x_half = (nx * VOXEL_SIZE_M) / 2.0
    z_max = nz * VOXEL_SIZE_M

    fig, axes = plt.subplots(1, 3, figsize=(30, 9), dpi=180)

    cmap = plt.cm.jet.copy()
    norm = TwoSlopeNorm(vmin=-shared_lim, vcenter=0.0, vmax=shared_lim)

    # Panel 1: one top-envelope line at time n, colored by W(n) on that line.
    idx_plot_w = np.where(np.isfinite(w_line) & valid_x_curr & (z_line_curr >= 0))[0]
    im0 = axes[0].scatter(
        x_coords_m[idx_plot_w],
        z_line_m[idx_plot_w],
        c=w_line[idx_plot_w],
        cmap=cmap,
        norm=norm,
        s=220,
        marker="o",
        linewidths=0,
    )
    axes[0].plot(x_coords_m[idx_plot_w], z_line_m[idx_plot_w], color="white", linewidth=0.8, alpha=0.4)
    idx_plot_bottom = np.where(valid_x_bottom & (z_bottom_curr >= 0))[0]
    axes[0].plot(x_coords_m[idx_plot_bottom], z_bottom_m[idx_plot_bottom], color="red", linewidth=3.5, alpha=0.95)
    axes[0].set_title("Envelope Vertical Flow [m/s]", fontsize=title_fs)
    axes[0].set_xlabel("X [m]", fontsize=label_fs)
    axes[0].set_ylabel("Z [m]", fontsize=label_fs)
    axes[0].set_xlim(-x_half, x_half)
    axes[0].set_ylim(Z_MIN_FOCUS_M, Z_MAX_FOCUS_M)
    axes[0].set_xticks([-1000, -500, 0, 500, 1000])
    axes[0].set_yticks([400, 800, 1200, 1600])
    axes[0].tick_params(axis="both", labelsize=tick_fs)
    axes[0].set_facecolor("black")
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    c0.ax.tick_params(labelsize=cbar_tick_fs)

    # Panel 2: the same style line, colored by centered envelope vertical speed at n.
    idx_plot_v = np.where(np.isfinite(env_vz_line) & valid_x_all & (z_line_curr >= 0))[0]
    im1 = axes[1].scatter(
        x_coords_m[idx_plot_v],
        z_line_m[idx_plot_v],
        c=env_vz_line[idx_plot_v],
        cmap=cmap,
        norm=norm,
        s=220,
        marker="o",
        linewidths=0,
    )
    axes[1].plot(x_coords_m[idx_plot_v], z_line_m[idx_plot_v], color="white", linewidth=0.8, alpha=0.4)
    axes[1].plot(x_coords_m[idx_plot_bottom], z_bottom_m[idx_plot_bottom], color="red", linewidth=3.5, alpha=0.95)
    axes[1].set_title("Envelope Structure Change Speed [m/s]", fontsize=title_fs)
    axes[1].set_xlabel("X [m]", fontsize=label_fs)
    axes[1].set_xlim(-x_half, x_half)
    axes[1].set_ylim(Z_MIN_FOCUS_M, Z_MAX_FOCUS_M)
    axes[1].set_xticks([-1000, -500, 0, 500, 1000])
    axes[1].set_yticks([400, 800, 1200, 1600])
    axes[1].tick_params(axis="both", labelsize=tick_fs)
    axes[1].set_facecolor("black")
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    c1.ax.tick_params(labelsize=cbar_tick_fs)

    # Panel 3: difference on the same line (W - envelope speed).
    norm_diff = TwoSlopeNorm(vmin=-lim_diff, vcenter=0.0, vmax=lim_diff)
    idx_plot_d = np.where(
        np.isfinite(diff_line) & np.isfinite(w_line) & np.isfinite(env_vz_line) & valid_x_all & (z_line_curr >= 0)
    )[0]
    im2 = axes[2].scatter(
        x_coords_m[idx_plot_d],
        z_line_m[idx_plot_d],
        c=diff_line[idx_plot_d],
        cmap=cmap,
        norm=norm_diff,
        s=220,
        marker="o",
        linewidths=0,
    )
    axes[2].plot(x_coords_m[idx_plot_d], z_line_m[idx_plot_d], color="white", linewidth=0.8, alpha=0.4)
    axes[2].plot(x_coords_m[idx_plot_bottom], z_bottom_m[idx_plot_bottom], color="red", linewidth=3.5, alpha=0.95)
    axes[2].set_title("Difference [m/s]", fontsize=title_fs)
    axes[2].set_xlabel("X [m]", fontsize=label_fs)
    axes[2].set_xlim(-x_half, x_half)
    axes[2].set_ylim(Z_MIN_FOCUS_M, Z_MAX_FOCUS_M)
    axes[2].set_xticks([-1000, -500, 0, 500, 1000])
    axes[2].set_yticks([400, 800, 1200, 1600])
    axes[2].tick_params(axis="both", labelsize=tick_fs)
    axes[2].set_facecolor("black")
    c2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    c2.ax.tick_params(labelsize=cbar_tick_fs)

    plt.tight_layout()

    out_dir = os.path.dirname(OUTPUT_FIG)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=160)
    plt.close(fig)

    # Additional figure: 3 scatter panels for three different folder ranges.
    type_results = []
    for type_name, range_start, range_end in SCATTER_TYPE_RANGES:
        flow_vals, env_change_vals, z_vals, used_triplets = collect_multi_file_pairs(range_start, range_end)
        if flow_vals.size == 0:
            # Fallback to currently configured single triplet if a specific range has no data.
            pair_mask = np.isfinite(w_line) & np.isfinite(env_vz_line) & valid_x_all
            flow_vals = w_line[pair_mask]
            env_change_vals = env_vz_line[pair_mask]
            z_vals = z_line_m[pair_mask]
            used_triplets = 1 if flow_vals.size > 0 else 0
        avg_height_m = float(np.mean(z_vals)) if z_vals.size > 0 else float("nan")
        type_results.append((type_name, range_start, range_end, flow_vals, env_change_vals, avg_height_m, used_triplets))

    all_vals = []
    for _, _, _, fvals, evals, _, _ in type_results:
        if fvals.size > 0:
            all_vals.append(fvals)
            all_vals.append(evals)
    scatter_lim = robust_sym_limit(np.concatenate(all_vals), fallback=1.0) if all_vals else 1.0

    fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(22, 9), dpi=180)

    for ax_scatter, (_, _, _, flow_vals, env_change_vals, avg_height_m, _) in zip(axes_scatter, type_results):
        ax_scatter.scatter(flow_vals, env_change_vals, s=70, alpha=0.75, edgecolors="none", rasterized=True)
        ax_scatter.plot(
            [-8, 8],
            [-8, 8],
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.8,
            rasterized=True,
        )
        ax_scatter.set_xlim(-8, 8)
        ax_scatter.set_ylim(-8, 8)
        ax_scatter.set_xticks([-5, 0, 5])
        ax_scatter.set_yticks([-5, 0, 5])
        
        if np.isfinite(avg_height_m):
            ax_scatter.set_title(f"{avg_height_m:.0f} m", fontsize=title_fs, weight="bold")
        else:
            ax_scatter.set_title("Avg Envelope Height: N/A", fontsize=title_fs, weight="bold")
            
        ax_scatter.set_xlabel("Envelope Airflow [m/s]", fontsize=label_fs, weight="bold")
        
        # Apply general tick size first
        ax_scatter.tick_params(axis="both", labelsize=tick_fs)
        
        if ax_scatter is axes_scatter[0]:
            # First plot keeps y-axis label and ticks
            ax_scatter.set_ylabel("Envelope Motion [m/s]", fontsize=label_fs, weight="bold")
        else:
            # For middle and right plots: remove labels AND the tick lines (left=False)
            ax_scatter.tick_params(axis="y", left=False, labelleft=False)
            
        plt.setp(ax_scatter.get_xticklabels(), weight="bold")
        plt.setp(ax_scatter.get_yticklabels(), weight="bold")
        ax_scatter.grid(alpha=0.25)
        ax_scatter.set_aspect("equal", adjustable="box")

    # Reduce horizontal space between the subplots to bring them closer
    # Use wspace=0.0 for no gap at all, or a small value like 0.05
    plt.subplots_adjust(wspace=0.001)

    out_dir_scatter = os.path.dirname(OUTPUT_SCATTER_FIG)
    if out_dir_scatter:
        os.makedirs(out_dir_scatter, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_SCATTER_FIG, dpi=160)
    plt.close(fig_scatter)

    print("[INFO] Saved figure:", OUTPUT_FIG)
    print("[INFO] Saved scatter figure:", OUTPUT_SCATTER_FIG)
    for type_name, range_start, range_end, flow_vals, _, avg_height_m, used_triplets in type_results:
        print(
            f"[INFO] {type_name} range [{range_start}, {range_end}] -> "
            f"points: {int(flow_vals.size)}, triplets: {used_triplets}, avg_height_m: {avg_height_m:.2f}"
        )
    print("[INFO] n-1:", PKL_N_MINUS_1)
    print("[INFO] n:", PKL_N)
    print("[INFO] n+1:", PKL_N_PLUS_1)


if __name__ == "__main__":
    main()
