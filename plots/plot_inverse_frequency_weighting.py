#!/usr/bin/env python3
"""
Academic-style dual-axis plot for inverse-frequency bin weighting.

This script:
1) Loads vertical velocity (Vz) values from an NPZ file.
2) Builds uniform bins over [bin_min, bin_max].
3) Computes inverse-frequency weights:
      omega_bin ∝ N_total / (N_bin + eps)
4) Normalizes and clips weights for clean visualization.
5) Plots:
   - Left axis: histogram (density) of Vz
   - Right axis: bin weights over bin centers

Example:
    python plots/plot_inverse_frequency_weighting.py \
        --npz data/dataset_trajectory_sequences_samples_W_top_w_fixed_w.npz \
        --key auto \
        --save plots/inverse_frequency_bin_weighting.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# PDF font embedding settings (matching run_render.py)
mpl.rcParams.update({
    'pdf.fonttype': 42,           # embed TrueType fonts
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def infer_velocity_key(npz_obj: np.lib.npyio.NpzFile, preferred_key: str | None) -> str:
    """Select a key containing vertical velocity values from the NPZ archive."""
    if preferred_key not in (None, "auto"):
        if preferred_key not in npz_obj.files:
            raise KeyError(
                f"Requested key '{preferred_key}' not found. Available keys: {npz_obj.files}"
            )
        return preferred_key

    for candidate in ("W", "w", "Y", "y", "vz", "Vz", "V_z", "velocity", "vel", "target"):
        if candidate in npz_obj.files:
            return candidate

    if len(npz_obj.files) == 0:
        raise ValueError("NPZ archive has no arrays.")

    return npz_obj.files[0]


def compute_inverse_frequency_weights(
    values: np.ndarray,
    bin_min: float,
    bin_max: float,
    bin_width: float,
    eps: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute inverse-frequency weights per bin.

    Returns:
        centers: bin centers
        counts: raw counts per bin
        weights: normalized + clipped inverse-frequency weights
    """
    edges = np.arange(bin_min, bin_max + bin_width, bin_width, dtype=np.float64)
    if edges[-1] < bin_max:
        edges = np.append(edges, bin_max)

    clipped_values = np.clip(values, bin_min, bin_max - 1e-9)
    counts, _ = np.histogram(clipped_values, bins=edges)

    total = counts.sum()
    raw_weights = total / (counts.astype(np.float64) + eps)

    non_empty = counts > 0
    if np.any(non_empty):
        raw_weights[non_empty] /= raw_weights[non_empty].mean()

    weights = np.clip(raw_weights, 0.0, clip_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, weights


def build_plot(
    values: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    bin_min: float,
    bin_max: float,
    hist_bins: int,
    out_path: Path,
    dpi: int,
) -> None:
    """Create and save dual-axis academic-style figure."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax1 = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)

    # Left axis: velocity histogram (density)
    ax1.hist(
        values,
        bins=hist_bins,
        range=(bin_min, bin_max),
        density=True,
        color="#9ecae1",
        edgecolor="white",
        alpha=0.85,
    )

    hist_density, hist_edges = np.histogram(
        values,
        bins=hist_bins,
        range=(bin_min, bin_max),
        density=True,
    )
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    line_dist, = ax1.plot(
        hist_centers,
        hist_density,
        color="#1f77b4",
        linewidth=2.2,
        label="Velocity Distribution (Density)",
    )
    ax1.set_xlabel(r"Vertical Velocity $V_z$ (m/s)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Pixel Frequency (Density)", fontsize=14, color="#1f4e79", fontweight='bold')
    ax1.tick_params(axis="y", labelcolor="#1f4e79")

    # Right axis: inverse-frequency bin weights
    ax2 = ax1.twinx()

    # Ensure ax2 (red line) is drawn on top of ax1 (blue line)
    ax2.set_zorder(ax1.get_zorder() + 1)
    ax2.patch.set_visible(False)

    line_weight, = ax2.plot(
        centers,
        weights,
        color="#d62728",
        linewidth=2.8,
        label=r"Bin Weight ($\Omega_{bin}$)",
    )
    ax2.set_ylabel(r"Bin Weight ($\Omega_{bin}$)", fontsize=14, color="#d62728", fontweight='bold')
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlim(bin_min, bin_max)
    ax1.set_title("Inverse-Frequency Bin Weighting", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.35)

    # Unified legend across both axes
    legend = ax2.legend(
        [line_dist, line_weight],
        ["Velocity Distribution (Density)", r"Bin Weight ($\Omega_{bin}$)"],
        loc="upper right",
        ncol=1,
        fontsize=11,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.7",
    )
    for text in legend.get_texts():
        text.set_fontweight('bold')
    legend.set_zorder(1000)

    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    
    # Make tick labels bold
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dual-axis plot for inverse-frequency bin weighting."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="data/dataset_trajectory_sequences_samples_W_top_w_fixed_w.npz",
        help="Path to NPZ file containing vertical velocity values.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="auto",
        help="NPZ key for velocity values. Use 'auto' for key inference.",
    )
    parser.add_argument("--bin-min", type=float, default=-7.60, help="Minimum bin value.")
    parser.add_argument("--bin-max", type=float, default=8.78, help="Maximum bin value.")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Uniform bin width.")
    parser.add_argument("--hist-bins", type=int, default=220, help="Histogram bins for density plot.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for stable division.")
    parser.add_argument(
        "--clip-max",
        type=float,
        default=12.0,
        help="Maximum plotted bin weight after normalization.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="plots/inverse_frequency_bin_weighting.pdf",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI for paper-quality export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path) as data:
        key = infer_velocity_key(data, args.key)
        values = np.asarray(data[key], dtype=np.float64).reshape(-1)

    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    if values.size == 0:
        raise ValueError("No valid finite velocity values found.")

    centers, _, weights = compute_inverse_frequency_weights(
        values=values,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        bin_width=args.bin_width,
        eps=args.eps,
        clip_max=args.clip_max,
    )

    build_plot(
        values=values,
        centers=centers,
        weights=weights,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        hist_bins=args.hist_bins,
        out_path=Path(args.save),
        dpi=args.dpi,
    )

    print(f"Saved figure to: {args.save}")


if __name__ == "__main__":
    main()
