"""
Analyze Beta Distribution by Z-axis
====================================
Processes all pickle files in dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W) folder
and builds histograms of beta extinction values (where beta > 0)
according to the z-axis levels.
"""

import os
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def extract_beta_by_z(pkl_path):
    """
    Extract beta extinction values from a single pickle file.
    Returns a dictionary mapping z-level to list of beta values where beta > 0.

    Args:
        pkl_path: Path to the pickle file

    Returns:
        dict: {z_level: [beta_values_where_beta_>_0]}
        beta_array: the full beta array for shape information
    """
    beta_by_z = {}

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error opening {pkl_path}: {e}")
        return beta_by_z, None

    try:
        # Extract beta extinction data
        beta = data['beta_ext']  # Shape: (Z, Y, X) = (200, 128, 128)
        n_z = beta.shape[0]

        # Initialize dictionary for each z level
        for z_idx in range(n_z):
            beta_by_z[z_idx] = []

        # Extract beta values by z-level where beta > 0
        for z_idx in range(n_z):
            beta_slice = beta[z_idx, :, :]

            # Filter for beta > 0
            valid_beta = beta_slice[beta_slice > 0]

            if len(valid_beta) > 0:
                beta_by_z[z_idx].extend(valid_beta.tolist())

        return beta_by_z, beta

    except Exception as e:
        print(f"Error processing {pkl_path}: {e}")
        return beta_by_z, None


def process_all_pkl_files(input_folder, n_samples=100, random_seed=42):
    """
    Process pickle files and aggregate beta values by z-level.
    Recursively searches for .pkl files in all subfolders and randomly samples a fixed number.

    Args:
        input_folder: Path to folder containing subfolders with .pkl files
        n_samples: Number of files to randomly sample (default: 100)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        aggregated_beta_by_z: dict with aggregated beta values for each z-level
        n_z: number of z levels
        n_sampled: actual number of files sampled
    """
    # Find all .pkl files recursively (including in subfolders)
    pkl_files = glob.glob(os.path.join(input_folder, "**/*.pkl"), recursive=True)

    if not pkl_files:
        print(f"No .pkl files found in {input_folder} or its subfolders")
        return {}, None, 0

    print(f"Total pickle files found: {len(pkl_files)}")

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Sample a fixed number of files
    n_sampled = min(n_samples, len(pkl_files))
    pkl_files_sampled = random.sample(pkl_files, n_sampled)

    print(f"Randomly sampled {n_sampled} files (seed={random_seed})")

    # Sort sampled files numerically
    try:
        pkl_files_sampled.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
    except Exception as e:
        print(f"Warning: Could not sort numerically. Error: {e}")

    # Aggregate beta values across all files
    aggregated_beta_by_z = {}
    n_z = None

    print("\nProcessing pickle files...")
    for pkl_file in tqdm(pkl_files_sampled, desc="Processing sampled files"):
        beta_by_z, beta_array = extract_beta_by_z(pkl_file)

        if n_z is None and beta_array is not None:
            n_z = beta_array.shape[0]
            # Initialize aggregated dict
            for z_idx in range(n_z):
                aggregated_beta_by_z[z_idx] = []

        # Aggregate beta values
        for z_idx, beta_values in beta_by_z.items():
            aggregated_beta_by_z[z_idx].extend(beta_values)

    return aggregated_beta_by_z, n_z, n_sampled


def create_mean_median_plot(aggregated_beta_by_z, n_z, output_dir):
    """
    Create plots showing mean/median beta and count of beta > 0 by z-level.

    Args:
        aggregated_beta_by_z: dict with aggregated beta values for each z-level
        n_z: number of z levels
        output_dir: directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nGenerating mean/median and count plots...")

    # Calculate mean, median, and count for each z-level
    z_levels = list(range(n_z))
    mean_betas = []
    median_betas = []
    count_betas = []

    for z_idx in z_levels:
        beta_values = aggregated_beta_by_z[z_idx]
        if len(beta_values) > 0:
            mean_betas.append(np.mean(beta_values))
            median_betas.append(np.median(beta_values))
            count_betas.append(len(beta_values))
        else:
            mean_betas.append(0)
            median_betas.append(0)
            count_betas.append(0)

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=150)

    # Plot 1: Mean and Median by Z-level
    ax1.plot(z_levels, mean_betas, 'o-', color='red', linewidth=3, markersize=10, label='Mean')
    ax1.plot(z_levels, median_betas, 's-', color='orange', linewidth=3, markersize=10, label='Median')
    ax1.set_xlabel('Z-level Index', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Beta [1/m]', fontsize=16, fontweight='bold')
    ax1.set_title('Mean and Median Beta Extinction by Z-level', fontsize=18, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.legend(fontsize=14, loc='best')
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # Plot 2: Count of beta > 0 by Z-level
    ax2.bar(z_levels, count_betas, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Z-level Index', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Count (beta > 0)', fontsize=16, fontweight='bold')
    ax2.set_title('Number of Valid Beta Values by Z-level', fontsize=18, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linewidth=1.5, axis='y')
    ax2.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "beta_mean_median_by_z.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print(f"\nPlot complete!")


# --- Run Configuration ---
if __name__ == "__main__":
    # Input and output paths
    input_directory = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/'
    output_directory = '/home/danino/PycharmProjects/pythonProject/preprocessing/beta_analysis/'

    # Configuration for random sampling
    n_samples = 100  # Number of files to randomly sample
    random_seed = 42  # For reproducibility

    # Process pickle files
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Sampling configuration: n_samples={n_samples}, random_seed={random_seed}")

    aggregated_beta_by_z, n_z, n_sampled = process_all_pkl_files(
        input_directory,
        n_samples=n_samples,
        random_seed=random_seed
    )

    if len(aggregated_beta_by_z) > 0:
        # Create mean and median plot
        create_mean_median_plot(aggregated_beta_by_z, n_z, output_directory)
        print(f"\nAnalysis complete! Results saved to {output_directory}")
        print(f"Processed {n_sampled} files out of total dataset")
    else:
        print("No data was processed. Check input directory.")

