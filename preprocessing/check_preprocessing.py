import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure  # נדרש עבור Marching Cubes


def visualize_cloud_mesh(beta_array, title="Cloud Surface", save_path=None, downsample_factor=1):
    """
    Method 2: Isosurface (Marching Cubes).
    Creates a continuous surface mesh around the cloud data.
    Best for checking continuity and shape.
    
    Args:
        beta_array: 3D array of beta extinction values
        title: Plot title
        save_path: Path to save the plot (if None, only displays)
        downsample_factor: Factor to reduce resolution (set to 1 for no downsampling)
    """
    print(f">> Original shape: {beta_array.shape}")
    
    # 1. Threshold
    # We look for the surface where beta_ext is roughly this value
    level = 0

    # Check if we have enough density for a surface
    if beta_array.max() < level:
        print(">> Cloud too thin for isosurface visualization.")
        return

    # 2. Marching Cubes Algorithm with step_size for reduced detail
    print(">> Running marching cubes...")
    try:
        # step_size=2 reduces mesh detail significantly (4x faster)
        verts, faces, normals, values = measure.marching_cubes(beta_array, level=level, step_size=2)
    except Exception as e:
        print(f">> Could not generate mesh: {e}")
        return

    print(f">> Generated Mesh: {len(verts)} vertices, {len(faces)} faces")

    # 3. Plotting
    print(">> Creating plot...")
    fig = plt.figure(figsize=(18, 8))
    
    # Left: 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    # plot_trisurf creates the continuous skin with fixed color
    mesh = ax1.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0], triangles=faces,
                           color='dodgerblue', alpha=0.6, lw=0, edgecolor='none')

    # Formatting 3D plot
    ax1.set_xlabel('X (Width)')
    ax1.set_ylabel('Y (Depth)')
    ax1.set_zlabel('Z (Height)')
    ax1.set_title(f"3D Cloud Surface: {title}")

    # Keep aspect ratio reasonable
    d_z, d_y, d_x = beta_array.shape
    ax1.set_xlim(0, d_x)
    ax1.set_ylim(0, d_y)
    ax1.set_zlim(0, min(100, d_z))
    
    # Right: 2D side view (Y-Z plane, looking from X direction)
    ax2 = fig.add_subplot(122)
    
    # Create 2D projection by taking max projection along X axis
    side_view = np.max(beta_array, axis=2)  # Max along X (width)
    
    # Display the side view
    im = ax2.imshow(side_view, cmap='Blues', aspect='auto', origin='lower',
                    extent=[0, d_y, 0, min(100, d_z)])
    ax2.set_xlabel('Y (Depth)')
    ax2.set_ylabel('Z (Height)')
    ax2.set_title(f"2D Side View (X-projection): {title}")
    plt.colorbar(im, ax=ax2, label='Max Beta Value')
    
    plt.tight_layout()
    
    if save_path:
        print(f">> Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Reduced DPI for faster saving
        print(f">> Plot saved!")
        plt.close(fig)  # Close without showing for faster execution
    else:
        plt.show()


def inspect_and_visualize(folder_path, specific_file, save_dir=None):
    file_path = os.path.join(folder_path, specific_file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    beta = data['beta_ext']

    print("Option 2: Continuous Surface (Marching Cubes)")
    
    # Create save path if save_dir is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{os.path.splitext(specific_file)[0]}_cloud_mesh.png")
    
    visualize_cloud_mesh(beta, title=specific_file, save_path=save_path)


# --- Run ---
dataset_folder = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/0000007000/'
save_folder = './cloud_visualizations'  # Set to None to disable saving

# Increase downsample_factor (e.g., 4) for even faster processing
inspect_and_visualize(dataset_folder, "sample_010.pkl", save_dir=save_folder)