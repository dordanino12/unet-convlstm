import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure  # נדרש עבור Marching Cubes


def visualize_cloud_mesh(beta_array, title="Cloud Surface"):
    """
    Method 2: Isosurface (Marching Cubes).
    Creates a continuous surface mesh around the cloud data.
    Best for checking continuity and shape.
    """
    # 1. Threshold
    # We look for the surface where beta_ext is roughly this value
    level = 0

    # Check if we have enough density for a surface
    if beta_array.max() < level:
        print(">> Cloud too thin for isosurface visualization.")
        return

    # 2. Marching Cubes Algorithm
    # This creates triangles (vertices and faces) to connect the voxels
    try:
        verts, faces, normals, values = measure.marching_cubes(beta_array, level=level)
    except Exception as e:
        print(f">> Could not generate mesh: {e}")
        return

    print(f">> Generated Mesh: {len(verts)} vertices, {len(faces)} faces")

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # plot_trisurf creates the continuous skin
    mesh = ax.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0], triangles=faces,
                           cmap='Blues', alpha=0.6, lw=0, edgecolor='none')

    # Formatting
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_title(f"Continuous Cloud Surface: {title}")

    # Keep aspect ratio reasonable
    d_z, d_y, d_x = beta_array.shape
    ax.set_xlim(0, d_x)
    ax.set_ylim(0, d_y)
    ax.set_zlim(0, d_z)

    plt.colorbar(mesh, ax=ax, label='Cloud Altitude/Depth')
    plt.show()


def inspect_and_visualize(folder_path, specific_file):
    file_path = os.path.join(folder_path, specific_file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    beta = data['beta_ext']

    print("Option 2: Continuous Surface (Marching Cubes)")
    visualize_cloud_mesh(beta, title=specific_file)


# --- Run ---
dataset_folder = '/wdata_visl/danino/dataset_128_overlap/'
inspect_and_visualize(dataset_folder, "sample_006.pkl")