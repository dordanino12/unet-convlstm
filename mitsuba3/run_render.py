
from rednder_from_udi_class import MitsubaRenderer
import numpy as np
from PIL import Image   # <-- Add this import at the top of your file

# --- 1. Define Your Input Data Paths ---
# You must have these files.
csv_file = '/home/danino/PycharmProjects/pythonProject/data/Udi_3satellites_overpass.csv'
cloud_data_file = '/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_3D/BOMEX_512x512x200_20m_20m_1s_512_0000002000_3_3'  # This is the pkl file you have
output_vol_file = 'temp/my_cloud.vol'  # A temporary file this script will create

# Define which rows from your CSV to use
# This example assumes 2 timestamps and 3 satellites (6 total)
overpass_indices = [0,1,2]
# overpass_indices = [0,1,2,3,4,5]
# overpass_indices = [6,7,8,9,10,11]
# overpass_indices = [12,13,14,15,16,17]
# overpass_indices = [18,19,20,21,22,23]
# overpass_indices = [24,25,26,27,22,23]
# overpass_indices = [30,31,32,33,34,35]




# --- 2. Set Up the Renderer Parameters ---
# These are the arguments for the __init__ method.
# You will need to get these values from your project's specifications.
renderer_params = {
    'overpass_csv': csv_file,
    'overpass_indices': overpass_indices,
    'spp': 16,  # Samples per pixel (image quality). Start low (e.g., 16).
    'g_value': 0,  # Cloud anisotropy (a physical property)
    'cloud_width': 128,  # Width of the cloud volume in voxels
    'voxel_res': 0.02,  # Resolution of each voxel (in km)
    'scene_scale': 1000.0,
    'cloud_zrange': [0.0, 4.0],  # Altitude of the cloud (2km to 6km)
    'satellites': 3,
    'timestamps': 1,
    'pad_image': True,
    'dynamic_emitter': True,
    'centralize_cloud': True,
    'bitmaps_required': True,
    'vol_path': output_vol_file  # Tell the renderer where to save the .vol file
}

print("Initializing MitsubaRenderer...")
# --- 3. Create an instance of the renderer ---
renderer = MitsubaRenderer(**renderer_params)

print("Reading CSV and setting up cameras...")
# --- 4. Run the setup methods ---
renderer.read_overpass_csv()
renderer.camera_params()
renderer.create_sensors()

print(f"Loading cloud data from '{cloud_data_file}'...")
# --- 5. Load your cloud data from the .pkl file ---
# This will create the 'temp/my_cloud.vol' file.
# We are passing 'sample_path' to load the pkl, and 'vol_path' to save the .vol
renderer.write_vol_file(sample_path=cloud_data_file,
                        vol_path=output_vol_file,
                        param_type='beta_ext',  # This must match a key in your pkl file!
                        sample_ext='pkl')

print("Setting up the 3D scene(s)...")
# --- 6. Build the 3D scene(s) in memory ---
renderer.set_scenes()

print("Rendering... This may take a while.")
# --- 7. Run the render! ---
# This will render all 6 images (2 timestamps * 3 satellites)
# 'tensors' will be a list of NumPy arrays (image data)
# 'bitmaps' will be a list of PIL-compatible images
tensor_stacks, bitmap_stacks = renderer.render_scenes()

print("Rendering complete!")

# --- 8. Do something with the output ---
import numpy as np
import matplotlib.pyplot as plt

# Check if rendering produced any output
if not bitmap_stacks or bitmap_stacks[0] is None:
    print("No images were rendered.")
else:
    print("Processing images for plotting...")

    # Get dimensions from the renderer object
    n_timestamps = renderer.timestamps
    n_satellites = renderer.satellites

    # Create a figure with a grid of subplots
    # (rows, cols) = (timestamps, satellites)
    # 'squeeze=False' ensures 'axes' is always a 2D array, even if you have only 1 row or col
    fig, axes = plt.subplots(n_timestamps, n_satellites,
                             figsize=(n_satellites * 4, n_timestamps * 4),
                             squeeze=False)

    fig.suptitle('All Rendered Satellite Images', fontsize=16)

    # Loop through all timestamps and satellites
    for t in range(n_timestamps):
        for s in range(n_satellites):
            # Get the specific image (it's 32-bit float data)
            float_data = bitmap_stacks[t][s]

            # --- Normalize and convert to 8-bit integer ---
            # (This is the same fix as before, applied to each image)
            if float_data.max() > 0:
                normalized_data = float_data / float_data.max()
            else:
                normalized_data = float_data

            # Convert to 8-bit integer (0-255)
            image_data_uint8 = (normalized_data * 255).astype(np.uint8)
            # --- End of conversion ---

            # Display the image on its subplot
            ax = axes[t, s]
            ax.imshow(image_data_uint8, cmap='gray')
            ax.set_title(f'Timestamp {t}, Satellite {s}')
            ax.axis('off')  # Hide the x/y axis ticks

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for the suptitle

    print("Displaying plot...")
    plt.show()  # Show the plot window

# You can still access the raw data if needed:
first_tensor_data = tensor_stacks[0][0]
print(f"Shape of the first tensor: {first_tensor_data.shape}")