import numpy as np
import h5py
import glob
import cv2  # Used for resizing the images

# --- Configuration ---
# Set to True to apply the specific crop before resizing, or False to resize the entire image directly
USE_CROP = True

# Set to True to flip the raw image horizontally (left-to-right) before processing
FLIP_IMAGE = True

# Set to True to add an additional horizontal flip after the image is cropped and resized
FLIP_AFTER_CROP = False

# --- Multiplier Configuration ---
# The center image will be multiplied by MAX_MULTIPLIER.
# The multiplier will decrease by DROP_OFF_STEP for each frame moving away from the center.
# For 9 frames with MAX=45 and STEP=5, the multipliers will be: [25, 30, 35, 40, 45, 40, 35, 30, 25]
MAX_MULTIPLIER = 50.0
DROP_OFF_STEP = 3.5


# ---------------------

def crop_and_resize_specific(image):
    # Define the exact bounding box for the crop
    start_y, end_y = 1300, 2050
    start_x, end_x = 750, 1550
    # start_y, end_y = 1547, 1803
    # start_x, end_x = 1022, 1278

    # Crop the specific region based on provided coordinates
    cropped_region = image[start_y:end_y, start_x:end_x]

    # Resize the crop to 128x128
    # This effectively changes the resolution to match the model's input
    final_image = cv2.resize(cropped_region, (128, 128), interpolation=cv2.INTER_AREA)

    # Apply the additional horizontal flip (left-to-right) after cropping and resizing
    if FLIP_AFTER_CROP:
        final_image = np.fliplr(final_image)

    return final_image


# Define input directory and format
directory = '/wdata/yaelsc/AirMSPI_raw_data/raw_data/'
format_ = '*.hdf'
paths = sorted(glob.glob(directory + '/' + format_))

# Limit to exactly 9 frames to match the required sequence length
num_frames = 9
paths = paths[:num_frames]

sequence_images = []

# Find the middle index (for 9 frames, center_idx is 4)
center_idx = num_frames // 2

for i, path in enumerate(paths):
    f = h5py.File(path, 'r')
    channels_data = f['HDFEOS']['GRIDS']

    # Extract only the red channel (660nm band)
    img_660 = np.array(channels_data['660nm_band']['Data Fields']['I'])

    # Handle invalid -999 values
    img_660[img_660 == -999] = 0

    # Flip the raw image horizontally (left-to-right) BEFORE cropping if enabled
    # This reverses the apparent viewing angle / motion direction
    if FLIP_IMAGE:
        img_660 = np.fliplr(img_660)

    # Calculate the dynamic multiplier for the current frame
    # Distance from center determines how much the multiplier is reduced
    distance_from_center = abs(i - center_idx)
    current_multiplier = MAX_MULTIPLIER - (distance_from_center * DROP_OFF_STEP)

    # Multiply every pixel value by the calculated dynamic multiplier
    img_660 = img_660 * current_multiplier

    # Apply specific crop and resize, or just resize based on the USE_CROP flag
    if USE_CROP:
        image_processed = crop_and_resize_specific(img_660)
    else:
        # Resize the entire original image directly to 128x128 without cropping
        # Note: This changes the aspect ratio since the original is not square.
        image_processed = cv2.resize(img_660, (128, 128), interpolation=cv2.INTER_AREA)

        # We also need to check the post-crop flip flag here if USE_CROP is False
        if FLIP_AFTER_CROP:
            image_processed = np.fliplr(image_processed)

    sequence_images.append(image_processed)

# Convert the list to a numpy array: shape becomes (9, 128, 128)
sequence_images = np.array(sequence_images)

# Duplicate the single camera view to simulate 2 stereo inputs
# We stack along a new axis to get shape: (9, 2, 128, 128)
sequence_views = np.stack((sequence_images, sequence_images), axis=1)

# Add the batch dimension at the beginning
# Final X shape: (1, 9, 2, 128, 128)
X = np.expand_dims(sequence_views, axis=0)

# Create a dummy Y array with random uniform values between -7 and 7
# Final Y shape: (1, 9, 1, 128, 128)
Y = np.random.uniform(low=-7.0, high=7.0, size=(1, 9, 1, 128, 128)).astype(X.dtype)

# Save the arrays to an NPZ file
output_filename = 'airmspi_experiment_data.npz'
np.savez(output_filename, X=X, Y=Y)

print("------------------------------------")
print(f"Data successfully saved to {output_filename}")
print(f"Applied Cropping: {USE_CROP}")
print(f"Applied Initial Flipping: {FLIP_IMAGE}")
print(f"Applied Post-Crop Flipping: {FLIP_AFTER_CROP}")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Y min value: {Y.min():.2f}")
print(f"Y max value: {Y.max():.2f}")
print("------------------------------------")