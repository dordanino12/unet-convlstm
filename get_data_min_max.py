import numpy as np

# Path to your npz file
npz_path = "dataset_sequences_original.npz"

# Load the npz file
data = np.load(npz_path)
print("Available keys:", data.files)

# Assuming the targets are stored in Y_all
Y = data["Y"]  # shape: [N, T, 1, H, W]
X = data["X"]  # shape: [N, T, 1, H, W]


# Compute min and max velocity over all sequences
min_velocity_Y = Y.min()
max_velocity_Y = Y.max()
min_velocity_X = X.min()
max_velocity_X = X.max()

print(f"Min velocity in dataset X: {min_velocity_X}")
print(f"Max velocity in dataset X: {max_velocity_X}")
print(f"Min velocity in dataset Y: {min_velocity_Y}")
print(f"Max velocity in dataset Y: {max_velocity_Y}")


