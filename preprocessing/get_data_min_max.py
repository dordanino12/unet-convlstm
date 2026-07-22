import numpy as np
import matplotlib.pyplot as plt

# Path to your npz file
npz_path = "/home/danino/PycharmProjects/pythonProject/data/dataset_trajectory_sequences_samples_500m_slices_w.npz"

# Load the npz file
print(f"[INFO] Loading data from {npz_path}...")
data = np.load(npz_path)
print("Available keys:", data.files)

# Assuming the targets are stored in Y
Y = data["Y"]  # shape: [N, T, 1, H, W]
X = data["X"]

# Compute min and max (Global)
min_velocity_Y = Y.min()
max_velocity_Y = Y.max()

print(f"Min velocity in dataset Y: {min_velocity_Y}")
print(f"Max velocity in dataset Y: {max_velocity_Y}")

# ---------------------------------------------------------
# HISTOGRAM PLOT (NON-ZERO VALUES ONLY)
# ---------------------------------------------------------
print("[INFO] Filtering zeros and generating histogram...")

# 1. Flatten the array
y_flat = Y.flatten()

# 2. Filter out ZERO values (Background)
# We use a small epsilon or exact inequality if data is truly 0
y_non_zero = y_flat[y_flat != 0]

print(f"Original count: {len(y_flat)}")
print(f"Non-zero count: {len(y_non_zero)}")

if len(y_non_zero) == 0:
    print("[WARN] No non-zero values found! Check your data.")
else:
    plt.figure(figsize=(10, 6))

    # Plot histogram of valid pixels only
    plt.hist(y_non_zero, bins=100, color='mediumseagreen', edgecolor='black', alpha=0.7)

    plt.title(f"Histogram of Y Values (Zeros Excluded)\nMin: {y_non_zero.min():.2f}, Max: {y_non_zero.max():.2f}")
    plt.xlabel("Velocity Value")
    plt.ylabel("Frequency (Pixel Count)")
    plt.grid(axis='y', alpha=0.5)

    plt.show()