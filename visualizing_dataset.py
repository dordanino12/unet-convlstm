import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the dataset
data = np.load("moving_mnist_2.npz")['data']  # Shape: [num_samples, seq_len, 2, H, W]

# Select a sample sequence
sample_idx = 3
sequence_digits = data[sample_idx, :, 0]  # Shape: [seq_len, H, W]
sequence_vx = data[sample_idx, :, 1]      # Shape: [seq_len, H, W]

# Create a figure for visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
frame_digits = ax1.imshow(sequence_digits[0], cmap='gray', vmin=0, vmax=1)
frame_vx = ax2.imshow(sequence_vx[0], cmap='hot', vmin=-5, vmax=5)

ax1.set_title("Digit")
ax2.set_title("Velocity (vx)")
ax1.axis('off')
ax2.axis('off')

# Update function for animation
def update(t):
    frame_digits.set_data(sequence_digits[t])
    frame_vx.set_data(sequence_vx[t])
    return frame_digits, frame_vx

# Create an animation
ani = FuncAnimation(fig, update, frames=sequence_digits.shape[0], interval=200, blit=True)

# Show the animation
plt.show()