import numpy as np
import torch
from torchvision import datasets, transforms

def generate_moving_mnist(seq_len=10, num_samples=1000, image_size=64, num_digits=2):
    mnist = datasets.MNIST(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())
    digits = mnist.data.numpy()

    data = np.zeros((num_samples, seq_len, 2, image_size, image_size), dtype=np.float32)

    for i in range(num_samples):
        seq = np.zeros((seq_len, image_size, image_size), dtype=np.float32)
        velocity_map = np.zeros((seq_len, image_size, image_size), dtype=np.float32)

        for _ in range(num_digits):
            digit = digits[np.random.randint(0, len(digits))]

            # Ensure x and y are within valid bounds
            x, y = np.random.randint(0, image_size - 28 + 1, size=2)
            vx, vy = np.random.randint(-5, 6, size=2)

            for t in range(seq_len):

                # Normalize digit to [0,1]
                digit_norm = digit / 255.0

                # Digit mask (True where digit is present)
                mask = digit_norm > 0

                # Place the digit where it belongs
                seq[t, y:y + 28, x:x + 28][mask] = digit_norm[mask]

                # Assign velocity only where digit pixels are (not the whole patch)
                velocity_map[t, y:y + 28, x:x + 28][mask] += vx

                # Update position
                x += vx
                y += vy

                # Handle bouncing
                if x < 0 or x > image_size - 28:
                    vx = -vx
                    x = max(0, min(x, image_size - 28))
                if y < 0 or y > image_size - 28:
                    vy = -vy
                    y = max(0, min(y, image_size - 28))

        data[i, :, 0, :, :] = seq
        # Normalize vx with a safeguard against division by zero
        # range_val = velocity_map.max() - velocity_map.min()
        # if range_val == 0:
        #     data[i, :, 1, :, :] = 0  # If no variation, set the velocity map to 0
        # else:
        #     data[i, :, 1, :, :] = (velocity_map - velocity_map.min()) / range_val
        # Just store the raw velocity map instead:
        data[i, :, 1, :, :] = velocity_map
    return data

if __name__ == "__main__":
    seq_len = 40
    num_samples = 10000
    data = generate_moving_mnist(seq_len=seq_len, num_samples=num_samples)

    # Save in npz format
    np.savez_compressed("moving_mnist_2dig_40seq.npz", data=data)
    print("✅ Dataset with vx map saved to moving_mnist.npz")