import numpy as np

# Create a random array of shape (1, 4, 84, 84, 6)
arr = np.random.rand(1, 4, 84, 84, 6)

# Split along the last axis (axis=4)
matrices = np.split(arr, 6, axis=4)

# Remove the last singleton dimension
matrices = [m.squeeze(axis=4) for m in matrices]

# Print shapes of the resulting matrices
for i, mat in enumerate(matrices):
    print(f"Matrix {i+1} shape: {mat.shape}")
