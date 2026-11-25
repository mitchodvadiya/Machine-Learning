# Experiment 12: Kernel-Based Image Filtering
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.ndimage import convolve

# Load a sample grayscale image
image = color.rgb2gray(data.astronaut())
image = image[::4, ::4]  # Resize for faster processing

# Sharpening kernel
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

filtered_image = convolve(image, kernel)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.clip(filtered_image, 0, 1), cmap="gray")
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()
plt.show()
