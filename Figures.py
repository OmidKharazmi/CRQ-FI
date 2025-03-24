

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image in grayscale
img_original = cv2.imread("C:/Users/FARHANG/Desktop/image/lake.png", cv2.IMREAD_GRAYSCALE)

# Create a list of adjusted images
img_list = [
    img_original,
    (0.5 * img_original + 0.3).astype(np.uint8),
    (1.75 * img_original).astype(np.uint8),
    (np.sqrt(img_original)).astype(np.uint8)
]
labels = ["X", "Y", "Z", "W"]

# === FIGURES ===
# Plot histograms
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.subplots_adjust(hspace=0.5)
for j, (adjusted_img, ax) in enumerate(zip(img_list, axs.flatten())):
    hist_values, bins, _ = ax.hist(adjusted_img.flatten(), bins=256, color='blue', alpha=0.5, density=True)
    ax.set_xlabel('Pixel Value', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_title(f'{labels[j]}', fontsize=7, loc='left', y=0.9, x=0.02)
plt.tight_layout()
plt.show()

# Plot images
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)
for j, (adjusted_img, ax) in enumerate(zip(img_list, axs.flatten())):
    ax.imshow(adjusted_img, cmap='gray')
    ax.axis('off')
    ax.set_title(f'{labels[j]}', fontsize=10)
plt.tight_layout()
plt.show()

# Data from the table
q_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
JCF_XY = np.array([0.000963, 0.000589, 0.000414, 0.000305, 0.000224, 0.000168, 0.000131, 0.000101])
JCF_XZ = np.array([0.000323, 0.000230, 0.000180, 0.000141, 0.000113, 0.000093, 0.000077, 0.000065])
JCF_XW = np.array([0.020015, 0.013308, 0.009776, 0.007375, 0.005604, 0.004294, 0.003282, 0.002492])

# Plot settings
plt.figure(figsize=(10, 6))
plt.plot(q_values, JCF_XY, 'o-', label=r'$\mathcal{JCF}_q(F_X, F_Y)$')
plt.plot(q_values, JCF_XZ, 's-', label=r'$\mathcal{JCF}_q(F_X, F_Z)$')
plt.plot(q_values, JCF_XW, '^-', label=r'$\mathcal{JCF}_q(F_X, F_W)$')

# Log scale for better visualization
plt.yscale("log")

# Labels and grid
plt.xlabel(r"$q$", fontsize=14)
plt.ylabel(r"$\mathcal{JCF}_q$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show the plot
plt.show()
