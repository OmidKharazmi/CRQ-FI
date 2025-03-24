import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

# Read the image
image_path = "C:/Users/FARHANG/Desktop/image/68077.jpg"
try:
    im = imread(image_path)
    print("Image successfully loaded.")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}.")
    exit(1)

# Ensure the image is in grayscale if it has multiple channels
if im.ndim == 3:  # If the image is RGB
    im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    print("Converted RGB image to grayscale.")

# Display the original grayscale image
plt.figure()
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title("Grayscale Image")
plt.show()

# Display the histogram
plt.figure()
hist, bins = np.histogram(im.flatten(), bins=np.arange(257), density=True)
plt.hist(im.flatten(), bins=np.arange(257), density=True, color='blue', alpha=0.7)
plt.title("Histogram of Pixel Intensities")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# Function to compute JCRE for a histogram
def compute_JCRE_histogram(h, t, w):
    h = h / np.sum(h)  # Normalize histogram to probabilities
    p1 = h[:t]
    p2 = h[t:]

    # Normalize sub-histograms
    p1 =p1 / np.sum(p1) if np.sum(p1) > 0 else p1
    p2 =p2 / np.sum(p2) if np.sum(p2) > 0 else p2
    ee =1e-10
    # Compute terms
    term_1 = w *len(p2)* np.sum([np.sum(p1[:s]) * np.log(np.sum(p1[:s]) + ee) for s in range(1, len(p1) + 1)])
    term_2 = (1 - w) **len(p1)* np.sum([np.sum(p2[:r]) * np.log(np.sum(p2[:r]) + ee) for r in range(1, len(p2) + 1)])

    term_3 = 0
    for s in range(1, len(p1) + 1):
        for r in range(1, len(p2) + 1):
            value = (w *(np.sum(p1[:s])) + (1 - w) *(np.sum(p2[:r]))) * np.log(w *(np.sum(p1[:s])) + (1 - w) *(np.sum(p2[:r]))+ee)
            term_3 += value

    return term_1 + term_2 - term_3

# JCRE-based thresholding function
def jcre_threshold(h):
    ee =1e-10
    h = h / h.sum()
    # Normalize histogram
    total_w = np.sum([np.sum(h[:t]) for t in range(1, len(h))])
    jcre_values = np.zeros(len(h))

    for t in range(1, len(h) - 1):
        w = np.sum(h[:t])
        jcre_values[t] = compute_JCRE_histogram(h, t, w)

    best_t = np.argmax(jcre_values)
    return best_t, jcre_values

# Compute the JCRE threshold
h, bins = np.histogram(im.flatten(), bins=np.arange(257))
threshold, jcre_values = jcre_threshold(h)
print(f"JCRE threshold: {threshold}")

# Apply the threshold to segment the image using the threshold found by JCRE
segmented_image_jcre = (im > threshold).astype(int)

# Apply Otsu's thresholding method to determine the optimal threshold
otsu_threshold = threshold_otsu(im)
print(f"Otsu's threshold: {otsu_threshold}")

# Apply the threshold to segment the image using Otsu's method
segmented_image_otsu = (im > otsu_threshold).astype(int)

# K-means clustering for image segmentation
# Reshape the image into a 2D array (pixels, features)
pixels = im.flatten().reshape(-1, 1)

# Apply K-means with 2 clusters (binary segmentation)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(pixels)
segmented_image_kmeans = kmeans.labels_.reshape(im.shape)

# Reverse the K-means labels (invert segmentation)
kmeans_cluster_centers = kmeans.cluster_centers_.flatten()
if kmeans_cluster_centers[0] > kmeans_cluster_centers[1]:
    segmented_image_kmeans = 1 - segmented_image_kmeans

# Ground truth placeholder (replace with actual binary image if available)
ground_truth = (im > 100).astype(int)

# Compute evaluation metrics
def compute_metrics(ground_truth, segmented_image):
    ground_truth_flat = ground_truth.flatten()
    segmented_image_flat = segmented_image.flatten()

    TP = np.sum((segmented_image_flat == 1) & (ground_truth_flat == 1))
    FP = np.sum((segmented_image_flat == 1) & (ground_truth_flat == 0))
    TN = np.sum((segmented_image_flat == 0) & (ground_truth_flat == 0))
    FN = np.sum((segmented_image_flat == 0) & (ground_truth_flat == 1))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    dice_coefficient = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    jaccard_index = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    ari_value = adjusted_rand_score(ground_truth_flat, segmented_image_flat)

    return accuracy, recall, precision, specificity, f1_score, dice_coefficient, jaccard_index, ari_value

# Calculate metrics for JCRE thresholding
metrics_jcre = compute_metrics(ground_truth, segmented_image_jcre)
metrics_otsu = compute_metrics(ground_truth, segmented_image_otsu)
metrics_kmeans = compute_metrics(ground_truth, segmented_image_kmeans)

metric_names = ["Accuracy", "Recall", "Precision", "Specificity", "F1-Score", "Dice Coefficient", "Jaccard Index", "Adjusted Rand Index (ARI)"]

print("\nMetrics for JCRE Thresholding:")
for name, value in zip(metric_names, metrics_jcre):
    print(f"{name}: {value}")

print("\nMetrics for Otsu Thresholding:")
for name, value in zip(metric_names, metrics_otsu):
    print(f"{name}: {value}")

print("\nMetrics for K-means Segmentation:")
for name, value in zip(metric_names, metrics_kmeans):
    print(f"{name}: {value}")

# Compute SSIM and PSNR for both methods
ssim_jcre = ssim(im, segmented_image_jcre.astype(float), data_range=im.max() - im.min())
psnr_jcre = psnr(im, segmented_image_jcre.astype(float), data_range=im.max() - im.min())

ssim_otsu = ssim(im, segmented_image_otsu.astype(float), data_range=im.max() - im.min())
psnr_otsu = psnr(im, segmented_image_otsu.astype(float), data_range=im.max() - im.min())

ssim_kmeans = ssim(im, segmented_image_kmeans.astype(float), data_range=im.max() - im.min())
psnr_kmeans = psnr(im, segmented_image_kmeans.astype(float), data_range=im.max() - im.min())

print(f"\nSSIM for JCRE: {ssim_jcre}")
print(f"PSNR for JCRE: {psnr_jcre}")

print(f"\nSSIM for Otsu: {ssim_otsu}")
print(f"PSNR for Otsu: {psnr_otsu}")

print(f"\nSSIM for K-means: {ssim_kmeans}")
print(f"PSNR for K-means: {psnr_kmeans}")



# Display Segmentation Results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Ground Truth
axes[0, 0].imshow(ground_truth, cmap=plt.cm.gray)
axes[0, 0].axis('off')
axes[0, 0].set_title("Ground Truth (GT)")

# Jensen-GS Segmented Image
axes[0, 1].imshow(segmented_image_jcre, cmap=plt.cm.gray)
axes[0, 1].axis('off')
axes[0, 1].set_title("Segmented Image ( Algorithm $I_{S}^{*}$)")

# Otsu Segmented Image
axes[1, 0].imshow(segmented_image_otsu, cmap=plt.cm.gray)
axes[1, 0].axis('off')
axes[1, 0].set_title("Segmented Image (Otsu)")

# K-means Segmented Image
axes[1, 1].imshow(segmented_image_kmeans, cmap=plt.cm.gray)
axes[1, 1].axis('off')
axes[1, 1].set_title("Segmented Image (K-means)")

plt.tight_layout()
# Save plot
save_path = "C:/Users/FARHANG/Desktop/image/T351093.png"
plt.savefig(save_path, format='png', dpi=300)
print(f"Plot saved successfully at {save_path}")
plt.show()

# Display Original, Grayscale, and Histogram in One Window
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Colored Image
original_image = imread(image_path)
axes[0].imshow(original_image)
axes[0].axis('off')
axes[0].set_title("Original Colored Image")

# Grayscale Image
axes[1].imshow(im, cmap=plt.cm.gray)
axes[1].axis('off')
axes[1].set_title("Grayscale Image")

# Histogram of Grayscale Image
hist, bins = np.histogram(im.flatten(), bins=np.arange(257), density=True)
axes[2].hist(im.flatten(), bins=np.arange(257), density=True, color='blue', alpha=0.7)
axes[2].set_title("Histogram of Pixel Intensities")
axes[2].set_xlabel("Pixel Value")
axes[2].set_ylabel("Frequency")
plt.tight_layout()

# Save the plot
save_path_tt1 = "C:/Users/FARHANG/Desktop/image/TT351093.png"
plt.savefig(save_path_tt1, format='png', dpi=300)
print(f"Combined plot saved successfully at {save_path_tt1}")
plt.show()
