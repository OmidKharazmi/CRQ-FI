
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad

# Function to compute the empirical survival function
def empirical_survival_function(data):
    sorted_data = np.sort(data)
    n = len(data)
    survival_function = 1.0 - np.arange(1, n + 1) / n
    return sorted_data, survival_function

    # Calculate the JCF divergence
    jcf_div = alpha * term1 + (1-alpha) * term2 - term3

    return jcf_div

# Function to calculate JCF divergence between two images
def jcf_divergence_images(img1, img2, q_param=0.5):
    return JCF_divergence(p, q, sf_p_kde, sf_q_kde, q_param)

# Function to calculate PSNR between two images
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # No difference
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Main function
def main():
    # Load the original image in grayscale
    img_original = cv2.imread("Path", cv2.IMREAD_GRAYSCALE)

    # Create a list of adjusted images
    img_list = [X,Y,Z,W]
    # Test JCF divergence for different q values
    q_values = [0.5, 0.6]
    for q_value in q_values:
        print(f"\nJCF Divergence for q = {q_value}:")
        jcf_divergences_original_to_adjusted = []
        jcf_divergences_adjusted_to_original = []

        for adjusted_img in img_list[1:]:
            jcf_div = jcf_divergence_images(img_original, adjusted_img, q_param=q_value)
            jcf_divergences_original_to_adjusted.append(jcf_div)
            jcf_div = jcf_divergence_images(adjusted_img, img_original, q_param=q_value)
            jcf_divergences_adjusted_to_original.append(jcf_div)

        print("JCF Divergence from original image to adjusted images:")
        for i, jcf_div in enumerate(jcf_divergences_original_to_adjusted):
            print(f"Adjusted Image {i + 1}: {jcf_div}")

        print("\nJCF Divergence from each adjusted image to original:")
        for i, jcf_div in enumerate(jcf_divergences_adjusted_to_original):
            print(f"Adjusted Image {i + 1}: {jcf_div}")

    # Calculate PSNR between original and adjusted images
    print("\nPSNR between original and adjusted images:")
    psnr_values = []
    for i, adjusted_img in enumerate(img_list[1:], start=1):
        psnr_value = psnr(img_original, adjusted_img)
        psnr_values.append(psnr_value)
        print(f"Adjusted Image {i}: {psnr_value} dB")


