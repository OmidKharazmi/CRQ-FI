
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

# JCF divergence
def JCF_divergence(f0, f1, F0_bar, F1_bar, q, num_points=1000, epsrel=1e-4, epsabs=1e-4):
    bandwidth = 0.5  # Adjust bandwidth as needed
    f0_kde = gaussian_kde(f0, bw_method=bandwidth)
    f1_kde = gaussian_kde(f1, bw_method=bandwidth)

    # Define the integrands for the divergence calculation
    integrand1 = lambda x: (f0_kde(x) ** 2) * (F0_bar(x) ** (2 * q - 1))
    integrand2 = lambda x: (f1_kde(x) ** 2) * (F1_bar(x) ** (2 * q - 1))
    integrand3 = lambda x: (f_mix) ** 2 * (Fmix_bar(x)) ** (2 

    # Integrate the terms
    min_val, max_val = min(np.min(f0), np.min(f1)), max(np.max(f0), np.max(f1))
    term1, _ = int(integrand1, min_val, max_val, epsrel=epsrel, epsabs=epsabs)
    term2, _ = int(integrand2, min_val, max_val, epsrel=epsrel, epsabs=epsabs)
    term3, _ = int(integrand3, min_val, max_val, epsrel=epsrel, epsabs=epsabs)

    # Calculate the JCF divergence
    jcf_div = alpha * term1 + (1-alpha) * term2 - term3

    return jcf_div

# Function to calculate JCF divergence between two images
def jcf_divergence_images(img1, img2, q_param=0.5):
    p = img1.flatten()
    q = img2.flatten()

    sorted_p, sf_p = empirical_survival_function(p)
    sorted_q, sf_q = empirical_survival_function(q)

    # Interpolation for survival function
    sf_p_kde = lambda x: np.interp(x, sorted_p, sf_p)
    sf_q_kde = lambda x: np.interp(x, sorted_q, sf_q)

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

    # Plot histograms of pixel values
    print("\nPlotting histograms...")
    labels = ["X", "Y", "Z", "W"]
    for i in range(0, len(img_list), 4):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.5)

        for j, (adjusted_img, ax) in enumerate(zip(img_list[i:i + 4], axs.flatten())):
            ax.hist(adjusted_img.flatten(), bins=256, color='blue', alpha=0.5, density=True)
            ax.set_xlabel('Pixel Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(f'{labels[j]}', fontsize=7, loc='left', y=0.9, x=0.02)

        plt.tight_layout()
        plt.show()

        # Display images
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)

        for j, (adjusted_img, ax) in enumerate(zip(img_list[i:i + 4], axs.flatten())):
            ax.imshow(adjusted_img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'{labels[j]}', fontsize=10)

        plt.tight_layout()
        plt.show()

# Call the main function to execute the program
if __name__ == "__main__":
    main()
