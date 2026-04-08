from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import cv2
import os

def calculate_metrics_for_folders(folder1_path, folder2_path, output_file_path):
    """
    Reads images from two folders, calculates PSNR and SSIM for images
    with the same name, computes their mean and variance, and logs results
    to a file.

    Args:
        folder1_path (str): Path to the first folder containing images.
        folder2_path (str): Path to the second folder containing images.
        output_file_path (str): Path to the file where results will be saved.
    """
    psnr_values = []
    ssim_values = []

    # Get list of files in both directories
    try:
        files_folder1 = set(os.listdir(folder1_path))
        files_folder2 = set(os.listdir(folder2_path))
    except FileNotFoundError as e:
        print(f"Error: One or both folders not found. {e}")
        return
    except Exception as e:
        print(f"An error occurred while listing files: {e}")
        return

    # Find common filenames
    common_files = sorted(list(files_folder1.intersection(files_folder2)))

    if not common_files:
        print("No common image files found in the specified folders.")
        return

    print(f"Found {len(common_files)} common files for comparison.")

    with open(output_file_path, 'w') as f_out:
        f_out.write("Filename\tPSNR\tSSIM\n")
        print("Filename\tPSNR\tSSIM")

        for filename in common_files:
            img1_path = os.path.join(folder1_path, filename)
            img2_path = os.path.join(folder2_path, filename)

            try:
                # Read images in grayscale
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

                if img1 is None:
                    print(f"Warning: Could not read image {img1_path}. Skipping.")
                    f_out.write(f"{filename}\tError reading image1\tError reading image1\n")
                    continue
                if img2 is None:
                    print(f"Warning: Could not read image {img2_path}. Skipping.")
                    f_out.write(f"{filename}\tError reading image2\tError reading image2\n")
                    continue

                # Ensure images have the same dimensions
                if img1.shape != img2.shape:
                    # Option 1: Resize the larger image to the smaller one's dimensions
                    # Or resize both to a predefined size, e.g., (256, 256)
                    # For this example, let's resize img1 to img2's shape if different.
                    # You might want a more sophisticated strategy depending on your needs.
                    print(f"Warning: Image shapes for {filename} differ: {img1.shape} vs {img2.shape}. Resizing img1 to match img2.")
                    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)
                    # If you prefer to skip:
                    # print(f"Warning: Image shapes for {filename} differ: {img1.shape} vs {img2.shape}. Skipping.")
                    # f_out.write(f"{filename}\tShape mismatch\tShape mismatch\n")
                    # continue


                # Calculate PSNR
                # peak_signal_noise_ratio requires the data_range to be specified
                # if the images are not in the default range (e.g., 0-255 for uint8).
                # Assuming uint8 images, so data_range is implicitly 255.
                # If images are float, data_range might be 1.0 or max_val - min_val.
                data_range = np.iinfo(img1.dtype).max if np.issubdtype(img1.dtype, np.integer) else img1.max() - img1.min()
                if data_range == 0 : # handle case where image is flat
                    psnr_value = float('inf') if np.array_equal(img1,img2) else 0
                else:
                    psnr_value = peak_signal_noise_ratio(img1, img2, data_range=data_range)


                # Calculate SSIM
                # For SSIM, data_range should also be specified.
                current_ssim, _ = ssim(img1, img2, full=True, data_range=data_range)

                psnr_values.append(psnr_value)
                ssim_values.append(current_ssim)

                result_line = f"{filename}\t{psnr_value:.2f}\t{current_ssim:.4f}"
                print(result_line)
                f_out.write(result_line + "\n")

            except Exception as e:
                error_message = f"Error processing {filename}: {e}"
                print(error_message)
                f_out.write(f"{filename}\tError\tError ({e})\n")

        if psnr_values and ssim_values:
            mean_psnr = np.mean(psnr_values)
            var_psnr = np.var(psnr_values)
            mean_ssim = np.mean(ssim_values)
            var_ssim = np.var(ssim_values)

            summary_header = "\n--- Summary Statistics ---"
            summary_psnr = f"Mean PSNR: {mean_psnr:.2f}, Variance PSNR: {var_psnr:.2f}"
            summary_ssim = f"Mean SSIM: {mean_ssim:.4f}, Variance SSIM: {var_ssim:.4f}"

            print(summary_header)
            print(summary_psnr)
            print(summary_ssim)

            f_out.write(summary_header + "\n")
            f_out.write(summary_psnr + "\n")
            f_out.write(summary_ssim + "\n")
        else:
            no_data_message = "\nNo valid data to calculate mean and variance."
            print(no_data_message)
            f_out.write(no_data_message + "\n")

    print(f"\nProcessing complete. Results saved to {output_file_path}")



folder1_path = r'D:\CT_rec\SiT\AAPM_AAPM_mat\ori' # Path to the first set of images (e.g., reconstructed)
folder2_path = r'D:\CT_rec\SiT\AAPM_AAPM_mat\rec' # Path to the second set of images (e.g., original/ground truth)
output_file = r'D:\CT_rec\SiT\AAPM_AAPM_mat\comparison_results.txt' # Path to save the results

# Ensure output directory exists if it's nested
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

calculate_metrics_for_folders(folder1_path, folder2_path, output_file)