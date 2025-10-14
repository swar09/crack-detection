import os
import cv2
import numpy as np
import glob
import time
from pathlib import Path


base_dir = Path.cwd() 
pattern = "Dataset/*/masks"
path_pattern = base_dir.glob(pattern)

BINARY_THRESHOLD = 127


DELETE_ORIGINALS = True

print("--- Starting CPU Transformation Phase ---")
if DELETE_ORIGINALS:
    print("This script will convert masks to binary format (0 or 255) (PNG) and  DELETE the original JPGs.")
else:
    print("This script will convert masks to binary format (0 or 255) (PNG) and keep the original JPGs.")

start_time = time.time()
files_processed = 0

mask_directories = glob.glob(path_pattern)
if not mask_directories:
    print(f"Warning: No directories found matching: '{path_pattern}'")

for dir_path in mask_directories:
    print(f"\nProcessing folder: {dir_path}")

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        if not os.path.isfile(file_path) or not filename.lower().endswith(('.jpg', '.jpeg')):
            continue

        try:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Skipping corrupted or invalid file: {filename}")
                continue

            _, binary_image = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

            base_filename, _ = os.path.splitext(file_path)
            new_png_path = base_filename + '.png'

            cv2.imwrite(new_png_path, binary_image)
            print(f"  - Converted {filename} to {os.path.basename(new_png_path)}")
            files_processed += 1

            if DELETE_ORIGINALS:
                os.remove(file_path)
                print(f"Deleted original: {filename}")

        except Exception as e:
            print(f"An error occurred with {filename}: {e}")

end_time = time.time()
print(f"\nTransformation complete. Processed {files_processed} files in {end_time - start_time:.2f} seconds.")


print("\n--- Starting Verification Phase ---")

files_to_fix = []
total_files_checked = 0

for dir_path in mask_directories:
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        if not os.path.isfile(file_path) or not filename.lower().endswith('.png'):
            continue

        total_files_checked += 1
        mask_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if mask_image is None:
            continue

        unique_values = np.unique(mask_image)
        for value in unique_values:
            if value != 0 and value != 255:
                files_to_fix.append(file_path)
                break

print("\n" + "="*40)
print("Verification Complete.")
print(f"Total PNG files checked: {total_files_checked}")

if not files_to_fix:
    print("\n✅ Success! All mask files were successfully transformed and verified as binary.")
else:
    print(f"\n❌ Verification FAILED. {len(files_to_fix)} PNG files are still not binary.")
    print("This is unexpected. Please check the following files:")
    for f in files_to_fix:
        print(f" - {f}")

print("="*40)