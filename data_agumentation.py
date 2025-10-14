import cv2
from tqdm import tqdm
import multiprocessing
import numpy as np
import random
from pathlib import Path

base_dir = Path.cwd() / "Dataset"
splits = ['train', 'val']
paths = {}

for split in splits:
    paths[split] = {
        'input_images': base_dir / split / "images",
        'input_masks':  base_dir / split / "masks",
        'output_images': base_dir / split / "images",
        'output_masks':  base_dir / split / "masks",
    }

def augment_pair(image_path, mask_path, output_image_dir, output_mask_dir):
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    output_image_dir = Path(output_image_dir)
    output_mask_dir = Path(output_mask_dir)
    
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    angle = random.uniform(-20, 20)
    kernel_size = random.choice([3, 5, 7])

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    blurred_image = cv2.GaussianBlur(rotated_image, (kernel_size, kernel_size), 0)

    base_name = image_path.stem
    new_filename_suffix = f"_aug_rot{int(angle)}_blur{kernel_size}"

    new_image_path = output_image_dir / (base_name + new_filename_suffix + '.jpg')
    cv2.imwrite(str(new_image_path), blurred_image)

    new_mask_path = output_mask_dir / (base_name + new_filename_suffix + '.png')
    cv2.imwrite(str(new_mask_path), rotated_mask)

    return f"Processed {base_name}"

def gather_jobs(input_image_dir, input_mask_dir, output_image_dir, output_mask_dir):
    job_list = []
    for image_path in Path(input_image_dir).glob('*.[jp][pn]g'): 
        mask_path = Path(input_mask_dir) / (image_path.stem + '.png')
        if mask_path.exists():
            job_list.append((str(image_path), str(mask_path), str(output_image_dir), str(output_mask_dir)))
    return job_list

if __name__ == "__main__":
    all_jobs = []
    for split_name, split_paths in paths.items():
        print(f"Gathering jobs from '{split_name}' directory...")
        jobs = gather_jobs(
            split_paths['input_images'], 
            split_paths['input_masks'],
            split_paths['output_images'],
            split_paths['output_masks']
        )
        all_jobs.extend(jobs)
    
    if not all_jobs:
        print("No image-mask pairs found to process. Please check your directories.")
    else:
        num_processes = multiprocessing.cpu_count()
        print(f"\nFound {len(all_jobs)} total pairs. Starting augmentation with {num_processes} processes...")

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.starmap(augment_pair, all_jobs), total=len(all_jobs)))

        print("\nAugmentation complete! ✅")