import kagglehub
from pathlib import Path
import os
import shutil
import random
import torch

# 1. Download dataset and get the path

download_root = Path(kagglehub.dataset_download("lakshaymiddha/crack-segmentation-dataset"))
print(f"Dataset downloaded to: {download_root}")

main_dirs = ["Dataset", "results", "weights"]
for dir_name in main_dirs:
    Path(dir_name).mkdir(exist_ok=True)

base_dir = Path("Dataset")
splits = ["test", "train", "val"]
sub_dirs = ["images", "masks"]

for split in splits:
    for sub_dir in sub_dirs:
        (base_dir / split / sub_dir).mkdir(parents=True, exist_ok=True)


source_a = download_root / "crack_segmentation_dataset" / "images"
source_b = download_root / "crack_segmentation_dataset" / "masks"


base_dest_dir = Path.cwd() / "Dataset"

dest_A = base_dest_dir / "train" / "images"
dest_B = base_dest_dir / "train" / "masks"

dest_AA = base_dest_dir / "val" / "images"
dest_BB = base_dest_dir / "val" / "masks"

torch.cuda.is_available() == True

# The split ratio. 0.8 means 80% of files will go to A/B.
split_ratio = 0.8


print("Setting up destination directories...")
dirs_to_create = [dest_A, dest_B, dest_AA, dest_BB]
for dir_path in dirs_to_create:
    os.makedirs(dir_path, exist_ok=True)
try:

    all_filenames = [f for f in os.listdir(source_a) if os.path.isfile(os.path.join(source_a, f))]
    print(f"Found {len(all_filenames)} files in '{source_a}'.")
except FileNotFoundError:
    print(f"Error: Source directory '{source_a}' not found. Please create it and add files.")
    exit()
random.shuffle(all_filenames)
print("Shuffling file list...")

split_point = int(len(all_filenames) * split_ratio)

train_files = all_filenames[:split_point]
test_files = all_filenames[split_point:]

print(f"Splitting into {len(train_files)} (80%) and {len(test_files)} (20%) files.")
print("-" * 20)


print(f"Moving {len(train_files)} files to '{dest_A}' and '{dest_B}'...")
for filename in train_files:

    source_path_a = os.path.join(source_a, filename)
    dest_path_A = os.path.join(dest_A, filename)

    source_path_b = os.path.join(source_b, filename)
    dest_path_B = os.path.join(dest_B, filename)

    shutil.move(source_path_a, dest_path_A)
    shutil.move(source_path_b, dest_path_B)

print(f"Moving {len(test_files)} files to '{dest_AA}' and '{dest_BB}'...")
for filename in test_files:
    source_path_a = os.path.join(source_a, filename)
    dest_path_AA = os.path.join(dest_AA, filename)

    source_path_b = os.path.join(source_b, filename)
    dest_path_BB = os.path.join(dest_BB, filename)

    shutil.move(source_path_a, dest_path_AA)
    shutil.move(source_path_b, dest_path_BB)
print(" successfully split and moved ")

