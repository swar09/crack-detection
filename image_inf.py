#!/usr/bin/env python
import os
import argparse
import glob
import shutil
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

def preprocess_image(image_path: str, resize_dim: tuple = (448, 448)) -> torch.Tensor:
  
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    image = cv2.resize(image, (resize_dim[1], resize_dim[0]), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    
    return image_tensor

def create_overlay(original_image: np.ndarray, prediction_mask: np.ndarray) -> np.ndarray:
    
    # Create a red color mask for the predicted cracks
    color_mask = np.zeros_like(original_image, dtype=np.uint8)
    color_mask[prediction_mask == 1] = [255, 0, 0]  # Red color for cracks

    # Blend the original image and the color mask
    overlay = cv2.addWeighted(original_image, 1.0, color_mask, 0.5, 0)
    
    return overlay

def run_inference(args: argparse.Namespace):
   
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'
    print(f"Using device: {device}")

    print("Loading model...")
    try:
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        ).to(device)
        
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not os.path.isdir(args.image_path):
        print(f"Error: Input path '{args.image_path}' is not a valid directory.")
        return

    positive_dir = os.path.join(args.output_path, 'positive')
    negative_dir = os.path.join(args.output_path, 'negative')
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(args.image_path, '*.jpg')) + \
                  glob.glob(os.path.join(args.image_path, '*.jpeg')) + \
                  glob.glob(os.path.join(args.image_path, '*.png'))

    if not image_files:
        print("No images found at the specified path.")
        return

    print(f"Found {len(image_files)} image(s) to process.")
    for img_path in tqdm(image_files, desc="Classifying images"):
        try:
            input_tensor = preprocess_image(img_path)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(input_batch)

            pred_mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            crack_pixel_count = np.sum(pred_mask)
            base_name = os.path.basename(img_path)

            if crack_pixel_count > args.crack_threshold:
                original_img_resized = cv2.resize(cv2.imread(img_path), (448, 448))
                overlay_image = create_overlay(original_img_resized, pred_mask)
                file_name, _ = os.path.splitext(base_name)
                output_file_path = os.path.join(positive_dir, f"{file_name}_overlay.png")
                cv2.imwrite(output_file_path, overlay_image)
            else:
                destination_path = os.path.join(negative_dir, base_name)
                shutil.copy(img_path, destination_path)

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    print(f"\nClassification complete.")
    print(f"Positive overlays saved to: '{positive_dir}'")
    print(f"Negative originals saved to: '{negative_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crack Classification and Visualization Script.")
    
    parser.add_argument(
        "--image_path", type=str, required=True, 
        help="Path to the directory containing input images."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, 
        help="Path to the trained PyTorch model file (.pth)."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, 
        help="Path to the base output directory where 'positive' and 'negative' subdirectories will be created."
    )
    parser.add_argument(
        "--crack_threshold", type=int, default=100,
        help="Minimum number of crack pixels to classify an image as positive. Default: 100."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=['cuda', 'cpu'],
        help="Device to use for inference ('cuda' or 'cpu')."
    )
    
    args = parser.parse_args()
    run_inference(args)