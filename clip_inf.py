#!/usr/bin/env python

import os
import argparse
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

RESIZE_DIM = (448, 448)  

def preprocess_frame(frame: np.ndarray, resize_dim: tuple = RESIZE_DIM) -> torch.Tensor:
    """
    Resizes and preprocesses a video frame for model inference.

    Args:
        frame (np.ndarray): The input video frame (H, W, C) in BGR format.
        resize_dim (tuple): The target dimensions (height, width) for resizing.

    Returns:
        torch.Tensor: The preprocessed frame tensor ready for the model.
    """
    image = cv2.resize(frame, (resize_dim[1], resize_dim[0]), interpolation=cv2.INTER_AREA)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    
    return image_tensor

def create_overlay(original_image: np.ndarray, prediction_mask: np.ndarray) -> np.ndarray:
    """
    Creates a visual overlay of the prediction mask on the original image.

    Args:
        original_image (np.ndarray): The original input image (H, W, C).
        prediction_mask (np.ndarray): The binary prediction mask (H, W) from the model.

    Returns:
        np.ndarray: An image with the detected cracks highlighted in red.
    """
    color_mask = np.zeros_like(original_image, dtype=np.uint8)
    color_mask[prediction_mask == 1] = [0, 0, 255]  

    overlay = cv2.addWeighted(original_image, 1.0, color_mask, 0.5, 0)
    
    return overlay

def process_video(args: argparse.Namespace):
    """
    Main function to run the video processing.
    """
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

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (RESIZE_DIM[1], RESIZE_DIM[0]))

    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  

            try:
                input_tensor = preprocess_frame(frame)
                input_batch = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    prediction = model(input_batch)

                pred_mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                frame_resized = cv2.resize(frame, (RESIZE_DIM[1], RESIZE_DIM[0]))
                
                overlay_frame = create_overlay(frame_resized, pred_mask)

                out.write(overlay_frame)

            except Exception as e:
                print(f"An error occurred while processing a frame: {e}")
                continue
            
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output video saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Crack Segmentation in Videos.")
    
    parser.add_argument(
        "--video_path", type=str, required=True, 
        help="Path to the input video file."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, 
        help="Path to the trained PyTorch model file (.pth)."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, 
        help="Path to save the output MP4 video file."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=['cuda', 'cpu'],
        help="Device to use for inference ('cuda' or 'cpu')."
    )
    
    args = parser.parse_args()
    process_video(args)