import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- Configuration ---
RESIZE_DIM = (448, 448)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/best_model.pth"

# --- Load the Model (once) ---
print("Loading model...")
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,  # Not needed for inference if weights are loaded
    in_channels=3,
    classes=1,
    activation='sigmoid'
).to(DEVICE)

# Load the state dictionary. This handles both simple and checkpoint files.
try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    # Check if it's a checkpoint dictionary or a raw state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded successfully on {DEVICE}.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Helper Functions (from your previous scripts) ---

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Resizes and preprocesses a video frame for model inference."""
    image = cv2.resize(frame, (RESIZE_DIM[1], RESIZE_DIM[0]), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    return image_tensor

def create_overlay(original_image: np.ndarray, prediction_mask: np.ndarray) -> np.ndarray:
    """Creates a visual overlay of the prediction mask on the original image."""
    color_mask = np.zeros_like(original_image, dtype=np.uint8)
    color_mask[prediction_mask == 1] = [0, 0, 255]  # Red for cracks
    overlay = cv2.addWeighted(original_image, 1.0, color_mask, 0.5, 0)
    return overlay

# --- Main Processing Functions ---

def process_image(input_path: str, output_path: str):
    """Processes a single image file."""
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot process image.")
        
    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Could not read image from {input_path}")

    input_tensor = preprocess_frame(frame)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_batch)

    pred_mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    
    frame_resized = cv2.resize(frame, (RESIZE_DIM[1], RESIZE_DIM[0]))
    overlay_frame = create_overlay(frame_resized, pred_mask)

    cv2.imwrite(output_path, overlay_frame)
    print(f"Image processed and saved to {output_path}")

def process_video(input_path: str, output_path: str):
    """Processes a single video file."""
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot process video.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file at {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (RESIZE_DIM[1], RESIZE_DIM[0]))

    with tqdm(total=total_frames, desc="Processing video for web app") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = preprocess_frame(frame)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prediction = model(input_batch)

            pred_mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            frame_resized = cv2.resize(frame, (RESIZE_DIM[1], RESIZE_DIM[0]))
            overlay_frame = create_overlay(frame_resized, pred_mask)
            out.write(overlay_frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Video processed and saved to {output_path}")