
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
import torchvision
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # You can increase according to GPU
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
RESIZE_H = 448
RESIZE_W = 448


BASE_DIR = Path.cwd()

DATASET_DIR = BASE_DIR / "Dataset"
TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_MASK_DIR = DATASET_DIR / "train" / "masks"
VAL_IMG_DIR = DATASET_DIR / "val" / "images"
VAL_MASK_DIR = DATASET_DIR / "val" / "masks"
TEST_IMG_DIR = DATASET_DIR / "test" / "images"
TEST_MASK_DIR = DATASET_DIR / "test" / "masks"

WEIGHTS_DIR = BASE_DIR / "weights"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for path in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, 
             TEST_IMG_DIR, TEST_MASK_DIR, WEIGHTS_DIR, LOGS_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "overlays"), exist_ok=True)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = []

        all_images = os.listdir(image_dir)
        for img_filename in all_images:
            img_path = os.path.join(image_dir, img_filename)
            mask_filename = img_filename.replace(".jpg", ".png")
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                self.images.append(img_filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_filename = self.images[index].replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return image, mask

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1, # Binary segmentation
    activation='sigmoid' # Sigmoid for binary output
).to(DEVICE)

# --- Loss Function (0.5 * BCE + 0.5 * Dice) ---
class CombinedLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')

    def forward(self, inputs, targets, smooth=1e-6):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return 0.5 * bce + 0.5 * dice

loss_fn = CombinedLoss()

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

def check_metrics(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).float()

    tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), targets.long(), mode='binary')

    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    dice_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

    return iou_score, dice_score, precision, recall

scaler = torch.amp.GradScaler()
best_val_iou = -1.0
patience_counter = 0
log_data = []


train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


def save_predictions_as_imgs(loader, model, epoch, folder="results/"):
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(DEVICE), masks.to(DEVICE)

    with torch.no_grad():
        preds = torch.sigmoid(model(images))
        preds = (preds > 0.5).float()

   
    grid = torchvision.utils.make_grid(torch.cat([images.cpu(), masks.cpu(), preds.cpu()], dim=0), nrow=BATCH_SIZE)
    plt.figure(figsize=(15,10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Epoch {epoch} Predictions")
    plt.savefig(f"{folder}/epoch_{epoch}_samples.png")
    plt.close()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
        data, targets = data.to(DEVICE), targets.float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    all_iou, all_dice, all_prec, all_recall = [], [], [], []

    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            data, targets = data.to(DEVICE), targets.float().to(DEVICE)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

            iou, dice, prec, recall = check_metrics(predictions, targets)
            all_iou.append(iou.item())
            all_dice.append(dice.item())
            all_prec.append(prec.item())
            all_recall.append(recall.item())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_iou = np.mean(all_iou)
    avg_dice = np.mean(all_dice)

    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_iou:.4f}")

    log_data.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_iou": avg_iou, "val_dice": avg_dice})

    scheduler.step(avg_iou)

    if avg_iou > best_val_iou:
        print(f"IoU improved from {best_val_iou:.4f} to {avg_iou:.4f}. Saving model...")
        best_val_iou = avg_iou
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
        patience_counter = 0 
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"epoch_{epoch+1}_checkpoint.pth"))
        save_predictions_as_imgs(val_loader, model, epoch+1, folder=RESULTS_DIR)

    if patience_counter >= 10:
        print("Early stopping triggered after 10 epochs with no improvement.")
        break

log_df = pd.DataFrame(log_data)
log_df.to_csv(os.path.join(LOGS_DIR, "metrics.csv"), index=False)