# TFG Skin Lesion Segmentation
# Training

#1. Libraries
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}") #check PyTorch version and GPU availability


#2. Configuration
IMG_SIZE = 320
BATCH_SIZE = 8
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 25
NUM_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
SEED = 42

#ensure reproducible results
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#set seed for GPUs to ensure reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#force execution on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


#3. Paths
TRAIN_IMG_DIR = "/workspace/datasets/ISIC2018/images/ISIC2018_Task1-2_Training_Input"
TRAIN_MASK_DIR = "/workspace/datasets/ISIC2018/masks/ISIC2018_Task1_Training_GroundTruth"
TEST_IMG_DIR = "/workspace/datasets/ISIC2018/images/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR = "/workspace/datasets/ISIC2018/masks/ISIC2018_Task1_Test_GroundTruth"

#get sorted list of all training and test images
train_img_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".jpg") and not f.startswith("._")])
test_img_files  = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".jpg") and not f.startswith("._")])

print(f"Training images : {len(train_img_files)}")
print(f"Test images     : {len(test_img_files)}")

#4. Dataset
#loads and pairs the raw skin lesion images with their binary segmentation masks, also applies data augmentation during training to increase diversity 
#inspired by the official PyTorch custom dataset tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class ISICSegmentationDataset(Dataset):

    def __init__(self, files, img_dir, mask_dir, img_size=320, augment=False):
        self.files    = files
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask  = Image.open(os.path.join(self.mask_dir, fname.replace(".jpg", "_segmentation.png"))).convert("L")

        #resize the image and mask to fixed resolution
        img  = TF.resize(img,  (self.img_size, self.img_size), antialias=True)
        mask = TF.resize(mask, (self.img_size, self.img_size), antialias=True)

        #apply same augmentation to both image and mask
        if self.augment:
            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() < 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-30, 30)
            img   = TF.rotate(img,  angle)
            mask  = TF.rotate(mask, angle)
            if random.random() < 0.5:
                scale    = random.uniform(1.0, 1.2)
                new_size = int(self.img_size * scale)
                img  = TF.resize(img,  (new_size, new_size), antialias=True)
                mask = TF.resize(mask, (new_size, new_size), antialias=True)
                img  = TF.center_crop(img,  self.img_size)
                mask = TF.center_crop(mask, self.img_size)
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
                img = TF.adjust_hue(img,        random.uniform(-0.1, 0.1))
            if random.random() < 0.3:
                img = TF.gaussian_blur(img, kernel_size=3)

        #convert to tensor and normalize with ImageNet mean and std
        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)

        #binarize mask with 1 = lesion and 0 = background
        mask = (mask > 0.5).float()

        return img, mask, fname
    


#5. Split train/val and official test set

#85% train and 15% validation
train_files, val_files = train_test_split(train_img_files, test_size=0.15, random_state=SEED)
#100% official test set
test_files = test_img_files

print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")
print(f"Test: {len(test_files)} images")

train_ds = ISICSegmentationDataset(train_files, TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE, augment=True)
val_ds = ISICSegmentationDataset(val_files, TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE, augment=False)
test_ds = ISICSegmentationDataset(test_files, TEST_IMG_DIR, TEST_MASK_DIR, IMG_SIZE, augment=False)

# how, how many, and how fast images will move from hard drive to GPU
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

#6. U-Net++ model
#encoder frozen during phase 1 and unfrozen during phase 2
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b5",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}  ({n_params/1e6:.1f}M)")

#7. Loss and metrics
def dice_loss(logits, targets, eps=1e-7):
    # measures overlap between predicted and ground truth masks
    probs   = torch.sigmoid(logits).view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter   = (probs * targets).sum(1)
    return 1 - ((2 * inter + eps) / (probs.sum(1) + targets.sum(1) + eps)).mean()

bce_fn = nn.BCEWithLogitsLoss()

def loss_fn(logits, targets):
    #combined BCE + Dice loss to prevent trivial solutions and measure overlap
    return bce_fn(logits, targets) + dice_loss(logits, targets)

def compute_metrics(masks, preds, eps=1e-7):
    #flatten 2D masks into 1D arrays
    m  = masks.view(masks.size(0), -1)
    p  = preds.view(preds.size(0), -1)

    #count true positives, false positives, false negatives, true negatives pixel by pixel
    TP = (p * m).sum(1)
    FP = (p * (1 - m)).sum(1)
    FN = ((1 - p) * m).sum(1)
    TN = ((1 - p) * (1 - m)).sum(1)

    #calculate overlap and alignment metrics
    dice      = ((2 * TP + eps) / (2 * TP + FP + FN + eps)).mean().item()
    iou       = ((TP + eps) / (TP + FP + FN + eps)).mean().item()
    precision = ((TP + eps) / (TP + FP + eps)).mean().item()
    recall    = ((TP + eps) / (TP + FN + eps)).mean().item()
    spec      = ((TN + eps) / (TN + FP + eps)).mean().item()

    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "specificity": spec}


#8. Optimization
#freeze encoder and train only decoder
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)

#automatically cuts the learning rate in half if Dice stops improving for 3 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

#speed up GPU training using mixed precision
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


#9. Training loop
history   = {"epoch": [], "train_loss": [], "val_dice": [], "val_iou": [], "lr": []}
best_dice = -1.0

for epoch in range(1, NUM_EPOCHS + 1):

    #unfreeze encoder at epoch 6
    if epoch == PHASE1_EPOCHS + 1:
        print("\n--- Phase 2: unfreezing encoder ---")
        for param in model.encoder.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

    #standard PyTorch training loop
    model.train()
    train_loss = 0.0
    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    #standard PyTorch evaluation loop
    model.eval()
    val_metrics = {"dice": 0., "iou": 0., "precision": 0., "recall": 0., "specificity": 0.}
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.sigmoid(model(imgs))
            # 1 if confidence above 50% else 0
            preds = (probs > 0.5).float()
            batch = compute_metrics(masks, preds)
            for k in val_metrics:
                val_metrics[k] += batch[k]
    for k in val_metrics:
        val_metrics[k] /= len(val_loader)

    #update learning rate
    scheduler.step(val_metrics["dice"])
    current_lr = optimizer.param_groups[0]["lr"]

    #save best model
    if val_metrics["dice"] > best_dice:
        best_dice = val_metrics["dice"]
        torch.save(model.state_dict(), "unet_best.pth")

    #save epoch history
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_dice"].append(val_metrics["dice"])
    history["val_iou"].append(val_metrics["iou"])
    history["lr"].append(current_lr)

    print(f"[{epoch:02d}/{NUM_EPOCHS}] "
          f"loss: {train_loss:.4f} | "
          f"val Dice: {val_metrics['dice']:.4f} | "
          f"IoU: {val_metrics['iou']:.4f} | "
          f"Prec: {val_metrics['precision']:.4f} | "
          f"Rec: {val_metrics['recall']:.4f} | "
          f"LR: {current_lr:.2e}")

#export training history to CSV
with open("training_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=history.keys())
    writer.writeheader()
    for row in zip(*history.values()):
        writer.writerow(dict(zip(history.keys(), row)))

print(f"\nBest Val Dice: {best_dice:.4f}  →  model saved as unet_best.pth")

#10. Learning curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history["epoch"], history["train_loss"], "b-o", markersize=4, label="Train Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (BCE + Dice)")
ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve_loss.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history["epoch"], history["val_dice"], "g-o", markersize=4, label="Val Dice")
ax.plot(history["epoch"], history["val_iou"],  "r-s", markersize=4, label="Val IoU")
best_epoch = history["val_dice"].index(max(history["val_dice"])) + 1
ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
ax.set_title("Validation Metrics"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("learning_curve_val.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nTraining complete.")
print("Model saved: unet_best.pth")