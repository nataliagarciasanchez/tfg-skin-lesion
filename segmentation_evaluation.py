# TFG Skin Lesion Segmentation
# Evaluation

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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}") #check PyTorch version and GPU availability

#2. Configuration
IMG_SIZE = 320
BATCH_SIZE = 8
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

#4. Dataset
#loads and pairs the raw skin lesion images with their binary segmentation masks
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

        #resize image and mask to fixed resolution
        img  = TF.resize(img,  (self.img_size, self.img_size), antialias=True)
        mask = TF.resize(mask, (self.img_size, self.img_size), antialias=True)

        #convert to tensor and normalize with ImageNet mean and std
        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)

        #binarize mask: 1 = lesion, 0 = background
        mask = (mask > 0.5).float()

        return img, mask, fname

#5. Split + dataloaders
#same 85/15 split used during training
_, val_files = train_test_split(train_img_files, test_size=0.15, random_state=SEED)
#100% official ISIC 2018 test set
test_files = test_img_files

print(f"Val: {len(val_files)} images")
print(f"Test: {len(test_files)} images")

val_ds  = ISICSegmentationDataset(val_files,  TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE, augment=False)
test_ds = ISICSegmentationDataset(test_files, TEST_IMG_DIR,  TEST_MASK_DIR,  IMG_SIZE, augment=False)

#how, how many, and how fast images will move from hard drive to GPU
val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

#6. Load trained model
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b5",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(device)

#load the weights of the best saved checkpoint
model.load_state_dict(torch.load("unet_best.pth", map_location=device, weights_only=True))
model.eval()
print("U-Net++ loaded successfully")

#7. Metrics
def compute_metrics(masks, preds, eps=1e-7):
    #flatten 2D masks into 1D arrays
    m = masks.view(masks.size(0), -1)
    p = preds.view(preds.size(0), -1)

    #count true positives, false positives, false negatives, true negatives pixel by pixel
    TP = (p * m).sum(1)
    FP = (p * (1 - m)).sum(1)
    FN = ((1 - p) * m).sum(1)
    TN = ((1 - p) * (1 - m)).sum(1)

    #calculate overlap and alignment metrics
    dice = ((2 * TP + eps) / (2 * TP + FP + FN + eps)).mean().item()
    iou = ((TP + eps) / (TP + FP + FN + eps)).mean().item()
    precision = ((TP + eps) / (TP + FP + eps)).mean().item()
    recall = ((TP + eps) / (TP + FN + eps)).mean().item()
    spec = ((TN + eps) / (TN + FP + eps)).mean().item()

    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "specificity": spec}

# 8. Final evaluation on official test set
test_metrics = {"dice": 0., "iou": 0., "precision": 0., "recall": 0., "specificity": 0.}
all_dice_scores = []

with torch.no_grad():
    for imgs, masks, _ in test_loader:

        imgs, masks = imgs.to(device), masks.to(device)
        probs = torch.sigmoid(model(imgs))

        #1 if confidence above 50% else 0
        preds = (probs > 0.5).float()

        batch = compute_metrics(masks, preds)
        for k in test_metrics:
            test_metrics[k] += batch[k]

        #compute per-image Dice scores for distribution analysis
        m = masks.view(masks.size(0), -1)
        p = preds.view(preds.size(0), -1)
        eps = 1e-7
        per_img_dice = ((2 * (p * m).sum(1) + eps) / (p.sum(1) + m.sum(1) + eps)).cpu().numpy()
        all_dice_scores.extend(per_img_dice.tolist())

#calculate the final average score
for k in test_metrics:
    test_metrics[k] /= len(test_loader)

dice_std = np.std(all_dice_scores)

print("=" * 50)
print("  FINAL RESULTS — OFFICIAL TEST SET (ISIC 2018)")
print("=" * 50)

print(f"  Dice / F1      : {test_metrics['dice']:.4f}  (±{dice_std:.4f})")
print(f"  IoU / Jaccard  : {test_metrics['iou']:.4f}")
print(f"  Precision      : {test_metrics['precision']:.4f}")
print(f"  Recall (Sens.) : {test_metrics['recall']:.4f}")
print(f"  Specificity    : {test_metrics['specificity']:.4f}")
print(f"  Total test images: {len(all_dice_scores)}")

print("=" * 50)

#save the results to a CSV
with open("test_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in test_metrics.items():
        writer.writerow([k, f"{v:.6f}"])
    writer.writerow(["dice_std", f"{dice_std:.6f}"])
print("Results saved: test_results.csv")

#9. Dice score distribution
all_dices = all_dice_scores
fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(all_dices, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(np.mean(all_dices),   color="red",    linestyle="--", label=f"Mean  {np.mean(all_dices):.3f}")
ax.axvline(np.median(all_dices), color="orange", linestyle="--", label=f"Median {np.median(all_dices):.3f}")
ax.set_xlabel("Dice Score"); ax.set_ylabel("Nº images")
ax.set_title("Distribution of Dice Score — Official Test Set")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("histograma_dice.png", dpi=150, bbox_inches="tight")
plt.show()

#10. Qualitative visualization
per_image_results = []
with torch.no_grad():
    for imgs, masks, fnames in test_loader:
        imgs_d, masks_d = imgs.to(device), masks.to(device)
        probs_d = torch.sigmoid(model(imgs_d))
        preds_d = (probs_d > 0.5).float()
        m = masks_d.view(masks_d.size(0), -1)
        p = preds_d.view(preds_d.size(0), -1)
        eps = 1e-7
        dice_per = ((2*(p*m).sum(1)+eps) / (p.sum(1)+m.sum(1)+eps)).cpu()
        for i in range(len(fnames)):
            per_image_results.append((
                dice_per[i].item(),
                imgs[i].cpu(),
                masks[i].cpu(),
                probs_d[i].cpu(),
                fnames[i]
            ))

#sort by Dice score to select difficult, limit, and good cases
per_image_results.sort(key=lambda x: x[0])
good = [r for r in per_image_results if r[0] > 0.90]
mid = [r for r in per_image_results if 0.75 <= r[0] <= 0.90]
hard = [r for r in per_image_results if r[0] < 0.75]
samples = hard[:3] + mid[:3] + good[:3]
group_labels = (["Difficult"]* min(3, len(hard)) + ["Limit"]* min(3, len(mid))  + ["Good"]* min(3, len(good)))
group_colors = {"Difficult": "#d62728", "Limit": "#ff7f0e", "Good": "#2ca02c"}

#denormalize images for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm(t):
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

n = len(samples)
fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
fig.suptitle("Qualitative Analysis of Segmentation — Official Test Set", fontsize=14, fontweight="bold", y=1.01)
col_titles = ["Original image", "Ground Truth", "Probability", "Prediction (t=0.5)"]

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=12, fontweight="bold")

for row, (result, glabel) in enumerate(zip(samples, group_labels)):
    dice_val, img_t, mask_t, prob_t, fname = result
    color   = group_colors[glabel]

    pred_t  = (prob_t > 0.5).float()
    img_np  = denorm(img_t)

    mask_np = mask_t.squeeze().numpy()
    prob_np = prob_t.squeeze().numpy()
    pred_np = pred_t.squeeze().numpy()

    axes[row, 0].imshow(img_np)
    axes[row, 0].axis("off")
    axes[row, 1].imshow(mask_np, cmap="gray")
    axes[row, 1].axis("off")
    axes[row, 2].imshow(prob_np, cmap="hot")
    axes[row, 2].axis("off")
    axes[row, 3].imshow(pred_np, cmap="gray")
    axes[row, 3].axis("off")
    axes[row, 0].set_ylabel(f"{glabel}\nDice={dice_val:.3f}", fontsize=10, color=color, fontweight="bold", rotation=0, labelpad=60, va="center")

    for col in range(4):
        for spine in axes[row, col].spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("visualizacion_cualitativa.png", dpi=150, bbox_inches="tight")
plt.show()

# 11. Save all results
import shutil

output_folder = "TFG_Segmentation_Results"
os.makedirs(output_folder, exist_ok=True)

files_to_save = [
    "unet_best.pth",
    "test_results.csv",
    "histograma_dice.png",
    "visualizacion_cualitativa.png",
]

for f in files_to_save:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(output_folder, f))
        print(f"{f}")
    else:
        print(f"{f} not found")

shutil.make_archive("TFG_Segmentation_Results", "zip", output_folder)
print("\nZIP created: TFG_Segmentation_Results.zip")
print("\nEvaluation complete.")