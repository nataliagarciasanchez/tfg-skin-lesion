# TFG Skin Lesion Classification
# Evaluation 

#1. Libraries

import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF_func
import torchvision.models as models
import segmentation_models_pytorch as smp
from statsmodels.stats.contingency_tables import mcnemar
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset as TorchDataset

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}") #check PyTorch version and GPU availability

#2. Configuration

IMG_SIZE = 224
SEG_SIZE = 320 
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASS_NAMES)

#ensure reproducible results
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#set seed for GPUs to ensure reproducibility
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED) 

#force execution on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#3. Paths 
IMG_DIR_TRAIN = "/workspace/datasets/ISIC2019/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
IMG_DIR_TEST = "/workspace/datasets/ISIC2019/images/ISIC_2019_Test_Input/ISIC_2019_Test_Input"
CSV_TRAIN = "/workspace/datasets/ISIC2019/labels/ISIC_2019_Training_GroundTruth.csv"
CSV_TEST = "/workspace/datasets/ISIC2019/labels/ISIC_2019_Test_GroundTruth.csv"
UNET_PATH = "/workspace/unet_best.pth"
MASK_DIR_TRAIN = "/workspace/datasets/ISIC2019/masks_train"
MASK_DIR_TEST = "/workspace/datasets/ISIC2019/masks_test"
ISIC2018_IMG_DIR = "/workspace/datasets/ISIC2018/images/ISIC2018_Task1-2_Training_Input"
ISIC2018_MASK_DIR = "/workspace/datasets/ISIC2018/masks/ISIC2018_Task1_Training_GroundTruth"

#4. Load, filter, and prepare data and labels
df_train = pd.read_csv(CSV_TRAIN)
df_train = df_train[df_train['UNK'] == 0.0].reset_index(drop=True)
df_train['label'] = df_train[CLASS_NAMES].values.argmax(axis=1)
df_train['filename'] = df_train['image'].apply(lambda x: x + '.jpg')
df_train = df_train[df_train['filename'].apply(lambda x: os.path.exists(os.path.join(IMG_DIR_TRAIN, x)))].reset_index(drop=True)
print(f"Training images found: {len(df_train)}")

df_test = pd.read_csv(CSV_TEST)
df_test = df_test[df_test['UNK'] == 0.0].reset_index(drop=True)
df_test['label'] = df_test[CLASS_NAMES].values.argmax(axis=1)
df_test['filename'] = df_test['image'].apply(lambda x: x + '.jpg')
df_test = df_test[df_test['filename'].apply(lambda x: os.path.exists(os.path.join(IMG_DIR_TEST, x)))].reset_index(drop=True)
print(f"Test images found: {len(df_test)}")

# 5. DATASETS
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#loads and returns the raw skin lesion images and their classification labels
#inspired by the official PyTorch custom dataset tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class ISIC2019Original(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(self.img_dir, row['filename'])).convert("RGB")
        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label, row['filename']

#returns the masked skin lesion images
#inspired by the official PyTorch custom dataset tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class ISIC2019Masked(Dataset):
    def __init__(self, dataframe, img_dir, mask_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img       = Image.open(os.path.join(self.img_dir, row['filename'])).convert("RGB")
        mask_name = row['filename'].replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask    = Image.open(mask_path).convert("L")
            mask    = mask.resize(img.size, Image.NEAREST)
            mask_np = np.array(mask) > 127
            img_np  = np.array(img)
            img_np[~mask_np] = 0
            img = Image.fromarray(img_np)
        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label, row['filename']
    


#6. Splits and dataloaders

#85% train, 15% validation
train_df, val_df = train_test_split(df_train, test_size=0.15, random_state=SEED, stratify=df_train['label'])

#100% test 
test_df = df_test

#how, how many, and how fast images will move from the hard drive to your GPU
loader_args = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

#loads RAW, original images
val_loader_A = DataLoader(ISIC2019Original(val_df,  IMG_DIR_TRAIN, val_transform), shuffle=False, **loader_args)
test_loader_A = DataLoader(ISIC2019Original(test_df, IMG_DIR_TEST,  val_transform), shuffle=False, **loader_args)

#loads MASKED images
val_loader_B = DataLoader(ISIC2019Masked(val_df,  IMG_DIR_TRAIN, MASK_DIR_TRAIN, val_transform), shuffle=False, **loader_args)
test_loader_B = DataLoader(ISIC2019Masked(test_df, IMG_DIR_TEST,  MASK_DIR_TEST,  val_transform), shuffle=False, **loader_args)


#7. Model template 

def build_model():
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = m.classifier[1].in_features #get the number of input connections coming from the network

    #replace head of the network w/ custom classifier
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3), #turn-off 30% of neurons randomly to prevent overfitting 
        nn.Linear(in_features, NUM_CLASSES) #substitude last layer to match the number of types of lesions
    )
    return m.to(device)

#8. Load trained models

model_A = build_model()
model_A.load_state_dict(torch.load("efficientnet_A_original.pth", map_location=device, weights_only=True))
model_A.to(device)
print("Model A loaded")

model_B = build_model()
model_B.load_state_dict(torch.load("efficientnet_B_masked.pth", map_location=device, weights_only=True))
model_B.to(device)
print("Model B loaded")

#9. Baseline Evaluation
def evaluate_model(model, test_loader, model_name):

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            
            #move images to GPU 
            imgs = imgs.to(device) 

            #switch to lighter numbers to speed up predictions and save GPU space
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")): 
                out = model(imgs)

            #turns raw scores into percentages
            probs = torch.softmax(out, dim=1).cpu().numpy()
            #picks the class with the highest score
            preds = out.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    #convert the lists into NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    #calculate accuracy, balance accuracy and macro F1 
    acc = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_mac = f1_score(all_labels, all_preds, average="macro")

    #compute Area Under the ROC Curve (AUC)
    try:
        probs_norm = all_probs.astype(np.float64)
        probs_norm = probs_norm / probs_norm.sum(axis=1, keepdims=True)
        auc = roc_auc_score(all_labels, probs_norm, multi_class="ovr", average="macro")
    except Exception as e:
        auc = float("nan")

    print(f"\n{'='*60}")
    print(f"MODEL {model_name} — TEST SET RESULTS")
    print(f"{'='*60}")

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {f1_mac:.4f}")
    print(f"Macro AUC: {auc:.4f}")

    #compares the doctor's real answers all_labels against the model's guesses all_preds to get the performance report
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)}")

    return all_preds, all_labels, all_probs, acc, bal_acc, f1_mac, auc

preds_A, labels_A, probs_A, acc_A, bal_acc_A, f1_A, auc_A = evaluate_model(model_A, test_loader_A, "A_original")
preds_B, labels_B, probs_B, acc_B, bal_acc_B, f1_B, auc_B = evaluate_model(model_B, test_loader_B, "B_masked")

#10. TTA 
def evaluate_model_tta(model, test_loader, model_name, n_augments=6):

    model.eval()

    #create 6 different points of view for the exact same image
    def tta_transforms(img_tensor):
        versions = [img_tensor]
        versions.append(TF_func.hflip(img_tensor))
        versions.append(TF_func.vflip(img_tensor))
        versions.append(TF_func.hflip(TF_func.vflip(img_tensor)))
        versions.append(torch.rot90(img_tensor, k=1, dims=[-2,-1]))
        versions.append(torch.rot90(img_tensor, k=3, dims=[-2,-1]))
        return versions[:n_augments]
    
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in test_loader:

            #move images to GPU 
            imgs = imgs.to(device)
            
            probs_sum = None

            for img_aug in tta_transforms(imgs):

                #switch to lighter numbers to speed up predictions and save GPU space
                with torch.amp.autocast("cuda", enabled=(device.type=="cuda")): 
                    out = model(img_aug)

                #turn raw scores into probabilities for each class
                p = torch.softmax(out, dim=1)

                #accumulate the probability percentages from all 6 versions
                probs_sum = p if probs_sum is None else probs_sum + p

            #calculate the final average percentage and pick the winning class
            probs_avg = (probs_sum / n_augments).cpu().numpy()
            preds = probs_avg.argmax(axis=1)

            all_probs.extend(probs_avg)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    #convert the lists into NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    #calculate accuracy, balance accuracy and macro F1 
    acc = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_mac = f1_score(all_labels, all_preds, average="macro")

    #compute Area Under the ROC Curve (AUC)
    try:
        probs_norm = all_probs.astype(np.float64)
        probs_norm = probs_norm / probs_norm.sum(axis=1, keepdims=True)
        auc = roc_auc_score(all_labels, probs_norm, multi_class="ovr", average="macro")
    except Exception as e:
        auc = float("nan")
    
    print(f"\n{'='*60}")
    print(f"MODEL {model_name} — TTA ({n_augments}) — TEST SET")
    print(f"{'='*60}")

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {f1_mac:.4f}")
    print(f"Macro AUC-ROC: {auc:.4f}")

    #compares the doctor's real answers all_labels against the model's guesses all_preds to get the performance report
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)}")

    return all_preds, all_labels, all_probs, acc, bal_acc, f1_mac, auc

preds_A_tta, labels_A_tta, probs_A_tta, acc_A_tta, bal_acc_A_tta, f1_A_tta, auc_A_tta = evaluate_model_tta(model_A, test_loader_A, "A_original")
preds_B_tta, labels_B_tta, probs_B_tta, acc_B_tta, bal_acc_B_tta, f1_B_tta, auc_B_tta = evaluate_model_tta(model_B, test_loader_B, "B_masked")

#11. McNEMAR

print("\n" + "="*60)
print("McNEMAR TEST — Model A vs Model B")
print("="*60)

#create a 1/0 score for each model 1=correct guess and 0=wrong guess
correct_A = (preds_A == labels_A).astype(int)
correct_B = (preds_B == labels_B).astype(int)

#count how many times they agreed or disagreed 
both_correct = np.sum((correct_A == 1) & (correct_B == 1))
only_A = np.sum((correct_A == 1) & (correct_B == 0))
only_B = np.sum((correct_A == 0) & (correct_B == 1))
both_wrong = np.sum((correct_A == 0) & (correct_B == 0))

#group the 4 counts into a table and run the McNemar statistical test
contingency_table = [[both_correct, only_A], [only_B, both_wrong]]
result = mcnemar(contingency_table, exact=False, correction=True)

print(f"Both correct: {both_correct}")
print(f"Only A correct: {only_A}")
print(f"Only B correct: {only_B}")
print(f"Both wrong: {both_wrong}")
print(f"McNemar stat: {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.6f}")

#12. Threshold tuning 
def tune_thresholds_on_val(model, val_loader):

    model.eval()

    val_probs, val_labels_list = [], []

    with torch.no_grad():
        for imgs, labels, _ in val_loader:

            #move images to GPU 
            imgs = imgs.to(device)

            #switch to lighter numbers to speed up predictions and save GPU space
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                out = model(imgs)

            #convert the raw guesses into percentages
            p = torch.softmax(out, dim=1).cpu().numpy()

            val_probs.extend(p)
            val_labels_list.extend(labels.numpy())

    return np.array(val_probs), np.array(val_labels_list)

def find_optimal_thresholds(val_probs, val_labels, n_classes):

    thresholds   = np.arange(0.5, 3.01, 0.1)
    best_scales  = np.ones(n_classes)

    #get the model's starting score before any tweaking
    best_bal_acc = balanced_accuracy_score(val_labels, val_probs.argmax(axis=1))
    print(f"Baseline val balanced accuracy: {best_bal_acc:.4f}")

    #apply brute force to each multiplier of each disease to find the optimal point.
    for cls in range(n_classes):
        for scale in thresholds:

            scaled = val_probs.copy()

            #boost or shrink the probability for the current class
            scaled[:, cls] *= scale
            
            #re-normalize so all probabilities still add up to 100%
            scaled = scaled / scaled.sum(axis=1, keepdims=True)

            #check if this tweak makes the overall score better
            ba = balanced_accuracy_score(val_labels, scaled.argmax(axis=1))
            if ba > best_bal_acc:
                best_bal_acc = ba
                best_scales[cls] = scale

    print(f"Best val balanced accuracy: {best_bal_acc:.4f}")
    print(f"Optimal scales: {dict(zip(CLASS_NAMES, best_scales.round(2)))}")

    return best_scales

def apply_thresholds(test_probs, test_labels, scales, tag):

    scaled = test_probs.copy()

    #apply the multiplier to each column to balance the model's bias
    for cls, scale in enumerate(scales):
        scaled[:, cls] *= scale

    #re-normalize so all probabilities still add up to 100%
    scaled = scaled / scaled.sum(axis=1, keepdims=True)

    #picks the class with the highest score
    preds_tuned = scaled.argmax(axis=1)

    #calculate accuracy, balance accuracy and macro F1 
    acc = (preds_tuned == test_labels).mean()
    bal_acc = balanced_accuracy_score(test_labels, preds_tuned)
    f1_mac = f1_score(test_labels, preds_tuned, average="macro")

    print(f"\n  {tag} — TEST RESULTS AFTER THRESHOLD TUNING")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {f1_mac:.4f}")

    #compares the doctor's real answers all_labels against the model's guesses all_preds to get the performance report
    print(f"\n{classification_report(test_labels, preds_tuned, target_names=CLASS_NAMES, digits=4)}")

    return preds_tuned, acc, bal_acc, f1_mac

#find the best multipliers using the Validation Set
print("\n--- Tuning thresholds for Model A ---")
val_probs_A, val_labels_A_val = tune_thresholds_on_val(model_A, val_loader_A)
scales_A = find_optimal_thresholds(val_probs_A, val_labels_A_val, NUM_CLASSES)
#test the multipliers on the standard Test Set
preds_A_tuned, acc_A_tuned, bal_A_tuned, f1_A_tuned = apply_thresholds(probs_A, labels_A, scales_A, "Model A")

#find the best multipliers using the Validation Set
print("\n--- Tuning thresholds for Model B ---")
val_probs_B, val_labels_B_val = tune_thresholds_on_val(model_B, val_loader_B)
scales_B = find_optimal_thresholds(val_probs_B, val_labels_B_val, NUM_CLASSES)
#test the multipliers on the standard Test Set
preds_B_tuned, acc_B_tuned, bal_B_tuned, f1_B_tuned = apply_thresholds(probs_B, labels_B, scales_B, "Model B")

#Combine 6 angle TTA + tuned multipliers
preds_A_combined, acc_A_combined, bal_A_combined, f1_A_combined = apply_thresholds(probs_A_tta, labels_A_tta, scales_A, "Model A TTA+Thresh")
preds_B_combined, acc_B_combined, bal_B_combined, f1_B_combined = apply_thresholds(probs_B_tta, labels_B_tta, scales_B, "Model B TTA+Thresh")


#13. Full comparison table 

print("\n" + "="*80)
print("  FULL COMPARISON — todas las variantes")
print("="*80)
print(f"  {'Metric':<22} {'A base':>10} {'A+TTA':>10} {'A+Thresh':>10} {'A+TTA+T':>10}")
print("-"*65)
print(f"  {'Accuracy':<22} {acc_A:>10.4f} {acc_A_tta:>10.4f} {acc_A_tuned:>10.4f} {acc_A_combined:>10.4f}")
print(f"  {'Balanced Accuracy':<22} {bal_acc_A:>10.4f} {bal_acc_A_tta:>10.4f} {bal_A_tuned:>10.4f} {bal_A_combined:>10.4f}")
print(f"  {'Macro F1':<22} {f1_A:>10.4f} {f1_A_tta:>10.4f} {f1_A_tuned:>10.4f} {f1_A_combined:>10.4f}")
print()
print(f"  {'Metric':<22} {'B base':>10} {'B+TTA':>10} {'B+Thresh':>10} {'B+TTA+T':>10}")
print("-"*65)
print(f"  {'Accuracy':<22} {acc_B:>10.4f} {acc_B_tta:>10.4f} {acc_B_tuned:>10.4f} {acc_B_combined:>10.4f}")
print(f"  {'Balanced Accuracy':<22} {bal_acc_B:>10.4f} {bal_acc_B_tta:>10.4f} {bal_B_tuned:>10.4f} {bal_B_combined:>10.4f}")
print(f"  {'Macro F1':<22} {f1_B:>10.4f} {f1_B_tta:>10.4f} {f1_B_tuned:>10.4f} {f1_B_combined:>10.4f}")
print("="*80)

#14. Segmentation internal split evaluation 

#loads and pairs raw skin lesion images with their binary segmentation masks
class ISICSegmentationDataset(TorchDataset):
    def __init__(self, files, img_dir, mask_dir, img_size=320):
        self.files    = files
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask  = Image.open(os.path.join(self.mask_dir, fname.replace(".jpg", "_segmentation.png"))).convert("L")
        img  = TF.resize(img,  (self.img_size, self.img_size), antialias=True)
        mask = TF.resize(mask, (self.img_size, self.img_size), antialias=True)
        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return img, mask, fname


def compute_metrics(masks, preds, eps=1e-7):

    #flatten the 2D images into 1D arrays
    m  = masks.view(masks.size(0), -1)
    p  = preds.view(preds.size(0), -1)

    #count true positives, false positives, false negatives, and true negatives pixel by pixel
    TP = (p * m).sum(1)
    FP = (p * (1 - m)).sum(1)
    FN = ((1 - p) * m).sum(1)
    TN = ((1 - p) * (1 - m)).sum(1)

    #calculate overlap and alignment metrics
    dice = ((2 * TP + eps) / (2 * TP + FP + FN + eps)).mean().item()
    iou  = ((TP + eps) / (TP + FP + FN + eps)).mean().item()
    prec = ((TP + eps) / (TP + FP + eps)).mean().item()
    rec  = ((TP + eps) / (TP + FN + eps)).mean().item()
    spec = ((TN + eps) / (TN + FP + eps)).mean().item()

    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec, "specificity": spec}

#get a clean and sorted list of all images
train_img_files = sorted([f for f in os.listdir(ISIC2018_IMG_DIR) if f.endswith(".jpg") and not f.startswith("._")])

#put aside a random 15% of the images for the evaluation test
_, internal_test_files = train_test_split(train_img_files, test_size=0.15, random_state=42)
print(f"Internal split size: {len(internal_test_files)} images")

#test files into the custom dataset loader
internal_ds = ISICSegmentationDataset(internal_test_files, ISIC2018_IMG_DIR, ISIC2018_MASK_DIR, img_size=320)

#delivers the ready images and masks to the model in batches of 8
internal_loader = DataLoader(internal_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

unet_model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b5",
    encoder_weights=None,
    in_channels=3, classes=1, activation=None
).to(device)

#load the weights of the U-NET++ model 
unet_model.load_state_dict(torch.load(UNET_PATH, map_location=device, weights_only=True))
unet_model.eval()

internal_metrics = {"dice": 0., "iou": 0., "precision": 0., "recall": 0., "specificity": 0.}

with torch.no_grad():
    for imgs, masks, _ in internal_loader:
        
        #move images to GPU 
        imgs, masks = imgs.to(device), masks.to(device)

        #convert the raw guesses into percentages
        probs = torch.sigmoid(unet_model(imgs))
        #1 if confidence is above 50% else 0
        preds = (probs > 0.5).float()

        #grade these 8 images and add the points to the scorecard
        batch = compute_metrics(masks, preds)
        for k in internal_metrics:
            internal_metrics[k] += batch[k]

#calculate the final average score by dividing the total sum by the number of batches
for k in internal_metrics:
    internal_metrics[k] /= len(internal_loader)

print("\nU-Net++ evaluated on INTERNAL split (same set as U-Net baseline):")
print(f"Dice: {internal_metrics['dice']:.4f}")
print(f"IoU: {internal_metrics['iou']:.4f}")
print(f"Precision: {internal_metrics['precision']:.4f}")
print(f"Recall: {internal_metrics['recall']:.4f}")
print(f"Specificity: {internal_metrics['specificity']:.4f}")

#15. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Confusion Matrices — Model A vs Model B", fontsize=14, fontweight="bold")

for ax, preds, labels, title in zip(axes, [preds_A, preds_B], [labels_A, labels_B], ["Model A — Original", "Model B — Masked"]):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

#16. F1 per class

#calculate the F1 score for each individual disease class, not average
f1_A_per = f1_score(labels_A, preds_A, average=None)
f1_B_per = f1_score(labels_B, preds_B, average=None)

x = np.arange(NUM_CLASSES)
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))

#plot side by side bars
bars_A = ax.bar(x - width/2, f1_A_per, width, label="Model A — Original", color="steelblue", alpha=0.85)
bars_B = ax.bar(x + width/2, f1_B_per, width, label="Model B — Masked", color="coral", alpha=0.85)

ax.set_xlabel("Class"); ax.set_ylabel("F1 Score")
ax.set_title("Per-class F1 Score — Model A vs Model B", fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

for bar in list(bars_A) + list(bars_B):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("perclass_f1_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# 17. SAVE RESULTS
import shutil
folder = "TFG_Classification_Results"

os.makedirs(folder, exist_ok=True)

files = [ "confusion_matrices.png", "perclass_f1_comparison.png"]

for f in files:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(folder, f))
        print(f"{f}")

shutil.make_archive("TFG_Classification_Results", "zip", folder)
print("\nZIP created: TFG_Classification_Results.zip")
print("\nEvaluation complete.")