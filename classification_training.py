# TFG Skin Lesion Classification
# Training

#1. Libraries

import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import segmentation_models_pytorch as smp

print(f"PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}") # Check PyTorch version and GPU availability

#2. Configuration

IMG_SIZE = 224
SEG_SIZE = 320 
BATCH_SIZE = 32
NUM_EPOCHS = 30 
LR = 1e-4
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

os.makedirs(MASK_DIR_TRAIN, exist_ok=True)
os.makedirs(MASK_DIR_TEST,  exist_ok=True)

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

#check number of images per class (class balance)
print("Class distribution (train):")
for i, cls in enumerate(CLASS_NAMES):
    n = (df_train['label'] == i).sum()
    print(f"  {cls}: {n} ({100*n/len(df_train):.1f}%)")

#5. U-NET for mask generation
unet = smp.UnetPlusPlus(
    encoder_name="efficientnet-b5",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(device)
unet.load_state_dict(torch.load(UNET_PATH, map_location=device, weights_only=True))
unet.eval()
print("U-Net++ loaded successfully")

#6. Generate masks
#ImageNet mean and std reshaped 
seg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
seg_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

def generate_masks(df, img_dir, mask_dir):

    already_done = set(os.listdir(mask_dir))
    to_process   = [f for f in df['filename'].tolist() if f.replace('.jpg', '_mask.png') not in already_done]

    MASK_BATCH = 16
    errors = 0

    with torch.no_grad():
        for start in range(0, len(to_process), MASK_BATCH):

            batch_files = to_process[start:start + MASK_BATCH]
            imgs, valid_files = [], []

            for fname in batch_files:
                try:
                    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
                    img = img.resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
                    img_t = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
                    imgs.append(img_t)
                    valid_files.append(fname)
                except Exception:
                    errors += 1

            if not imgs:
                continue

            batch_t = torch.stack(imgs).to(device)
            batch_t = (batch_t - seg_mean) / seg_std

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = unet(batch_t)

            masks = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy()

            for fname, mask in zip(valid_files, masks):
                mask_img  = Image.fromarray((mask * 255).astype(np.uint8))
                mask_name = fname.replace('.jpg', '_mask.png')
                mask_img.save(os.path.join(mask_dir, mask_name))

            if (start // MASK_BATCH) % 100 == 0:
                print(f"  {start}/{len(to_process)}")

    print(f"Done. Errors: {errors} | Total masks: {len(os.listdir(mask_dir))}")

generate_masks(df_train, IMG_DIR_TRAIN, MASK_DIR_TRAIN)
generate_masks(df_test,  IMG_DIR_TEST,  MASK_DIR_TEST)


#7. Datasets 

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#loads and returns the raw skin lesion images and their classification labels
#inspired by the official PyTorch custom dataset tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class ISIC2019Original(Dataset):

    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
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
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert("RGB")
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
    

#8. Splits and dataloaders

#85% train, 15% validation
train_df, val_df = train_test_split(
    df_train, test_size=0.15, random_state=SEED, stratify=df_train['label']
)
#100% test 
test_df = df_test

print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")

#how, how many, and how fast images will move from the hard drive to your GPU
loader_args    = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

#loads RAW, original images
train_loader_A = DataLoader(ISIC2019Original(train_df, IMG_DIR_TRAIN, train_transform), shuffle=True,  **loader_args)
val_loader_A   = DataLoader(ISIC2019Original(val_df,   IMG_DIR_TRAIN, val_transform), shuffle=False, **loader_args)
test_loader_A  = DataLoader(ISIC2019Original(test_df,  IMG_DIR_TEST,  val_transform), shuffle=False, **loader_args)

#loads MASKED images
train_loader_B = DataLoader(ISIC2019Masked(train_df, IMG_DIR_TRAIN, MASK_DIR_TRAIN, train_transform), shuffle=True,  **loader_args)
val_loader_B   = DataLoader(ISIC2019Masked(val_df,   IMG_DIR_TRAIN, MASK_DIR_TRAIN, val_transform), shuffle=False, **loader_args)
test_loader_B  = DataLoader(ISIC2019Masked(test_df,  IMG_DIR_TEST,  MASK_DIR_TEST,  val_transform), shuffle=False, **loader_args)


#9. Model and Loss
#penalize rare class mistakes heavily, balancing the model's judgment and uses label_smoothing to prevent overconfident predictions.
class_counts = df_train['label'].value_counts().sort_index().values
class_weights = 1.0/class_counts 
class_weights = class_weights/class_weights.sum() * NUM_CLASSES
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

def build_model():
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = m.classifier[1].in_features #get the number of input connections coming from the network

    #replace head of the network w/ custom classifier
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3), #turn-off 30% of neurons randomly to prevent overfitting 
        nn.Linear(in_features, NUM_CLASSES) #substitude last layer to match the number of types of lesions
    )
    return m.to(device)

def compute_accuracy(outputs, labels):
    #1.get highest-probability choices with argmax
    #2.compare with true labels with ==
    #3.convert True/False to 1.0/0.0 with .float
    #4.average them with .mean and convert to a standard Python number with .item
    return (outputs.argmax(dim=1) == labels).float().mean().item()

def mixup_data(x, y, alpha=0.2):
    #randomly decides the blending percentage 
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1

    #match anothe batch of images 
    index = torch.randperm(x.size(0)).to(device)

    #blend the images together 
    mixed_x = lam*x + (1-lam)*x[index]

    return mixed_x, y, y[index], lam

#calculates the loss proportionally, blanding the penalties based on the lam of the original labels 
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#10. Training Function
def train_model(train_loader, val_loader, model_name, num_epochs=NUM_EPOCHS):

    print(f"\n{'='*60}")
    print(f"  TRAINING MODEL {model_name}")
    print(f"{'='*60}")

    model  = build_model()

    #speed up GPU training using mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    #keeps all of the information of the training 
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "phase": []}

    best_acc = -1.0
    UNFREEZE = 5

    #freeze the backbone weights to train only the head first
    for param in model.features.parameters():
        param.requires_grad = False

    #configures the Adam optimizer to adjust just the unfrozen weights and using weight decay to prevent overfitting
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    
    #automatically cuts the learning rate in half if the score stops improving for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    for epoch in range(1, num_epochs + 1):
        if epoch == UNFREEZE:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR/10, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
            print(f"\n Epoch {epoch}: Thawed Backbone, LR={LR/10:.2e}")

        phase = "frozen" if epoch < UNFREEZE else "finetuning"
        model.train()
        
        #standard PyTorch training loop
        t_loss, t_acc = 0.0, 0.0
        for imgs, labels, _ in train_loader:

            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                out  = model(imgs)
                loss = mixup_criterion(criterion, out, labels_a, labels_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            t_acc  += compute_accuracy(out, labels_a)

        t_loss /= len(train_loader)
        t_acc  /= len(train_loader)

        model.eval()

        #standard PyTorch evaluation loop
        v_loss, v_acc = 0.0, 0.0
        with torch.no_grad():

            for imgs, labels, _ in val_loader:

                imgs, labels = imgs.to(device), labels.to(device)

                with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                    out  = model(imgs)
                    loss = criterion(out, labels)

                v_loss += loss.item()
                v_acc  += compute_accuracy(out, labels)

        v_loss /= len(val_loader)
        v_acc  /= len(val_loader)

        #update learning rate
        scheduler.step(v_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        #check the best learning rate 
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(),
                       f"efficientnet_{model_name}.pth")
        
        #save values of epoch 
        history["epoch"].append(epoch)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["phase"].append(phase)
        print(f"[{epoch:02d}/{num_epochs}] "
              f"loss:{t_loss:.4f} acc:{t_acc:.4f} | "
              f"val_loss:{v_loss:.4f} val_acc:{v_acc:.4f} | "
              f"LR:{lr_now:.2e} | {phase}")

    print(f"\n Best Val Acc [{model_name}]: {best_acc:.4f}")

    #export the history into a permanent CSV
    with open(f"history_{model_name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()

        for row in zip(*history.values()):
            writer.writerow(dict(zip(history.keys(), row)))

    return model, history, best_acc

#11. Train
model_A, history_A, best_acc_A = train_model(train_loader_A, val_loader_A, model_name="A_original")
model_B, history_B, best_acc_B = train_model(train_loader_B, val_loader_B, model_name="B_masked")

#12. Learning curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history_A["epoch"], history_A["val_loss"], "b-o", markersize=4, label="Val Loss — A")
ax.plot(history_B["epoch"], history_B["val_loss"], "r-s", markersize=4, label="Val Loss — B")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Validation Loss — Model A vs Model B")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve_loss_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history_A["epoch"], history_A["val_acc"], "b-o", markersize=4, label="Val Acc — A")
ax.plot(history_B["epoch"], history_B["val_acc"], "g-s", markersize=4, label="Val Acc — B")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Validation Accuracy — Model A vs Model B")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("learning_curve_acc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nTraining complete.")
print("Models saved: efficientnet_A_original.pth, efficientnet_B_masked.pth")
