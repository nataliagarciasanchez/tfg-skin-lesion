#TFG MCP Models
#Core engine

#1. Libraries

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

#2. Configuration

IMG_SIZE = 224
SEG_SIZE = 320

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    'MEL':  'Melanoma - malignant lesion, requires urgent attention',
    'NV':   'Melanocytic nevus - benign mole',
    'BCC':  'Basal cell carcinoma - slow-growing malignant lesion',
    'AK':   'Actinic keratosis - premalignant lesion',
    'BKL':  'Benign keratosis - benign lesion',
    'DF':   'Dermatofibroma - benign lesion',
    'VASC': 'Vascular lesion - benign',
    'SCC':  'Squamous cell carcinoma - malignant',
}

MALIGNANT = {'MEL', 'BCC', 'SCC', 'AK'}

#save results to current file 
RESULTS_DIR = Path(__file__).parent / "Results"

#force execution on CPU
device = torch.device("cpu")

#ImageNet baseline for normalization
seg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
seg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

#classification preprocessing of image 
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#3. Model initialization and weight load 
def load_models(unet_path, effnet_path):

    #initialize U-Net++ structure
    unet = smp.UnetPlusPlus(
        encoder_name="efficientnet-b5",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    #load the segementation trained weights 
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()

    #initialize EfficientNet-B4 structure 
    effnet = models.efficientnet_b4(weights=None)
    
    #replace the head with the custom 8 class classifier
    effnet.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(effnet.classifier[1].in_features, NUM_CLASSES)
    )
    
    #load the trained classification weights 
    effnet.load_state_dict(torch.load(effnet_path, map_location=device, weights_only=True))
    effnet.eval()

    return unet, effnet

#4. Core engine
def predict(image_path, unet, effnet):

    #open skin lesion image file and ensure the RGB channels
    img = Image.open(image_path).convert("RGB")
     
    #SEGMENTATION PIPELINE
    #segmentation preprocessing of image
    img_seg = img.resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
    img_t = torch.tensor(np.array(img_seg)).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0)
    img_t = (img_t - seg_mean) / seg_std

    #run the image through the segmentation model without updating gradients
    with torch.no_grad():
        prob_map = torch.sigmoid(unet(img_t)).squeeze().numpy()
        mask = (prob_map > 0.5).astype(np.uint8)

    #ratio of white pixels to determine total area covered by the lesion
    lesion_pct = float(mask.mean() * 100)

    #save mask image
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    #unique filenames using timestamps to prevent file overwrites
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    mask_filename = f"{image_name}_mask_{timestamp}.png"
    mask_path = RESULTS_DIR / mask_filename

    #rebuild the image 
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    #resize the black and white mask 
    mask_img = mask_img.resize(img.size, Image.NEAREST)
    mask_img.save(str(mask_path))

    #CLASSIFICATION PIPELINE
    #prepare the image 
    img_cls = val_transform(img).unsqueeze(0)

    #run the image through the clasification EfficientNet-B4 
    with torch.no_grad():
        probs = torch.softmax(effnet(img_cls), dim=1).squeeze().numpy()

    #locate the highest score index to get the winner prediction
    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    #pack the 8 individual decimals into a readable map
    all_probs  = {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)}

    return {
        "predicted_class":     pred_class,
        "confidence":          round(confidence, 4),
        "all_probabilities":   all_probs,
        "lesion_coverage_pct": round(lesion_pct, 2),
        "class_description":   CLASS_DESCRIPTIONS[pred_class],
        "is_malignant":        pred_class in MALIGNANT,
        "mask_saved_path":     str(mask_path),
    }