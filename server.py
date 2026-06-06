#TFG MCP Server
#Communication bridge between LLM and models

#1. Libraries

from pathlib import Path
import os
from mcp.server.fastmcp import FastMCP
from models import load_models, predict, CLASS_NAMES

#2. Paths and load models 

#find the exact folder where this file is
BASE_DIR = Path(__file__).parent 

#create the full file paths for the trained models
UNET_PATH = str(BASE_DIR / "unet_best.pth")
EFFNET_PATH = str(BASE_DIR / "efficientnet_A_original.pth")

#load models into the memory of the computer using those paths
unet, effnet = load_models(UNET_PATH, EFFNET_PATH)

#3. Server 

#initialize the MCP Server and give it a name
mcp = FastMCP("skin-lesion-analyzer")

#4. Create a public button so the LLM can see and execute this function 
@mcp.tool()
async def analyze_skin_lesion(image_path: str) -> str:
    
    """Analyzes a dermoscopic image of a skin lesion.
    Uses a U-Net++ with EfficientNet-B5 encoder for segmentation
    and EfficientNet-B4 for classification.
    Returns diagnosis, confidence, and probabilities for 8 classes:
    MEL (melanoma), NV (nevus), BCC (basal cell carcinoma), 
    AK (actinic keratosis), BKL (benign keratosis), DF (dermatofibroma), 
    VASC (vascular lesion), SCC (squamous cell carcinoma). 
    Also saves the segmentation mask to the Results folder.

    Args:
        image_path: Absolute path to the dermoscopic image (JPG or PNG).
    """
    #verify the file actually exists
    if not os.path.exists(image_path):
        return "ERROR File not found: " + image_path
    
    #accept only valid image formats
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return "ERROR File must be JPG or PNG."
    
    try:
        #run the core models from models.py
        result = predict(image_path, unet, effnet)

        #format the probabilities for each of the 8 medical classes
        lines = []
        for cls in CLASS_NAMES:
            raw_decimal = result['all_probabilities'][cls]
            percentage = round(raw_decimal * 100, 1)
            lines.append(cls + ": " + str(percentage) + "%")
        
        probs_str = "\n".join(lines)

        if result['is_malignant'] == True:
            malignant_str = "YES - requires urgent medical evaluation"
        else:
            malignant_str = "NO - benign according to model"


        return (
            "DERMOSCOPIC ANALYSIS RESULT\n"
            "=====================================\n"
            "Diagnosis        : " + result['predicted_class'] + "\n"
            "Confidence       : " + str(round(result['confidence'] * 100, 1)) + "%\n"
            "Malignant        : " + malignant_str + "\n"
            "\nProbabilities per class:\n" + probs_str + "\n"
            "\nSegmentation mask saved to:\n  " + result['mask_saved_path'] + "\n"
            "\nNOTE: This is a decision-support tool only."
        )
    
    except Exception as e:
        return "ERROR during analysis: " + str(e)

#5. Server execution 
if __name__ == "__main__":
    mcp.run(transport="stdio")