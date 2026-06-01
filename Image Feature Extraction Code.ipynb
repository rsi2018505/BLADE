import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import entropy
from openpyxl import Workbook

# ============================================================
# INPUT DATASET PATH
# ============================================================

DATASET_PATH = r"C:\Users\RS\Desktop\Final BLD Dataset\Training\Banana Leafs"

# ============================================================
# OUTPUT EXCEL FILE
# ============================================================

OUTPUT_EXCEL = r"C:\Users\RS\Desktop\Final BLD Dataset\Training\banana_leaf_features.xlsx"

# ============================================================
# PARAMETERS
# ============================================================

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(image_path, class_name):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return None

    # --------------------------------------------------------
    # Basic Features
    # --------------------------------------------------------

    height, width, channels = image.shape
    image_size = os.path.getsize(image_path) / 1024  # KB

    # Filename
    filename = os.path.basename(image_path)

    # --------------------------------------------------------
    # RGB Features
    # --------------------------------------------------------

    mean_r = np.mean(image[:, :, 2])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 0])

    std_r = np.std(image[:, :, 2])
    std_g = np.std(image[:, :, 1])
    std_b = np.std(image[:, :, 0])

    # --------------------------------------------------------
    # HSV Features
    # --------------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    # --------------------------------------------------------
    # Grayscale Features
    # --------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()

    image_entropy = entropy(hist_norm + 1e-7)

    # --------------------------------------------------------
    # GLCM Texture Features
    # --------------------------------------------------------

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    glcm_dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_energy = graycoprops(glcm, 'energy')[0, 0]
    glcm_correlation = graycoprops(glcm, 'correlation')[0, 0]

    # --------------------------------------------------------
    # LBP Features
    # --------------------------------------------------------

    lbp = local_binary_pattern(
        gray,
        LBP_POINTS,
        LBP_RADIUS,
        method='uniform'
    )

    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)

    # --------------------------------------------------------
    # Edge Features
    # --------------------------------------------------------

    edges = cv2.Canny(gray, 100, 200)

    edge_pixel_count = np.sum(edges > 0)

    # --------------------------------------------------------
    # Disease Spot Approximation
    # --------------------------------------------------------

    # Dark spot mask
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    spot_area = np.sum(thresh > 0)

    infected_percentage = (spot_area / (height * width)) * 100

    # Count contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    spot_count = len(contours)

    # --------------------------------------------------------
    # Leaf Area Approximation
    # --------------------------------------------------------

    green_mask = cv2.inRange(
        hsv,
        (25, 20, 20),
        (100, 255, 255)
    )

    leaf_area = np.sum(green_mask > 0)

    leaf_area_percentage = (leaf_area / (height * width)) * 100

    # --------------------------------------------------------
    # Final Feature Dictionary
    # --------------------------------------------------------

    features = {

        # Metadata
        "Class": class_name,
        "Filename": filename,
        "Path": image_path,

        # Image Info
        "Width": width,
        "Height": height,
        "Channels": channels,
        "File_Size_KB": round(image_size, 2),

        # RGB
        "Mean_R": round(mean_r, 2),
        "Mean_G": round(mean_g, 2),
        "Mean_B": round(mean_b, 2),

        "Std_R": round(std_r, 2),
        "Std_G": round(std_g, 2),
        "Std_B": round(std_b, 2),

        # HSV
        "Mean_H": round(mean_h, 2),
        "Mean_S": round(mean_s, 2),
        "Mean_V": round(mean_v, 2),

        # Lighting
        "Brightness": round(brightness, 2),
        "Contrast": round(contrast, 2),
        "Entropy": round(image_entropy, 4),

        # Texture
        "GLCM_Contrast": round(glcm_contrast, 4),
        "GLCM_Dissimilarity": round(glcm_dissimilarity, 4),
        "GLCM_Homogeneity": round(glcm_homogeneity, 4),
        "GLCM_Energy": round(glcm_energy, 4),
        "GLCM_Correlation": round(glcm_correlation, 4),

        # LBP
        "LBP_Mean": round(lbp_mean, 4),
        "LBP_STD": round(lbp_std, 4),

        # Edge
        "Edge_Pixel_Count": int(edge_pixel_count),

        # Disease Approximation
        "Spot_Count": int(spot_count),
        "Spot_Area": int(spot_area),
        "Infected_Percentage": round(infected_percentage, 2),

        # Leaf Area
        "Leaf_Area": int(leaf_area),
        "Leaf_Area_Percentage": round(leaf_area_percentage, 2)
    }

    return features

# ============================================================
# MAIN PROCESSING
# ============================================================

all_features = []

class_folders = os.listdir(DATASET_PATH)

for class_name in class_folders:

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    image_files = os.listdir(class_path)

    print(f"\nProcessing Class: {class_name}")

    for image_name in tqdm(image_files):

        image_path = os.path.join(class_path, image_name)

        try:

            features = extract_features(image_path, class_name)

            if features is not None:
                all_features.append(features)

        except Exception as e:
            print(f"Error processing {image_path}")
            print(e)

# ============================================================
# SAVE TO EXCEL
# ============================================================

df = pd.DataFrame(all_features)

df.to_excel(OUTPUT_EXCEL, index=False)

print("\n================================================")
print("Feature Extraction Completed Successfully")
print("================================================")
print(f"Excel File Saved As: {OUTPUT_EXCEL}")
print(f"Total Images Processed: {len(df)}")
print("================================================")
