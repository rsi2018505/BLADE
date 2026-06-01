import os
import cv2

# ====== CONFIG ======
INPUT_FOLDER = r"C:\Users\RS\Desktop\BLD Dataset"
OUTPUT_FOLDER = r"C:\Users\RS\Desktop\BLD_Cropped5"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PATCH_SIZE = 2048
STRIDE = 2048   # =512 → no overlap, <512 → overlapping patches

# ====== FUNCTION ======
def tile_image(image_path, output_folder, patch_size=512, stride=512):
    img = cv2.imread(image_path)

    if img is None:
        return

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    count = 0

    # Slide window across image
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]

            patch_name = f"{base_name}_y{y}_x{x}.png"
            cv2.imwrite(os.path.join(output_folder, patch_name), patch)

            count += 1

    print(f"✅ {base_name}: {count} patches created")

# ====== BATCH PROCESS ======
for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        tile_image(os.path.join(INPUT_FOLDER, file), OUTPUT_FOLDER, PATCH_SIZE, STRIDE)

print("🚀 All images tiled into patches successfully!")
