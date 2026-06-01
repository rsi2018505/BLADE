import os
import cv2

# =========================
# CONFIGURATION
# =========================
INPUT_DIR = r"C:\Users\RS\Desktop\BLD Final Dataset"
OUTPUT_DIR = r"C:\Users\RS\Desktop\Final BLD Dataset"

TARGET_SIZE = (512, 512)   # (width, height)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# =========================
# FUNCTION
# =========================
def process_images(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        
        # Get relative path to preserve hierarchy
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Leaf folder name (used for renaming)
        leaf_folder = os.path.basename(root)

        count = 0

        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                input_path = os.path.join(root, file)

                # Read image
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Skipping corrupted file: {input_path}")
                    continue

                # Resize image
                resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                # Create new filename
                ext = os.path.splitext(file)[1]
                new_name = f"{leaf_folder}_{count}{ext}"

                output_path = os.path.join(output_subdir, new_name)

                # Save image
                cv2.imwrite(output_path, resized)

                count += 1

        print(f"Processed folder: {root}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR)
    print("✅ All images processed successfully!")
