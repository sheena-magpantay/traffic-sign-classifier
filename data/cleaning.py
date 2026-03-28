import os
import hashlib
from PIL import Image

dataset_path = "dataset"

MIN_WIDTH = 32
MIN_HEIGHT = 32

seen_hashes = set()
removed_corrupted = 0
removed_duplicates = 0
removed_small = 0
converted = 0

def get_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.isdir(class_path):
        continue
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # --- 1. Remove corrupted images ---
        try:
            img = Image.open(img_path)
            img.verify()
        except:
            print("Removing corrupted:", img_path)
            os.remove(img_path)
            removed_corrupted += 1
            continue
        
        # Re-open after verify (important)
        img = Image.open(img_path)

        # --- 2. Remove small/low-resolution images ---
        if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
            print("Removing small image:", img_path)
            os.remove(img_path)
            removed_small += 1
            continue

        # --- 3. Remove duplicates ---
        try:
            file_hash = get_hash(img_path)
            if file_hash in seen_hashes:
                print("Removing duplicate:", img_path)
                os.remove(img_path)
                removed_duplicates += 1
                continue
            else:
                seen_hashes.add(file_hash)
        except:
            continue

        # --- 4. Convert to RGB JPG ---
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")

            new_path = os.path.splitext(img_path)[0] + ".jpg"
            img.save(new_path, "JPEG", quality=95)

            if new_path != img_path:
                os.remove(img_path)
                converted += 1

        except Exception as e:
            print("Conversion failed:", img_path, e)

# --- 5. Show dataset summary ---
print("\n=== CLEANING SUMMARY ===")
print("Corrupted removed:", removed_corrupted)
print("Duplicates removed:", removed_duplicates)
print("Small images removed:", removed_small)
print("Converted to JPG:", converted)

print("\n=== CLASS COUNTS ===")
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        print(class_name, ":", len(os.listdir(class_path)))