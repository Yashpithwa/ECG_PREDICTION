import os
import shutil
import random

# Source folder jaha abhi saare classes ke folders hain
source_dir = "ECG_DATA"
target_dir = "data"

# Split ratios
split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

# Ensure target folders exist
for split in split_ratio.keys():
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # List all image files
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * split_ratio["train"])
    val_end = train_end + int(total * split_ratio["val"])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    # Copy files to new structure
    for split_name, split_files in splits.items():
        split_class_dir = os.path.join(target_dir, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img_name in split_files:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(split_class_dir, img_name)
            shutil.copy(src_path, dst_path)

print("✅ Dataset successfully split into train/val/test folders!")
