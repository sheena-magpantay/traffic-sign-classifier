import os
import random
import shutil

src_dir = "data/dataset"
dst_dir = "dataset_split"

split_ratio = (0.7, 0.15, 0.15)  # train, val, test

for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_end = int(len(images) * split_ratio[0])
    val_end = train_end + int(len(images) * split_ratio[1])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        split_class_dir = os.path.join(dst_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for f in files:
            src_file = os.path.join(class_path, f)
            dst_file = os.path.join(split_class_dir, f)
            shutil.copy(src_file, dst_file)

print("Dataset successfully split into train/val/test!")