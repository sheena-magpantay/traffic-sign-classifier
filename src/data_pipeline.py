import os  
import numpy as np 
from utils.image_utils import load_image_from_path

def load_datasets(dataset_paths):
    images = []
    labels = []

    label_map = {
        "greenlight": 0,
        "redlight": 1,
        "speedlimit_20": 2,
        "speedlimit_30": 3,
        "speedlimit_40": 4,
        "speedlimit_50": 5,
        "speedlimit_60": 6,
        "speedlimit_70": 7,
        "speedlimit_80": 8,
        "speedlimit_90": 9,
        "speedlimit_100": 10,
        "speedlimit_110": 11,
        "speedlimit_120": 12
    }

    for dataset_path in dataset_paths:
        for label in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, label)

            if not os.path.isdir(class_path) or label not in label_map:
                continue

            for file in os.listdir(class_path):

                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(class_path, file)

                try:
                    img = load_image_from_path(img_path)

                    if img is None:
                        continue

                    images.append(img)
                    labels.append(label_map[label])

                except Exception as e:
                    print(f"Skipped {img_path}: {e}")
                    continue

    X = np.array(images)
    y = np.array(labels)

    print("Total images:", len(X))
    print("Total labels:", len(y))

    if len(X) != len(y):
        raise ValueError("Mismatch between images and labels!")

    return X, y, label_map
