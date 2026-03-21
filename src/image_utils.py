import cv2
import numpy as np

IMG_SIZE = 32

def preprocess_image(img):
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = img / 255.0

    return img

def load_image_from_path(path):
    img = cv2.imread(path)

    if img is None:
        return None

    return preprocess_image(img)
