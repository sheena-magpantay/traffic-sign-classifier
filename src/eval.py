import numpy as np
import json
import cv2
from tensorflow.keras.models import load_model
from utils.image_utils import load_image_from_path

# ----------------------------
# Load model
# ----------------------------
model = load_model("models/traffic_model.h5")

# ----------------------------
# Load label map
# ----------------------------
with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}

# ----------------------------
# Predict
# ----------------------------
def predict_image(path):

    # Load image (make sure it's RGB or BGR consistent)
    img = load_image_from_path(path)

    img = cv2.resize(img, (32, 32))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)  # (1, 32, 32, 3)

    pred = model.predict(img, verbose=0)

    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    print("Raw prediction:", pred)
    print("Predicted class_id:", class_id)

    class_name = id_to_label.get(class_id, "Unknown")

    return class_name, confidence

# ----------------------------
# for testing
# ----------------------------
if __name__ == "__main__":
    label, confidence = predict_image("test.jpg")

    print("Traffic Sign:", label)
    print("Confidence:", round(confidence, 2))
