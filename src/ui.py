
from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "traffic_model.h5")
model = load_model(MODEL_PATH)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file.filename == '':
        return "No file uploaded"

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return f"Class: {class_id} | Confidence: {confidence:.2f}"


if __name__ == "__main__":
    app.run(debug=True)
