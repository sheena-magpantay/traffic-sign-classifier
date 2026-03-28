import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import os

MODEL_PATH = "models/traffic_sign_mobilenetv2_finetuned.h5"
CLASS_INDICES_PATH = "models/class_indices.pkl"
RL_MEMORY_PATH = "models/rl_memory.pkl"

model = load_model(MODEL_PATH)
print("CNN model loaded successfully.")

with open(CLASS_INDICES_PATH, "rb") as f:
    class_indices = pickle.load(f)

label_map = {v: k.replace("_", " ").title() for k, v in class_indices.items()}

if os.path.exists(RL_MEMORY_PATH):
    with open(RL_MEMORY_PATH, "rb") as f:
        rl_memory = pickle.load(f)
else:
    rl_memory = []

def predict_with_rl(image_path):
    # Load image
    img = image.load_img(image_path, target_size=(160,160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    pred_probs = model.predict(img_array)[0]
    pred_class = int(np.argmax(pred_probs))
    confidence = float(pred_probs[pred_class])
    readable_label = label_map[pred_class] 

    nlp_sentence = f"This is a {readable_label.lower()}" 

    print(f"\nCNN predicts: {readable_label} ({confidence*100:.2f}%)")
    print(f"NLP Output: {nlp_sentence}")

    while True:
        user_input = input("Is this correct? (yes/no): ").strip().lower()
        if user_input in ["yes", "no"]:
            break
        print("Please type 'yes' or 'no'.")
    reward = 1 if user_input == "yes" else 0

    rl_memory.append((pred_class, reward))
    with open(RL_MEMORY_PATH, "wb") as f:
        pickle.dump(rl_memory, f)

    print(f"Feedback recorded. Total entries in RL memory: {len(rl_memory)}\n")
    return readable_label, confidence, nlp_sentence, reward

if __name__ == "__main__":
    print("=== Traffic Sign RL Loop with NLP-style UI output ===")
    print("Type 'exit' to quit.\n")

    while True:
        img_path = input("Enter image path: ").strip()
        if img_path.lower() == "exit":
            print("Exiting RL loop.")
            break
        if not os.path.exists(img_path):
            print("File not found. Try again.\n")
            continue

        predict_with_rl(img_path)
