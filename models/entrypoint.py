import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os

MODEL_PATH = "models/traffic_sign_mobilenetv2_finetuned.h5"
CLASS_INDICES_PATH = "models/class_indices.pkl"
RL_MEMORY_PATH = "models/rl_memory.pkl"

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "rb") as f:
    class_indices = pickle.load(f)

label_map = {v: k.replace("_", " ").title() for k, v in class_indices.items()}

if os.path.exists(RL_MEMORY_PATH):
    with open(RL_MEMORY_PATH, "rb") as f:
        rl_memory = pickle.load(f)
else:
    rl_memory = []

current_pred_class = None  # to store prediction for RL 

def open_file():
    global current_pred_class
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
        
    img = Image.open(file_path)
    img = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img)
    img_label.config(image=tk_img)
    img_label.image = tk_img

    img_array = tf.keras.preprocessing.image.load_img(file_path, target_size=(160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    pred_probs = model.predict(img_array)[0]
    pred_class = int(np.argmax(pred_probs))
    confidence = float(pred_probs[pred_class])
    readable_label = label_map[pred_class]
    nlp_sentence = f"This is a {readable_label.lower()}"

    prediction_label.config(text=f"CNN predicts: {readable_label} ({confidence*100:.2f}%)")
    nlp_label.config(text=f"NLP Output: {nlp_sentence}")

    current_pred_class = pred_class

def give_feedback(is_correct):
    global current_pred_class
    if current_pred_class is None:
        messagebox.showwarning("No prediction", "Please select an image first.")
        return

    reward = 10 if is_correct else "0 (Penalty)"
    rl_memory.append((current_pred_class, reward))

    with open(RL_MEMORY_PATH, "wb") as f:
        pickle.dump(rl_memory, f)

    messagebox.showinfo("Feedback recorded", f"Reward = {reward}\nTotal entries: {len(rl_memory)}")
    current_pred_class = None
    prediction_label.config(text="")
    nlp_label.config(text="")
    img_label.config(image="")


root = tk.Tk()
root.title("Traffic Sign Recognition")
root.configure(bg="#FFC0CB") 
root.geometry("500x600")

img_label = tk.Label(root, bg="#FFC0CB")
img_label.pack(pady=15)

prediction_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#FFC0CB")
prediction_label.pack(pady=2)

nlp_label = tk.Label(root, text="", font=("Arial", 11, "italic"), bg="#FFC0CB")
nlp_label.pack(pady=2)

open_button = tk.Button(root, text="Open Image", command=open_file, 
                        bg="#FF69B4", fg="white", font=("Arial", 11, "bold"),
                        relief="flat", cursor="hand2")
open_button.pack(pady=20, ipadx=15, ipady=5)

feedback_frame = tk.Frame(root, bg="#FFC0CB")
feedback_frame.pack(pady=10)

yes_button = tk.Button(feedback_frame, text="Yes", command=lambda: give_feedback(True), 
                       bg="#32CD32", fg="white", font=("Arial", 11, "bold"),
                       width=8) 
yes_button.pack(side="left", padx=10, ipady=8)

no_button = tk.Button(feedback_frame, text="No", command=lambda: give_feedback(False), 
                      bg="#FF4500", fg="white", font=("Arial", 11, "bold"),
                      width=8)
no_button.pack(side="right", padx=10, ipady=8)

root.mainloop()








