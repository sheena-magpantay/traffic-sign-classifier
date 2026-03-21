import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(BASE_DIR, "data", "dataset")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "docs", "screenshots")

IMG_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 30 

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# ----------------------------
# DATA LOADING & PREPROCESSING
# ----------------------------
def load_and_preprocess_data():
    images = []
    labels = []
    
    if not os.path.exists(DATADIR):
        raise FileNotFoundError(f"Directory {DATADIR} not found at {DATADIR}")

    categories = sorted(os.listdir(DATADIR))
    label_map = {category: i for i, category in enumerate(categories)}
    
    print(f"Loading {len(categories)} classes...")

    for category, label_index in label_map.items():
        path = os.path.join(DATADIR, category)
        if not os.path.isdir(path): continue
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label_index)
            except Exception:
                continue

    X = np.array(images) / 255.0  
    y = np.array(labels)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), label_map

# ----------------------------
# CNN MODEL
# ----------------------------
def build_model(num_classes):
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        data_augmentation, 
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------------------------
# REPORTING
# ----------------------------
def generate_checkpoint_reports(model, history, X_val, y_val, label_map):
    print("\n--- Generating Checkpoint Artifacts ---")
    
    # 1. Predictions
    y_pred = np.argmax(model.predict(X_val), axis=1)
    
    # FIX: Extract STRINGS as names, not INTS
    target_names = [str(k) for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    # 2. Save Classification Report
    report = classification_report(y_val, y_pred, target_names=target_names)
    report_path = os.path.join(SCREENSHOT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # 3. Accuracy & Loss Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(os.path.join(SCREENSHOT_DIR, "learning_curves.png"))
    plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, target_names, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"Artifacts successfully saved to: {SCREENSHOT_DIR}")

# ----------------------------
# RL 
# ----------------------------
class TrafficRLAgent:
    def __init__(self, label_map):
        self.actions = ["BRAKE", "MAINTAIN", "ACCELERATE"]
        self.label_map = label_map

    def decide_action(self, pred_idx, current_speed):
        # 1. Parse the Limit from the Class Name
        inv_map = {v: k for k, v in self.label_map.items()}
        class_name = inv_map.get(pred_idx, "").lower()
        
        # Determine logical limit
        limit = 100 # Default
        if "stop" in class_name: limit = 0
        elif "speedlimit" in class_name:
            digits = ''.join(filter(str.isdigit, class_name))
            limit = int(digits) if digits else 100

        # 2. Decision Logic: Compare Current Speed to Sign Limit
        # Buffer of 5 units to avoid jittering
        if current_speed > limit + 5:
            action_idx = 0  # BRAKE
            reason = f"Vehicle speed {current_speed} exceeds limit {limit}."
        elif current_speed < limit - 5:
            action_idx = 2  # ACCELERATE
            reason = f"Vehicle speed {current_speed} is below limit {limit}."
        else:
            action_idx = 1  # MAINTAIN
            reason = f"Vehicle speed {current_speed} is optimal for limit {limit}."

        return action_idx, reason, limit

# ----------------------------
# EXECUTION for PIPELINE
# ----------------------------
if __name__ == "__main__":
    try:
        (X_train, X_val, y_train, y_val), label_map = load_and_preprocess_data()

        num_classes = len(label_map)

        model = build_model(num_classes)
        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        print("\n--- CNN Training ---")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[stop_early, reduce_lr]
        )

        generate_checkpoint_reports(model, history, X_val, y_val, label_map)

        print("\n--- Phase 2: Explainable Inference ---")
        rl_agent = TrafficRLAgent(label_map)
        simulated_car_speed = 90 

        for i in range(min(5, len(X_val))):
            img = X_val[i:i+1]
            pred_idx = np.argmax(model.predict(img, verbose=0))
            act_idx, reason, limit = rl_agent.decide_action(pred_idx, simulated_car_speed)
            print(f"Sample {i} | Detected: {limit} | Action: {rl_agent.actions[act_idx]}")
            print(f"   ↳ Rationale: {reason}")

        model.save('traffic_sign_model_final.keras')
        print("\nCheckpoint Complete.")

    except Exception as e:
        print(f"Critical Error: {e}")
