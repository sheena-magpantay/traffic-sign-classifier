import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "dataset_split/train"
val_dir   = "dataset_split/val"
test_dir  = "dataset_split/test"

IMG_HEIGHT, IMG_WIDTH = 160, 160
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

def get_generators(use_augmentation=True):
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=(0.7, 1.3),
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


train_generator, val_generator, test_generator = get_generators(
    use_augmentation=True
)
num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())

train_counts = Counter(
    [
        d
        for d in os.listdir(train_dir)
        for _ in os.listdir(os.path.join(train_dir, d))
    ]
)
total = sum(train_counts.values())
class_weights = {
    i: total / (len(train_counts) * count)
    for i, (cls, count) in enumerate(train_counts.items())
}

#DL Baseline
def build_model(num_classes, use_dropout=True):
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model, base_model


#Non-DL Baseline
print(
    f"Non-DL Baseline Accuracy (Random Guess floor): {100/num_classes:.2f}%"
)

print("\nTraining Top Layers (Feature Extraction)...")
model, base_model = build_model(num_classes)

callbacks = [
    EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    ),
    ModelCheckpoint(
        "best_model.h5", monitor="val_accuracy", save_best_only=True
    ),
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

print("\nFine-tuning last layers of MobileNetV2...")
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],
    class_weight=class_weights,
    callbacks=callbacks,
)

print("\nGenerating Predictions for Analysis...")
test_generator.reset()
y_probs = model.predict(test_generator)
y_true = test_generator.classes
y_pred = np.argmax(y_probs, axis=1)

#Calibration
print("\nGenerating Calibration Curve...")
prob_pos = np.max(y_probs, axis=1)
correct_mask = (y_pred == y_true).astype(int)
fraction_of_positives, mean_predicted_value = calibration_curve(
    correct_mask, prob_pos, n_bins=10
)

plt.figure(figsize=(6, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.title("Calibration Curve")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Confusion Matrix
print("\nGenerating Slice Analysis (Confusion Matrix)...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels,
    cmap="Blues",
)
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_true, y_pred, target_names=class_labels))

#Ablation
print("\nRunning Ablation Experiments...")

#No Dropout
print("Training No-Dropout Ablation...")
model_no_dropout, _ = build_model(num_classes, use_dropout=False)
model_no_dropout.fit(
    train_generator, epochs=3, verbose=1, class_weight=class_weights
)
_, acc_no_dropout = model_no_dropout.evaluate(test_generator, verbose=0)

#No Augmentation
print("Training No-Augmentation Ablation...")
train_gen_no_aug, _, _ = get_generators(use_augmentation=False)
model_no_aug, _ = build_model(num_classes)
model_no_aug.fit(
    train_gen_no_aug, epochs=3, verbose=1, class_weight=class_weights
)
_, acc_no_aug = model_no_aug.evaluate(test_generator, verbose=0)

# Full Model eval
_, acc_full = model.evaluate(test_generator, verbose=0)

results = {
    "Full Model": acc_full,
    "No Dropout": acc_no_dropout,
    "No Augmentation": acc_no_aug,
}

plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=["green", "orange", "red"])
plt.ylabel("Accuracy")
plt.title("Ablation Study Results")
plt.show()

model.save("traffic_sign_mobilenetv2_finetuned.h5")
print("Model saved as traffic_sign_mobilenetv2_finetuned.h5")

os.makedirs("models", exist_ok=True)
with open("models/class_indices.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)
print("Class indices saved to models/class_indices.pkl")



