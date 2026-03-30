import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
import os

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32
train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
test_dir = "dataset_split/test"

def get_generators(use_augmentation=True):
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

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


# DL Baseline
def build_model(num_classes, use_dropout=True):
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            ),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
        ]
    )
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


train_generator, val_generator, test_generator = get_generators(
    use_augmentation=True
)
num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())

# Non-DL Baseline
print(f"Non-DL Baseline Accuracy (Random Guess floor): {100/num_classes:.2f}%")

print("\nTraining Full Model")
model = build_model(num_classes)
history = model.fit(train_generator, validation_data=val_generator, epochs=20)

# Calibration
print("\nGenerating Calibration Curve")
test_generator.reset()
y_probs = model.predict(test_generator)
y_true = test_generator.classes

prob_pos = np.max(y_probs, axis=1)
correct_mask = (np.argmax(y_probs, axis=1) == y_true).astype(int)
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

# Confusion Matrix
print("\nGenerating Slice Analysis")
y_pred = np.argmax(y_probs, axis=1)
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

print("\nRunning Ablation Experiments")
# No Dropout
model_no_dropout = build_model(num_classes, use_dropout=False)
model_no_dropout.fit(train_generator, epochs=5, verbose=0)
_, acc_no_dropout = model_no_dropout.evaluate(test_generator, verbose=0)

# No Augmentation
train_gen_no_aug, _, _ = get_generators(use_augmentation=False)
model_no_aug = build_model(num_classes)
model_no_aug.fit(train_gen_no_aug, epochs=5, verbose=0)
_, acc_no_aug = model_no_aug.evaluate(test_generator, verbose=0)

_, acc_full = model.evaluate(test_generator, verbose=0)

# Ablation
results = {
    "Full Model": acc_full,
    "No Dropout": acc_no_dropout,
    "No Augmentation": acc_no_aug,
}
plt.bar(results.keys(), results.values(), color=["green", "orange", "red"])
plt.ylabel("Accuracy")
plt.title("Ablation Study Results")
plt.show()
