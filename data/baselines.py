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
