import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter
import os

train_dir = "dataset_split/train"
val_dir   = "dataset_split/val"
test_dir  = "dataset_split/test"

IMG_HEIGHT, IMG_WIDTH = 160, 160
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

train_counts = Counter([d for d in os.listdir(train_dir) for _ in os.listdir(os.path.join(train_dir, d))])
total = sum(train_counts.values())
class_weights = {i: total/(len(train_counts)*count) for i, (cls, count) in enumerate(train_counts.items())}

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.7,1.3)
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training top layers...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

base_model.trainable = True
fine_tune_at = 100 

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(5e-6),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Fine-tuning last layers of MobileNetV2...")
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],
    class_weight=class_weights,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

label_map = {v: k.replace("_", " ").title() for k, v in train_generator.class_indices.items()}

def predict_readable(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)
    
    prediction = model.predict(img_array)
    class_idx = prediction.argmax(axis=-1)[0]
    readable_label = label_map[class_idx]
    confidence = prediction[0][class_idx]
    return readable_label, confidence

model.save("traffic_sign_mobilenetv2_finetuned.h5")
print("Model saved as traffic_sign_mobilenetv2_finetuned.h5")

import pickle
with open("models/class_indices.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)
print("Class indices saved to models/class_indices.pkl")