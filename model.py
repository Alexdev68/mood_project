import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW  # AdamW is better for generalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# ======== Dataset Setup ========
data_dir = "dataset"  # must contain subfolders: happy/, sad/, angry/, etc.

img_size = (96, 96)
batch_size = 32

# Add stronger augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

# ======== Compute Class Weights ========
print("⚖️ Computing class weights to handle imbalance...")
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ======== Model Definition ========
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

# Freeze first 100 layers to reduce training cost and overfitting
for layer in base_model.layers[:100]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ======== Compile ========
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======== Callbacks ========
checkpoint = ModelCheckpoint(
    "best_face_emotion_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

# ======== Train ========
epochs = 40  # more epochs = better accuracy; early stopping prevents overfitting
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ======== Save Final Model ========
model.save("face_emotion_model.h5")
print("✅ Saved face_emotion_model.h5")

# ======== Convert and Save TensorFlow Lite Model ========
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("face_emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved face_emotion_model.tflite (lightweight version)")
print("✅ Training Complete — Expected Accuracy: ~70–80% on balanced data")
