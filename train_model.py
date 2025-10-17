# train_model.py (Windows-friendly, RGB + fallback)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Paths
train_dir = "data/train"
val_dir = "data/val"

# Parameters
img_size = (224, 224)
batch_size = 16

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Flow from directory (convert grayscale -> RGB)
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="rgb"   # Convert to 3-channel RGB
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="rgb"
)

# Try to load pretrained weights, fallback to scratch if fails
try:
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    print("✅ Pretrained EfficientNetB0 loaded successfully.")
except Exception as e:
    print("⚠️ Failed to load pretrained weights. Training from scratch.")
    print(e)
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))

base_model.trainable = False

# Custom head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save model
model.save("ecg_model.h5")
print("✅ Model training complete and saved as ecg_model.h5")
print("Classes:", train_gen.class_indices)
