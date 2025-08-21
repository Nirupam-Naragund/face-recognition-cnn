import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json
import os

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
DATASET_PATH = "data/"
MODEL_PATH = "model/face_cnn.h5"
CLASS_INDICES_PATH = "class_indices.json"


datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation"
)


base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=10)


os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)


with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"Model saved at {MODEL_PATH}")
print(f"Class indices saved at {CLASS_INDICES_PATH}")
