import tensorflow as tf
from tensorflow.keras import layers, models
import zipfile, os, glob
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import random

#1. DOWNLOAD AND EXTRACT DATASET
!wget -q https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

with zipfile.ZipFile("cats_and_dogs_filtered.zip", 'r') as zip_ref:
    zip_ref.extractall()

base_dir = "cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

#2. LOAD DATASET
img_size = (160, 160)
batch_sz = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_sz,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_sz,
    label_mode="binary",
    shuffle=False
)

#3. DEFINE MODEL
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(160, 160, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

#4. COMPILE AND TRAIN
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=20)

#5. EVALUATE MODEL
loss, acc = model.evaluate(val_ds)
print(f"\nValidation Accuracy: {acc * 100:.2f}%")

#6. TEST ON RANDOM IMAGE
test_path = random.choice(list(glob.glob(os.path.join(val_dir, "*", "*.jpg"))))
img = Image.open(test_path).resize(img_size)
arr = np.expand_dims(np.array(img) / 255.0, 0)

plt.imshow(img)
plt.axis('off')
plt.title("Actual: " + os.path.basename(os.path.dirname(test_path)))
plt.show()

pred = model.predict(arr)[0][0]
label = "Dog" if pred > 0.5 else "Cat"
print(f"Predicted: {label} ({pred:.2f})")
