import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class_labels = ['i', 'ii', 'iii', 'iv', 'v']  
img_size = (28, 28)

def load_images(folder_path):
    images = []
    labels = []
    num_images_per_class = [0] * len(class_labels)  
    for class_label in class_labels:
        class_folder = os.path.join(folder_path, class_label)
        for filename in os.listdir(class_folder):
            if not filename.endswith('.png'):
                continue
            img_path = os.path.join(class_folder, filename)
            img = Image.open(img_path).convert('L') 
            img = img.resize(img_size) 
            img_array = np.array(img) / 255.0  
            images.append(img_array)
            labels.append(class_labels.index(class_label))  
            num_images_per_class[class_labels.index(class_label)] += 1  
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, num_images_per_class

train_images, train_labels, train_num_images_per_class = load_images('dataset/train')
test_images, test_labels, test_num_images_per_class = load_images('dataset/test')
val_images, val_labels, val_num_images_per_class = load_images('dataset/val')

import tensorflow as tf
from tensorflow.keras import layers

num_classes = len(class_labels)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images.reshape((-1, 28, 28, 1)), train_labels, epochs=10,
                    validation_data=(val_images.reshape((-1, 28, 28, 1)), val_labels))

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("models.h5")