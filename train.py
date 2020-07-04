import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

PATH='/tmp/'

train_dir=os.path.join(PATH, 'Train')
val_dir=os.path.join(PATH, 'Validation')

train_image_generator=ImageDataGenerator(rescale=1./255, shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
validation_image_generator=ImageDataGenerator(rescale=1./255)

batch_size = 128
epochs = 75
IMG_HEIGHT = 300
IMG_WIDTH = 300

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH))
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH))

sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


#plotImages(sample_training_images[:5])

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    tensorflow.keras.layers.MaxPooling2D(pool_size=2),
    tensorflow.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tensorflow.keras.layers.MaxPooling2D(pool_size=2),
    tensorflow.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tensorflow.keras.layers.MaxPooling2D(),
    tensorflow.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tensorflow.keras.layers.MaxPooling2D(),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(64, activation='relu'),
    tensorflow.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=1927 // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=600 // batch_size
)
model.save('final_model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
