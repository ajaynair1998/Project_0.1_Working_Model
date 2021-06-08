import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, ReLU, Dropout, Dense, Softmax
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
import os
import numpy as np


predict_folder = r'predict'
img = r'image.jpg'
saved_model = r'saved_model'

# functions for accessing directory path
__dirname = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(__dirname, predict_folder, img)



img = load_img(image_path, target_size=(512, 512),
               color_mode='grayscale',
               )

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)


model = Sequential([
    Conv2D(256, input_shape=(512, 512, 1), kernel_size=(2, 2),
           strides=(2, 2), padding='same', activation=tf.keras.activations.sigmoid),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
    Conv2D(512, strides=(1, 1), padding='same', kernel_size=(2, 2), input_shape=(256, 256, 1), activation=tf.keras.
           activations.sigmoid),
    ReLU(),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
    GlobalAveragePooling2D(),
    ReLU(),
    Dense(256),
    ReLU(),
    Dropout(0.5),
    Dense(2),
    Softmax(),

])

model.summary()

model.load_weights(os.path.join(__dirname, saved_model, 'my_model.h5'))

prediction = model.predict(img_array)

print("The Probability of the Image being Positive is \t" + str(prediction[0][0] * 100) + '\nand negative is\t' + str(
    prediction[0][1] * 100))
