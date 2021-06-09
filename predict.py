import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, ReLU, Dropout, Dense, Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# raw strings
predict_folder = r'predict'
img = r'image.jpg'
saved_model = r'saved_model'

# functions for accessing directory path
__dirname = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(__dirname, predict_folder)

# Use image Data Generator to Take in Image to Predict
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    image_path,
    target_size=(512, 512),
    batch_size=1,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False)

# Predefined Layers of the Model
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

# Print the summary of the model
model.summary()

# Load the optimised weights
model.load_weights(os.path.join(__dirname, saved_model, 'my_model.h5'))

# Predict from generator (returns probabilities)
prediction = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Print the probability using a single image
print("The Probability of the Image being Negative is \t" + str(prediction[0][0] * 100) + '\nand Positive is\t' + str(
    prediction[0][1] * 100))
