import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from keras.layers import  GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import glob2
import tensorflow as tf
from keras.layers import Input, Lambda
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

# [96, 128, 160, 192, 224
image_width, image_height = 19, 19
batch_size = 64


from keras.layers import Lambda

def plot_history(history):

  # Plot the training and validation loss
  plt.figure()
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.grid(True)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.savefig('loss.png')

  # Plot the training and validation accuracy
  plt.figure()
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.grid(True)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig('accuracy.png')


def count_png_images(folder_path):
    # Use glob to find all PGM files in the folder
    png_files = glob2.glob(os.path.join(folder_path, '*.png'))

    # Get the count of PGM images
    png_count = len(png_files)

    return png_count





# Function to convert grayscale images to RGB by duplicating the single channel
def expand_grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)



#--------------
#import cv2
#import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Step 1: Load the 19x19 PNG image
image_path = "D:\\Downloads\\faces\\Imagespng\\train\\face\\face00001.png"  # Replace with the actual image file path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Convert the image to a numpy array (19x19)
image_array = np.array(image)

# Step 3: Resize the image to 96x96 using cv2.resize
desired_width = 96
desired_height = 96
resized_image = cv2.resize(image_array, (desired_width, desired_height))

# Step 4: Add channel dimension to match MobileNet input format
resized_image = np.expand_dims(resized_image, axis=-1)

# Step 5: Convert the grayscale image to a three-channel (RGB) image
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

# Step 6: Prepare the input image for MobileNet
input_image = np.expand_dims(preprocess_input(rgb_image), axis=0)

def getMobileNetModel():
    # Step 7: Load MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

    # Freeze the first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.005))(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

model= getMobileNetModel()
# Step 7: Make predictions using the loaded model
predictionsSec = model.predict(input_image)

# Step 8: Display the predictions
print(predictionsSec)

#--------------------
# train phase

batch_size=64
train_path ='D:\\Downloads\\faces\\Imagespng\\train'
model_path = 'D:\\code2018\\model3' # where model saved

face_train_path = os.path.join(train_path,'face')
nonface_train_path = os.path.join(train_path,'non-face')
negative_folder = nonface_train_path
positive_folder = face_train_path


# Step 1: Create an ImageDataGenerator for preprocessing and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    # Add other augmentations as needed (rotation, width_shift_range, etc.)
    rotation_range = 15,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=[0.7,1.3],
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Step 2: Create the train_generator using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_path,  # Replace with the path to the folder containing your training data
    target_size=(96, 96),  # Resize images to the desired size
    batch_size=batch_size,  # Set your desired batch size
    color_mode='rgb',  # Convert images to RGB (3 channels)
    class_mode='binary',  # For binary classification use 'binary', for multi-class 'categorical'
    shuffle=True,  # Shuffle the data during training
    classes=['non-face', 'face'],
    subset = 'training'
)

validation_generator = val_datagen.flow_from_directory(train_path,
                                           target_size=(96, 96),
                                           color_mode='rgb',
                                           batch_size=batch_size,
                                           class_mode='binary',
                                           shuffle=False,
                                           classes=['non-face', 'face'],
                                           subset = 'validation')

# create network

# Define the filepath where the best weights will be saved

checkpoint_filepath = os.path.join(model_path,'best_model_weights.h5')

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                      monitor='val_accuracy',  # Choose the validation metric to monitor
                                      save_best_only=True,    # Save only the best model weights
                                      mode='max',             # For 'accuracy', use 'max'; for 'loss', use 'min'
                                      verbose=1)

# Define the EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


total_train_samples = train_generator.n #5583
total_val_samples =  validation_generator.n #1394

epochs=50 # just see it works

# Calculate class weights to handle class imbalance
total_negative_images = count_png_images(negative_folder)
total_positive_images = count_png_images(positive_folder)
total_images = total_negative_images + total_positive_images
class_weight = {0: total_images / (2 * total_negative_images), 1: total_images / (2 * total_positive_images)}


# Reduce the learning rate of Adam optimizer to 0.0001
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit_generator(
      generator=train_generator,
      validation_data=validation_generator,
      steps_per_epoch=total_train_samples // batch_size,
      validation_steps  = total_val_samples // batch_size,
      epochs=epochs,
      class_weight=class_weight,
      callbacks=[checkpoint_callback,early_stop])


model.save(os.path.join(model_path,'model.keras'))

# Save history to a JSON file
history_path = os.path.join(model_path,'training_history.json')
with open(history_path, 'w') as file:
    json.dump(history.history, file)

plot_history(history)

plt.show()

print('.')
