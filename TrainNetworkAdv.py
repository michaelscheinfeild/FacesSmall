from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import os
from matplotlib import pyplot
import glob2
import matplotlib.pyplot as plt

image_width=19
image_height=19

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

def getSimpleModel(image_width, image_height):

  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  # Add the remaining layers as needed for your task
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  return model


def getSimpleModelDropOut(image_width, image_height):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 1)))
    model.add(MaxPooling2D((2, 2)))
    BatchNormalization(),

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    BatchNormalization(),

    # Add Dropout with 20% rate
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))  # Add 20% Dropout
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))  # Add L2 regularization

    return model


batch_size=64
train_path ='D:\\Downloads\\faces\\Imagespng\\train'

#==========================
# Define the filepath where the best weights will be saved
model_path = 'D:\\code2018\\model2'
checkpoint_filepath = os.path.join(model_path,'best_model_weights.h5')


face_train_path = os.path.join(train_path,'face')
nonface_train_path = os.path.join(train_path,'non-face')
negative_folder = nonface_train_path
positive_folder = face_train_path

model=getSimpleModel(image_width,image_height)
model.summary()

# Reduce the learning rate of Adam optimizer to 0.0001
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


#==============
train_datagen = ImageDataGenerator( rescale=1.0 / 255,
                                    rotation_range = 10,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=[0.8,1.2],
                                    horizontal_flip=True,
                                    brightness_range=(0.8, 1.2),
                                    validation_split=0.2)  # val 20%


val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)



train_generator = train_datagen.flow_from_directory(train_path,
                                               target_size=(image_width, image_height),
                                               color_mode='grayscale',
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               shuffle=True,
                                               classes=['non-face', 'face'],
                                               subset = 'training')


validation_generator = val_datagen.flow_from_directory(train_path,
                                           target_size=(image_width, image_height),
                                           color_mode='grayscale',
                                           batch_size=batch_size,
                                           class_mode='binary',
                                           shuffle=False,
                                           classes=['non-face', 'face'],
                                           subset = 'validation')






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




history = model.fit_generator(
      generator=train_generator,
      validation_data=validation_generator,
      steps_per_epoch=total_train_samples // batch_size,
      validation_steps  = total_val_samples // batch_size,
      epochs=epochs,
      class_weight=class_weight,
      callbacks=[checkpoint_callback,early_stop])



model.save(os.path.join(model_path,'model.keras'))

plot_history(history)

plt.show()
