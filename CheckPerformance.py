
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np
# next view images not classified well ! +aoc
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import  confusion_matrix,ConfusionMatrixDisplay
from PIL import Image
import cv2
import random

def get_preditictions_data(model,generator,threshold = 0.5):

  # Predict classes for the training and validation datasets
  predictions = model.predict_generator(generator).flatten()

  # Convert probabilities to binary predictions (0 or 1)
  predictions_binary = (predictions > threshold).astype(int)

  # Calculate precision, recall, and confusion matrix for training data
  precision_recall = classification_report(generator.classes, predictions_binary)
  cm = confusion_matrix(generator.classes, predictions_binary)


  return predictions, predictions_binary, precision_recall,cm


def get_stat(generator):

    data_mean = 0.0
    data_var = 0.0

    num_batches = len(generator)

    for i in range(num_batches):
        batch_images, _ = generator[i]  # Get a batch of images (ignore labels, denoted by '_')
        batch_mean = np.mean(batch_images)  # Calculate the mean of the batch

        # Calculate the variance of the batch
        batch_var = np.mean((batch_images - batch_mean)**2)

        data_mean += batch_mean
        data_var += batch_var

    # Calculate the mean and standard deviation across all batches
    data_mean /= num_batches
    data_std = np.sqrt(data_var / num_batches)

    print("Mean pixel value of the  data:", data_mean)
    print("Standard deviation of the  data:", data_std)

def plot_roc_aoc_All(train_generator,train_predictions,validation_generator,valid_predictions,
                     test_generator,test_predictions):
    plt.figure()
    # Calculate ROC curve and AUC for training data
    train_roc_auc = roc_auc_score(train_generator.classes, train_predictions)
    fpr_train, tpr_train, _ = roc_curve(train_generator.classes, train_predictions)
    plt.plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {train_roc_auc:.2f})')

    # Calculate ROC curve and AUC for validation data
    valid_roc_auc = roc_auc_score(validation_generator.classes, valid_predictions)
    fpr_valid, tpr_valid, _ = roc_curve(validation_generator.classes, valid_predictions)
    plt.plot(fpr_valid, tpr_valid, label=f'Validation ROC (AUC = {valid_roc_auc:.2f})')

    # Calculate ROC curve and AUC for Test data
    valid_roc_auc = roc_auc_score(test_generator.classes, test_predictions)
    fpr_valid, tpr_valid, _ = roc_curve(test_generator.classes, test_predictions)
    plt.plot(fpr_valid, tpr_valid, label=f'Test ROC (AUC = {valid_roc_auc:.2f})')

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')


def plot_confusion_mtrix(train_cm,valid_cm,test_cm,class_labels):
    # Plot the confusion matrices using seaborn heatmap
    plt.figure(figsize=(12, 4))


    #   plt.imshow(np.log(cm+ 1e-5), interpolation='nearest', cmap=plt.cm.Blues)

    # Train confusion matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Train Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Validation confusion matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(valid_cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')


    # Test confusion matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')


    plt.tight_layout()

    # percents
    train_cm_percent = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis] * 100
    valid_cm_percent = valid_cm.astype('float') / valid_cm.sum(axis=1)[:, np.newaxis] * 100
    test_cm_percent = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 4))

    # Train confusion matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(train_cm_percent, annot=True, fmt='1.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Train Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Validation confusion matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(valid_cm_percent, annot=True, fmt='1.1f', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')


    # Test confusion matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(test_cm_percent, annot=True, fmt='1.1f', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')


    plt.tight_layout()

def evaluate_with_threshold(y_true, y_pred_probs, threshold):
    y_pred_binary = (y_pred_probs > threshold).astype(int)

    # TN FP
    # FN TP
    cm = confusion_matrix(y_true, y_pred_binary)
    '''
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
    '''
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision  = precision_score(y_true, y_pred_binary) #  TP/(TP+FP) #
    recall =recall_score(y_true, y_pred_binary)  #TP/(TP+FN)
    f1 =  f1_score(y_true, y_pred_binary) #  2*precision*recall/(precision+recall)


    return accuracy, precision, recall, f1, cm

def plot_metrics_vs_threshold(thresholds, accuracies, precisions, recalls, f1_scores,title):

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, label='Accuracy', marker='o')
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='o')
    plt.plot(thresholds, f1_scores, label='F1-score', marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title([title + ' Metrics vs. Threshold'])
    plt.legend()
    plt.grid(True)
    plt.show()

def get_pgn_data(file_path):
    try:
        img = Image.open(file_path)
        img_array = np.array(img)
        # Perform further operations on the 'img_array' here
    except Exception as e:
        print(f"Error: {e}")

    return img_array

def dataperThreshold(generator,predictions):

    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        accuracy, precision, recall, f1, cm = evaluate_with_threshold(generator.classes, predictions, threshold)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Optional: Print metrics and confusion matrix for each threshold
        print(f"Threshold: {threshold:.1f}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("Confusion Matrix:")
        print(cm)
        print("-" * 30)

    return thresholds, accuracies, precisions, recalls, f1_scores

def show16ImagesGrid(pathDiaplay):
  image_files = os.listdir(pathDiaplay )
  selected_files = np.random.choice(image_files, size=16, replace=False)
  window_size = (19*2, 19*2)

  fig, ax = plt.subplots(4, 4, figsize=(4, 4))

  for i, file_name in enumerate(selected_files):
      file_path = os.path.join(pathDiaplay, file_name)
      img = get_pgn_data(file_path)

      # Resize the image to the desired window size
      img = cv2.resize(img, window_size, interpolation=cv2.INTER_AREA)

      # Plot the image
      row_idx, col_idx = i // 4, i % 4
      ax[row_idx, col_idx].imshow(img, cmap='gray')
      ax[row_idx, col_idx].axis('off')

def view_misclassified_samples(test_generator, predictions_binary,true_label, threshold=0.5, num_samples=16):
    num_samples = min(num_samples, len(test_generator.classes))  # Ensure num_samples is not greater than the total samples
    misclassified_indices = np.where((test_generator.classes != predictions_binary) & (test_generator.classes == true_label))[0]
    random_indices = random.sample(list(misclassified_indices), num_samples)

    # Create a 4x4 grid to display the images
    plt.figure(figsize=(10, 10))
    for i, index in enumerate(random_indices):
        img_batch, label_batch = test_generator[index // test_generator.batch_size]
        img = img_batch[index % test_generator.batch_size]
        label = label_batch[index % test_generator.batch_size]
        prediction = predictions_binary[index]

        plt.subplot(4, 4, i+1)
        plt.imshow(img[:, :, 0], cmap='gray')
        #plt.title(f'True Label: {label}\nPredicted Label: {prediction}')
        plt.axis('off')

    plt.tight_layout()

#==================
#loaad model
typeModel=1

if typeModel==1:
    model_path='D:\\code2018\\model'  #  simple model with 64 filters
    model = tf.keras.models.load_model(os.path.join(model_path,'model.keras')) # h5 wasnt saved

else:
    model_path='D:\\code2018\\model2' #  model filters with dropout + Adam lr + l2 reg
    model = tf.keras.models.load_model(os.path.join(model_path,'best_model_weights.h5'))


model.summary()

train_path ='D:\\Downloads\\faces\\Imagespng\\train'
test_folder='D:\\Downloads\\faces\\Imagespng\\test'
image_width=19
image_height=19
batch_size=64

show16ImagesGrid(os.path.join(test_folder,'face'))

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
    # Add other preprocessing settings for test data if needed
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary',  # Use 'binary' for binary classification
    color_mode='grayscale',  # For grayscale images
    shuffle=False,
    classes=['non-face', 'face'])


test_generator_one = test_datagen.flow_from_directory(
    test_folder,
    target_size=(image_width, image_height),
    batch_size=1,
    class_mode='binary',  # Use 'binary' for binary classification
    color_mode='grayscale',  # For grayscale images
    shuffle=False,
    classes=['non-face', 'face'])


test_predictions, test_predictions_binary, test_precision_recall,test_confusion_matrix = get_preditictions_data(model,test_generator)

print("Test Precision & Recall:\n", test_precision_recall)
print("Test Confusion Matrix:\n", test_confusion_matrix)

view_misclassified_samples(test_generator_one, test_predictions_binary,1)
view_misclassified_samples(test_generator_one, test_predictions_binary,0)
#-----------train/validation

train_datagen = ImageDataGenerator( rescale=1.0 / 255,
                                    rotation_range=5,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=[0.9,1.1],
                                    horizontal_flip=True,
                                    brightness_range=(0.9, 1.1),
                                    validation_split=0.2)  # val 20%

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_path,
                                               target_size=(image_width, image_height),
                                               color_mode='grayscale',
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               shuffle=False,
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


train_predictions, train_predictions_binary, train_precision_recall,train_confusion_matrix=get_preditictions_data(model,train_generator)
print("Training Precision & Recall:\n", train_precision_recall)
print("Training Confusion Matrix:\n", train_confusion_matrix)



valid_predictions, valid_predictions_binary, valid_precision_recall,valid_confusion_matrix=get_preditictions_data(model,validation_generator)
print("Validation Precision & Recall:\n", valid_precision_recall)
print("Validation Confusion Matrix:\n", valid_confusion_matrix)

'''
Training Precision & Recall:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99      3639
           1       0.98      0.99      0.98      1944

    accuracy                           0.99      5583
   macro avg       0.99      0.99      0.99      5583
weighted avg       0.99      0.99      0.99      5583

Training Confusion Matrix:
 [[3601   38]
 [  24 1920]]
 
 Validation Precision & Recall:
               precision    recall  f1-score   support

           0       1.00      0.93      0.96       909
           1       0.89      1.00      0.94       485

    accuracy                           0.95      1394
   macro avg       0.94      0.96      0.95      1394
weighted avg       0.96      0.95      0.96      1394

Validation Confusion Matrix:
 [[848  61]
 [  2 483]]


Test Precision & Recall:
               precision    recall  f1-score   support

           0       0.99      0.96      0.98     23573
           1       0.27      0.68      0.39       472

    accuracy                           0.96     24045
   macro avg       0.63      0.82      0.68     24045
weighted avg       0.98      0.96      0.97     24045

Test Confusion Matrix:
 [[22712   861]
 [  153   319]]
 
 

'''

#----------------------

train_datagen = ImageDataGenerator(validation_split=0.2 )  # val 20%
val_datagen = ImageDataGenerator( validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_path,
                                               target_size=(image_width, image_height),
                                               color_mode='grayscale',
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               shuffle=False,
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

# no change images
test_datagen = ImageDataGenerator(
    # Add other preprocessing settings for test data if needed
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary',  # Use 'binary' for binary classification
    color_mode='grayscale',  # For grayscale images
    shuffle=False,
    classes=['non-face', 'face'])


print('Train')
get_stat(train_generator)
print('Validation')
get_stat(validation_generator)
print('Test')
get_stat(test_generator)

'''

Found 5583 images belonging to 2 classes. Train 
Found 1394 images belonging to 2 classes. Validation
Found 24045 images belonging to 2 classes. Test

Train
Mean pixel value of the  data: 112.09317794713107
Standard deviation of the  data: 54.67850386385896
Validation
Mean pixel value of the  data: 105.10203621604226
Standard deviation of the  data: 53.696102496734746
Test
Mean pixel value of the  data: 107.81450467414044
Standard deviation of the  data: 52.53765693988822

'''


#roc
plot_roc_aoc_All(train_generator,train_predictions,validation_generator,valid_predictions,
                         test_generator,test_predictions)

plt.savefig(os.path.join(model_path,'roc_train_validation.png'))


#view confusion matrix

class_labels = train_generator.class_indices.keys()
plot_confusion_mtrix(train_confusion_matrix,valid_confusion_matrix,test_confusion_matrix,class_labels)


# check best threshold

# Assuming you have the ground truth labels and the predicted probabilities for the test dataset



# Plot metrics vs. threshold
vthresholds, vaccuracies, vprecisions, vrecalls, vf1_scores = dataperThreshold(validation_generator,valid_predictions )
plot_metrics_vs_threshold(vthresholds, vaccuracies, vprecisions, vrecalls, vf1_scores,'validation')


tthresholds, taccuracies, tprecisions, trecalls, tf1_scores = dataperThreshold(train_generator,train_predictions )
plot_metrics_vs_threshold(tthresholds, taccuracies, tprecisions, trecalls, tf1_scores,'train')


plt.show()

print('.')

x=0.15*np.array(vprecisions) +0.85*np.array(vrecalls)
idx=x.argmax()

plt.figure()
plt.grid(True)
plt.plot(vthresholds,x)
plt.plot(vthresholds[idx],x[idx],'rs')
plt.xlabel('Threshold')
plt.ylabel('Weighted Score')
plt.title('Weighted Score 0.15*np.array(vprecisions) +0.85*np.array(vrecalls)')

#by validation
bestthreshold=vthresholds[idx]# 0.2

accuracy_train, precision_train, recall_train, f1_train, cm_train = evaluate_with_threshold(train_generator.classes,  train_predictions, bestthreshold)
accuracy_validation, precision_validation, recall_validation, f1_validation, cm_validation = evaluate_with_threshold(validation_generator.classes, valid_predictions, bestthreshold)
accuracy_test, precision_test, recall_test, f1_test, cm_test = evaluate_with_threshold(test_generator.classes, test_predictions, bestthreshold)


plot_confusion_mtrix(cm_train,cm_validation,cm_test,class_labels)

print('.')
