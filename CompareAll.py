import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import pyplot

import warnings
import numpy as np
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import  confusion_matrix

def CheckBestGM(gmeans,tprt, fprt, generator,predictions,thresholds):

    ix = np.argmax(gmeans)
    specificityt,recallt,precisiont = getMetrics(tprt[ix], fprt[ix])
    Sensitivityt = tprt[ix]#True Positive Rate
    Specificityt = (1-fprt[ix])#1 – False Positive Rate
    Recallt = Sensitivityt
    accuracyt, precisiont, recallt, f1t, cmt = evaluate_with_threshold(generator.classes, predictions, thresholds[ix])

    print('1 Best Threshold=%f, G-Mean=%.3f, Sensitivity=%.3f ,Specificity=%.3f ,Recall=%.3f,Precision=%.3f'
          % (thresholds[ix], gmeans[ix],Sensitivityt, Specificityt, Recallt, precisiont))

    print('Confusion1',cmt)

    return ix

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

def getPredictions(model,traingen,validgen,testgen):

    train_predictions = model.predict_generator(traingen).flatten()
    valid_predictions = model.predict_generator(validgen).flatten()
    test_predictions = model.predict_generator(testgen).flatten()

    train_roc_auc = roc_auc_score(traingen.classes, train_predictions)
    valid_roc_auc = roc_auc_score(validgen.classes, valid_predictions)
    test_roc_auc = roc_auc_score(testgen.classes, test_predictions)

    return train_predictions,valid_predictions,test_predictions,train_roc_auc,valid_roc_auc,test_roc_auc

def plot_fpr_tpr_test(fpr1, tpr1, fpr2, tpr2, fpr3,tpr3,ix1,ix2,ix3,title):

    pyplot.figure()
    # plot the roc curve for the model Train
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr1, tpr1, marker='.', label='model1')
    pyplot.plot(fpr2, tpr2, marker='.', label='model2')
    pyplot.plot(fpr3, tpr3, marker='.', label='model3')

    pyplot.scatter(fpr1[ix1], tpr1[ix1], s=60,marker='o', color='black', label='Best1')
    pyplot.scatter(fpr2[ix2], tpr2[ix2], s=60,marker='s', color='black', label='Best2')
    pyplot.scatter(fpr3[ix3], tpr3[ix3], s=60,marker='*', color='black', label='Best3')

    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    # show the plot
    plt.grid(True)
    pyplot.title(title)


def plot_fpr_tpr(fpr1, tpr1, fpr2, tpr2, fpr3,tpr3,title):

    pyplot.figure()
    # plot the roc curve for the model Train
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr1, tpr1, marker='.', label='Logistic1')
    pyplot.plot(fpr2, tpr2, marker='.', label='Logistic2')
    pyplot.plot(fpr3, tpr3, marker='.', label='Logistic3')

    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    # show the plot
    plt.grid(True)
    pyplot.title(title)

def getMetrics(fpr1v,tpr1v):

   # Calculate true negatives and false negatives from false positive rate (fpr) and true positive rate (tpr)
    true_negatives = 1 - fpr1v
    false_negatives = 1 - tpr1v

    # Calculate true positives and false positives
    true_positives = tpr1v
    false_positives = fpr1v

    # Calculate specificity
    specificity = true_negatives / (true_negatives + false_positives)

    # Calculate recall as sensetivity
    recall = true_positives / (true_positives + false_negatives)

    # Calculate precision
    precision = true_positives / (true_positives + false_positives)

    return specificity,recall,precision

#--------------------------------
# load models
model_path='D:\\code2018\\model'  #  simple model with 64 filters
model1 = tf.keras.models.load_model(os.path.join(model_path,'model.keras')) # h5 wasnt saved


model_path='D:\\code2018\\model2' #  model filters with dropout + Adam lr + l2 reg
model2 = tf.keras.models.load_model(os.path.join(model_path,'best_model_weights.h5'))

model_path='D:\\code2018\\model3' #mobilenet
model3 = tf.keras.models.load_model(os.path.join(model_path,'best_model_weights.h5'))

#---------------------------
train_path ='D:\\Downloads\\faces\\Imagespng\\train'
test_folder='D:\\Downloads\\faces\\Imagespng\\test'
#---------------------------
batch_size=64
image_width  = 19
image_height = 19

image_widthMobile=96
image_heightMobile=96

# generators
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


#---------------mobile
test_datagenMobile = ImageDataGenerator(
    rescale=1.0 / 255
    # Add other preprocessing settings for test data if needed
)

test_generatorMobile = test_datagenMobile.flow_from_directory(
    test_folder,
    target_size=(image_widthMobile, image_heightMobile),
    batch_size=batch_size,
    class_mode='binary',  # Use 'binary' for binary classification
    color_mode='rgb',  # For grayscale images
    shuffle=False,
    classes=['non-face', 'face'])



train_datagenMobile = ImageDataGenerator( rescale=1.0 / 255,
                                    rotation_range=5,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=[0.9,1.1],
                                    horizontal_flip=True,
                                    brightness_range=(0.9, 1.1),
                                    validation_split=0.2)  # val 20%

val_datagenMobile = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generatorMobile = train_datagenMobile.flow_from_directory(train_path,
                                               target_size=(image_widthMobile, image_heightMobile),
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               shuffle=False,
                                               classes=['non-face', 'face'],
                                               subset = 'training')

validation_generatorMobile = val_datagenMobile.flow_from_directory(train_path,
                                           target_size=(image_widthMobile, image_heightMobile),
                                           color_mode='rgb',
                                           batch_size=batch_size,
                                           class_mode='binary',
                                           shuffle=False,
                                           classes=['non-face', 'face'],
                                           subset = 'validation')


#--------------------------------
#https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/


# calculate scores

train_predictions1,valid_predictions1,test_predictions1,\
           train_roc_auc1,valid_roc_auc1,test_roc_auc1 = getPredictions(model1,train_generator,validation_generator,test_generator)


train_predictions2,valid_predictions2,test_predictions2,\
           train_roc_auc2,valid_roc_auc2,test_roc_auc2 = getPredictions(model2,train_generator,validation_generator,test_generator)


train_predictions3,valid_predictions3,test_predictions3,\
           train_roc_auc3,valid_roc_auc3,test_roc_auc3 = getPredictions(model3,train_generatorMobile,validation_generatorMobile,test_generatorMobile)


print('Model1,2,3 Train rocauc',train_roc_auc1,train_roc_auc2,train_roc_auc3)
print('Model1,2,3 Validation rocauc',valid_roc_auc1,valid_roc_auc2,valid_roc_auc3)
print('Model1,2,3 Test rocauc',test_roc_auc1,test_roc_auc2,test_roc_auc3)

fpr1, tpr1, thresholds1 = roc_curve(train_generator.classes, train_predictions1)
fpr2, tpr2, thresholds2 = roc_curve(train_generator.classes, train_predictions2)
fpr3, tpr3, thresholds3 = roc_curve(train_generatorMobile.classes, train_predictions3)

fpr1v, tpr1v, thresholds1v = roc_curve(validation_generator.classes, valid_predictions1)
fpr2v, tpr2v, thresholds2v = roc_curve(validation_generator.classes, valid_predictions2)
fpr3v, tpr3v, thresholds3v = roc_curve(validation_generatorMobile.classes, valid_predictions3)

fpr1t, tpr1t, thresholds1t = roc_curve(test_generator.classes, test_predictions1)
fpr2t, tpr2t, thresholds2t = roc_curve(test_generator.classes, test_predictions2)
fpr3t, tpr3t, thresholds3t = roc_curve(test_generatorMobile.classes, test_predictions3)



plot_fpr_tpr(fpr1, tpr1, fpr2, tpr2, fpr3, tpr3,'Train')
plot_fpr_tpr(fpr1v, tpr1v, fpr2v, tpr2v, fpr3v, tpr3v,'Validation')
plot_fpr_tpr(fpr1t, tpr1t, fpr2t, tpr2t, fpr3t, tpr3t,'Test')




'''
Sensitivity = TruePositive / (TruePositive + FalseNegative)
Specificity = TrueNegative / (FalsePositive + TrueNegative)
Where:

Sensitivity = True Positive Rate
Specificity = 1 – False Positive Rate
The Geometric Mean or G-Mean is a metric for imbalanced classification that, if optimized, will seek a balance between the sensitivity and the specificity.

G-Mean = sqrt(Sensitivity * Specificity)

'''

# calculate the g-mean for each threshold
gmeans1 = np.sqrt(tpr1t * (1-fpr1t))
gmeans2 = np.sqrt(tpr2t * (1-fpr2t))
gmeans3 = np.sqrt(tpr3t * (1-fpr3t))

plt.figure()
plt.plot(thresholds1t,gmeans1,label='model1')
plt.plot(thresholds2t,gmeans2,label='model2')
plt.plot(thresholds3t,gmeans3,label='model3')
plt.grid(True)
plt.legend()
plt.title('geomtric means')
plt.ylabel('G mean')
plt.xlabel('Threshold')
#print('Geometric mean',gmeans1,gmeans2,gmeans3)


ix1 = CheckBestGM(gmeans1,tpr1t, fpr1t, test_generator, test_predictions1, thresholds1t)
ix2 = CheckBestGM(gmeans2,tpr2t, fpr2t, test_generator, test_predictions2, thresholds2t)
ix3 = CheckBestGM(gmeans3,tpr3t, fpr3t, test_generatorMobile, test_predictions3, thresholds3t)

print('gmeans : ',gmeans1[ix1],gmeans2[ix2],gmeans3[ix3])

plot_fpr_tpr_test(fpr1t, tpr1t, fpr2t, tpr2t, fpr3t, tpr3t,ix1,ix2,ix3,'Test')

pyplot.show()

print('.')
