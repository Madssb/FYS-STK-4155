import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
import random

from tensorflow.keras import datasets, layers, models # my import
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer

data_dir = 'data/CCSN_v2/'
model_dir = 'models/cnn/'
figure_dir = 'figures/cnn/'
unique_dir = 'cloudnet128_100epochs_augmentFalse_centerTrue_optimizerSGD_lr0.001mom0.9/'
if not os.path.exists(figure_dir+unique_dir):
   os.makedirs(figure_dir+unique_dir)
target_size=(128,128)
#target_size=(227,227)

model = models.load_model(model_dir+unique_dir+'model.keras')
history = pd.read_csv(model_dir+unique_dir+'history.csv')
train_test_indices = np.load(model_dir+unique_dir+'train_test_indices.npy')
train_val_indices = np.load(model_dir+unique_dir+'train_val_indices.npy')

# Define function for splitting of training and test data
def train_test_split(X, Y, indices, test_size=0.2):
    n=len(X)
    stop_train = int(n*(1-test_size))
    trainx=X[indices[0:stop_train]]
    testx=X[indices[stop_train:]]
    trainy=Y[indices[0:stop_train]]
    testy=Y[indices[stop_train:]]

    return trainx, testx, trainy, testy

# Generate labels from folder names
Name=[]
for file in os.listdir(data_dir):
    Name+=[file]

N=[]
for i in range(len(Name)):
    N+=[i]
    
normal_mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name)) 

print(normal_mapping)
print(reverse_mapping)

# Load images

datax0=[]
datay0=[]
count=0
for file in Name:
    path=os.path.join(data_dir,file)
    for im in os.listdir(path):
        # Load images, rescaling to 227 x 227
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=target_size) 
        image=img_to_array(image)
        # Normalize RGB values to between 0 and 1
        image=image/255.0
        datax0.append(image)
        datay0.append(count)
    count=count+1

# Convert to numpy array
datax1=np.array(datax0)
datay1=np.array(datay0)

# Split into training and test data using shuffled indices
trainx0, testx0, trainy0, testy0 = train_test_split(datax1, datay1, train_test_indices, test_size=0.25)

# Create categorical output for training data
y_train=to_categorical(trainy0)

# Create categorical output for test data
testy=to_categorical(testy0)

# Split into training and test data using shuffled indices
trainx,valx,trainy,valy=train_test_split(trainx0,y_train,train_val_indices, test_size=0.2)


# Print shapes of train, validation and test data
print('Whole dataset')
print('X: ', np.shape(datax0))
print('Y: ', np.shape(datay0))
print('Test dataset')
print('X: ', np.shape(testx0))
print('Y: ', np.shape(testy0))
print('Validation dataset')
print('X: ', np.shape(valx))
print('Y: ', np.shape(valy))
print('Train dataset')
print('X: ', np.shape(trainx))
print('Y: ', np.shape(trainy))

# Validation data visualization
y_pred=model.predict(valx)
pred=np.argmax(y_pred,axis=1)
ground = np.argmax(valy,axis=1)
print(classification_report(ground,pred))

ConfusionMatrixDisplay.from_predictions(ground, pred, cmap=plt.cm.Blues, display_labels=Name)
plt.savefig(figure_dir+unique_dir+"confusion_matrix_val.png")
plt.show()

label_binarizer = LabelBinarizer().fit(trainy)
y_binary_test = label_binarizer.transform(valy)

fig, ax = plt.subplots(figsize=(6, 6))

colors = list(mcolors.TABLEAU_COLORS.keys()) + ["black"]
for class_id in range(len(Name)):
    RocCurveDisplay.from_predictions(
        y_binary_test[:, class_id],
        y_pred[:, class_id],
        color=colors[class_id],
        name=f"{reverse_mapping[class_id]}",
        ax=ax,
        plot_chance_level=(class_id == 2),
    )

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic\n One-vs-Rest")
plt.legend()
plt.grid()
plt.savefig(figure_dir+unique_dir+"roccurve_val.png")
plt.show()

# Test data visualization

y_pred=model.predict(testx0)
pred=np.argmax(y_pred,axis=1)
ground = np.argmax(testy,axis=1)
print(classification_report(ground,pred))

ConfusionMatrixDisplay.from_predictions(ground, pred, cmap=plt.cm.Blues, display_labels=Name)
plt.savefig(figure_dir+unique_dir+"confusion_matrix_test.png")
plt.show()

label_binarizer = LabelBinarizer().fit(trainy)
y_binary_test = label_binarizer.transform(testy)

fig, ax = plt.subplots(figsize=(6, 6))

colors = list(mcolors.TABLEAU_COLORS.keys()) + ["black"]
for class_id in range(len(Name)):
    RocCurveDisplay.from_predictions(
        y_binary_test[:, class_id],
        y_pred[:, class_id],
        color=colors[class_id],
        name=f"{reverse_mapping[class_id]}",
        ax=ax,
        plot_chance_level=(class_id == 2),
    )

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic\n One-vs-Rest")
plt.legend()
plt.grid()
plt.savefig(figure_dir+unique_dir+"roccurve_test.png")
plt.show()

# Visualize history

get_acc = history['accuracy']
value_acc = history['val_accuracy']
get_loss = history['loss']
validation_loss = history['val_loss']

epochs = np.arange(len(get_acc)) + 1
plt.figure()
plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.grid()
plt.savefig(figure_dir+unique_dir+"accuracy_training_validation.png")
plt.show()

epochs = np.arange(len(get_loss)) + 1
plt.figure()
plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss')
plt.grid()
plt.legend(loc=0)
plt.savefig(figure_dir+unique_dir+"loss_training_validation.png")
plt.show()