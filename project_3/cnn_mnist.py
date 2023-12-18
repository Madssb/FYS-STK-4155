# Author: Ahmad Zagar
# https://www.kaggle.com/code/ahmadzargar/inceptionv3

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import random

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_dir = 'models/cnn/'
num_epochs = 5
augmentation = False
center = False
optimizer_name = "SGD"
lr = 0.001
momentum = 0.9
unique_model_dir = model_dir+'mnist_{}epochs_augment{}_center{}_optimizer{}_lr{}mom{}'.format(num_epochs, 
                                                                                        augmentation,
                                                                                        center, 
                                                                                        optimizer_name,
                                                                                        lr,
                                                                                        momentum)
if not os.path.exists(unique_model_dir):
   os.makedirs(unique_model_dir)

# Define function for splitting of training and test data
def train_test_split(X, Y, indices, test_size=0.2):
    n=len(X)
    stop_train = int(n*(1-test_size))
    trainx=X[indices[0:stop_train]]
    testx=X[indices[stop_train:]]
    trainy=Y[indices[0:stop_train]]
    testy=Y[indices[stop_train:]]

    return trainx, testx, trainy, testy

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Create categorical output for training data
y_train=to_categorical(y_train)

# Reshape images
X_train = x_train[:,:,:,np.newaxis]

# Shuffle index images
K=np.arange(len(x_train))
random.shuffle(K)
np.save(unique_model_dir+'/train_val_indices.npy', K)

# Split into training and validation data using shuffled indices
trainx,valx,trainy,valy=train_test_split(X_train,y_train,K,test_size=0.2)

# Center training data
if center==True:
    trainx = trainx - np.mean(trainx, axis=0, keepdims=True)
    valx = valx - np.mean(valx, axis=0, keepdims=True)

# Print shapes of train, validation and test data
print('Test dataset')
print('X: ', np.shape(x_test))
print('Y: ', np.shape(y_test))
print('Validation dataset')
print('X: ', np.shape(valx))
print('Y: ', np.shape(valy))
print('Train dataset')
print('X: ', np.shape(trainx))
print('Y: ', np.shape(trainy))

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1), strides=(1,1)))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(units=1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.summary()

if optimizer_name == "SGD":
    optimizer = optimizers.SGD(learning_rate=lr, momentum=momentum)
elif optimizer_name == "Adam":
    optimizer = optimizers.Adam(learning_rate=lr)
elif optimizer_name == "AdaGrad":
    optimizer = optimizers.Adagrad(learning_rate=lr)
else:
    print("Supply optimizer name 'SGD' or 'Adam'")

#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

if augmentation:
    # Generate more variations of images
    n = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                        width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")
    hist=model.fit(n.flow(trainx,trainy,batch_size=32),validation_data=(valx,valy),epochs=num_epochs)
else:
    hist=model.fit(trainx,trainy,validation_data=(valx,valy),epochs=num_epochs)

# Save model
model.save(unique_model_dir+'/model.keras')

# Save model training history
hist_df = pd.DataFrame(hist.history)
hist_csv_file = unique_model_dir+'/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)



