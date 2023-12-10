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

data_dir = 'data/CCSN_v2/'
model_dir = 'models/cnn/'
num_epochs = 20
augmentation = False
optimizer_name = "SGD"
lr = 0.001
momentum = 0.9
unique_model_dir = model_dir+'cloudnet_{}epochs_augment{}_optimizer{}_lr{}mom{}'.format(num_epochs, 
                                                                                        augmentation, 
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
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(227,227)) 
        image=img_to_array(image)
        # Normalize RGB values to between 0 and 1
        image=image/255.0
        datax0.append(image)
        datay0.append(count)
    count=count+1

# Shuffle index images
M=np.arange(len(datax0))
random.shuffle(M)
np.save(unique_model_dir+'/train_test_indices.npy', M)

# Convert to numpy array
datax1=np.array(datax0)
datay1=np.array(datay0)

# Split into training and test data using shuffled indices
trainx0, testx0, trainy0, testy0 = train_test_split(datax1, datay1, M, test_size=0.25)

# Create categorical output for training data
y_train=to_categorical(trainy0)

# Shuffle index images
K=np.arange(len(trainx0))
random.shuffle(K)
np.save(unique_model_dir+'/train_val_indices.npy', K)

# Split into training and validation data using shuffled indices
trainx,valx,trainy,valy=train_test_split(trainx0,y_train,K,test_size=0.2)

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

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), activation='relu', input_shape=(227, 227, 3), strides=(4,4)))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(units=9216, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=4096, activation='relu'))
model.add(layers.Dense(units=11, activation='softmax'))

model.summary()

if optimizer_name == "SGD":
    optimizer = optimizers.SGD(learning_rate=lr, momentum=momentum)
elif optimizer_name == "Adam":
    optimizer = optimizers.Adam(learning_rate=lr)
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



