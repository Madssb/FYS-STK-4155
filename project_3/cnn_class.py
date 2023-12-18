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

np.random.seed(2023)

# If running on ML node set desired GPU id
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[2], 'GPU')
    gpu_id = '/gpu:2'


def train_test_split(X, Y, indices, test_size=0.2):
    n=len(X)
    stop_train = int(n*(1-test_size))
    trainx=X[indices[0:stop_train]]
    testx=X[indices[stop_train:]]
    trainy=Y[indices[0:stop_train]]
    testy=Y[indices[stop_train:]]

    return trainx, testx, trainy, testy

class CNNClassifier:
    def __init__(self, data_dir='data/CCSN_v2/', model_dir='models/cnn/', num_epochs=400,
                 augmentation=True, optimizer_name="AdaGrad", lr=0.01, momentum=0,
                 batch_size=8, target_size=(227, 227), lr_period=1):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.augmentation = augmentation
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.model = None
        self.target_size = target_size
        self.callbacks = None
        self.period = num_epochs/lr_period
        self.workers = 12# os.cpu_count()/4 # setting half avaible cpu cores
        np.random.seed(2023)
        self.unique_model_dir = model_dir+'cloudnet{}_{}epochs_augment{}_optimizer{}_lr{}mom{}batch{}'.format(target_size[0],
                                                                                        num_epochs, 
                                                                                        augmentation, 
                                                                                        optimizer_name,
                                                                                        lr,
                                                                                        momentum,
                                                                                        batch_size)
        if not os.path.exists(self.unique_model_dir):
            os.makedirs(self.unique_model_dir)

    def load_data(self):
        # Generate labels from folder names
        Name=[]
        for file in os.listdir(self.data_dir):
            Name+=[file]
        N=[]
        for i in range(len(Name)):
            N+=[i]

        datax0=[]
        datay0=[]
        count=0
        for file in Name:
            path=os.path.join(self.data_dir,file)
            for im in os.listdir(path):
                # Load images, rescaling to 227 x 227
                image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=self.target_size) 
                image=img_to_array(image)
                # Normalize RGB values to between 0 and 1
                image=image/255.0
                datax0.append(image)
                datay0.append(count)
            count=count+1

        M = np.arange(len(datax0))
        random.shuffle(M)
        np.save(self.unique_model_dir+'/train_test_indices.npy', M)

        datax1 = np.array(datax0)
        datay1 = np.array(datay0)
        # Split into training and test data using shuffled indices
        trainx0, testx0, trainy0, testy0 = train_test_split(datax1, datay1, M, test_size=0.2)

        # Create categorical output for training data
        y_train = to_categorical(trainy0)

        # Shuffle index images
        K=np.arange(len(trainx0))
        random.shuffle(K)
        np.save(self.unique_model_dir+'/train_val_indices.npy', K)

        # Split into training and validation data using shuffled indices
        self.trainx, self.valx, self.trainy, self.valy = train_test_split(trainx0, y_train, K, test_size=0.2)

    def create_model(self, summary=True):
        self.model = models.Sequential()
        pixels = self.target_size[0]
        # Add model layers
        self.model = tf.keras.models.Sequential()
        self.model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), activation='relu', input_shape=(pixels, pixels, 3), strides=(4,4)))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        self.model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        self.model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

        self.model.add(layers.Flatten())
        if pixels == 128:
            self.model.add(layers.Dense(units=1024, activation='relu'))
            self.model.add(layers.Dropout(0.5))
            self.model.add(layers.Dense(units=512, activation='relu'))
        elif pixels == 227:
            self.model.add(layers.Dense(units=9216, activation='relu'))
            self.model.add(layers.Dropout(0.5))
            self.model.add(layers.Dense(units=4096, activation='relu'))
        self.model.add(layers.Dense(units=11, activation='softmax'))
        if summary == True:
            self.model.summary()

    def compile_model(self, scheduler='period_dec', period=100):
        if self.optimizer_name == "SGD":
            optimizer = optimizers.SGD(learning_rate=self.lr, momentum=self.momentum)
            # setting lr scheduler
            if scheduler == 'period_dec':
                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.period_dec)
                self.callbacks = [lr_scheduler]
            elif scheduler == 'exp_dec':
                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.exp_dec)
                self.callbacks = [lr_scheduler]
        elif self.optimizer_name == "Adam":
            optimizer = optimizers.Adam(learning_rate=self.lr)
        elif self.optimizer_name == "AdaGrad":
            optimizer = optimizers.Adagrad(learning_rate=self.lr)
        else:
            raise ValueError("Supply optimizer name 'SGD', 'Adam', or 'AdaGrad'")
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        if self.augmentation:
            # Data augmentation
            n = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                                width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest",
                                featurewise_center=True, featurewise_std_normalization=True)
            n.fit(self.trainx)
            train_gen = n.flow(self.trainx, self.trainy, batch_size=self.batch_size)#, subset='training')
            val_gen =  n.flow(self.valx, self.valy, batch_size=self.batch_size)#, subset='validation')
            if gpus:
                with tf.device(gpu_id):
                    self.hist = self.model.fit(train_gen, validation_data=val_gen, epochs=self.num_epochs, use_multiprocessing=True,
                                callbacks=self.callbacks, verbose=2, workers=self.workers)
            else:
                self.hist = self.model.fit(train_gen, validation_data=val_gen, epochs=self.num_epochs, use_multiprocessing=True,
                                callbacks=self.callbacks, verbose=2, workers=self.workers)

        else:
            if gpus:
                with tf.device(gpu_id):
                    self.hist = self.model.fit(self.trainx, self.trainy, validation_data=(self.valx, self.valy), epochs=self.num_epochs,
                     callbacks=self.callbacks, batch_size=self.batch_size)
            else:
                self.hist = self.model.fit(self.trainx, self.trainy, validation_data=(self.valx, self.valy), epochs=self.num_epochs,
                 callbacks=self.callbacks, batch_size=self.batch_size)

    def save_model(self):
        # save model
        self.model.save(self.unique_model_dir+'/model.keras')

        # Save model training history
        hist_df = pd.DataFrame(self.hist.history)
        hist_csv_file = self.unique_model_dir+'/history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        print(hist_df['lr'])

    # Additional methods
    def exp_dec(self, epoch):
        """ exponential lr decay by 0.1 over a period """
        return lr*0.1**(epoch/self.period)

    def period_dec(self, epoch):
        """ decrease lr by factor 10 ever period number of epochs """
        if epoch > 0 and epoch % self.period == 0:
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            return lr/10
        else:
            return tf.keras.backend.get_value(self.model.optimizer.lr)


# def exp_dec(epoch):
#     return lr*0.1**(epoch/100)

# def period_dec(epoch):
#     if epoch > 0 and epoch % 100 == 0:
#         lr = tf.keras.backend.get_value(model.optimizer.lr)
#         return lr/10
#     else:
#         return tf.keras.backend.get_value(model.optimizer.lr)

# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exp_dec)
# # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(period_dec)
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy',
#                                               min_delta=1e-4,
#                                               patience=25,
#                                               verbose=1)

# callbacks = [early_stop]
# # callbacks = [lr_scheduler]
# # callbacks = [early_stop, lr_scheduler]

# data_dir = 'data/CCSN_v2/'
# model_dir = 'models/cnn/'
# num_epochs = 400
# augmentation = True
# optimizer_name = "AdaGrad"
# lr = 0.01
# momentum = 0
# batch_size = 8
# unique_model_dir = model_dir+'cloudnet128_{}epochs_augment{}_optimizer{}_lr{}mom{}batch{}'.format(num_epochs, 
#                                                                                         augmentation, 
#                                                                                         optimizer_name,
#                                                                                         lr,
#                                                                                         momentum,
#                                                                                         batch_size)
# if not os.path.exists(unique_model_dir):
#    os.makedirs(unique_model_dir)

# # Define function for splitting of training and test data
# def train_test_split(X, Y, indices, test_size=0.2):
#     n=len(X)
#     stop_train = int(n*(1-test_size))
#     trainx=X[indices[0:stop_train]]
#     testx=X[indices[stop_train:]]
#     trainy=Y[indices[0:stop_train]]
#     testy=Y[indices[stop_train:]]

#     return trainx, testx, trainy, testy

# # Generate labels from folder names
# Name=[]
# for file in os.listdir(data_dir):
#     Name+=[file]

# N=[]
# for i in range(len(Name)):
#     N+=[i]
    
# normal_mapping=dict(zip(Name,N)) 
# reverse_mapping=dict(zip(N,Name)) 

# print(normal_mapping)
# print(reverse_mapping)

# # Load images

# datax0=[]
# datay0=[]
# count=0
# for file in Name:
#     path=os.path.join(data_dir,file)
#     for im in os.listdir(path):
#         # Load images, rescaling to 227 x 227
#         image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(128, 128)) 
#         image=img_to_array(image)
#         # Normalize RGB values to between 0 and 1
#         image=image/255.0
#         datax0.append(image)
#         datay0.append(count)
#     count=count+1

# # Shuffle index images
# M=np.arange(len(datax0))
# random.shuffle(M)
# np.save(unique_model_dir+'/train_test_indices.npy', M)

# # Convert to numpy array
# datax1=np.array(datax0)
# datay1=np.array(datay0)
# # y_train=to_categorical(datay1)
# # Split into training and test data using shuffled indices
# trainx0, testx0, trainy0, testy0 = train_test_split(datax1, datay1, M, test_size=0.2)

# # Create categorical output for training data
# y_train=to_categorical(trainy0)

# # Shuffle index images
# K=np.arange(len(trainx0))
# random.shuffle(K)
# np.save(unique_model_dir+'/train_val_indices.npy', K)

# # Split into training and validation data using shuffled indices
# trainx,valx,trainy,valy=train_test_split(trainx0,y_train,K,test_size=0.2)

# # Print shapes of train, validation and test data
# print('Whole dataset')
# print('X: ', np.shape(datax0))
# print('Y: ', np.shape(datay0))
# print('Test dataset')
# print('X: ', np.shape(testx0))
# print('Y: ', np.shape(testy0))
# print('Validation dataset')
# print('X: ', np.shape(valx))
# print('Y: ', np.shape(valy))
# print('Train dataset')
# print('X: ', np.shape(trainx))
# print('Y: ', np.shape(trainy))

# model = tf.keras.models.Sequential()
# model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), activation='relu', input_shape=(128, 128, 3), strides=(4,4)))
# model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'))
# model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

# model.add(layers.Flatten())
# model.add(layers.Dense(units=1024, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=11, activation='softmax'))

# model.summary()
# # exit()
# if optimizer_name == "SGD":
#     optimizer = optimizers.SGD(learning_rate=lr, momentum=momentum)
# elif optimizer_name == "Adam":
#     optimizer = optimizers.Adam(learning_rate=lr)
# elif optimizer_name == "AdaGrad":
#     optimizer = optimizers.Adagrad(learning_rate=lr)
# else:
#     print("Supply optimizer name 'SGD' or 'Adam'")

# #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

# if augmentation:
#     # Generate more variations of images
#     n = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
#                         width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest",
#                         featurewise_center=True, featurewise_std_normalization=True)
#     n.fit(trainx)
#     train_gen = n.flow(trainx, trainy, batch_size=batch_size)#, subset='training')
#     val_gen =  n.flow(valx, valy, batch_size=batch_size)#, subset='validation')
#     with tf.device(gpu_id):
#         hist=model.fit(train_gen, validation_data=val_gen, epochs=num_epochs, use_multiprocessing=True,
#                        callbacks=callbacks, verbose=2, workers=12)
# else:
#     with tf.device(gpu_id):
#         hist=model.fit(trainx,trainy,validation_data=(valx,valy),epochs=num_epochs, callbacks=callbacks)

# # Save model
# model.save(unique_model_dir+'/model.keras')

# # Save model training history
# hist_df = pd.DataFrame(hist.history)
# hist_csv_file = unique_model_dir+'/history.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)

if __name__ == '__main__':
    # cnn = CNNClassifier(num_epochs=400, optimizer_name='SGD', lr=0.001, momentum=0.9, batch_size=32, lr_period=4, augmentation=True)
    # cnn.load_data()
    # cnn.create_model()
    # cnn.compile_model()
    # cnn.train_model()
    # cnn.save_model()
    cnn = CNNClassifier(num_epochs=400, optimizer_name='SGD', lr=0.001, momentum=0.9, batch_size=32, lr_period=4, augmentation=False)
    cnn.load_data()
    cnn.create_model()
    cnn.compile_model()
    cnn.train_model()
    cnn.save_model()

